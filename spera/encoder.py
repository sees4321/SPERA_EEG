from __future__ import annotations

import json
import os
import torch
import torch.nn as nn

from typing import Optional, Tuple

from .config import EEGModelConfig
from .layers import MultiheadSelfAttentionRoPE, SwiGLU, TimePatchEmbed
from .spatial import CoordMLPEmbedding, PairwiseLegendreBias, LegendreAnchor


def _gather_channel_features(x_ch: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
    B, C, Fdim = x_ch.shape
    B2, L = c_idx.shape
    assert B == B2
    idx = c_idx[..., None].expand(B, L, Fdim)
    return x_ch.gather(dim=1, index=idx)


class FullAttentionBlock(nn.Module):
    # Periodic full-attention block: 
    # attends jointly over all C * P_t tokens to allow cross-axis interactions that purely factorized blocks cannot model.
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.use_spatial_bias = bool(cfg.full_attn_use_spatial_bias)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.attn = MultiheadSelfAttentionRoPE(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            spatial_qk_dim=cfg.spatial_ak_feat_dim,
            spatial_qk_scale=cfg.spatial_ak_scale,
            use_spatial_qk=(str(cfg.spatial_ak).lower() == "legendre_anchor"),
            max_seq_len=cfg.max_tokens,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        rope_pos: torch.Tensor,
        chan_idx: torch.Tensor,
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],
        grid_channels: Optional[int] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        attn_bias = None
        spatial_q_add = None
        spatial_k_add = None

        if (spatial_qk_ch is not None) and (chan_idx is not None) and (self.attn.spatial_q_proj is not None):
            # Eq. (4): block-specific Q/K projections of shared geometry features.
            # Their addition after RoPE also creates content–geometry score terms.
            q_sp_ch = self.attn.spatial_q_proj(spatial_qk_ch)
            k_sp_ch = self.attn.spatial_k_proj(spatial_qk_ch)
            spatial_q_add = _gather_channel_features(q_sp_ch, chan_idx)
            spatial_k_add = _gather_channel_features(k_sp_ch, chan_idx)
            if padding_mask is not None:
                spatial_q_add = spatial_q_add.masked_fill(padding_mask[..., None], 0.0)
                spatial_k_add = spatial_k_add.masked_fill(padding_mask[..., None], 0.0)

        if self.use_spatial_bias and (spatial_bias_cc is not None) and (chan_idx is not None):
            c = chan_idx
            B, _ = c.shape
            b = torch.arange(B, device=c.device)[:, None, None]
            attn_bias = spatial_bias_cc[b, c[:, :, None], c[:, None, :]]

        y = self.attn(
            self.norm1(x),
            padding_mask=padding_mask,
            rope_pos=rope_pos,
            attn_bias=attn_bias,
            spatial_q_add=spatial_q_add,
            spatial_k_add=spatial_k_add,
        )
        x = x + self.dropout(y)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class DividedSpatiotemporalBlock(nn.Module):
    # Factorized spatio-temporal block: 
    # temporal attention is applied within each channel, followed by spatial attention across channels at each time patch.
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        D = cfg.d_model
        self.norm_t = nn.LayerNorm(D, eps=1e-6)
        self.attn_t = MultiheadSelfAttentionRoPE(
            d_model=D,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            max_seq_len=cfg.max_tokens,
        )

        self.norm_s = nn.LayerNorm(D, eps=1e-6)

        use_divided_spatial_qk = (
            (str(cfg.spatial_ak).lower() == "legendre_anchor")
            and (str(cfg.spatial_bias).lower() in ("none", "off", "disable", "disabled"))
        )
        # Anchor-only ablation: use geometry features in the spatial pass when the pairwise bias is disabled. 
        # Spatial attention has no temporal RoPE.
        self.attn_s = MultiheadSelfAttentionRoPE(
            d_model=D,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=0.0,
            spatial_qk_dim=cfg.spatial_ak_feat_dim,
            spatial_qk_scale=cfg.spatial_ak_scale,
            use_spatial_qk=use_divided_spatial_qk,
            max_seq_len=cfg.max_tokens,
        )

        self.norm_m = nn.LayerNorm(D, eps=1e-6)
        self.mlp = SwiGLU(D, cfg.mlp_ratio, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self._batch_rows_cache = {}
        self._temporal_rope_cache = {}

    @staticmethod
    def _device_key(device: torch.device):
        return (device.type, -1 if device.index is None else int(device.index))

    def _get_batch_rows(self, batch_size: int, device: torch.device) -> torch.Tensor:
        key = (self._device_key(device), int(batch_size))
        rows = self._batch_rows_cache.get(key)
        if rows is None:
            rows = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
            self._batch_rows_cache[key] = rows
        return rows

    def _get_temporal_rope(self, P: int, device: torch.device) -> torch.Tensor:
        key = (self._device_key(device), int(P))
        rope = self._temporal_rope_cache.get(key)
        if rope is None:
            rope = torch.arange(P, device=device, dtype=torch.long)
            self._temporal_rope_cache[key] = rope
        return rope

    def _scatter_to_grid(self, x: torch.Tensor, pad: torch.Tensor, c_idx: torch.Tensor, t_idx: torch.Tensor, C: int, P: int, b_rows: torch.Tensor):
        B, L, D = x.shape
        grid = x.new_zeros((B, C, P + 1, D))
        grid_pad = torch.ones((B, C, P + 1), dtype=torch.bool, device=x.device)
        b = b_rows.expand(B, L)
        trash = torch.full_like(t_idx, P)
        t_safe = torch.where(pad, trash, t_idx)
        grid[b, c_idx, t_safe] = x
        grid_pad[b, c_idx, t_safe] = pad
        return grid[:, :, :P, :], grid_pad[:, :, :P]

    def _gather_from_grid(self, grid: torch.Tensor, pad: torch.Tensor, c: torch.Tensor, t: torch.Tensor, b_rows: torch.Tensor) -> torch.Tensor:
        B, L = c.shape
        out = grid[b_rows.expand(B, L), c, t]
        return out.masked_fill(pad[..., None], 0.0)

    def _temporal_from_grid(self, grid: torch.Tensor, grid_pad: torch.Tensor, P: int) -> torch.Tensor:
        B, C, P2, D = grid.shape
        assert P2 == P
        x_t = grid.reshape(B * C, P, D)
        pad_t = grid_pad.reshape(B * C, P)
        all_pad = pad_t.all(dim=1)
        pad_t_safe = pad_t.clone()
        pad_t_safe[:, 0] = pad_t_safe[:, 0] & (~all_pad)

        rope = self._get_temporal_rope(P, grid.device)
        y_t = self.attn_t(x_t, padding_mask=pad_t_safe, rope_pos=rope, attn_bias=None)
        y_t = y_t.masked_fill(all_pad[:, None, None], 0.0)
        return y_t.reshape(B, C, P, D)

    def _spatial_from_grid(self, grid: torch.Tensor, grid_pad: torch.Tensor, spatial_bias_cc: Optional[torch.Tensor], spatial_qk_ch: Optional[torch.Tensor], P: int) -> torch.Tensor:
        B, C, P2, D = grid.shape
        assert P2 == P

        grid_tp = grid.permute(0, 2, 1, 3).contiguous()
        pad_tp = grid_pad.permute(0, 2, 1).contiguous()
        x_s = grid_tp.reshape(B * P, C, D)
        pad_s = pad_tp.reshape(B * P, C)

        all_pad = pad_s.all(dim=1)
        pad_s_safe = pad_s.clone()
        pad_s_safe[:, 0] = pad_s_safe[:, 0] & (~all_pad)

        bias = None
        spatial_q_add = None
        spatial_k_add = None
        if spatial_bias_cc is not None:
            bias = spatial_bias_cc[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, C)
        elif (spatial_qk_ch is not None) and (self.attn_s.spatial_q_proj is not None):
            q_sp_ch = self.attn_s.spatial_q_proj(spatial_qk_ch)
            k_sp_ch = self.attn_s.spatial_k_proj(spatial_qk_ch)
            spatial_q_add = q_sp_ch[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, D)
            spatial_k_add = k_sp_ch[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, D)

        y_s = self.attn_s(
            x_s,
            padding_mask=pad_s_safe,
            rope_pos=None,
            attn_bias=bias,
            spatial_q_add=spatial_q_add,
            spatial_k_add=spatial_k_add,
        )
        y_s = y_s.masked_fill(all_pad[:, None, None], 0.0)
        return y_s.reshape(B, P, C, D).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: torch.Tensor,
        chan_idx: Optional[torch.Tensor],
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],
        grid_channels: Optional[int] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if chan_idx is None:
            raise ValueError("DividedSpatiotemporalBlock requires chan_idx")

        b_rows = self._get_batch_rows(int(x.shape[0]), x.device)

        if grid_channels is not None:
            C = int(grid_channels)
        elif spatial_bias_cc is not None:
            C = int(spatial_bias_cc.shape[1])
        elif spatial_qk_ch is not None:
            C = int(spatial_qk_ch.shape[1])
        else:
            C = int(chan_idx.max().item()) + 1

        P = int(grid_patches) if grid_patches is not None else int(rope_pos.max().item()) + 1
        grid, grid_pad = self._scatter_to_grid(x, padding_mask, chan_idx, rope_pos, C=C, P=P, b_rows=b_rows)

        grid_normed = self.norm_t(grid).masked_fill(grid_pad[..., None], 0.0)
        grid = grid + self.dropout(self._temporal_from_grid(grid_normed, grid_pad, P=P))

        grid_normed = self.norm_s(grid).masked_fill(grid_pad[..., None], 0.0)
        grid = grid + self.dropout(self._spatial_from_grid(grid_normed, grid_pad, spatial_bias_cc=spatial_bias_cc, spatial_qk_ch=spatial_qk_ch, P=P))

        grid_normed = self.norm_m(grid).masked_fill(grid_pad[..., None], 0.0)
        grid = grid + self.dropout(self.mlp(grid_normed))
        return self._gather_from_grid(grid, padding_mask, chan_idx, rope_pos, b_rows=b_rows)


class EEGEncoder(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_samples = int(round(cfg.sample_rate * cfg.patch_seconds))

        self.time_embed = TimePatchEmbed(cfg)
        self.coord_embed = CoordMLPEmbedding(
            d_model=cfg.d_model,
            w_init=cfg.coord_w_init,
        )

        self.spatial_qk_feat = None
        if str(cfg.spatial_ak).lower() == "legendre_anchor":
            self.spatial_qk_feat = LegendreAnchor(cfg)

        self.spatial_bias = PairwiseLegendreBias(cfg)

        arch = str(cfg.encoder_arch).lower()
        full_every = int(cfg.full_attn_every)
        blocks = []
        if arch == "full":
            blocks = [FullAttentionBlock(cfg) for _ in range(cfg.n_layers)]
        elif arch == "factorized":
            blocks = [DividedSpatiotemporalBlock(cfg) for _ in range(cfg.n_layers)]
        elif arch == "hybrid":
            # Hybrid backbone: 
            # mostly factorized blocks, with periodic full-attention blocks controlled by full_attn_every.
            for i in range(cfg.n_layers):
                is_full = (full_every > 0) and ((i + 1) % full_every == 0)
                blocks.append(FullAttentionBlock(cfg) if is_full else DividedSpatiotemporalBlock(cfg))
        else:
            raise ValueError(f"Unknown encoder_arch: {arch}")
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model, eps=1e-6)

    def extract_patches_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(dimension=-1, size=self.patch_samples, step=self.patch_samples)

    @staticmethod
    def _safe_gather_channel(x: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape
        B2, L = c_idx.shape
        assert B == B2
        idx = c_idx[..., None].expand(B, L, D)
        return x.gather(dim=1, index=idx)
    
    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, ensure_ascii=False)
        torch.save(self.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    @staticmethod
    def from_pretrained(path: str, map_location: str = "cpu") -> "EEGEncoder":
        cfg = EEGModelConfig.from_json(os.path.join(path, "config.json"))
        model = EEGEncoder(cfg)
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location, weights_only=True)
        model.load_state_dict(sd, strict=True)
        return model

    def embed_from_indices(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        c_idx: torch.Tensor,
        t_idx: torch.Tensor,
        pad: torch.Tensor,
        coord_ch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Materialize only the selected packed tokens. 
        # Context and target sets are represented by their channel/time indices instead of dense C x P_t tensors.
        device = x.device
        B, _, _ = x.shape
        B2, L = c_idx.shape
        assert B == B2
        c_safe = c_idx.clamp(min=0)
        t_safe = t_idx.clamp(min=0)

        if coord_ch is None:
            coord_ch = self.coord_embed(coords)
        coord_tok = self._safe_gather_channel(coord_ch, c_safe)
        coord_tok = coord_tok.masked_fill(pad[..., None], 0.0)

        patches_view = self.extract_patches_view(x)
        b_idx = torch.arange(B, device=device)[:, None].expand(B, L)
        patches = patches_view[b_idx, c_safe, t_safe]
        patches = patches.masked_fill(pad[..., None], 0.0)

        tok = self.time_embed.forward_packed(patches) + coord_tok
        tok = tok.masked_fill(pad[..., None], 0.0)
        return tok, pad, t_safe, c_safe

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        rope_pos: torch.Tensor,
        chan_idx: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        spatial_bias_cc = None
        spatial_qk_ch = None
        grid_channels = None

        if coords is not None:
            grid_channels = int(coords.shape[1])
            spatial_bias_cc = self.spatial_bias(coords)
            if spatial_bias_cc is not None and (spatial_bias_cc.device != tokens.device or spatial_bias_cc.dtype != tokens.dtype):
                spatial_bias_cc = spatial_bias_cc.to(dtype=tokens.dtype, device=tokens.device)

            if self.spatial_qk_feat is not None:
                spatial_qk_ch = self.spatial_qk_feat(coords)
                if spatial_qk_ch.device != tokens.device or spatial_qk_ch.dtype != tokens.dtype:
                    spatial_qk_ch = spatial_qk_ch.to(dtype=tokens.dtype, device=tokens.device)

        if grid_patches is None and rope_pos.numel() > 0:
            grid_patches = int(rope_pos.max().item()) + 1

        x = tokens
        for blk in self.blocks:
            x = blk(
                x,
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                chan_idx=chan_idx,
                spatial_bias_cc=spatial_bias_cc,
                spatial_qk_ch=spatial_qk_ch,
                grid_channels=grid_channels,
                grid_patches=grid_patches,
            )
        x = self.norm(x)
        return x
    
    def forward_eeg(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        return_grid: bool = False,
    ) -> torch.Tensor:
        """
        Encode dense EEG input.

        Args:
            x:      EEG tensor of shape (B, C, T)
            coords: Electrode coordinates of shape (B, C, 3)
            return_grid: if True, return shape (B, C, P, D)

        Returns:
            Token features of shape (B, C * P, D), or (B, C, P, D) if return_grid=True.
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C, T), got {tuple(x.shape)}")
        if coords.ndim != 3:
            raise ValueError(f"coords must have shape (B, C, 3), got {tuple(coords.shape)}")

        B, C, _ = x.shape
        if coords.shape[0] != B or coords.shape[1] != C:
            raise ValueError(
                f"x and coords must agree on batch/channels: "
                f"x={tuple(x.shape)}, coords={tuple(coords.shape)}"
            )

        patches_view = self.extract_patches_view(x)
        P = int(patches_view.shape[2])
        if P <= 0:
            raise ValueError(
                f"Input is shorter than one patch: T={x.shape[-1]}, "
                f"patch_samples={self.patch_samples}"
            )

        if C * P > int(self.cfg.max_tokens):
            raise ValueError(
                f"Input has too many tokens: C * P = {C * P}, "
                f"but max_tokens={self.cfg.max_tokens}. "
                f"Please crop or split the input first."
            )

        device = x.device
        c_idx = torch.arange(C, device=device).repeat_interleave(P)
        t_idx = torch.arange(P, device=device).repeat(C)

        c_idx = c_idx[None, :].expand(B, -1)
        t_idx = t_idx[None, :].expand(B, -1)
        pad = torch.zeros((B, C * P), dtype=torch.bool, device=device)

        coord_ch = self.coord_embed(coords)
        tokens, pad, rope_pos, chan_idx = self.embed_from_indices(
            x=x,
            coords=coords,
            c_idx=c_idx,
            t_idx=t_idx,
            pad=pad,
            coord_ch=coord_ch,
        )

        z = self.forward_tokens(
            tokens=tokens,
            padding_mask=pad,
            rope_pos=rope_pos,
            chan_idx=chan_idx,
            coords=coords,
            grid_patches=P,
        )

        if return_grid:
            z = z.view(B, C, P, -1)

        return z

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        rope_pos: Optional[torch.Tensor] = None,
        chan_idx: Optional[torch.Tensor] = None,
        grid_patches: Optional[int] = None,
        return_grid: bool = False,
    ) -> torch.Tensor:
        """
        Usage:
            z = model(eeg, coords)  # eeg: (B, C, T), coords: (B, C, 3)
        Advanced usage for pretraining:
            z = model.forward_tokens(tokens, padding_mask, rope_pos, chan_idx, coords)
        """
        if padding_mask is None and rope_pos is None and chan_idx is None:
            if coords is None:
                raise ValueError("coords must be provided when forwarding raw EEG.")
            return self.forward_eeg(x, coords=coords, return_grid=return_grid)

        if padding_mask is None or rope_pos is None or chan_idx is None:
            raise ValueError(
                "padding_mask, rope_pos, and chan_idx must be provided together "
                "when forwarding pre-tokenized inputs."
            )

        return self.forward_tokens(
            tokens=x,
            padding_mask=padding_mask,
            rope_pos=rope_pos,
            chan_idx=chan_idx,
            coords=coords,
            grid_patches=grid_patches,
        )
