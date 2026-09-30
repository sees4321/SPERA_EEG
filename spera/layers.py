from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from .config import EEGModelConfig


def build_rope_cache(
    max_pos: int,
    rotary_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert rotary_dim % 2 == 0
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    if cos.dim() == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.dim() == 3:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class TimePatchEmbed(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.patch_samples = int(round(cfg.sample_rate * cfg.patch_seconds))
        self.proj = nn.Linear(self.patch_samples, cfg.d_model)

    def forward_packed(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.fc = nn.Linear(d_model, 2 * hidden)
        self.proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.drop(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        rope_theta: float,
        rotary_pct: float,
        spatial_qk_dim: int = 0,
        spatial_qk_scale: float = 1.0,
        use_spatial_qk: bool = False,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = float(attn_dropout)
        self.spatial_qk_scale = float(spatial_qk_scale)

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        if self.rotary_dim > 0:
            cos, sin = build_rope_cache(
                max_pos=max_seq_len,
                rotary_dim=self.rotary_dim,
                theta=float(rope_theta),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        if use_spatial_qk:
            self.spatial_q_proj = nn.Linear(spatial_qk_dim, d_model, bias=False)
            self.spatial_k_proj = nn.Linear(spatial_qk_dim, d_model, bias=False)
        else:
            self.spatial_q_proj = None
            self.spatial_k_proj = None

    def _get_rope(self, rope_pos: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._rope_cos.to(dtype=dtype)
        sin = self._rope_sin.to(dtype=dtype)
        if rope_pos.dim() == 1:
            return cos[rope_pos], sin[rope_pos]
        if rope_pos.dim() == 2:
            return cos[rope_pos], sin[rope_pos]
        raise ValueError(f"rope_pos must be 1D or 2D, got {rope_pos.shape}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        spatial_q_add: Optional[torch.Tensor] = None,
        spatial_k_add: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary_dim > 0:
            # RoPE is applied along the temporal patch index. 
            # Channel geometry is handled separately by coordinate embeddings and Legendre priors.
            if rope_pos is None:
                raise ValueError("rope_pos must be provided when rotary_dim > 0")
            cos, sin = self._get_rope(rope_pos, dtype=q.dtype)
            if self.rotary_dim == self.head_dim:
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)
            else:
                q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
                k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
                q = torch.cat([apply_rope(q_rot, cos, sin), q_pass], dim=-1)
                k = torch.cat([apply_rope(k_rot, cos, sin), k_pass], dim=-1)

        if (spatial_q_add is not None) or (spatial_k_add is not None):
            # Legendre anchor geometry features enter Q/K after RoPE (Eq. 4).
            # The resulting scores include content–geometry cross terms (F.3).
            if spatial_q_add is None or spatial_k_add is None:
                raise ValueError("spatial_q_add and spatial_k_add must be provided together")
            q_sp = spatial_q_add.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k_sp = spatial_k_add.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            q = q + self.spatial_qk_scale * q_sp
            k = k + self.spatial_qk_scale * k_sp

        attn_mask = None
        if attn_bias is None:
            if padding_mask is not None:
                attn_mask = (~padding_mask)[:, None, None, :]
        else:
            # Optional pairwise spatial bias, used mainly in factorized spatial attention where attention is only C x C per time step.
            attn_mask = attn_bias[:, None, :, :] if attn_bias.dim() == 3 else attn_bias
            if attn_mask.dtype != q.dtype:
                attn_mask = attn_mask.to(dtype=q.dtype)
            if padding_mask is not None:
                attn_mask = attn_mask.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)
    

class CrossAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, rope_theta: float, rotary_pct: float, max_seq_len: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = float(attn_dropout)

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        if self.rotary_dim > 0:
            cos, sin = build_rope_cache(
                max_pos=max_seq_len,
                rotary_dim=self.rotary_dim,
                theta=float(rope_theta),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

        self.q = nn.Linear(d_model, d_model, bias=True)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def _get_rope(self, rope_pos: torch.Tensor, dtype: torch.dtype):
        cos = self._rope_cos.to(dtype=dtype)
        sin = self._rope_sin.to(dtype=dtype)
        if rope_pos.dim() == 1:
            return cos[rope_pos], sin[rope_pos]
        if rope_pos.dim() == 2:
            return cos[rope_pos], sin[rope_pos]
        raise ValueError(f"rope_pos must be 1D or 2D, got {rope_pos.shape}")

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor],
        rope_pos_q: torch.Tensor,
        rope_pos_k: torch.Tensor,
    ) -> torch.Tensor:
        B, Lq, D = q_in.shape
        _, Lk, _ = kv_in.shape

        q = self.q(q_in).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv_in)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary_dim > 0:
            cos_q, sin_q = self._get_rope(rope_pos_q, dtype=q.dtype)
            cos_k, sin_k = self._get_rope(rope_pos_k, dtype=q.dtype)
            if self.rotary_dim == self.head_dim:
                q = apply_rope(q, cos_q, sin_q)
                k = apply_rope(k, cos_k, sin_k)
            else:
                q = torch.cat([apply_rope(q[..., : self.rotary_dim], cos_q, sin_q), q[..., self.rotary_dim :]], dim=-1)
                k = torch.cat([apply_rope(k[..., : self.rotary_dim], cos_k, sin_k), k[..., self.rotary_dim :]], dim=-1)

        attn_mask = None
        if kv_padding_mask is not None:
            attn_mask = (~kv_padding_mask)[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)
