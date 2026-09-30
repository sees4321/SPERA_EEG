import torch
import torch.nn as nn

from .config import EEGModelConfig
from .layers import CrossAttentionRoPE, SwiGLU


class CrossAttnPredictorBlock(nn.Module):
    # Cross-attention predictor block used to map context latents to target latent predictions, rather than reconstructing raw EEG patches.
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        d_model = cfg.d_model
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.xattn = CrossAttentionRoPE(
            d_model=d_model,
            n_heads=cfg.predictor_n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            max_seq_len=cfg.max_tokens,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = SwiGLU(d_model, cfg.predictor_mlp_ratio, cfg.dropout)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, ctx_pad: torch.Tensor, rope_q: torch.Tensor, rope_ctx: torch.Tensor) -> torch.Tensor:
        q = q + self.drop(self.xattn(self.norm1(q), ctx, ctx_pad, rope_pos_q=rope_q, rope_pos_k=rope_ctx))
        q = q + self.drop(self.mlp(self.norm2(q)))
        return q


class CrossAttentionPredictor(nn.Module):
    # JEPA predictor: 
    # Initialize each target query as a shared learnable token plus the target coordinate embedding.
    # Then refine it by cross-attending to context latents from the student encoder.
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        # Shared target query token used for all masked target positions; 
        # spatial specificity is provided by the target coordinate embedding.
        self.query_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.query_token, std=float(cfg.query_token_init_std))
        
        self.blocks = nn.ModuleList([CrossAttnPredictorBlock(cfg) for _ in range(int(cfg.predictor_layers))])
        self.norm = nn.LayerNorm(cfg.d_model, eps=1e-6)

    def forward(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        rope_ctx: torch.Tensor,
        tgt_coord_emb: torch.Tensor,
        tgt_pad: torch.Tensor,
        rope_tgt: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_token[None, None, :].to(tgt_coord_emb.dtype) + tgt_coord_emb
        q = q.masked_fill(tgt_pad[..., None], 0.0)
        for blk in self.blocks:
            q = blk(q, ctx, ctx_pad, rope_q=rope_tgt, rope_ctx=rope_ctx)
        q = self.norm(q)
        return q.masked_fill(tgt_pad[..., None], 0.0)