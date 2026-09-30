from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .config import EEGModelConfig


class CoordMLPEmbedding(nn.Module):
    # Direct coordinate embedding used as an additive per-channel positional signal.
    # Coordinates are projected after unit-sphere normalization and shared across all temporal patches of the same electrode.
    def __init__(self, d_model: int, w_init: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        self.emb_w = nn.Parameter(torch.tensor([w_init], dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = F.normalize(coords, p=2, dim=-1)
        return self.proj(coords) * self.emb_w.to(dtype=coords.dtype, device=coords.device)


class PairwiseLegendreBias(nn.Module):
    # Pairwise Legendre spatial bias, corresponding to Eq. (2) in the paper.
    # It parameterizes a learnable rotation-invariant function of the cosine similarity between two unit-normalized electrode coordinates.
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.enabled = str(cfg.spatial_bias).lower() not in ("none", "off", "disable", "disabled")
        self.use_unit = bool(cfg.spatial_bias_use_unit_sphere)
        self.degree = int(cfg.spatial_bias_degree)
        self.scale = float(cfg.spatial_bias_scale)
        self.eps = 1e-6
        if self.enabled:
            self.coeff = nn.Parameter(torch.zeros(self.degree + 1))
        else:
            self.register_parameter("coeff", None)

    def _cosine(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_unit:
            u = coords / coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            u = coords
        return torch.matmul(u, u.transpose(1, 2)).clamp(-1.0, 1.0)

    def forward(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        x = self._cosine(coords.float())
        Pm2 = torch.ones_like(x)
        bias = self.coeff[0].to(x.dtype) * Pm2
        if self.degree >= 1:
            Pm1 = x
            bias = bias + self.coeff[1].to(x.dtype) * Pm1
            # Recursively compute P_l(x) to avoid explicitly materializing polynomial bases with external dependencies.
            for l in range(2, self.degree + 1):
                Pl = ((2 * l - 1) * x * Pm1 - (l - 1) * Pm2) / float(l)
                bias = bias + self.coeff[l].to(x.dtype) * Pl
                Pm2, Pm1 = Pm1, Pl
        if self.scale != 1.0:
            bias = bias * self.scale
        return bias


class LegendreAnchor(nn.Module):
    """Legendre anchor geometry features phi_c, Eq. (3).

    Anchor directions and coefficients are shared across layers and heads.
    Each attention block projects these features into Q/K after temporal RoPE
    (Eq. (4)). This produces geometry–geometry and content–geometry score terms
    (Appendix F.3); it is not a low-rank approximation of the pairwise bias.
    """
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.num_anchors = int(cfg.spatial_ak_num_anchors)
        self.degree = int(cfg.spatial_ak_degree)
        self.out_dim = int(cfg.spatial_ak_feat_dim)
        self.use_unit = bool(cfg.spatial_bias_use_unit_sphere)
        self.eps = 1e-6

        anchors = torch.randn(self.num_anchors, 3)
        anchors = F.normalize(anchors, p=2, dim=-1)
        self.anchors = nn.Parameter(anchors)
        self.coeff = nn.Parameter(torch.zeros(self.num_anchors, self.degree + 1))
        self.proj = nn.Linear(self.num_anchors, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim, eps=1e-6)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_unit:
            u = coords / coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            u = coords

        a = F.normalize(self.anchors, p=2, dim=-1)
        x = torch.einsum("bcd,rd->bcr", u.float(), a.float()).clamp(-1.0, 1.0)

        basis = []
        Pm2 = torch.ones_like(x)
        basis.append(Pm2)
        if self.degree >= 1:
            Pm1 = x
            basis.append(Pm1)
            for l in range(2, self.degree + 1):
                Pl = ((2 * l - 1) * x * Pm1 - (l - 1) * Pm2) / float(l)
                basis.append(Pl)
                Pm2, Pm1 = Pm1, Pl

        basis = torch.stack(basis, dim=-1)
        coeff = self.coeff[None, None, :, :].to(dtype=basis.dtype, device=basis.device)
        feat = (basis * coeff).sum(dim=-1)
        feat = self.proj(feat)
        feat = self.norm(feat)
        return feat.to(coords.dtype)

LegendreAnchorKernel = LegendreAnchor