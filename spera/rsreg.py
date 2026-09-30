from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class ZProjector(nn.Module):
    # Project predicted latents into the RSReg relation space.
    # Nonzero weights are essential: an all-zero R gives zero gradients through
    # R @ R.T. The lambda schedule controls the onset of the auxiliary objective.
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, out_dim),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
@torch.no_grad()
def compute_logspec_view(
    patches: torch.Tensor,
    fs: int,
    f_min: float = 1.0,
    f_max: float = 45.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    # Spectral view for RSReg: 
    # Compute Hann-windowed log-power spectra over 1-45 Hz, then z-score across frequency bins. 
    # This target is used only to define pairwise spectral relations, not for direct spectrum reconstruction.
    x = patches - patches.mean(dim=-1, keepdim=True)
    win = torch.hann_window(x.shape[-1], periodic=True, device=x.device, dtype=x.dtype)
    x = x * win[None, :]
    X = torch.fft.rfft(x, dim=-1)
    P = (X.real ** 2 + X.imag ** 2).clamp_min(eps)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(fs)).to(device=x.device, dtype=torch.float32)
    sel = (freqs >= float(f_min)) & (freqs <= float(f_max))
    logp = torch.log(P[:, sel])
    logp = logp - logp.mean(dim=-1, keepdim=True)
    logp = logp / (logp.std(dim=-1, keepdim=True).clamp_min(1e-6))
    return logp


def relational_kl_loss(
    z: torch.Tensor, s: torch.Tensor, tau_z: float = 0.1, tau_s: float = 0.1,
    *, check_finite: bool = True,
) -> torch.Tensor:
    # RSReg aligns pairwise similarity distributions between projected predicted latents and spectral views. 
    # Diagonal entries are masked to avoid trivial self-similarity.
    
    z = z.float()
    s = s.float()
    n = int(z.shape[0])
    if n < 2:
        return z.new_zeros((), dtype=torch.float32)
    z_n = F.normalize(z, dim=-1, eps=1e-6)
    s_n = F.normalize(s, dim=-1, eps=1e-6)
    logits_z = (z_n @ z_n.T) / float(tau_z)
    logits_s = (s_n @ s_n.T) / float(tau_s)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    logits_z = logits_z.masked_fill(eye, float("-inf"))
    logits_s = logits_s.masked_fill(eye, float("-inf"))
    log_p = F.log_softmax(logits_z, dim=-1).masked_fill(eye, 0.0)
    log_q = F.log_softmax(logits_s, dim=-1).detach().masked_fill(eye, 0.0)
    loss = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
    if check_finite and not torch.isfinite(loss):
        raise FloatingPointError("spec_rel_loss became non-finite inside relational_kl_loss")
    return loss

def rsreg_weight(
    global_step: int,
    *,
    warmup_steps: int,
    ramp_steps: int,
    decay_step: int,
    max_steps: int,
    weight: float,
    final_weight: float,
) -> float:
    # Schedule lambda for L = L_JEPA + lambda * L_spec.
    # ramp_steps is a duration after warmup_steps, not the absolute peak step.
    warm = int(warmup_steps)
    ramp = int(ramp_steps)
    decay = int(decay_step)

    if global_step < warm:
        return 0.0

    if global_step < warm + ramp:
        u = min(1.0, float(global_step - warm) / float(max(1, ramp)))
        return float(weight) * u

    if global_step < decay:
        return float(weight)

    decay_span = max(1, int(max_steps) - decay)
    u = float(global_step - decay) / float(decay_span)
    u = min(max(u, 0.0), 1.0)
    return float(weight + (final_weight - weight) * u)


def _require_finite(ok: torch.Tensor, message: str, sync_distributed: bool) -> None:
    # Every rank checks at the same points, even when it has no eligible samples.
    # Fail together instead of allowing one rank to skip a DDP backward/reduction.
    flag = ok.detach().to(dtype=torch.int32)
    if sync_distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise FloatingPointError(message)


def compute_rsreg_sum(
    *,
    rel_proj: nn.Module,
    pred_tgt: torch.Tensor,
    cached_tgt_patches: torch.Tensor,
    valid_tgt: torch.Tensor,
    fs: int,
    subsample_tokens: int,
    tau_z: float,
    tau_s: float,
    f_min: float = 1.0,
    f_max: float = 45.0,
    sync_distributed: bool = False,
) -> tuple[torch.Tensor, int]:
    """Return the sum of per-recording RSReg losses and the eligible count.

    Positive subsample_tokens optionally caps tokens *per recording*. The paper
    configuration uses zero to include every valid target token. Recordings with
    fewer than two targets have no off-diagonal relations and are excluded.

    Divide by the global eligible count across the entire optimizer update.
    Empty inputs still call the projector and return a connected zero for DDP.
    With sync_distributed=True, all ranks must call this function together.
    """
    if 0 < int(subsample_tokens) < 2:
        raise ValueError("rsreg_subsample_tokens must be 0 or at least 2")

    latents, patches, lengths = [], [], []
    for z_b, x_b, valid_b in zip(pred_tgt, cached_tgt_patches, valid_tgt):
        indices = valid_b.nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            continue
        if subsample_tokens > 0 and indices.numel() > subsample_tokens:
            order = torch.randperm(indices.numel(), device=indices.device)
            indices = indices[order[:subsample_tokens]]
        latents.append(z_b[indices].float())
        patches.append(x_b[indices].float())
        lengths.append(indices.numel())

    if lengths:
        z_flat = torch.cat(latents)
        patches_flat = torch.cat(patches)
    else:
        z_flat = pred_tgt.new_zeros((1, pred_tgt.shape[-1]), dtype=torch.float32)
        patches_flat = cached_tgt_patches.new_zeros((1, cached_tgt_patches.shape[-1]), dtype=torch.float32)
    _require_finite(
        torch.isfinite(z_flat).all() & torch.isfinite(patches_flat).all(),
        "Non-finite target tokens in RSReg on at least one rank", sync_distributed,
    )
    spec_target = compute_logspec_view(patches_flat, fs=fs, f_min=f_min, f_max=f_max)
    # A single projector call also works when rel_proj is wrapped in DDP.
    z_rel = rel_proj(z_flat)
    _require_finite(
        torch.isfinite(spec_target).all() & torch.isfinite(z_rel).all(),
        "Non-finite spectral views or projected latents in RSReg on at least one rank", sync_distributed,
    )

    losses = [
        relational_kl_loss(z_b, s_b, tau_z=float(tau_z), tau_s=float(tau_s), check_finite=False)
        for z_b, s_b in zip(z_rel.split(lengths), spec_target.split(lengths))
    ] if lengths else []
    loss_sum = torch.stack(losses).sum() if losses else z_rel.sum() * 0.0
    _require_finite(
        torch.isfinite(loss_sum), "Non-finite RSReg loss on at least one rank", sync_distributed,
    )
    return loss_sum, len(lengths)


def compute_rsreg(**kwargs) -> Optional[torch.Tensor]:
    """Eq. (5) mean for one batch; training uses compute_rsreg_sum for accumulation."""
    loss_sum, count = compute_rsreg_sum(**kwargs)
    return loss_sum / count if count else None
