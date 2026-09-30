from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from spera.config import TrainConfig
from spera.encoder import EEGEncoder
from spera.predictor import CrossAttentionPredictor


def distributed_sum(value: int, device: torch.device) -> int:
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def distributed_mean(value: float, device: torch.device) -> float:
    t = torch.tensor([float(value)], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


@torch.no_grad()
def rescale_small_segments(
    x: torch.Tensor,
    target_amp: float = 1.0,
    quantile: float = 0.90,
    amp_floor: float = 1e-4,
    gain_max: float = 200.0,
    clip: float = 15.0,
) -> torch.Tensor:
    x32 = x.float()
    T = x32.shape[-1]
    k_idx = max(1, int(round(quantile * T)))
    robust_amp = x32.abs().kthvalue(k_idx, dim=-1, keepdim=True).values
    gain = (target_amp / robust_amp.clamp_min(amp_floor)).clamp(max=gain_max)
    x32 = x32 * gain
    if clip and clip > 0:
        x32 = x32.clamp(-clip, clip)
    return x32.to(dtype=x.dtype)


def gather_channel_embeddings(x: torch.Tensor, c_idx: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    B, C, D = x.shape
    B2, L = c_idx.shape
    assert B == B2
    idx = c_idx[..., None].expand(B, L, D)
    out = x.gather(dim=1, index=idx)
    return out.masked_fill(pad[..., None], 0.0)


@torch.no_grad()
def encode_teacher_targets(
    teacher: EEGEncoder,
    x: torch.Tensor,
    coords: torch.Tensor,
    c_idx: torch.Tensor,
    t_idx: torch.Tensor,
    pad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the full unaugmented view and select normalized target tokens."""
    b_idx = torch.arange(x.shape[0], device=x.device)[:, None]
    c_safe, t_safe = c_idx.clamp(min=0), t_idx.clamp(min=0)
    z_grid = teacher.forward_eeg(x, coords, return_grid=True)
    z_tgt = z_grid[b_idx, c_safe, t_safe].float()
    # V-JEPA target normalization: no learned affine parameters. Tokenwise
    # normalization commutes with selection, so only selected tokens need fp32.
    z_tgt = F.layer_norm(z_tgt, (z_tgt.shape[-1],))
    z_tgt = z_tgt.masked_fill(pad[..., None], 0.0)
    patches = teacher.extract_patches_view(x)[b_idx, c_safe, t_safe]
    return z_tgt, patches


def token_warmup_cosine_lr(tokens_next: int, warmup_tokens: int, total_tokens: int, base_lr: float, min_lr: float = 0.0,) -> float:
    t = max(0, int(tokens_next))
    T = max(1, int(total_tokens))
    W = min(max(0, int(warmup_tokens)), T)

    # 1) warmup: 0 -> base_lr
    if W > 0 and t < W:
        return float(base_lr) * float(t) / float(max(1, W))
    remain = max(1, T - W)  # warmup 이후 남은 길이

    # 2) cosine decay: base_lr -> min_lr
    if t < T:
        td = t - W
        frac = float(td) / float(remain)   # 0 ~ 1
        cos_frac = 0.5 * (1.0 + math.cos(math.pi * frac))
        return float(min_lr) + (float(base_lr) - float(min_lr)) * cos_frac

    return float(min_lr) # 3) training 끝난 뒤


def compile_modules(student: EEGEncoder, predictor: CrossAttentionPredictor, train_cfg: TrainConfig) -> None:
    if not bool(train_cfg.use_torch_compile):
        return
    mode = str(train_cfg.compile_mode)
    dynamic = bool(train_cfg.compile_dynamic)
    for blk in student.blocks:
        if hasattr(blk, "attn"):
            blk.attn = torch.compile(blk.attn, mode=mode, dynamic=dynamic)
        if hasattr(blk, "attn_t"):
            blk.attn_t = torch.compile(blk.attn_t, mode=mode, dynamic=dynamic)
        if hasattr(blk, "attn_s"):
            blk.attn_s = torch.compile(blk.attn_s, mode=mode, dynamic=dynamic)
        if hasattr(blk, "mlp"):
            blk.mlp = torch.compile(blk.mlp, mode=mode, dynamic=dynamic)
    for blk in predictor.blocks:
        blk.xattn = torch.compile(blk.xattn, mode=mode, dynamic=dynamic)
        blk.mlp = torch.compile(blk.mlp, mode=mode, dynamic=dynamic)


def count_params(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


class EMAUpdater:
    def __init__(self, teacher: nn.Module, student: nn.Module, m0: float):
        self.teacher_params = list(teacher.parameters())
        self.student_params = list(student.parameters())
        self.teacher_buffers = list(teacher.buffers())
        self.student_buffers = list(student.buffers())
        self.m = float(m0)

    @torch.no_grad()
    def copy_from_student(self) -> None:
        """Initialize from the student after DDP has synchronized its weights.

        Parameter order also works with compiled student submodules, whose
        state_dict keys contain extra ``_orig_mod`` components.
        """
        for targets, sources in (
            (self.teacher_params, self.student_params),
            (self.teacher_buffers, self.student_buffers),
        ):
            for target, source in zip(targets, sources, strict=True):
                if target.shape != source.shape:
                    raise ValueError("Teacher and student tensor shapes do not match")
                target.copy_(source)

    def set_momentum(self, step: int, total_steps: int, m0: float, m1: float) -> None:
        progress = step / max(1, total_steps)
        self.m = float(m1 - (m1 - m0) * (0.5 * (1.0 + math.cos(math.pi * progress))))

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_mul_(self.teacher_params, self.m)
        torch._foreach_add_(self.teacher_params, self.student_params, alpha=(1.0 - self.m))
