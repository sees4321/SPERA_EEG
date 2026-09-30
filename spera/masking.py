from __future__ import annotations

import torch

from typing import Tuple


@torch.no_grad()
def _random_nonneg_composition(total: int, parts: int) -> torch.Tensor:
    out = torch.zeros((parts,), dtype=torch.long)
    if total <= 0 or parts <= 0:
        return out
    idx = torch.randint(0, parts, (total,), dtype=torch.long)
    out.scatter_add_(0, idx, torch.ones((total,), dtype=torch.long))
    return out


@torch.no_grad()
def temporal_masking_SSP(
    B: int,
    C: int,
    P: int,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    # Temporal masking follows the SSP strategy: 
    # preserve contiguous temporal context blocks instead of masking isolated random time patches.
    mask = torch.zeros((B, C, P), dtype=torch.bool)
    if B <= 0 or C <= 0 or P <= 1:
        return mask

    frac = torch.empty((B,), dtype=torch.float32).uniform_(ratio_min, ratio_max)
    num_masked = torch.round(frac * P).to(torch.long).clamp_(1, P - 1)
    num_kept = P - num_masked
    num_blocks = max(1, P // 10)

    for b in range(B):
        nk = int(num_kept[b])
        nm = int(num_masked[b])

        if nk <= 0:
            mask[b].fill_(True)
            continue
        if nm <= 0:
            continue

        actual_blocks = min(num_blocks, nk)
        base = nk // actual_blocks
        rem = nk - base * actual_blocks

        kept_lengths = torch.full((actual_blocks,), base, dtype=torch.long)
        kept_lengths[-1] += rem
        gaps = _random_nonneg_composition(nm, actual_blocks + 1)

        row = torch.ones((P,), dtype=torch.bool)
        pos = int(gaps[0])
        for i in range(actual_blocks):
            li = int(kept_lengths[i])
            row[pos:pos + li] = False
            pos += li + int(gaps[i + 1])

        mask[b] = row.unsqueeze(0).expand(C, P)

    return mask


@torch.no_grad()
def spatial_masking(
    coords: torch.Tensor,   # (B, C, 3), CPU
    P_t: int,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    # Spatial target mask: 
    # sample a center electrode and mask its nearest neighbors in 3D coordinate space, making the target region montage-aware.
    B, C, _ = coords.shape
    base = torch.zeros((B, C), dtype=torch.bool)
    if B <= 0 or C <= 1 or P_t <= 0:
        return base.unsqueeze(-1).expand(B, C, P_t)

    frac = torch.empty((B,), dtype=torch.float32).uniform_(ratio_min, ratio_max)
    k = torch.round(frac * C).to(torch.long).clamp_(1, C - 1)
    centers = torch.randint(0, C, (B,), dtype=torch.long)

    for b in range(B):
        center = coords[b, centers[b]]
        d = torch.sum((coords[b] - center[None, :]) ** 2, dim=-1)
        nn = torch.topk(d, int(k[b]), largest=False).indices
        base[b, nn] = True

    return base.unsqueeze(-1).expand(B, C, P_t)


@torch.no_grad()
def target_masking(
    coords: torch.Tensor,
    P_t: int,
    mask_time_prob: float,
    mask_spatial_prob: float,
    time_ratio_range: Tuple[float, float],
    spatial_ratio_range: Tuple[float, float],
) -> torch.Tensor:
    assert coords.device.type == "cpu"
    B, C, _ = coords.shape
    target = torch.zeros((B, C, P_t), dtype=torch.bool)
    if B <= 0:
        return target
    if C * P_t < 2:
        raise ValueError("JEPA masking requires at least two tokens per recording")

    # Only use axes that can leave both context and target. Each active mask
    # keeps at least one position, so their union also leaves context tokens.
    use_time = (torch.rand((B,)) < float(mask_time_prob)) & (P_t > 1)
    use_spat = (torch.rand((B,)) < float(mask_spatial_prob)) & (C > 1)
    none = ~(use_time | use_spat)
    if P_t > 1:
        use_time[none] = True
    else:
        use_spat[none] = True

    if bool(use_time.any()):
        tmask = temporal_masking_SSP(
            B=B,
            C=C,
            P=P_t,
            ratio_min=float(time_ratio_range[0]),
            ratio_max=float(time_ratio_range[1]),
        )
        target |= tmask & use_time[:, None, None]

    if bool(use_spat.any()):
        smask = spatial_masking(
            coords=coords,
            P_t=P_t,
            ratio_min=float(spatial_ratio_range[0]),
            ratio_max=float(spatial_ratio_range[1]),
        )
        target |= smask & use_spat[:, None, None]
    return target


@torch.no_grad()
def mask_to_packed_indices(mask: torch.Tensor):
    """
    mask: (B, C, P) bool on CPU
    returns: c_idx, t_idx, pad on CPU
    order: time-major (t, c)
    """
    # Pack context inputs and target positions; the teacher encodes the full grid.
    assert mask.device.type == "cpu"
    B, C, P = mask.shape
    mask_tc = mask.permute(0, 2, 1).reshape(B, P * C)
    lengths = mask_tc.sum(dim=1, dtype=torch.long)
    Lmax = int(lengths.max()) if lengths.numel() > 0 else 0
    if Lmax <= 0:
        Lmax = 1

    c_idx = torch.zeros((B, Lmax), dtype=torch.long)
    t_idx = torch.zeros((B, Lmax), dtype=torch.long)
    pad = torch.ones((B, Lmax), dtype=torch.bool)

    nz = mask_tc.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return c_idx, t_idx, pad

    b = nz[:, 0]
    tc = nz[:, 1]
    starts = torch.cumsum(lengths, dim=0) - lengths
    pos = torch.arange(nz.shape[0], dtype=torch.long) - torch.repeat_interleave(starts, lengths)

    t_idx[b, pos] = torch.div(tc, C, rounding_mode="floor")
    c_idx[b, pos] = tc.remainder(C)
    pad[b, pos] = False
    return c_idx, t_idx, pad
