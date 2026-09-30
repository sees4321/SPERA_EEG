from __future__ import annotations

import torch

@torch.no_grad()
def apply_student_augmentations(
    x: torch.Tensor,
    gain_min: float,
    gain_max: float,
    noise_std_min: float,
    noise_std_max: float,
) -> torch.Tensor:
    # JEPA-style view asymmetry: 
    # augment only the student context branch while keeping the teacher target branch clean.
    if x.numel() == 0:
        return x

    device = x.device
    B, _, _ = x.shape
    x_fp32 = x.to(torch.float32)

    if (gain_min != 1.0) or (gain_max != 1.0):
        g = torch.empty((B, 1, 1), device=device).uniform_(gain_min, gain_max)
        x_fp32 = x_fp32 * g

    if noise_std_max and noise_std_max > 0:
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=(1, 2), keepdim=True) + 1e-8)
        ns = torch.empty((B, 1, 1), device=device).uniform_(float(noise_std_min), float(noise_std_max))
        noise = torch.randn_like(x_fp32) * (rms * ns)
        x_fp32 = x_fp32 + noise

    return x_fp32.to(x.dtype)

@torch.no_grad()
def apply_coord_jitter(
    coords: torch.tensor,
    coord_jitter_std: float = 0.05, 
    coord_jitter_prob: float = 0.5, 
) -> torch.Tensor:
    # Coordinate jitter regularizes the model against small montage localization errors while preserving the global spherical geometry.
    B, _, _ = coords.shape

    if (coord_jitter_std > 0) and (coord_jitter_prob > 0):
        gate = (torch.rand((B,), device=coords.device) < coord_jitter_prob).to(coords.dtype)
        coords = coords + torch.randn_like(coords) * coord_jitter_std * gate[:, None, None]
    
    return coords