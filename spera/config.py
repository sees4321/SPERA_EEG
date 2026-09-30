from __future__ import annotations

import json
import os

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, Tuple


@dataclass
class EEGModelConfig:
    # Tokenization
    sample_rate: int = 200
    patch_seconds: float = 1.0
    max_tokens: int = 4096

    mlp_type: str = "swiglu"
    norm_type: str = "layernorm"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    rope_theta: float = 10000.0
    rotary_pct: float = 1.0

    # Encoder Architecture 
    encoder_arch: str = "hybrid" # full / factorized / hybrid
    full_attn_every: int = 4
    full_attn_use_spatial_bias: bool = False

    # Legendre Bias
    spatial_bias: str = "legendre" # legendre / none
    spatial_bias_degree: int = 8
    spatial_bias_use_unit_sphere: bool = True
    spatial_bias_scale: float = 1.0
    
    # Coord embedding weight init
    coord_w_init: float = 0.0

    # Legendre Anchor Kernel
    spatial_ak: str = "legendre_anchor" # legendre_anchor / none
    spatial_ak_num_anchors: int = 32
    spatial_ak_degree: int = 8
    spatial_ak_feat_dim: int = 64
    spatial_ak_scale: float = 1.0

    predictor_layers: int = 2
    predictor_n_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    query_token_init_std: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "EEGModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        allowed = {fd.name for fd in fields(EEGModelConfig)}
        d_f = {k: v for k, v in d.items() if k in allowed}
        return EEGModelConfig(**d_f)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class TrainConfig:
    seed: int = 42
    torch_deterministic: bool = False
    cudnn_benchmark: bool = True
    mixed_precision: str = "bf16"

    # Data
    shards_txt: str = ""
    shard_shuffle: int = 0
    sample_shuffle: int = 256
    post_split_shuffle: int = 128

    tokens_per_batch: int = 21504  # per-rank valid-token target for ShapeBatcher
    max_samples_per_batch: int = 256
    num_workers: int = 8

    # Masking
    mask_time_prob: float = 0.8
    mask_spatial_prob: float = 0.8
    time_mask_ratio_min: float = 0.15
    time_mask_ratio_max: float = 0.35
    spatial_mask_ratio_min: float = 0.10
    spatial_mask_ratio_max: float = 0.30

    # Student augmentations
    aug_gain_min: float = 0.8
    aug_gain_max: float = 1.2
    aug_noise_std_min: float = 0.00
    aug_noise_std_max: float = 0.03
    coord_jitter_std: float = 0.05
    coord_jitter_prob: float = 0.5

    # Optimization
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    # NOTE: global effective target tokens per optimizer step (summed across ranks)
    tokens_per_update: int = 131072
    max_steps: int = 47000
    warmup_steps: int = 2350
    min_lr: float = 3.0e-5
    ema_momentum: float = 0.996
    ema_momentum_final: float = 0.9999

    # RSReg
    rsreg_proj_dim: int = 128
    rsreg_subsample_tokens: int = 0  # 0: all target tokens within each recording (Eq. 5)
    rsreg_tau_z: float = 0.1
    rsreg_tau_s: float = 0.1
    rsreg_weight: float = 0.02
    rsreg_warmup_steps: int = 2350
    rsreg_ramp_steps: int = 2350  # duration: peak at step 4700 (10%)
    rsreg_decay_step: int = 23500
    rsreg_final_weight: float = 0.002

    # Logging / checkpoints
    output_dir: str = "./checkpoints/spera"
    log_every: int = 50
    save_every: int = 5000
    use_wandb: bool = True
    wandb_project: str = "SPERA"
    run_name: Optional[str] = None

    # Execution
    use_torch_compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = True
    resume_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        allowed = {fd.name for fd in fields(TrainConfig)}
        d_f = {k: v for k, v in d.items() if k in allowed}
        return TrainConfig(**d_f)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
