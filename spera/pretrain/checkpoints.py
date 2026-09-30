from __future__ import annotations

import json
import os
import torch

from torch.optim import AdamW
from typing import Dict, Optional

from ..encoder import EEGEncoder
from ..predictor import CrossAttentionPredictor
from ..rsreg import ZProjector


def save_checkpoint(
    ckpt_dir: str,
    student: EEGEncoder,
    teacher: EEGEncoder,
    predictor: CrossAttentionPredictor,
    rel_proj: Optional[ZProjector],
    optimizer: AdamW,
    trainer_state: Dict[str, int],
) -> None:
    # Save student, teacher, predictor, optional RSReg projector, optimizer, and trainer state separately.
    # Final student encoder can be reused for downstream evaluation.
    os.makedirs(ckpt_dir, exist_ok=True)
    student.save_pretrained(os.path.join(ckpt_dir, "student"))
    teacher.save_pretrained(os.path.join(ckpt_dir, "teacher"))
    torch.save(predictor.state_dict(), os.path.join(ckpt_dir, "predictor.pt"))
    if rel_proj is not None:
        torch.save(rel_proj.state_dict(), os.path.join(ckpt_dir, "rel_projector.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)


def load_checkpoint(
    ckpt_dir: str,
    student: EEGEncoder,
    teacher: EEGEncoder,
    predictor: CrossAttentionPredictor,
    rel_proj: Optional[ZProjector],
    optimizer: AdamW,
) -> Dict[str, int]:
    student_sd = torch.load(os.path.join(ckpt_dir, "student", "pytorch_model.bin"), map_location="cpu", weights_only=True)
    teacher_sd = torch.load(os.path.join(ckpt_dir, "teacher", "pytorch_model.bin"), map_location="cpu", weights_only=True)
    pred_sd = torch.load(os.path.join(ckpt_dir, "predictor.pt"), map_location="cpu", weights_only=True)
    opt_sd = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu", weights_only=False)

    student.load_state_dict(student_sd, strict=True)
    teacher.load_state_dict(teacher_sd, strict=True)
    predictor.load_state_dict(pred_sd, strict=True)
    if rel_proj is not None and os.path.exists(os.path.join(ckpt_dir, "rel_projector.pt")):
        rel_sd = torch.load(os.path.join(ckpt_dir, "rel_projector.pt"), map_location="cpu", weights_only=True)
        rel_proj.load_state_dict(rel_sd, strict=True)
    optimizer.load_state_dict(opt_sd)
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "r", encoding="utf-8") as f:
        return json.load(f)