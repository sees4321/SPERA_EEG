from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from spera.config import EEGModelConfig
from spera.encoder import EEGEncoder


def downstream_example(
    checkpoint_dir: str = None, # will be released at Huggingface upon acceptance.
    model_cfg: str = "./configs/model/spera_tiny_18m.json",
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if checkpoint_dir is not None:
        model = EEGEncoder.from_pretrained(checkpoint_dir, map_location=str(device)).to(device)
        print(f"Loaded checkpoint: {checkpoint_dir}")
    else:
        cfg = EEGModelConfig.from_json(model_cfg)
        model = EEGEncoder(cfg).to(device)
        print("No checkpoint supplied. Using randomly initialized model.")

    model.eval()

    # example: 4-class
    classifier = nn.Sequential(
        nn.LayerNorm(model.cfg.d_model),
        nn.Linear(model.cfg.d_model, 4),
    ).to(device)

    classifier.eval()

    # (batch_size, num_channels, time_samples)
    # sampling rate = 200 Hz
    eeg = torch.randn((8, 22, 4 * 200), device=device)

    # Synthetic normalized 3D electrode coordinates.
    coords = F.normalize(torch.randn((8, 22, 3), device=device), dim=-1)

    with torch.no_grad():
        z = model(eeg, coords)
        features = z.mean(dim=1)
        logits = classifier(features)

    print(f"eeg.shape = {tuple(eeg.shape)}")
    print(f"features.shape = {tuple(features.shape)}")
    print(f"logits.shape   = {tuple(logits.shape)}")


def pretrain_smoke_test(
    model_cfg: str = "./configs/model/spera_tiny_18m.json",
    train_cfg: str = "./configs/pretrain/pretrain_smoke.json",
    shards_txt: str = "./example_data/shards.txt",
    output_dir: str = "./runs/quickstart",
    max_steps: int = 5,
) -> None:
    cmd = [
        sys.executable,
        "pretrain.py",
        "--model_cfg",
        model_cfg,
        "--train_cfg",
        train_cfg,
        "--shards_txt",
        shards_txt,
        "--output_dir",
        output_dir,
        "--max_steps",
        str(max_steps),
        "--no_wandb",
    ]

    print("Running pretraining smoke test:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["downstream", "pretrain", "all"])
    parser.add_argument(
        "--mode",
        dest="mode_option",
        choices=["downstream", "pretrain", "all"],
        help="Run downstream forward example, pretraining smoke test, or both.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None) # will be released at Huggingface upon acceptance.
    parser.add_argument("--model_cfg", type=str, default="./configs/model/spera_tiny_18m.json")
    parser.add_argument("--train_cfg", type=str, default="./configs/pretrain/pretrain_smoke.json")
    parser.add_argument("--shards_txt", type=str, default="./example_data/shards.txt")
    parser.add_argument("--output_dir", type=str, default="./runs/quickstart")
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()
    mode = args.mode_option or args.mode or "downstream"

    if mode in ("downstream", "all"):
        downstream_example(
            checkpoint_dir=args.checkpoint_dir,
            model_cfg=args.model_cfg,
        )

    if mode in ("pretrain", "all"):
        pretrain_smoke_test(
            model_cfg=args.model_cfg,
            train_cfg=args.train_cfg,
            shards_txt=args.shards_txt,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
