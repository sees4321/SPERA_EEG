from __future__ import annotations

import argparse
import io
import math
import tarfile
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_size(size: str) -> int:
    s = str(size).strip().lower()
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }

    for unit, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * mult)

    return int(float(s))


def npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def make_unit_sphere_coords(num_channels: int, rng: np.random.Generator) -> np.ndarray:
    coords = rng.normal(size=(num_channels, 3)).astype(np.float32)
    coords /= np.linalg.norm(coords, axis=-1, keepdims=True).clip(min=1e-6)
    return coords.astype(np.float32)


def make_synthetic_eeg(
    num_channels: int,
    num_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    eeg = np.empty((num_channels, num_samples), dtype=np.float32)

    # EEG-like mixture of slow oscillations, alpha/beta components, and noise.
    for c in range(num_channels):
        f_delta_theta = rng.uniform(1.0, 7.0)
        f_alpha = rng.uniform(8.0, 13.0)
        f_beta = rng.uniform(14.0, 30.0)

        p1 = rng.uniform(0.0, 2.0 * math.pi)
        p2 = rng.uniform(0.0, 2.0 * math.pi)
        p3 = rng.uniform(0.0, 2.0 * math.pi)

        signal = (
            0.8 * np.sin(2.0 * math.pi * f_delta_theta * t + p1)
            + 0.6 * np.sin(2.0 * math.pi * f_alpha * t + p2)
            + 0.3 * np.sin(2.0 * math.pi * f_beta * t + p3)
            + 0.15 * rng.normal(size=num_samples)
        )

        eeg[c] = signal.astype(np.float32)

    # Match the rough assumptions of preprocessed EEG.
    eeg -= eeg.mean(axis=-1, keepdims=True)
    eeg /= eeg.std(axis=-1, keepdims=True).clip(min=1e-6)
    eeg = np.clip(eeg, -15.0, 15.0)
    return eeg.astype(np.float32)


def make_one_sample(
    num_channels: int,
    num_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    eeg = make_synthetic_eeg(
        num_channels=num_channels,
        num_samples=num_samples,
        sample_rate=sample_rate,
        rng=rng,
    )
    coords = make_unit_sphere_coords(num_channels, rng)
    return eeg, coords


def write_shards(
    out_dir: Path,
    num_shards: int,
    target_shard_bytes: int,
    num_channels: int,
    num_samples: int,
    sample_rate: int,
    seed: int,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    shard_paths = []
    global_sample_id = 0

    for shard_idx in range(num_shards):
        shard_path = out_dir / f"{prefix}-{shard_idx:06d}.tar"
        shard_paths.append(shard_path)

        num_samples_in_shard = 0

        with tarfile.open(shard_path, "w") as tar:
            while True:
                key = f"{global_sample_id:010d}"
                eeg, coords = make_one_sample(
                    num_channels=num_channels,
                    num_samples=num_samples,
                    sample_rate=sample_rate,
                    rng=rng,
                )

                add_bytes_to_tar(tar, f"{key}.eeg.npy", npy_bytes(eeg))
                add_bytes_to_tar(tar, f"{key}.coords.npy", npy_bytes(coords))

                global_sample_id += 1
                num_samples_in_shard += 1

                # tarfile writes in blocks, so this is approximate.
                if shard_path.exists() and shard_path.stat().st_size >= target_shard_bytes:
                    break

        size_gb = shard_path.stat().st_size / 1000**3
        print(
            f"[done] {shard_path} | "
            f"{size_gb:.3f} GB | "
            f"{num_samples_in_shard} samples"
        )

    shards_txt = out_dir / "shards.txt"
    with open(shards_txt, "w", encoding="utf-8") as f:
        for p in shard_paths:
            f.write(str(p.as_posix()) + "\n")

    print(f"[done] wrote shard list: {shards_txt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create synthetic EEG WebDataset shards for SPERA quickstart."
    )
    p.add_argument("--out_dir", type=str, default="./example_data")
    p.add_argument("--num_shards", type=int, default=2)
    p.add_argument("--target_shard_size", type=str, default="1GB")
    p.add_argument("--channels", type=int, default=19)
    p.add_argument("--length", type=int, default=200 * 30)
    p.add_argument("--sample_rate", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", type=str, default="synthetic")
    return p


def main() -> None:
    args = build_parser().parse_args()

    write_shards(
        out_dir=Path(args.out_dir),
        num_shards=int(args.num_shards),
        target_shard_bytes=parse_size(args.target_shard_size),
        num_channels=int(args.channels),
        num_samples=int(args.length),
        sample_rate=int(args.sample_rate),
        seed=int(args.seed),
        prefix=str(args.prefix),
    )


if __name__ == "__main__":
    main()