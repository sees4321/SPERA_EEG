from __future__ import annotations

import argparse

from typing import Optional, Sequence

from spera.pretrain.trainer import run_train


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=add_help)
    p.add_argument("--model_cfg", type=str, required=True)
    p.add_argument("--train_cfg", type=str, required=True)
    p.add_argument("--shards_txt", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--tokens_per_batch", type=int, default=None)
    p.add_argument("--tokens_per_update", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume_shard_overlap", type=int, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_train(args)


if __name__ == "__main__":
    main()