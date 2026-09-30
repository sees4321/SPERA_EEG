"""LPFT (Linear Probing then Fine-Tuning) for EEG foundation model.

Single-GPU:
    python eval_lpft.py --data_root /path/to/task --pretrained /path/to/encoder \
                   --n_classes 4 --amp

Multi-GPU (2 GPUs):
    torchrun --nproc_per_node=2 eval_lpft.py --data_root /path/to/task \
             --pretrained /path/to/encoder --n_classes 4 --amp --do_ft
"""
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Adjust the import path to your project layout.
from spera.encoder import EEGEncoder


# ---------------- Classifier ----------------
class EEGClassifier(nn.Module):
    """Mean-pooled encoder features, LayerNorm, and a linear classifier."""
    def __init__(self, model, n_classes: int):
        super().__init__()
        self.feature_model = model
        d = int(getattr(model, "feat_dim", model.cfg.d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, int(n_classes)),
        )

    def forward(self, eeg: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        feat = self.feature_model(eeg, coord).mean(dim=1)
        return self.head(feat)


# ---------------- Dataset ----------------
class EEGDataset(Dataset):
    """Loads eeg.npy (N,C,T), coords.npy (N,C,3), label.npy (N,) from {root}/{split}/."""
    def __init__(self, root: str, split: str):
        d = Path(root) / split
        self.eeg = np.load(d / "eeg.npy", mmap_mode="r")
        self.coords = np.load(d / "coords.npy", mmap_mode="r")
        self.label = np.load(d / "label.npy")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return (
            torch.from_numpy(np.ascontiguousarray(self.eeg[i])).float(),
            torch.from_numpy(np.ascontiguousarray(self.coords[i])).float(),
            torch.tensor(int(self.label[i]), dtype=torch.long),
        )


# ---------------- DDP / utils ----------------
def setup_ddp():
    """Initialize DDP if launched via torchrun, else fall back to single-process."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def is_main(rank): return rank == 0


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------- Metrics ----------------
def compute_metrics(y_true: np.ndarray, logits: np.ndarray, n_classes: int) -> dict:
    """Return the metric set appropriate for the task."""
    pred = logits.argmax(axis=1)
    out = {"bacc": balanced_accuracy_score(y_true, pred)}
    if n_classes == 2:
        prob = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
        out["auroc"] = roc_auc_score(y_true, prob)
        out["aucpr"] = average_precision_score(y_true, prob)
    else:
        out["f1w"] = f1_score(y_true, pred, average="weighted")
        out["kappa"] = cohen_kappa_score(y_true, pred)
    return out


@torch.no_grad()
def evaluate(model, loader, device, n_classes, use_amp):
    """Compute metrics over the full dataset (call from rank 0 only)."""
    model.eval()
    logits_all, y_all = [], []
    for eeg, coords, y in loader:
        eeg = eeg.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(eeg, coords)
        logits_all.append(logits.float().cpu().numpy())
        y_all.append(y.numpy())
    logits_all = np.concatenate(logits_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return compute_metrics(y_all, logits_all, n_classes)


# ---------------- Phase trainer ----------------
def run_phase(
    model_ddp, train_loader, val_loader, optimizer, scaler, criterion,
    device, n_epochs, n_classes, use_amp, rank, ddp,
    selection_metric, phase_name, log_fn, best_state,
):
    """Run one phase (LP or FT). Update best_state when validation improves."""
    for epoch in range(1, n_epochs + 1):
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        # ----- train -----
        model_ddp.train()
        if phase_name == "lp":
            (model_ddp.module if ddp else model_ddp).feature_model.eval()
        total_loss, total_n = 0.0, 0
        for eeg, coords, y in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model_ddp(eeg, coords)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)

        # Aggregate train loss across ranks.
        if ddp:
            t = torch.tensor([total_loss, total_n], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, total_n = t[0].item(), int(t[1].item())
        train_loss = total_loss / max(total_n, 1)

        # ----- eval (rank 0 only; DDP keeps weights in sync every step) -----
        if is_main(rank):
            eval_model = model_ddp.module if ddp else model_ddp
            val_metrics = evaluate(eval_model, val_loader, device, n_classes, use_amp)
            log_fn({
                "phase": phase_name, "epoch": epoch, "train_loss": train_loss,
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })
            if val_metrics[selection_metric] > best_state["score"]:
                best_state["score"] = val_metrics[selection_metric]
                best_state["epoch"] = epoch
                best_state["phase"] = phase_name
                src = model_ddp.module if ddp else model_ddp
                best_state["state_dict"] = {k: v.detach().cpu().clone()
                                            for k, v in src.state_dict().items()}
        if ddp:
            dist.barrier()


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--pretrained", required=True, help="Path passed to EEGEncoder.from_pretrained")
    p.add_argument("--output_dir", default="./lpft_run")
    p.add_argument("--n_classes", type=int, required=True)
    # Optimization
    p.add_argument("--lp_lr", type=float, default=1e-3)
    p.add_argument("--ft_lr", type=float, default=1e-5)
    p.add_argument("--lp_epochs", type=int, default=20)
    p.add_argument("--ft_epochs", type=int, default=20)
    p.add_argument("--do_ft", action="store_true", help="Run FT after LP")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    # Misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    p.add_argument("--selection_metric", default="bacc",
                   choices=["bacc", "kappa", "auroc"])
    # Logging
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="eeg-lpft")
    p.add_argument("--run_name", default="run")
    args = p.parse_args()

    rank, world_size, local_rank, ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    # Make sure the chosen selection_metric is compatible with the task.
    valid = {"bacc"} | ({"auroc"} if args.n_classes == 2 else {"kappa"})
    if args.selection_metric not in valid:
        raise ValueError(f"selection_metric '{args.selection_metric}' is not valid "
                         f"for n_classes={args.n_classes}. Choose from {valid}.")

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Logger (wandb or csv) ----
    use_wandb = (not args.no_wandb) and is_main(rank)
    csv_file, csv_writer = None, None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    if is_main(rank) and args.no_wandb:
        csv_file = open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "phase", "epoch", "train_loss",
            "val/bacc", "val/auroc", "val/aucpr", "val/f1w", "val/kappa",
            "test/bacc", "test/auroc", "test/aucpr", "test/f1w", "test/kappa",
        ])
        csv_writer.writeheader()

    def log_fn(d):
        if not is_main(rank): return
        if use_wandb:
            import wandb; wandb.log(d)
        if csv_writer is not None:
            csv_writer.writerow({k: d.get(k, "") for k in csv_writer.fieldnames})
            csv_file.flush()

    # ---- Data ----
    train_ds = EEGDataset(args.data_root, "train")
    val_ds   = EEGDataset(args.data_root, "validation")
    test_ds  = EEGDataset(args.data_root, "test")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=ddp,
    )
    eval_kw = dict(batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True)
    val_loader  = DataLoader(val_ds, **eval_kw)
    test_loader = DataLoader(test_ds, **eval_kw)

    # ---- Model ----
    encoder = EEGEncoder.from_pretrained(args.pretrained)
    model = EEGClassifier(encoder, n_classes=args.n_classes).to(device)

    # ---- Loss (class_weight) ----
    if args.class_weight == "balanced":
        cw = compute_class_weight("balanced",
                                  classes=np.arange(args.n_classes),
                                  y=np.asarray(train_ds.label))
        cw = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        cw = None
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_state = {"score": -float("inf"), "epoch": -1, "phase": "", "state_dict": None}
    scaler = GradScaler(enabled=args.amp)

    # ---- LP phase: freeze encoder, train head only ----
    for prm in model.feature_model.parameters():
        prm.requires_grad = False
    model_ddp = DDP(model, device_ids=[local_rank]) if ddp else model
    lp_params = [p_ for p_ in model.head.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(lp_params, lr=args.lp_lr,
                                  weight_decay=args.weight_decay)
    if is_main(rank):
        print(f"[LP] trainable params: {sum(p_.numel() for p_ in lp_params):,}")
    run_phase(model_ddp, train_loader, val_loader, optimizer, scaler, criterion,
              device, args.lp_epochs, args.n_classes, args.amp,
              rank, ddp, args.selection_metric, "lp", log_fn, best_state)

    # ---- FT phase (optional): unfreeze everything with a smaller lr ----
    if args.do_ft:
        # DDP reconstruction below broadcasts rank 0's restored LP weights.
        if is_main(rank) and best_state["state_dict"] is not None:
            model.load_state_dict(best_state["state_dict"])
            print(f"[FT] restored LP epoch {best_state['epoch']} "
                  f"(val_{args.selection_metric}={best_state['score']:.4f})")
        for prm in model.feature_model.parameters():
            prm.requires_grad = True
        # Trainable params changed, so rebuild DDP and optimizer.
        if ddp:
            del model_ddp
            model_ddp = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr,
                                      weight_decay=args.weight_decay)
        if is_main(rank):
            print(f"[FT] trainable params: {sum(p_.numel() for p_ in model.parameters()):,}")
        run_phase(model_ddp, train_loader, val_loader, optimizer, scaler, criterion,
                  device, args.ft_epochs, args.n_classes, args.amp,
                  rank, ddp, args.selection_metric, "ft", log_fn, best_state)

    # ---- Test (rank 0): load weights from best validation epoch ----
    if is_main(rank):
        if best_state["state_dict"] is not None:
            (model_ddp.module if ddp else model_ddp).load_state_dict(best_state["state_dict"])
        eval_model = model_ddp.module if ddp else model_ddp
        test_metrics = evaluate(eval_model, test_loader, device, args.n_classes, args.amp)
        print(f"[BEST] phase={best_state['phase']} epoch={best_state['epoch']} "
              f"val_{args.selection_metric}={best_state['score']:.4f}")
        print(f"[TEST] {test_metrics}")
        log_fn({"phase": "test", "epoch": best_state["epoch"], "train_loss": "",
                **{f"test/{k}": v for k, v in test_metrics.items()}})
        torch.save(best_state["state_dict"],
                   os.path.join(args.output_dir, "best.pt"))

    if csv_file is not None:
        csv_file.close()
    if use_wandb:
        import wandb; wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
