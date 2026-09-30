from __future__ import annotations

import argparse
import copy
import numpy as np
import os
import random
import torch
import webdataset as wds

from accelerate import Accelerator
from accelerate.utils import set_seed
from contextlib import ExitStack
from pathlib import Path
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict

from spera.augment import apply_student_augmentations, apply_coord_jitter
from spera.config import EEGModelConfig, TrainConfig
from spera.data import ShapeBatcher, build_webdataset, read_shards_txt, build_shard_index_map
from spera.encoder import EEGEncoder
from spera.predictor import CrossAttentionPredictor
from spera.rsreg import ZProjector, rsreg_weight, compute_rsreg_sum
from .checkpoints import *
from .utils import *


def run_train(args: argparse.Namespace) -> Dict[str, str]:
    model_cfg = EEGModelConfig.from_json(args.model_cfg)
    train_cfg = TrainConfig.from_json(args.train_cfg)

    if args.shards_txt is not None:
        train_cfg.shards_txt = args.shards_txt
    if args.output_dir is not None:
        train_cfg.output_dir = args.output_dir
    if args.resume_from is not None:
        train_cfg.resume_from = args.resume_from
    if args.tokens_per_batch is not None:
        train_cfg.tokens_per_batch = int(args.tokens_per_batch)
    if args.tokens_per_update is not None:
        train_cfg.tokens_per_update = int(args.tokens_per_update)
    if args.max_steps is not None:
        train_cfg.max_steps = int(args.max_steps)
    if args.lr is not None:
        train_cfg.lr = float(args.lr)
    if args.weight_decay is not None:
        train_cfg.weight_decay = float(args.weight_decay)
    if args.num_workers is not None:
        train_cfg.num_workers = int(args.num_workers)
    if args.run_name is not None:
        train_cfg.run_name = args.run_name
    if args.no_wandb:
        train_cfg.use_wandb = False

    if not train_cfg.shards_txt:
        raise ValueError("train_cfg.shards_txt (or --shards_txt) is required in the run-only project")

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=train_cfg.mixed_precision,
        log_with="wandb" if train_cfg.use_wandb else None,
        project_dir=train_cfg.output_dir,
    )
    dev = accelerator.device

    seed = int(train_cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed, device_specific=True)

    if bool(train_cfg.torch_deterministic): 
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = bool(train_cfg.cudnn_benchmark)

    if accelerator.is_main_process:
        os.makedirs(train_cfg.output_dir, exist_ok=True)
        model_cfg.save_json(os.path.join(train_cfg.output_dir, "model_config.json"))
        train_cfg.save_json(os.path.join(train_cfg.output_dir, "train_config.json"))

    if train_cfg.use_wandb:
        accelerator.init_trackers(
            train_cfg.wandb_project,
            config={**model_cfg.to_dict(), **train_cfg.to_dict()},
            init_kwargs={"wandb": {"name": train_cfg.run_name}} if train_cfg.run_name else None,
        )

    student = EEGEncoder(model_cfg).to(dev)
    predictor = CrossAttentionPredictor(model_cfg).to(dev)
    rel_proj = None
    if float(train_cfg.rsreg_weight) > 0:
        rel_proj = ZProjector(model_cfg.d_model, int(train_cfg.rsreg_proj_dim)).to(dev)

    # Teacher encoder is an EMA copy of the student. 
    # Gradients flow only through the student encoder and predictor.
    teacher = copy.deepcopy(student).to(dev)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student_raw = student
    predictor_raw = predictor
    compile_modules(student_raw, predictor_raw, train_cfg)

    params = list(student.parameters()) + list(predictor.parameters())
    if rel_proj is not None:
        params += list(rel_proj.parameters())
    optimizer = AdamW(params, lr=train_cfg.lr, betas=train_cfg.betas, weight_decay=train_cfg.weight_decay)

    student, predictor, optimizer = accelerator.prepare(student, predictor, optimizer)
    if rel_proj is not None:
        rel_proj = accelerator.prepare(rel_proj)

    teacher.to(dev)
    ema_updater = EMAUpdater(teacher, accelerator.unwrap_model(student), train_cfg.ema_momentum)

    # prepare() broadcasts the student after rank-specific initialization.
    # Copy now; checkpoint loading below restores the saved EMA teacher on resume.
    ema_updater.copy_from_student()

    trainer_state = {
        "global_step": 0,
        "passes_completed": 0,
        "tokens_seen_total": 0,
    }
    if train_cfg.resume_from:
        ckpt_dir = str(Path(train_cfg.resume_from).expanduser().resolve())
        trainer_state = load_checkpoint(
            ckpt_dir,
            accelerator.unwrap_model(student),
            teacher,
            accelerator.unwrap_model(predictor),
            accelerator.unwrap_model(rel_proj) if rel_proj is not None else None,
            optimizer,
        )
        accelerator.wait_for_everyone()

    patch_samples = int(round(model_cfg.sample_rate * model_cfg.patch_seconds))
    all_shards = read_shards_txt(train_cfg.shards_txt)
    resume_shard_overlap = int(
        args.resume_shard_overlap
        if getattr(args, "resume_shard_overlap", None) is not None
        else os.environ.get("EEGFM_RESUME_SHARD_OVERLAP", "8")
    )
    resume_shard_overlap = max(0, int(resume_shard_overlap))
    resume_next_shard_index = int(trainer_state.get("data_next_shard_index", 0))
    resume_next_shard_index = max(0, min(len(all_shards), resume_next_shard_index))
    resume_midpass_active = bool(train_cfg.resume_from) and (resume_next_shard_index > 0)
    active_start_shard_index = max(0, min(len(all_shards), resume_next_shard_index) - resume_shard_overlap) if resume_midpass_active else 0
    active_shards = all_shards[active_start_shard_index:] if active_start_shard_index > 0 else all_shards

    def _make_loader(shards_subset):
        ds = build_webdataset(
            shards=shards_subset,
            shard_shuffle=train_cfg.shard_shuffle,
            sample_shuffle=train_cfg.sample_shuffle,
            max_tokens=model_cfg.max_tokens,
            patch_samples=patch_samples,
            post_split_shuffle=train_cfg.post_split_shuffle,
            seed=train_cfg.seed,
            shard_to_index=build_shard_index_map(all_shards),
        )
        ds_batched = ShapeBatcher(
            dataset=ds,
            tokens_per_batch=train_cfg.tokens_per_batch,
            max_samples_per_batch=train_cfg.max_samples_per_batch,
            patch_samples=patch_samples,
            target_mask_cfg=dict(
                mask_time_prob=float(train_cfg.mask_time_prob),
                mask_spatial_prob=float(train_cfg.mask_spatial_prob),
                time_ratio_range=(float(train_cfg.time_mask_ratio_min), float(train_cfg.time_mask_ratio_max)),
                spatial_ratio_range=(float(train_cfg.spatial_mask_ratio_min), float(train_cfg.spatial_mask_ratio_max)),
            ),
            max_wait_samples=5000,
            flush_check_every=256,
            max_pending_samples=512,
            max_pending_tokens=0,
            shuffle_within_bucket=True,
            yield_incomplete=False,
        )
        return wds.WebLoader(
            ds_batched,
            batch_size=None,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            persistent_workers=(train_cfg.num_workers > 0),
            prefetch_factor=4 if train_cfg.num_workers > 0 else None,
        )

    loader = _make_loader(active_shards)

    if accelerator.is_main_process:
        enc_params = count_params(accelerator.unwrap_model(student))
        pred_params = count_params(accelerator.unwrap_model(predictor))
        rel_params = count_params(accelerator.unwrap_model(rel_proj)) if rel_proj is not None else 0
        print(f"[params] encoder={enc_params/1e6:.3f}M predictor={pred_params/1e6:.3f}M aux={rel_params/1e6:.3f}M")
        if resume_midpass_active:
            print(
                f"[data-resume] mid-pass resume enabled: next_shard_index={resume_next_shard_index} "
                f"overlap={resume_shard_overlap} start_from={active_start_shard_index} "
                f"remaining_shards={len(active_shards)}/{len(all_shards)}"
            )

    budget_tokens = int(train_cfg.tokens_per_update)
    if budget_tokens <= 0:
        raise ValueError("tokens_per_update must be positive")
    total_sched_tokens = int(train_cfg.max_steps) * budget_tokens
    warmup_tokens = int(train_cfg.warmup_steps) * budget_tokens
    world_size = accelerator.num_processes

    global_step = int(trainer_state.get("global_step", 0))
    passes_completed = int(trainer_state.get("passes_completed", 0))
    tokens_seen_total = int(trainer_state.get("tokens_seen_total", 0))
    current_pass_last_shard_seen = int(trainer_state.get("data_last_shard_index_seen", active_start_shard_index - 1))

    optimizer.zero_grad(set_to_none=True)

    pass_input_tokens_global = 0
    pass_target_tokens_global = 0
    pass_context_tokens_global = 0

    log_l1_sum = 0.0
    log_total_sum = 0.0
    log_rsreg_sum = 0.0
    log_rsreg_count = 0
    log_count = 0

    it = iter(loader)
    pbar = tqdm(total=int(train_cfg.max_steps), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
    if global_step > 0:
        pbar.update(global_step)

    last_logged_step = global_step

    while global_step < int(train_cfg.max_steps):
        # Keep one update's batches on CPU to count eligible recordings before
        # backward. GPU activations still live for only one microbatch at a time.
        update_batches = []
        pending_target_tokens = 0
        update_recordings_local = 0
        while pending_target_tokens < budget_tokens:
            try:
                batch = next(it)
                local_exhausted = 0
            except StopIteration:
                batch = None
                local_exhausted = 1

            exhausted_any = distributed_sum(local_exhausted, dev) > 0
            if exhausted_any:
                if pass_input_tokens_global == 0:
                    raise RuntimeError(
                        "No complete training batches were produced on at least one rank. "
                        "Check shard paths and reduce tokens_per_batch or num_workers."
                    )
                passes_completed += 1
                if accelerator.is_main_process:
                    print(
                        f"[data] pass {passes_completed} completed. Step: {global_step}\n"
                        f"Input Tokens: {pass_input_tokens_global}, Target Tokens: {pass_target_tokens_global}, "
                        f"Context Tokens: {pass_context_tokens_global}"
                    )

                pass_input_tokens_global = 0
                pass_target_tokens_global = 0
                pass_context_tokens_global = 0
                current_pass_last_shard_seen = -1
                if resume_midpass_active:
                    loader = _make_loader(all_shards)
                    resume_midpass_active = False
                    active_start_shard_index = 0
                    if accelerator.is_main_process:
                        print("[data-resume] switched back to full shard list for the next pass")
                it = iter(loader)
                continue

            local_input_tokens = int(batch["valid_tokens"])
            local_target_tokens = int(batch["target_tokens"])
            local_context_tokens = int(batch["context_tokens"])

            global_input_tokens = distributed_sum(local_input_tokens, dev)
            global_target_tokens = distributed_sum(local_target_tokens, dev)
            global_context_tokens = distributed_sum(local_context_tokens, dev)

            pass_input_tokens_global += global_input_tokens
            pass_target_tokens_global += global_target_tokens
            pass_context_tokens_global += global_context_tokens

            shard_indices_cpu = batch.get("shard_indices", None)
            if shard_indices_cpu is not None:
                try:
                    if isinstance(shard_indices_cpu, torch.Tensor):
                        valid_si = shard_indices_cpu[shard_indices_cpu >= 0]
                        if valid_si.numel() > 0:
                            current_pass_last_shard_seen = max(current_pass_last_shard_seen, int(valid_si.max().item()))
                    else:
                        valid_si = [int(s) for s in shard_indices_cpu if int(s) >= 0]
                        if len(valid_si) > 0:
                            current_pass_last_shard_seen = max(current_pass_last_shard_seen, max(valid_si))
                except Exception:
                    pass

            if global_target_tokens <= 0:
                raise RuntimeError("A training batch has no target tokens")
            update_batches.append((batch, global_target_tokens))
            pending_target_tokens += global_target_tokens
            update_recordings_local += int(((~batch["pad_tgt"]).sum(dim=1) >= 2).sum().item())

        update_recordings_global = distributed_sum(update_recordings_local, dev)
        accum_tokens_global = 0
        accum_tokens_eff_global = 0
        update_l1_sum_local = torch.zeros((), device=dev)
        update_rsreg_sum_local = torch.zeros((), device=dev)
        spec_lam = rsreg_weight(
            global_step,
            warmup_steps=train_cfg.rsreg_warmup_steps,
            ramp_steps=train_cfg.rsreg_ramp_steps,
            decay_step=train_cfg.rsreg_decay_step,
            max_steps=train_cfg.max_steps,
            weight=train_cfg.rsreg_weight,
            final_weight=train_cfg.rsreg_final_weight,
        ) if rel_proj is not None else 0.0

        for microbatch_index, (batch, global_target_tokens) in enumerate(update_batches):
            will_step = microbatch_index == len(update_batches) - 1
            x = batch["eeg"].to(dev, non_blocking=True)
            coords = batch["coord"].to(dev, non_blocking=True)

            c_ctx = batch["c_ctx"].to(dev, non_blocking=True)
            t_ctx = batch["t_ctx"].to(dev, non_blocking=True)
            pad_ctx = batch["pad_ctx"].to(dev, non_blocking=True)
            c_tgt = batch["c_tgt"].to(dev, non_blocking=True)
            t_tgt = batch["t_tgt"].to(dev, non_blocking=True)
            pad_tgt = batch["pad_tgt"].to(dev, non_blocking=True)

            P_max = int(batch["P_max_cpu"])
            # x = rescale_small_segments(x, target_amp=1.0, quantile=0.90, amp_floor=1e-4, gain_max=200.0, clip=15.0)

            no_sync = (not will_step) and (world_size > 1)
            weight = 1.0
            tokens_eff_this_global = global_target_tokens
            if will_step and budget_tokens > 0:
                remain = max(0, budget_tokens - accum_tokens_global)
                if remain > 0 and global_target_tokens > remain:
                    weight = float(remain) / float(global_target_tokens)
                    tokens_eff_this_global = int(remain)

            with ExitStack() as stack:
                if no_sync:
                    if hasattr(student, "no_sync"):
                        stack.enter_context(student.no_sync())
                    if hasattr(predictor, "no_sync"):
                        stack.enter_context(predictor.no_sync())
                    if rel_proj is not None and hasattr(rel_proj, "no_sync"):
                        stack.enter_context(rel_proj.no_sync())

                x_aug = apply_student_augmentations(
                    x,
                    gain_min=train_cfg.aug_gain_min,
                    gain_max=train_cfg.aug_gain_max,
                    noise_std_min=train_cfg.aug_noise_std_min,
                    noise_std_max=train_cfg.aug_noise_std_max,
                )
                coord_aug = apply_coord_jitter(
                    coords,
                    coord_jitter_std=train_cfg.coord_jitter_std,
                    coord_jitter_prob=train_cfg.coord_jitter_prob
                )

                with accelerator.autocast():
                    # Student branch sees the augmented context view.
                    student_raw = accelerator.unwrap_model(student)

                    coord_ch_student = student_raw.coord_embed(coord_aug)
                    tok_ctx, pad_ctx2, rope_ctx, chan_ctx = student_raw.embed_from_indices(
                        x=x_aug,
                        coords=coord_aug,
                        c_idx=c_ctx,
                        t_idx=t_ctx,
                        pad=pad_ctx,
                        coord_ch=coord_ch_student,
                    )
                    z_ctx = student(tok_ctx, padding_mask=pad_ctx2, rope_pos=rope_ctx, chan_idx=chan_ctx, coords=coord_aug, grid_patches=P_max)

                    z_tgt, cached_tgt_patches = encode_teacher_targets(
                        teacher, x, coords, c_tgt, t_tgt, pad_tgt,
                    )
                    rope_tgt = t_tgt.clamp(min=0)

                    coord_tgt = gather_channel_embeddings(coord_ch_student, c_tgt.clamp(min=0), pad_tgt)
                    pred_tgt = predictor(
                        ctx=z_ctx,
                        ctx_pad=pad_ctx2,
                        rope_ctx=rope_ctx,
                        tgt_coord_emb=coord_tgt,
                        tgt_pad=pad_tgt,
                        rope_tgt=rope_tgt,
                    )

                    valid_tgt = ~pad_tgt
                    pred_main = pred_tgt[valid_tgt].float()
                    tgt_main = z_tgt[valid_tgt].float()

                    # Main JEPA objective, Eq. (1): L1 distance to normalized
                    # EMA-teacher target latents.
                    l1_sum_local = torch.abs(pred_main - tgt_main).sum()
                    loss_scaled = l1_sum_local * (weight * world_size / (budget_tokens * model_cfg.d_model))

                    # Eq. (5): equal weight per eligible recording across this entire
                    # optimizer update, including all ranks and microbatches.
                    rsreg_sum_local = l1_sum_local.new_zeros(())
                    if rel_proj is not None:
                        with torch.autocast(device_type=dev.type, enabled=False):
                            rsreg_sum_local, _ = compute_rsreg_sum(
                                rel_proj=rel_proj,
                                pred_tgt=pred_tgt,
                                cached_tgt_patches=cached_tgt_patches,
                                valid_tgt=valid_tgt,
                                fs=model_cfg.sample_rate,
                                subsample_tokens=train_cfg.rsreg_subsample_tokens,
                                tau_z=train_cfg.rsreg_tau_z,
                                tau_s=train_cfg.rsreg_tau_s,
                                f_min=1.0,
                                f_max=45.0,
                                sync_distributed=True,
                            )
                        # DDP averages gradients, so compensate with world_size.
                        # The fractional token weight applies to L1 only; each
                        # recording contributes once to the RSReg recording mean.
                        loss_scaled = loss_scaled + rsreg_sum_local * (
                            spec_lam * world_size / max(1, update_recordings_global)
                        )

                accelerator.backward(loss_scaled)

            accum_tokens_global += int(global_target_tokens)
            accum_tokens_eff_global += int(tokens_eff_this_global)

            update_l1_sum_local += l1_sum_local.detach() * weight
            update_rsreg_sum_local += rsreg_sum_local.detach()

        tokens_next = int(tokens_seen_total) + int(accum_tokens_eff_global)
        lr_now = token_warmup_cosine_lr(
            tokens_next=tokens_next,
            warmup_tokens=warmup_tokens,
            total_tokens=total_sched_tokens,
            base_lr=float(train_cfg.lr),
            min_lr=float(train_cfg.min_lr),
        )

        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        if float(train_cfg.grad_clip) > 0:
            accelerator.clip_grad_norm_(params, float(train_cfg.grad_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_updater.set_momentum(global_step, int(train_cfg.max_steps), float(train_cfg.ema_momentum), float(train_cfg.ema_momentum_final))
        ema_updater.update()

        tokens_seen_total = tokens_next
        global_step += 1
        pbar.update(1)

        # Reduce update-wide sums on every rank, including empty RSReg ranks.
        metric_sums = torch.stack((update_l1_sum_local, update_rsreg_sum_local))
        metric_sums = accelerator.reduce(metric_sums, reduction="sum")
        l1_mean = float(metric_sums[0].item()) / (budget_tokens * model_cfg.d_model)
        rsreg_mean = float(metric_sums[1].item()) / max(1, update_recordings_global)
        total_loss_mean = l1_mean + spec_lam * rsreg_mean

        log_l1_sum += l1_mean
        log_total_sum += total_loss_mean
        log_count += 1
        if rel_proj is not None and update_recordings_global > 0:
            log_rsreg_sum += rsreg_mean
            log_rsreg_count += 1

        if (global_step - last_logged_step) >= int(train_cfg.log_every):
            denom = max(1, log_count)

            logs = {
                "train/l1_mean": log_l1_sum / denom,
                "train/total_loss": log_total_sum / denom,
                "train/lr": float(lr_now),
                "train/ema_momentum": float(ema_updater.m),
                "train/tokens_seen_total": int(tokens_seen_total),
            }

            if log_rsreg_count > 0:
                logs["train/rsreg"] = log_rsreg_sum / log_rsreg_count
                logs["train/rsreg_weight"] = float(spec_lam)

            if accelerator.is_main_process and not train_cfg.use_wandb:
                print(
                    f"[step {global_step:07d}] l1={logs['train/l1_mean']:.6f} "
                    f"total={logs['train/total_loss']:.6f} "
                    f"lr={logs['train/lr']:.3e} ema={logs['train/ema_momentum']:.6f}"
                )
            accelerator.log(logs, step=global_step)
            log_l1_sum = 0.0
            log_total_sum = 0.0
            log_rsreg_sum = 0.0
            log_rsreg_count = 0
            log_count = 0
            last_logged_step = global_step

        if (int(train_cfg.save_every) > 0) and (global_step % int(train_cfg.save_every) == 0):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(train_cfg.output_dir, f"step_{global_step:07d}")
                save_checkpoint(
                    ckpt_dir,
                    accelerator.unwrap_model(student),
                    teacher,
                    accelerator.unwrap_model(predictor),
                    accelerator.unwrap_model(rel_proj) if rel_proj is not None else None,
                    optimizer,
                    {
                        "global_step": global_step,
                        "passes_completed": passes_completed,
                        "tokens_seen_total": tokens_seen_total,
                        "data_next_shard_index": int(min(len(all_shards), max(0, current_pass_last_shard_seen + 1))),
                        "data_last_shard_index_seen": int(current_pass_last_shard_seen),
                        "data_resume_overlap": int(resume_shard_overlap),
                        "data_shards_total": int(len(all_shards)),
                    },
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(train_cfg.output_dir, "final")
        save_checkpoint(
            final_dir,
            accelerator.unwrap_model(student),
            teacher,
            accelerator.unwrap_model(predictor),
            accelerator.unwrap_model(rel_proj) if rel_proj is not None else None,
            optimizer,
            {
                "global_step": global_step,
                "passes_completed": passes_completed,
                "tokens_seen_total": tokens_seen_total,
                "data_next_shard_index": int(min(len(all_shards), max(0, current_pass_last_shard_seen + 1))),
                "data_last_shard_index_seen": int(current_pass_last_shard_seen),
                "data_resume_overlap": int(resume_shard_overlap),
                "data_shards_total": int(len(all_shards)),
            },
        )
    accelerator.wait_for_everyone()
    return {
        "output_dir": str(train_cfg.output_dir),
        "final_dir": os.path.join(train_cfg.output_dir, "final"),
        "student_dir": os.path.join(train_cfg.output_dir, "final", "student"),
        "teacher_dir": os.path.join(train_cfg.output_dir, "final", "teacher"),
        "predictor_path": os.path.join(train_cfg.output_dir, "final", "predictor.pt"),
    }
