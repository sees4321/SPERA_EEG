from __future__ import annotations

import io
import json
import math
import numpy as np
import os
import random
import sys
import torch
import webdataset as wds

from functools import partial
from torch.utils.data import IterableDataset
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .masking import target_masking, mask_to_packed_indices


EEG_KEY_ALIASES = (
    "eeg.npy", "eeg", "x.npy", "x", "signal.npy", "signal",
)
COORD_KEY_ALIASES = (
    "coords.npy", "coord.npy", "coords", "coord"
)
_BAD_SAMPLE_LOG = os.environ.get("EEGFM_BAD_SAMPLE_LOG", os.environ.get("EEGFM_DEBUG_LOG", ""))
_WARN_MAX = int(os.environ.get("EEGFM_BAD_SAMPLE_WARN_MAX", "32"))
_WARN_COUNT = 0

def read_shards_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        shards = [ln.strip() for ln in f if ln.strip()]
    shards = list(dict.fromkeys(shards))
    if not shards:
        raise RuntimeError(f"No shard paths found in {path}")
    return shards

def _norm_path(p: Any) -> str:
    try:
        return os.path.realpath(os.path.expanduser(str(p)))
    except Exception:
        return str(p)


def build_shard_index_map(shards: List[str]) -> Dict[str, int]:
    return {_norm_path(s): int(i) for i, s in enumerate(shards)}


def _attach_shard_index(sample: Dict[str, Any], shard_to_index: Dict[str, int]) -> Dict[str, Any]:
    try:
        u = sample.get("__url__", "")
        sample["__shard_index__"] = int(shard_to_index.get(_norm_path(u), -1))
    except Exception:
        sample["__shard_index__"] = -1
    return sample


def compute_num_patches(T: int, patch_samples: int) -> int:
    # SPERA tokenization: each EEG channel is split into non-overlapping temporal patches. 
    # So each token is indexed by a (channel, patch-time) pair.
    if T < patch_samples:
        return 0
    return int(T // patch_samples)


def _annotate_tokens(ex: Dict[str, Any], patch_samples: int) -> Dict[str, Any]:
    eeg = ex["eeg"]
    C, T = eeg.shape
    P_t = compute_num_patches(T, patch_samples=patch_samples)
    ex["n_tokens"] = int(C * P_t)
    ex["n_patches"] = int(P_t)
    ex["n_channels"] = int(C)
    return ex


def _as_torch(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (bytes, bytearray)):
        bio = io.BytesIO(x)
        arr = np.load(bio, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if len(arr.files) == 0:
                raise ValueError("Empty NPZ payload")
            arr = arr[arr.files[0]]
        return torch.from_numpy(arr)
    raise TypeError(type(x))


def _key_matches_alias(key: str, alias: str) -> bool:
    base = os.path.basename(str(key)).lower()
    alias_l = alias.lower()
    if base == alias_l:
        return True
    for sep in (".", "_", "-"):
        if base.endswith(sep + alias_l):
            return True
    return base.endswith(alias_l)


def _remove_alias_suffix(key: str, alias: str) -> str:
    base = os.path.basename(str(key))
    base_l = base.lower()
    alias_l = alias.lower()
    if base_l == alias_l:
        return ""
    for sep in (".", "_", "-"):
        token = sep + alias_l
        if base_l.endswith(token):
            return base[: -len(token)]
    if base_l.endswith(alias_l):
        return base[: -len(alias_l)].rstrip("._-")
    return base


def _resolve_candidates(sample: Dict[str, Any], aliases: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for k, v in sample.items():
        ks = str(k)
        if ks.startswith("__"):
            continue
        for alias in aliases:
            if _key_matches_alias(ks, alias):
                base = os.path.basename(ks)
                out.append({
                    "key": ks,
                    "value": v,
                    "alias": alias,
                    "exact": base.lower() == alias.lower(),
                    "stem": _remove_alias_suffix(base, alias),
                })
                break
    out.sort(key=lambda d: (0 if d["exact"] else 1, len(d["key"]), d["key"]))
    return out


def _resolve_pair(sample: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str], Optional[Any], Optional[str]]:
    eegs = _resolve_candidates(sample, EEG_KEY_ALIASES)
    coords = _resolve_candidates(sample, COORD_KEY_ALIASES)
    if not eegs or not coords:
        return None, None, None, None

    for e in eegs:
        for c in coords:
            if e["stem"] and c["stem"] and e["stem"] == c["stem"]:
                return e["value"], e["key"], c["value"], c["key"]

    return eegs[0]["value"], eegs[0]["key"], coords[0]["value"], coords[0]["key"]


def _report_bad_sample(sample: Dict[str, Any], reason: str) -> None:
    global _WARN_COUNT
    payload = {
        "reason": str(reason),
        "__key__": sample.get("__key__"),
        "__url__": sample.get("__url__"),
        "keys": sorted([str(k) for k in sample.keys()]),
    }
    msg = "[bad-sample] " + json.dumps(payload, ensure_ascii=False)
    if _WARN_COUNT < _WARN_MAX:
        print(msg, file=sys.stderr, flush=True)
        _WARN_COUNT += 1
    if _BAD_SAMPLE_LOG:
        try:
            os.makedirs(os.path.dirname(_BAD_SAMPLE_LOG) or ".", exist_ok=True)
            with open(_BAD_SAMPLE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass


def decode_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eeg_raw, eeg_key, coord_raw, coord_key = _resolve_pair(sample)
    if eeg_raw is None or coord_raw is None:
        eeg_cands = [d["key"] for d in _resolve_candidates(sample, EEG_KEY_ALIASES)]
        coord_cands = [d["key"] for d in _resolve_candidates(sample, COORD_KEY_ALIASES)]
        _report_bad_sample(sample, f"missing required field(s): eeg_candidates={eeg_cands}, coord_candidates={coord_cands}")
        return None

    try:
        eeg = _as_torch(eeg_raw).to(torch.float32)
        coord = _as_torch(coord_raw).to(torch.float32)
        norm = torch.linalg.norm(coord, dim=-1, keepdim=True)
        coord = coord / torch.clamp(norm, min=1e-6)
    except Exception as e:
        _report_bad_sample(sample, f"tensor decode failed: {type(e).__name__}: {e}")
        return None

    if eeg.ndim != 2:
        _report_bad_sample(sample, f"unexpected eeg ndim={eeg.ndim}, expected 2 (picked {eeg_key!r})")
        return None
    if coord.ndim != 2:
        _report_bad_sample(sample, f"unexpected coord ndim={coord.ndim}, expected 2 (picked {coord_key!r})")
        return None
    if eeg.shape[0] != coord.shape[0]:
        _report_bad_sample(sample, f"channel mismatch: eeg C={eeg.shape[0]} ({eeg_key!r}) vs coord C={coord.shape[0]} ({coord_key!r})")
        return None
    if eeg.shape[1] <= 0:
        _report_bad_sample(sample, f"empty time dimension T={eeg.shape[1]} (picked {eeg_key!r})")
        return None
    if not torch.isfinite(eeg).all():
        _report_bad_sample(sample, "nonfinite_eeg")
        return None
    if not torch.isfinite(coord).all():
        _report_bad_sample(sample, "nonfinite_coord")
        return None
    return {
        "eeg": eeg.contiguous(),
        "coord": coord.contiguous(),
        "__url__": sample.get("__url__"),
        "__key__": sample.get("__key__"),
        "__shard_index__": int(sample.get("__shard_index__", -1)),
    }


def _flatmap_stage(mapper):
    def _iter(src):
        for ex in src:
            out = mapper(ex)
            if out is None:
                continue
            if isinstance(out, dict):
                yield out
            else:
                for y in out:
                    if y is not None:
                        yield y
    return _iter


def _split_one_view_to_fit(
    ex: Dict[str, Any],
    max_tokens: int,
    patch_samples: int,
    prefer_chunk_sec=(30, 10),
) -> List[Dict[str, Any]]:
    eeg = ex["eeg"]
    fs = 200
    C, T = eeg.shape
    P = compute_num_patches(T, patch_samples=patch_samples)
    if C * P < 2:
        return []
    if C * P <= max_tokens:
        return [_annotate_tokens(ex, patch_samples)]

    P_max = max_tokens // C
    if P_max < 1:
        return []

    chunk_P = None
    for sec in prefer_chunk_sec:
        P_sec = compute_num_patches(sec * fs, patch_samples=patch_samples)
        if 0 < P_sec <= P_max and P_sec <= P:
            chunk_P = P_sec
            break
    if chunk_P is None:
        chunk_P = min(P, P_max)

    outs = []
    n_chunks = math.ceil(P / chunk_P)
    for i in range(n_chunks):
        p0 = i * chunk_P
        curP = min(chunk_P, P - p0)
        if C * curP < 2:
            continue
        start = p0 * patch_samples
        T_need = curP * patch_samples
        out = dict(ex)
        out["eeg"] = eeg[:, start:start + T_need].contiguous()
        outs.append(_annotate_tokens(out, patch_samples))
    return outs


def split_and_fit(
    ex: Dict[str, Any],
    max_tokens: int,
    patch_samples: int,
) -> List[Dict[str, Any]]:
    # Keep each example under the encoder token budget. 
    # This follows the pretraining recipe where long 60-s EEG segments are split into shorter contiguous chunks when C * P_t exceeds max_tokens.
    return _split_one_view_to_fit(
        ex,
        max_tokens=max_tokens,
        patch_samples=patch_samples,
        prefer_chunk_sec=(30, 10),
    )


def collate_stack(
    batch: List[Dict[str, Any]],
    patch_samples: int,
    target_mask_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    # ShapeBatcher groups examples with identical (C, P_t).
    # This allows dense C x P_t token grids without padding across different montages.
    assert len(batch) > 0
    C = int(batch[0]["n_channels"])
    P = int(batch[0]["n_patches"])
    for b in batch:
        assert int(b["n_channels"]) == C and int(b["n_patches"]) == P

    T_need = P * patch_samples if P > 0 else 0
    eeg = torch.stack([b["eeg"][:, :T_need].contiguous() for b in batch], dim=0)
    coord = torch.stack([b["coord"].contiguous() for b in batch], dim=0)

    B = len(batch)
    n_channels = torch.full((B,), C, dtype=torch.long)
    n_patches = torch.full((B,), P, dtype=torch.long)

    target_mask = target_masking(
        coords=coord,
        P_t=P,
        mask_time_prob=float(target_mask_cfg["mask_time_prob"]),
        mask_spatial_prob=float(target_mask_cfg["mask_spatial_prob"]),
        time_ratio_range=target_mask_cfg["time_ratio_range"],
        spatial_ratio_range=target_mask_cfg["spatial_ratio_range"],
    )
    context_mask = ~target_mask
    c_ctx, t_ctx, pad_ctx = mask_to_packed_indices(context_mask)
    c_tgt, t_tgt, pad_tgt = mask_to_packed_indices(target_mask)

    return {
        "eeg": eeg,
        "coord": coord,
        "n_channels": n_channels,
        "n_patches": n_patches,
        "valid_tokens": int(B * C * P),
        "target_tokens": int(target_mask.sum()),
        "context_tokens": int(context_mask.sum()),
        "P_max_cpu": int(P),
        "C_max_cpu": int(C),
        "target_mask": target_mask,
        "context_mask": context_mask,
        "c_ctx": c_ctx,
        "t_ctx": t_ctx,
        "pad_ctx": pad_ctx,
        "c_tgt": c_tgt,
        "t_tgt": t_tgt,
        "pad_tgt": pad_tgt,
        "shard_indices": torch.tensor([int(b.get("__shard_index__", -1)) for b in batch], dtype=torch.long),
        "sample_keys": [str(b.get("__key__", "")) for b in batch],
        "sample_urls": [str(b.get("__url__", "")) for b in batch],
    }


class ShapeBatcher(IterableDataset):
    """
    Exact same-shape batching by (n_channels, n_patches).
    We deliberately default to dropping incomplete or expired buffers, because that is more DDP-friendly than emitting tiny tail batches.
    """
    def __init__(
        self,
        dataset: Iterable[Dict[str, Any]],
        tokens_per_batch: int,
        max_samples_per_batch: int,
        patch_samples: int,
        target_mask_cfg: Dict[str, Any],
        max_wait_samples: int = 5000,
        flush_check_every: int = 256,
        max_pending_samples: int = 512,
        max_pending_tokens: int = 0,
        shuffle_within_bucket: bool = True,
        yield_incomplete: bool = False,
    ):
        self.dataset = dataset
        self.tokens_per_batch = int(tokens_per_batch)
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.patch_samples = int(patch_samples)
        self.target_mask_cfg = target_mask_cfg

        self.max_wait_samples = int(max_wait_samples)
        self.flush_check_every = int(flush_check_every)
        self.max_pending_samples = int(max_pending_samples)
        self.max_pending_tokens = int(max_pending_tokens)
        self.shuffle_within_bucket = bool(shuffle_within_bucket)
        self.yield_incomplete = bool(yield_incomplete)

    def _target_bs(self, C: int, P: int) -> int:
        tokens_per_sample = C * P
        bs = self.tokens_per_batch // max(1, tokens_per_sample)
        bs = max(1, bs)
        return min(bs, self.max_samples_per_batch)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        first_seen: Dict[Tuple[int, int], int] = {}
        seen = 0
        pending_samples = 0
        pending_tokens = 0

        def _pop_bucket(key: Tuple[int, int]) -> List[Dict[str, Any]]:
            nonlocal pending_samples, pending_tokens
            buf = buckets.pop(key, [])
            first_seen.pop(key, None)
            if buf:
                pending_samples -= len(buf)
                pending_tokens -= sum(int(x.get("n_tokens", 0)) for x in buf)
            return buf

        def _flush_key(key: Tuple[int, int]):
            buf = _pop_bucket(key)
            if not buf:
                return
            if self.yield_incomplete:
                yield collate_stack(
                    buf,
                    patch_samples=self.patch_samples,
                    target_mask_cfg=self.target_mask_cfg,
                )

        def _flush_oldest_until_under_limits():
            while True:
                over_samples = (self.max_pending_samples > 0) and (pending_samples > self.max_pending_samples)
                over_tokens = (self.max_pending_tokens > 0) and (pending_tokens > self.max_pending_tokens)
                if not (over_samples or over_tokens):
                    break
                if not first_seen:
                    break
                oldest = min(first_seen, key=first_seen.get)
                for out in _flush_key(oldest):
                    yield out

        def _flush_expired():
            if self.max_wait_samples <= 0:
                return
            expired = []
            for k, t0 in first_seen.items():
                if (seen - t0) >= self.max_wait_samples:
                    expired.append(k)
            for k in expired:
                for out in _flush_key(k):
                    yield out

        for ex in self.dataset:
            seen += 1
            C = int(ex.get("n_channels", 0))
            P = int(ex.get("n_patches", 0))
            if C <= 0 or P <= 0 or C * P < 2:
                continue

            key = (C, P)
            if key not in buckets:
                buckets[key] = []
                first_seen[key] = seen

            buckets[key].append(ex)
            pending_samples += 1
            pending_tokens += int(ex.get("n_tokens", C * P))

            bs_target = self._target_bs(C, P)
            buf = buckets[key]
            if self.shuffle_within_bucket and len(buf) == bs_target:
                random.shuffle(buf)

            while len(buf) >= bs_target:
                batch = buf[:bs_target]
                del buf[:bs_target]
                pending_samples -= bs_target
                pending_tokens -= sum(int(x.get("n_tokens", C * P)) for x in batch)

                yield collate_stack(
                    batch,
                    patch_samples=self.patch_samples,
                    target_mask_cfg=self.target_mask_cfg,
                )
                if len(buf) > 0:
                    first_seen[key] = seen
                else:
                    buckets.pop(key, None)
                    first_seen.pop(key, None)
                    break

            for out in _flush_oldest_until_under_limits():
                yield out

            if (seen % self.flush_check_every) == 0:
                for out in _flush_expired():
                    yield out

        if self.yield_incomplete:
            for k in list(buckets.keys()):
                for out in _flush_key(k):
                    yield out


def _map_attach_shard_index(sample, shard_to_index):
    return _attach_shard_index(sample, shard_to_index)


def _map_split_and_fit(ex, max_tokens, patch_samples):
    return split_and_fit(ex, max_tokens=max_tokens, patch_samples=patch_samples)


def _map_annotate_tokens(ex, patch_samples):
    return _annotate_tokens(ex, patch_samples=patch_samples)


def build_webdataset(
    shards: List[str],
    shard_shuffle: int,
    sample_shuffle: int,
    max_tokens: int,
    patch_samples: int,
    post_split_shuffle: int = 128,
    seed: int = 0,
    shard_to_index: Optional[Dict[str, int]] = None,
) -> Iterable[Dict[str, Any]]:
    if wds is None:
        raise RuntimeError("webdataset is not installed")

    src = wds.SimpleShardList(shards) if hasattr(wds, "SimpleShardList") else wds.shardlists.SimpleShardList(shards)
    if shard_to_index is None:
        shard_to_index = build_shard_index_map(shards)

    pipeline = [src]

    if shard_shuffle > 0:
        if hasattr(wds, "detshuffle"):
            pipeline.append(wds.detshuffle(bufsize=shard_shuffle, initial=min(shard_shuffle, 32), seed=seed))
        else:
            pipeline.append(wds.shuffle(shard_shuffle))

    pipeline += [wds.split_by_node, wds.split_by_worker]

    pipeline += [
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.map(lambda s: _attach_shard_index(s, shard_to_index)),
        wds.shuffle(sample_shuffle),
        wds.decode(),
        wds.map(decode_sample),
        _flatmap_stage(partial(_map_split_and_fit,
                               max_tokens=max_tokens,
                               patch_samples=patch_samples)),
        # _flatmap_stage(lambda ex: split_and_fit(
        #     ex, 
        #     max_tokens=max_tokens, 
        #     patch_samples=patch_samples,)),
        wds.shuffle(post_split_shuffle),
        wds.map(partial(_map_annotate_tokens, patch_samples=patch_samples)),
        # wds.map(lambda ex: _annotate_tokens(ex, patch_samples=patch_samples,)),
    ]
    return wds.DataPipeline(*pipeline)
