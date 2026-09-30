import argparse
import os
import re
from glob import glob

import mne
import numpy as np
import pandas as pd


TARGET_CHANNELS = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
STAGE_MAP = {
    # AASM letters
    'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4,
    # Numeric (some ISRUC variants use these in their .txt/.xlsx)
    '0': 0, '1': 1, '2': 2, '3': 3, '5': 4,
}


def build_coords(channels, montage_name="standard_1005"):
    pos = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    return np.asarray([pos[ch][:3] for ch in channels])


def clean_ch(ch):
    """ISRUC channel like 'F3-A2' -> 'F3'."""
    if ch[0] not in 'FCO':
        return ch
    return ch.split('-')[0].strip().upper()


def find_eeg_path(raw_dir, subject_id):
    """Find file ending with {subject_id}.rec or {subject_id}.edf (no digit immediately before)."""
    pat = re.compile(rf'(?<!\d){subject_id}\.(rec|edf)$', re.IGNORECASE)
    for f in glob(os.path.join(raw_dir, f"*{subject_id}*")):
        if pat.search(f):
            return f
    return None


def process_subject(subject_id, raw_dir, target_dir, seq_len, win_s=30, target_sr=200):
    eeg_path = find_eeg_path(raw_dir, subject_id)
    label_path = os.path.join(target_dir, f"{subject_id}_1.xlsx")
    if not eeg_path or not os.path.exists(label_path):
        return None, None

    # MNE only reads .edf extension; rename .rec -> .edf if needed
    if eeg_path.lower().endswith('.rec'):
        new_path = eeg_path[:-4] + ".edf"
        if not os.path.exists(new_path):
            os.rename(eeg_path, new_path)
        eeg_path = new_path

    stages = pd.read_excel(label_path, engine='openpyxl').iloc[:, 1].values

    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    # raw.resample(target_sr, verbose=False)            # already 200 Hz
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)    # band-pass
    raw.notch_filter(60, verbose=False)                 # notch
    try:
        raw.rename_channels({ch: clean_ch(ch) for ch in raw.ch_names})
    except Exception:
        pass

    if not all(ch in raw.ch_names for ch in TARGET_CHANNELS):
        print(f"  subject {subject_id}: missing channels, skip ({raw.ch_names})")
        return None, None
    raw.pick(TARGET_CHANNELS)
    raw.reorder_channels(TARGET_CHANNELS)

    data = raw.get_data(units='uV').astype(np.float32)
    win_len = win_s * target_sr   # 6000

    eegs, labels = [], []
    for i, stage in enumerate(stages):
        label = STAGE_MAP.get(str(stage).strip().upper())
        if label is None:                          # skip unknown labels (NaN, MT, etc.)
            continue
        s, e = i * win_len, (i + 1) * win_len
        if e > data.shape[-1]:                     # signal exhausted
            break
        eegs.append(data[:, s:e])
        labels.append(label)

    # trim per-subject to a multiple of seq_len so sequences come from the same file,
    # and trial counts stay identical whether `--sequence` is set or not.
    n_keep = (len(eegs) // seq_len) * seq_len
    if n_keep == 0:
        return None, None
    return np.stack(eegs[:n_keep]), np.asarray(labels[:n_keep])


def preprocess_isruc(args):
    out_dir = args.output_dir
    raw_dir = os.path.join(args.root_dir, "raw_data")
    target_dir = os.path.join(args.root_dir, "target_data")
    seq_len = args.seq_len

    files_dict = {
        "train":      list(range(1, 81)),    # subjects 1-80
        "validation": list(range(81, 91)),   # subjects 81-90
        "test":       list(range(91, 101)),  # subjects 91-100
    }

    coords = build_coords(TARGET_CHANNELS)

    for split in ["train", "validation", "test"]:
        eeg_list, label_list = [], []
        for sid in files_dict[split]:
            eeg, label = process_subject(sid, raw_dir, target_dir, seq_len)
            if eeg is None:
                continue
            eeg_list.append(eeg)
            label_list.append(label)

        eeg_arr   = np.concatenate(eeg_list)                    # (T, C, t)
        label_arr = np.concatenate(label_list)                  # (T,)
        coords_arr = np.stack([coords] * len(label_arr))        # (T, C, 3)

        if args.sequence:
            # group consecutive epochs from same subject -> (N, S, C, t)
            eeg_arr    = eeg_arr.reshape(-1, seq_len, *eeg_arr.shape[1:])
            label_arr  = label_arr.reshape(-1, seq_len)
            coords_arr = coords_arr.reshape(-1, seq_len, *coords_arr.shape[1:])

        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        np.save(os.path.join(out_dir, split, "eeg.npy"), eeg_arr)
        np.save(os.path.join(out_dir, split, "label.npy"), label_arr)
        np.save(os.path.join(out_dir, split, "coords.npy"), coords_arr)

        print(split, eeg_arr.shape, label_arr.shape)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/isruc/archive/")
    p.add_argument("--output_dir", type=str, default="/isruc_preprocessed/")
    p.add_argument("--seq_len", type=int, default=20,
                   help="epochs per sequence (also used to trim trailing trials)")
    p.add_argument("--sequence", action="store_true",
                   help="reshape outputs into (sequences, seq_len, ...) for seq-to-seq sleep staging")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_isruc(args)


if __name__ == "__main__":
    main()