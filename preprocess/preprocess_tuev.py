import argparse
import mne
import numpy as np
import os
from glob import glob


# 16 bipolar derivations (BIOT/CBraMod convention)
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
# legacy (T3/T4/T5/T6) <-> 10-05 (T7/T8/P7/P8), FP1/FP2 -> Fp1/Fp2
MONTAGE_MAP = {"FP1": "Fp1", "FP2": "Fp2", "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
LABEL_MAP = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
             "spsw": 0, "gped": 1, "pled": 2, "eyem": 3, "artf": 4, "bckg": 5}


def build_bipolar_coords(montage_name="standard_1005"):
    pos = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    coords = [((pos[MONTAGE_MAP.get(a, a)] + pos[MONTAGE_MAP.get(b, b)]) / 2)[:3]
              for a, b in BIPOLAR_PAIRS]
    return np.asarray(coords)


def find_ref_indices(ch_names):
    """Map referential names (FP1/F7/.../O2) to channel indices, accepting -REF/-LE suffixes and T7/T8/P7/P8 aliases."""
    idx = {}
    for i, ch in enumerate(ch_names):
        s = ch.upper().replace("EEG ", "").split("-")[0].strip()
        s = ALIAS.get(s, s)
        idx.setdefault(s, i)
    return idx


def parse_rec(rec_path):
    """TUEV .rec line format: channel,start_sec,stop_sec,label"""
    events = []
    with open(rec_path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 4:
                continue
            try:
                start, stop = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            label = LABEL_MAP.get(parts[3].lower())
            if label is not None:
                events.append((start, stop, label))
    return events


def process_edf(edf_path, win_s=5, target_sr=200):
    rec_path = edf_path.replace(".edf", ".rec")
    if not os.path.exists(rec_path):
        return None, None
    events = parse_rec(rec_path)
    if not events:
        return None, None

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(target_sr, verbose=False)      # resample
    raw.filter(0.3, 75., verbose=False)         # band-pass
    raw.notch_filter(60., verbose=False)        # notch

    idx = find_ref_indices(raw.ch_names)
    data = raw.get_data(units="uV")
    bipolar = np.stack([data[idx[a]] - data[idx[b]] for a, b in BIPOLAR_PAIRS])

    win_len = win_s * target_sr
    n = bipolar.shape[-1]
    eegs, labels = [], []
    for start, stop, label in events:
        center = int(round((start + stop) / 2 * target_sr))
        s, e = center - win_len // 2, center + win_len // 2
        window = bipolar[:, np.arange(s, e) % n]   # circular wrap at edges
        eegs.append(window.astype(np.float32))
        labels.append(label)
    return np.stack(eegs), np.asarray(labels)


def preprocess_tuev(args):
    out_dir = args.output_dir
    root_dir = args.root_dir

    train_files = sorted(glob(os.path.join(root_dir, "train", "**", "*.edf"), recursive=True))
    test_files  = sorted(glob(os.path.join(root_dir, "eval",  "**", "*.edf"), recursive=True))

    # 80:20 train/val split by subject (filename prefix before first '_')
    subjects = sorted({os.path.basename(f).split("_")[0] for f in train_files})
    n_train = int(len(subjects) * 0.8)
    train_subj = set(subjects[:n_train])
    files_dict = {
        "train":      [f for f in train_files if os.path.basename(f).split("_")[0] in train_subj],
        "validation": [f for f in train_files if os.path.basename(f).split("_")[0] not in train_subj],
        "test":       test_files,
    }

    coords = build_bipolar_coords()

    for split in ["train", "validation", "test"]:
        eeg_list, label_list = [], []
        for file in files_dict[split]:
            eeg, label = process_edf(file)
            if eeg is None:
                continue
            eeg_list.append(eeg)
            label_list.append(label)

        eeg_list = np.concatenate(eeg_list)
        label_list = np.concatenate(label_list)
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        np.save(os.path.join(out_dir, split, "eeg.npy"), eeg_list)
        np.save(os.path.join(out_dir, split, "label.npy"), label_list)
        np.save(os.path.join(out_dir, split, "coords.npy"), np.stack([coords] * len(label_list)))

        print(split, eeg_list.shape, label_list.shape)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/tuev/edf/")
    p.add_argument("--output_dir", type=str, default="/tuev_preprocessed/")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_tuev(args)


if __name__ == "__main__":
    main()