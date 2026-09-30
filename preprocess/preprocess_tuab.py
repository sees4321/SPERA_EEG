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
LABEL_MAP = {"normal": 0, "abnormal": 1}


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


def subject_id(path):
    """TUAB filename 'aaaaaguk_s002_t001.edf' -> 'aaaaaguk'."""
    return os.path.basename(path).split("_")[0]


def label_of(path):
    """Infer label from any 'normal'/'abnormal' folder in the path."""
    parts = path.lower().replace("\\", "/").split("/")
    if "abnormal" in parts:
        return 1
    if "normal" in parts:
        return 0
    raise ValueError(f"Cannot infer label from {path}")


def process_edf(edf_path, win_s=10, target_sr=200):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(target_sr, verbose=False)      # resample
    raw.filter(0.3, 75., verbose=False)         # band-pass
    raw.notch_filter(60., verbose=False)        # notch

    idx = find_ref_indices(raw.ch_names)
    data = raw.get_data(units="uV")
    bipolar = np.stack([data[idx[a]] - data[idx[b]] for a, b in BIPOLAR_PAIRS])

    win_len = win_s * target_sr
    leftover = bipolar.shape[-1] % win_len
    if leftover != 0:                                # drop trailing remainder < win_s
        bipolar = bipolar[:, :-leftover]
    eeg = bipolar.reshape(16, -1, win_len).transpose(1, 0, 2).astype(np.float32)
    label = np.full(eeg.shape[0], label_of(edf_path), dtype=np.int64)
    return eeg, label


def preprocess_tuab(args):
    out_dir = args.output_dir
    root_dir = args.root_dir
    coords = build_bipolar_coords()

    # classwise 80:20 train/val split (abnormal/normal subjects split independently)
    files_dict = {"train": [], "validation": [], "test": []}
    for cls in ["abnormal", "normal"]:
        train_cls = sorted(glob(os.path.join(root_dir, "train", cls, "**", "*.edf"), recursive=True))
        test_cls  = sorted(glob(os.path.join(root_dir, "eval",  cls, "**", "*.edf"), recursive=True))
        subjects = sorted({subject_id(f) for f in train_cls})
        n_train = int(len(subjects) * 0.8)
        train_subj = set(subjects[:n_train])
        files_dict["train"]      += [f for f in train_cls if subject_id(f) in train_subj]
        files_dict["validation"] += [f for f in train_cls if subject_id(f) not in train_subj]
        files_dict["test"]       += test_cls

    for split in ["train", "validation", "test"]:
        eeg_list, label_list = [], []
        for file in files_dict[split]:
            eeg, label = process_edf(file)
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
    p.add_argument("--root_dir", type=str, default="/tuab/edf/")
    p.add_argument("--output_dir", type=str, default="/tuab_preprocessed/")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_tuab(args)


if __name__ == "__main__":
    main()