import argparse
import mne
import numpy as np
import os
import warnings

from scipy.signal import butter, lfilter, resample
from moabb.datasets import BNCI2014_001


def bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def _matrix_inv_sqrt_spd(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Regularized inverse square root, matching the reference EA script."""
    mat = 0.5 * (mat + mat.T)
    c = mat.shape[0]
    reg = eps * max(float(np.trace(mat)) / max(c, 1), 1.0)
    mat = mat + reg * np.eye(c, dtype=mat.dtype)

    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt.astype(np.float32)


def euclidean_align_trials(trials: np.ndarray) -> np.ndarray:
    """Align a subject's trials (N, C, T) using their mean covariance.

    Pass signals in volts to match the reference script: its absolute ridge
    floor makes the transform depend on the input amplitude unit.
    """
    if trials.ndim != 3:
        raise ValueError(f"Expected [N, C, T], got {trials.shape}")
    n_trials, n_ch, _ = trials.shape
    if n_trials == 0:
        return trials

    cov_sum = np.zeros((n_ch, n_ch), dtype=np.float64)
    for x in trials:
        x = x.astype(np.float64, copy=False)
        x = x - x.mean(axis=-1, keepdims=True)
        cov = (x @ x.T) / max(x.shape[-1], 1)
        cov_sum += 0.5 * (cov + cov.T)

    ref_cov = (cov_sum / n_trials).astype(np.float32)
    ref_inv_sqrt = _matrix_inv_sqrt_spd(ref_cov)
    aligned = np.einsum("ij,njt->nit", ref_inv_sqrt, trials.astype(np.float32), optimize=True)
    return aligned.astype(np.float32)


def build_channel_coords(channel_names, montage_name="standard_1005"):
    NUM_TO_CH_NAME = {
        "0": "FC3", "1": "FC1", "2": "FCz", "3": "FC2", "4": "FC4",
        "5": "C5", "6": "C1", "7": "C2", "8": "C6",
        "9": "CP3", "10": "CP1", "11": "CPz", "12": "CP2", "13": "CP4",
        "14": "P1", "15": "P2", "16": "POz",
    }

    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    coords = []
    for ch in channel_names:
        ch = ch.replace("EEG-", "")

        if ch in NUM_TO_CH_NAME:
            ch = NUM_TO_CH_NAME[ch]

        for m_name, m_pos in pos.items():
            if m_name.upper() == ch.upper():
                coords.append(m_pos[:3])
                break

    return np.asarray(coords)


def preprocess_bciciv2a(args):
    out_dir = args.output_dir

    splits = {
        "train": range(1, 6),
        "validation": range(6, 8),
        "test": range(8, 10),
    }

    fs = 250

    classes = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "tongue": 3,
    }

    dataset = BNCI2014_001()
    b, a = bandpass(0.5, 99.5, fs)

    coords = None

    for split, subject_range in splits.items():
        eeg_list = []
        label_list = []

        for subject in subject_range:
            subject_trials = []
            subject_labels = []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                subject_data = dataset.get_data(subjects=[subject])[subject]

            for session in subject_data.values():
                for raw in session.values():
                    if coords is None:
                        coords = build_channel_coords(raw.info["ch_names"][:22])

                    raw.pick("eeg")
                    raw_data = raw.get_data(units="uV")
                    events, event_id = mne.events_from_annotations(raw, verbose=False)

                    events_triggers, events_labels = [], []

                    for e in events:
                        event_type = e[2]

                        desc = None
                        for k, v in event_id.items():
                            if v == event_type:
                                desc = k
                                break

                        if desc in classes:
                            events_triggers.append(e[0])
                            events_labels.append(classes[desc])

                    for i in range(len(events_triggers)):
                        start_idx = events_triggers[i]
                        end_idx = (
                            events_triggers[i + 1]
                            if i < len(events_triggers) - 1
                            else raw_data.shape[-1]
                        )

                        sample = raw_data[:22, start_idx:end_idx]
                        sample = sample - np.mean(sample, axis=0, keepdims=True)
                        sample = lfilter(b, a, sample, axis=-1)
                        sample = sample[:, 0:4*fs] # MOABB event onset is cue onset; 0–4s here equals trial [2, 6]s.
                        sample = resample(sample, 800, axis=-1)

                        subject_trials.append(sample)
                        subject_labels.append(events_labels[i])

            if not subject_trials:
                continue
            # Reference default: one EA transform for both sessions of this
            # subject. No covariance is pooled across subjects or data splits.
            # Existing extraction uses microvolts; the reference EA uses volts.
            trials = np.stack(subject_trials).astype(np.float32) * 1e-6
            aligned = euclidean_align_trials(trials)
            eeg_list.extend(aligned)
            label_list.extend(subject_labels)

        eeg_list = np.asarray(eeg_list)
        label_list = np.asarray(label_list)

        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, "eeg.npy"), eeg_list)
        np.save(os.path.join(split_dir, "label.npy"), label_list)
        np.save(os.path.join(split_dir, "coords.npy"), np.stack([coords] * len(label_list)))

        print(split, eeg_list.shape, label_list.shape)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="/bciciv2a_preprocessed/")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_bciciv2a(args)


if __name__ == "__main__":
    main()
