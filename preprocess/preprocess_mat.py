import argparse
import mne
import numpy as np
import os
import re


def build_channel_coords(channel_names, montage_name="standard_1005"):
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    coords = []
    for ch in channel_names:
        ch = ch.replace("EEG ", "")

        if "-" in ch:  # bipolar channel (A2-A1), use midpoint of two electrodes
            a, b = ch.split("-")
            pos_a = next((p for n, p in pos.items() if n.upper() == a.upper()), None)
            pos_b = next((p for n, p in pos.items() if n.upper() == b.upper()), None)
            if pos_a is not None and pos_b is not None:
                coords.append(((pos_a + pos_b) / 2)[:3])
            continue

        for m_name, m_pos in pos.items():
            if m_name.upper() == ch.upper():
                coords.append(m_pos[:3])
                break

    return np.asarray(coords)


def preprocess_mat(args):
    out_dir = args.output_dir
    root_dir = args.root_dir

    files_list = sorted(
        (f for f in os.listdir(root_dir) if f.endswith('.edf')),
        key=lambda name: tuple(int(x) for x in re.findall(r"\d+", name)),
    )
    files_dict = { # (2 files per subjects)
        'train': files_list[:56], # Subjects 1-28 
        'validation': files_list[56:64], # Subjects 29-32
        'test': files_list[64:], # Subjects 33-36
    }
    raw = mne.io.read_raw_edf(os.path.join(root_dir, files_dict['train'][0]), preload=False, verbose=False)
    coords = build_channel_coords(raw.ch_names[:20])

    for split in ['train', 'validation', 'test']:
        eeg_list = []
        label_list = []

        for file in files_dict[split]:
            raw = mne.io.read_raw_edf(os.path.join(root_dir, file), preload=True, verbose=False)
            raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
            raw.resample(200)
            eeg = raw.get_data(units='uV')[:20] # remove ECG channel
            leftover = eeg.shape[-1] % (5*200) # window = 5 s
            if leftover != 0: # remove leftover samples to fit 200Hz * 5s segments
                eeg = eeg[:,:-leftover]
            eeg = eeg.reshape(20, -1, 5*200).transpose(1, 0, 2)
            label = int(file[-5]) - 1
            eeg_list.append(eeg)
            label_list.append([label]*eeg.shape[0])

        eeg_list = np.concatenate(eeg_list)
        label_list = np.concatenate(label_list)
        os.makedirs(os.path.join(out_dir,split), exist_ok=True)
        np.save(os.path.join(out_dir,split,"eeg.npy"), eeg_list)
        np.save(os.path.join(out_dir,split,"label.npy"), label_list)
        np.save(os.path.join(out_dir,split,"coords.npy"), np.stack([coords]*len(label_list)))

        print(split, eeg_list.shape, label_list.shape)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/mat/")
    p.add_argument("--output_dir", type=str, default="/mat_preprocessed/")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_mat(args)


if __name__ == "__main__":
    main()
