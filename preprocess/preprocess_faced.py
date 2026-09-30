import argparse
import mne
import numpy as np
import os
import pickle

from scipy.signal import resample

def build_channel_coords(montage_name="standard_1005"):
    # The original recordings were collected using two cohort-specific channel layouts.
    # After preprocessing, the first cohort's channel order is remapped to match the second cohort order.
    channel_names =  [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4",
    "T7", "T8",
    "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4",
    "P7", "P8",
    "PO3", "PO4",
    "Oz", "O1", "O2",
    "A2", "A1",
    ]
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    coords = []
    for ch in channel_names:
        coords.append(pos[ch][:3])

    return np.asarray(coords)

def preprocess_faced(args):
    out_dir = args.output_dir
    root_dir = args.root_dir
    files_list = sorted(os.listdir(root_dir))
    files_dict = {
        'train': files_list[:80],           # Subjects 1-80
        'validation': files_list[80:100],   # Subjects 81-100
        'test': files_list[100:],           # Subjects 101-123
    }
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])
    coords = build_channel_coords()

    for split in ['train', 'validation', 'test']:
        eeg_list = []
        label_list = []

        for file in files_dict[split]:
            with open(os.path.join(root_dir, file), 'rb') as f:
                eeg = pickle.load(f)
            eeg = resample(eeg, 6000, axis=-1)

            for i, (sample, label) in enumerate(zip(eeg, labels)):
                for j in range(3): # 10 s window * 3
                    eeg_list.append(sample[:,10*j*200:10*(j+1)*200]) 
                    label_list.append(label)

        eeg_list = np.array(eeg_list)
        label_list = np.array(label_list)
        os.makedirs(os.path.join(out_dir,split), exist_ok=True)
        np.save(os.path.join(out_dir,split,"eeg.npy"), eeg_list)
        np.save(os.path.join(out_dir,split,"label.npy"), label_list)
        np.save(os.path.join(out_dir,split,"coords.npy"), np.stack([coords]*len(label_list)))

        print(split, eeg_list.shape, label_list.shape)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/faced/")
    p.add_argument("--output_dir", type=str, default="/faced_preprocessed/")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_faced(args)


if __name__ == "__main__":
    main()