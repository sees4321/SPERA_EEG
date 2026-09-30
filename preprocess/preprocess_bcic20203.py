import argparse
import h5py
import mne
import numpy as np
import os
import pandas as pd

from scipy.signal import resample
from scipy.io import loadmat


def build_channel_coords(channel_names, montage_name="standard_1005"):
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    coords = []
    for ch in channel_names:
        for m_name, m_pos in pos.items():
            if m_name.upper() == ch.upper():
                coords.append(m_pos[:3])
                break

    return np.asarray(coords)


def extract_test_labels(file_path="Track3_Answer Sheet_Test.xlsx"):
    df = pd.read_excel(file_path, header=None)
    labels = df.iloc[3:53, 2::2].values
    labels = labels.astype(int)
    result_array = labels.T.flatten()

    return np.array(result_array) - 1


def preprocess_bcic20203(args):
    split_dir = {
        'train': os.path.join(args.root_dir, "Training set"),
        'validation': os.path.join(args.root_dir, "Validation set"),
        'test': os.path.join(args.root_dir, "Test set")
    }
    out_dir = args.output_dir
    test_lab_dir = args.test_label_dir

    files_dict = {
        'train': sorted([file for file in os.listdir(split_dir['train'])]),
        'validation': sorted([file for file in os.listdir(split_dir['validation'])]),
        'test': sorted([file for file in os.listdir(split_dir['test'])]),
    }

    # fs and channel coords are identical across the splits
    data = loadmat(os.path.join(split_dir['train'], files_dict['train'][0]))['epo_train'][0][0]
    fs = int(data[1][0][0])
    coords = build_channel_coords(np.concatenate(data[0][0]))


    for split in ['train', 'validation', 'test']:
        eeg_list = []
        label_list = []

        for file in files_dict[split]:
            if split != 'test':
                data = loadmat(os.path.join(split_dir[split], file))['epo_'+split][0][0]
                eeg = data[4].transpose(2,1,0)[:,:,-(3*fs):] # window = 3 s
                label = np.argmax(data[5].transpose(1,0), axis=1)
                label_list.append(label)
            else:
                data = h5py.File(os.path.join(split_dir[split], file))
                eeg = data['epo_'+split]['x'][:,:,-(3*fs):]

            eeg = resample(eeg, 600, axis=2)
            eeg_list.append(eeg)

        eeg_list = np.concatenate(eeg_list)
        label_list = np.concatenate(label_list) if split != 'test' else extract_test_labels(test_lab_dir)
        os.makedirs(os.path.join(out_dir,split), exist_ok=True)
        np.save(os.path.join(out_dir,split,"eeg.npy"), eeg_list)
        np.save(os.path.join(out_dir,split,"label.npy"), label_list)
        np.save(os.path.join(out_dir,split,"coords.npy"), np.stack([coords]*len(label_list)))

        print(split, eeg_list.shape, label_list.shape)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/bcic2020_03/")
    p.add_argument("--output_dir", type=str, default="/bcic20203_preprocessed/")
    p.add_argument("--test_label_dir", type=str, default="/bcic2020_03/Track3_Answer Sheet_Test.xlsx")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_bcic20203(args)


if __name__ == "__main__":
    main()
