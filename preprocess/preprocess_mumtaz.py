import argparse
import mne
import numpy as np
import os


def build_channel_coords(channel_names, montage_name="standard_1005"):
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    coords = []
    for ch in channel_names:
        ch = ch.replace("EEG ", "").replace("-LE", "")
        coords.append(pos[ch][:3])

    return np.asarray(coords)


def iter_files(rootDir):
    files_H, files_MDD = [], []
    for file in os.listdir(rootDir):
        if file.endswith('.edf') and 'TASK' not in file:
            if 'MDD' in file:
                files_MDD.append(file)
            else:
                files_H.append(file)
    return files_H, files_MDD


def preprocess_mumtaz(args):
    out_dir = args.output_dir
    root_dir = args.root_dir
    files_h, files_m = iter_files(root_dir)
    # Lex sort (matches prior work's released code): S1, S10, S11, ..., S2, S20, ...
    files_h = sorted(files_h)
    files_m = sorted(files_m)

    # Subject-level split following the released code of CBraMod (Wang et al., 2024) and REVE (Ouahidi et al., 2025).
    # Under lex sort, no subject's recordings span multiple splits.
    # Resulting subject counts:
    #   train: 21 control (H) / 22 MDD
    #   val:    4 control (H) /  5 MDD
    #   test:   5 control (H) /  6 MDD
    files_dict = {
        'train':      files_h[:40]   + files_m[:42],
        'validation': files_h[40:48] + files_m[42:52],
        'test':       files_h[48:]   + files_m[52:],
    }
    selected_channels = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                        'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                        'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']
    coords = build_channel_coords(selected_channels)

    for split in ['train', 'validation', 'test']:
        eeg_list = []
        label_list = []

        for file in files_dict[split]:
            raw = mne.io.read_raw_edf(os.path.join(root_dir, file), preload=True, verbose=False)
            raw.reorder_channels(selected_channels)
            raw.resample(200)
            raw.filter(l_freq=0.3, h_freq=75, verbose=False)
            raw.notch_filter((50), verbose=False)
            eeg = raw.get_data(units='uV')
            leftover = eeg.shape[-1] % (5*200) # window = 5 s
            if leftover != 0: # remove leftover samples to fit 200Hz * 5s segments
                eeg = eeg[:,:-leftover]
            eeg = eeg.reshape(19, -1, 5*200).transpose(1, 0, 2)
            label = 1 if 'MDD' in file else 0
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
    p.add_argument("--root_dir", type=str, default="/mumtaz/")
    p.add_argument("--output_dir", type=str, default="/mumtaz_preprocessed/")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_mumtaz(args)


if __name__ == "__main__":
    main()