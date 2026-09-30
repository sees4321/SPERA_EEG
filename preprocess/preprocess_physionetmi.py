import argparse
import mne
import numpy as np
import os


def build_channel_coords(channel_names, montage_name="standard_1005"):
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    coords = []
    for ch in channel_names:
        ch = ch.rstrip('.') # remove . and .. from ch_names

        for m_name, m_pos in pos.items():
            if m_name.upper() == ch.upper():
                coords.append(m_pos[:3])
                break

    return np.asarray(coords)


def preprocess_physionetmi(args):
    tasks = ['04', '06', '08', '10', '12', '14'] # select the data for motor imagery

    out_dir = args.output_dir
    root_dir = args.root_dir
    subj_dict = {
        'train': [f'S{i:03d}' for i in range(1, 71)],
        'validation': [f'S{i:03d}' for i in range(71, 90)],
        'test': [f'S{i:03d}' for i in range(90, 110)],
    }

    coords = None

    for split in ['train', 'validation', 'test']:
        eeg_list = []
        label_list = []
        
        for subj in subj_dict[split]:
            for task in tasks:
                raw = mne.io.read_raw_edf(os.path.join(root_dir, subj, f'{subj}R{task}.edf'), preload=True, verbose=False)
                if coords is None:
                    coords = build_channel_coords(raw.ch_names)
                if raw.info['bads']:
                    print(f'interpolate_bads: {subj}, {task}')
                    raw.interpolate_bads()
                raw.set_eeg_reference(ref_channels='average', verbose=False)
                raw.filter(l_freq=None, h_freq=64.0, verbose=False)
                raw.resample(200)
                events_annot, event_dict = mne.events_from_annotations(raw, verbose=False)
                epochs = mne.Epochs(raw, events_annot, event_dict, 
                                    tmin=0, tmax=4. - 1.0 / raw.info['sfreq'], # subtract 1 sample to get exactly 800 samples instead of 801
                                    baseline=None, preload=True, verbose=False)
                eeg = epochs.get_data(units='uV')
                events = epochs.events[:, 2]
                eeg_list.append(eeg[events != 1])
                # event labels: 
                # 1 (T0) is rest (not used for downstream task)
                # 2 (T1) corresponds to left-fist in 04,08,12 and both-fists in others
                # 3 (T2) corresponds to right-fist in 04,08,12 and both-feet in others
                label_list.append(events[events != 1] - 2 if task in ['04', '08', '12'] else events[events != 1])
        eeg_list = np.concatenate(eeg_list)
        label_list = np.concatenate(label_list)
        os.makedirs(os.path.join(out_dir,split), exist_ok=True)
        np.save(os.path.join(out_dir,split,"eeg.npy"), eeg_list)
        np.save(os.path.join(out_dir,split,"label.npy"), label_list)
        np.save(os.path.join(out_dir,split,"coords.npy"), np.stack([coords]*len(label_list)))

        print(split, eeg_list.shape, label_list.shape)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/physionetmi/")
    p.add_argument("--output_dir", type=str, default="/physionetmi_preprocessed/")
    return p


def main():
    args = build_parser().parse_args()
    preprocess_physionetmi(args)


if __name__ == "__main__":
    main()
