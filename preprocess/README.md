# Preprocessing

This folder includes preprocessing scripts for each downstream task dataset.

Implementation references: [CBraMod](https://github.com/wjq-learning/CBraMod), [REVE](https://github.com/elouayas/reve_eeg).

All scripts share the following preprocessing conventions:
- Target sampling rate is generally `200 Hz`.
- Labels are mapped to integer class IDs.
- Splits are usually materialized as `train` / `validation` / `test`.

Run scripts from the repository root. Basic dependencies:

```bash
pip install numpy scipy mne pandas h5py openpyxl
# BCIC-IV-2a additionally uses MOABB:
pip install moabb
```