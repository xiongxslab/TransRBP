import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiRBPDatasetH5(Dataset):
    """
    Reads pre-computed inputs and labels from an HDF5 file created by preprocess_h5.py.

    __getitem__ is a single array slice — no sequence/BigWig file I/O at training time.

    The h5py file handle is opened lazily inside each DataLoader worker (after fork)
    so it is safe with num_workers > 0.
    """

    def __init__(self, h5_path, mode='train'):
        self.h5_path = h5_path
        self.mode    = mode
        self._h5     = None   # opened lazily per worker after fork

        # Read metadata without keeping the handle open
        with h5py.File(h5_path, 'r') as f:
            self._len     = f[f'{mode}/inputs'].shape[0]
            self.rbp_list = list(f.attrs['rbp_list'])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Open handle on first access in this worker process
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

        seq   = torch.from_numpy(self._h5[f'{self.mode}/inputs'][idx].astype(np.float32))
        label = torch.from_numpy(self._h5[f'{self.mode}/labels'][idx].astype(np.float32))
        return seq, label

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
