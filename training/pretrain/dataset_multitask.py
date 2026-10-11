from torch.utils.data import Dataset
import torch
import pyBigWig as pbw
import numpy as np
import gzip
import random
import os
from tqdm import tqdm


class MultiRBPDataset(Dataset):
    """
    Multi-task dataset with per-RBP quota sampling and m6A as a 5th input channel.

    For each RBP in rbp_list:
      1. Slide 800bp windows over that RBP's own BED file
      2. Sample a quota of (total_num // num_rbps) windows, split 70/20/10 by chromosome
    Then union + deduplicate across all RBPs to form the final train/val/test pools.

    __getitem__ returns:
        seq:   (5, 800)         4-ch one-hot RNA + 1-ch m6A BigWig signal
        label: (num_tasks, 800) stacked RBP BigWig signals for all RBPs in rbp_list
    """

    def __init__(self, rbp_list, bed_root,
                 mode='train',
                 bw_root=None,
                 chrom_root=None,
                 total_num=100000,
                 m6A_binding=1,
                 precomputed_windows=None):

        self.rbp_list = rbp_list
        self.bed_root = bed_root
        self.bw_root = bw_root
        self.chrom_root = chrom_root
        self.input_len = 800
        self.mode = mode
        self.m6A_binding = m6A_binding

        self.val_chr  = ['chr2', 'chr3', 'chr4']
        self.test_chr = ['chr1', 'chr8', 'chr9']

        if precomputed_windows is not None:
            windows = precomputed_windows
        else:
            windows = self._get_window_per_rbp(bed_root, rbp_list, total_num)

        self._all_windows = windows   # store so caller can reuse for other splits

        if mode == 'train':
            self.peak = windows[0]
        elif mode == 'val':
            self.peak = windows[1]
        elif mode == 'test':
            self.peak = windows[2]

    def __len__(self):
        return len(self.peak)

    def __getitem__(self, idx):
        start, end, chrom, direction = self.peak[idx]

        # 4-ch one-hot RNA sequence
        rna_seq = self._get_seq_npy(self.chrom_root, chrom, start, end, direction)
        seq = self._get_onehot(rna_seq)                          # (4, 800)

        # Concatenate m6A signal as 5th channel
        if self.m6A_binding:
            m6a = self._read_bigwig(self.bw_root, 'm6A', chrom, start, end, direction)
            seq = torch.cat((seq, m6a), dim=0)                   # (5, 800)

        # Stack BigWig labels for all 32 RBPs → (num_tasks, 800)
        tracks = [
            self._read_bigwig(self.bw_root, rbp, chrom, start, end, direction)
            for rbp in self.rbp_list
        ]
        label = torch.cat(tracks, dim=0)

        return seq, label

    # ------------------------------------------------------------------
    # Per-RBP quota sampling
    # ------------------------------------------------------------------

    def _get_window_per_rbp(self, bed_root, rbp_list, total_num):
        """
        For each RBP: slide windows over its BED → sample per-RBP quota.
        Union + deduplicate across all RBPs.
        Returns (train_list, val_list, test_list).
        """
        num_rbps = len(rbp_list)
        per_rbp_quota = total_num // num_rbps

        n_train = int(per_rbp_quota * 0.7)
        n_val   = int(per_rbp_quota * 0.2)
        n_test  = int(per_rbp_quota * 0.1)

        train_pool = set()
        val_pool   = set()
        test_pool  = set()

        for rbp_idx, rbp in enumerate(rbp_list):
            bed_path = os.path.join(bed_root, f'{rbp}.bed')
            trains, vals, tests = self._slide_windows(bed_path)

            # Per-RBP seed so each RBP samples different windows
            seed = 43 + rbp_idx
            train_pool.update(self._sample_choice(sorted(trains), n_train, seed=seed))
            val_pool.update(  self._sample_choice(sorted(vals),   n_val,   seed=seed))
            test_pool.update( self._sample_choice(sorted(tests),  n_test,  seed=seed))

            print(f'  {rbp}: {len(trains)} train / {len(vals)} val / {len(tests)} test candidates '
                  f'-> sampled {min(n_train, len(trains))}/{min(n_val, len(vals))}/{min(n_test, len(tests))}',
                  flush=True)

        print(f'Final pool after union+dedup: '
              f'{len(train_pool)} train / {len(val_pool)} val / {len(test_pool)} test',
              flush=True)

        return sorted(train_pool), sorted(val_pool), sorted(test_pool)

    def _slide_windows(self, bed_path):
        """Slide 800bp windows over all peaks in a BED file, split by chromosome."""
        trains, vals, tests = [], [], []

        with open(bed_path, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f'  sliding {os.path.basename(bed_path)}', leave=False):
            fields = line.split('\t')
            if len(fields) < 6:
                continue
            chrom       = fields[0]
            direction   = fields[5].replace('\n', '')
            start       = int(fields[1])
            end         = int(fields[2])
            peak_length = end - start

            for bp in range(start - self.input_len // 2,
                            end + self.input_len,
                            int(peak_length / 10) + 1):
                item = (bp - self.input_len + 1, bp + 1, chrom, direction)

                if chrom in self.val_chr:
                    vals.append(item)
                elif chrom in self.test_chr:
                    tests.append(item)
                else:
                    trains.append(item)

        return set(trains), set(vals), set(tests)

    def _sample_choice(self, seq, sample_size, seed=43):
        rng = random.Random(seed)
        if len(seq) <= sample_size:
            return seq
        return rng.sample(seq, sample_size)

    # ------------------------------------------------------------------
    # Sequence / BigWig helpers (same logic as dataset_windowneg.py)
    # ------------------------------------------------------------------

    def _get_seq_npy(self, chrom_root, chrom, start, end, direction):
        with gzip.open(f'{chrom_root}/{chrom}.fa.gz', 'r') as f:
            seq = f.read().decode('utf-8')
            seq = seq[seq.find('\n'):]
            seq = seq.replace('\n', '').lower()

        if direction == '-':
            complement = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
            com_seq = ''.join(complement[b] for b in seq[start:end])
            return com_seq[::-1]

        return seq[start:end].replace('t', 'u')

    def _get_onehot(self, rna_seq):
        en_dict = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
        en_seq  = [en_dict[nt] for nt in rna_seq]
        np_seq  = np.array(en_seq, dtype=int)

        seq_emb = np.zeros((len(rna_seq), 5))
        seq_emb[np.arange(len(rna_seq)), np_seq] = 1
        seq = torch.from_numpy(np.transpose(seq_emb))
        return seq[:4, :]   # drop 'N' channel → (4, 800)

    def _read_bigwig(self, bw_root, rbp, chrom, start, end, direction):
        strand = 'plus' if direction == '+' else 'minus'
        bw = pbw.open(f'{bw_root}/{rbp}_{strand}.bw')
        signals = bw.values(chrom, start, end)
        bw.close()

        if direction == '-':
            signals = signals[::-1]

        arr      = np.reshape(signals, (1, -1))
        features = np.nan_to_num(arr, nan=0)
        features[features < 0] = 0
        return torch.from_numpy(features)   # (1, 800)

