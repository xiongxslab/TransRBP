from torch.utils.data import Dataset
import torch
import pyBigWig as pbw
import numpy as np
import gzip
import random
from tqdm import tqdm


class RBPDataset(Dataset):
    """
    Sliding-window dataset for a single RBP.

    Chromosome split:
        train  : all except val/test chromosomes
        val    : chr2, chr3, chr4
        test   : chr1, chr8, chr9

    Args:
        RBP_name   : RBP identifier (must match BigWig and BED file names)
        peak_bed   : path to BED6 file of RBP peaks
        bw_root    : directory containing {RBP}_plus.bw / {RBP}_minus.bw
        chrom_root : directory containing {chrom}.fa.gz per-chromosome FASTA files
        m6A_bw_plus  : path to m6A MeRIP BigWig (plus strand)
        m6A_bw_minus : path to m6A MeRIP BigWig (minus strand)
        mode       : 'train', 'val', or 'test'
        m6A_binding: 1 = 5-channel input (seq + m6A); 0 = 4-channel seq only
        total_num  : maximum windows to sample (default 20000; 70/20/10 split)
    """

    def __init__(self, RBP_name, peak_bed, bw_root, chrom_root,
                 m6A_bw_plus=None, m6A_bw_minus=None,
                 mode='train', m6A_binding=1, total_num=20000):

        self.RBP_name    = RBP_name
        self.bw_root     = bw_root
        self.chrom_root  = chrom_root
        self.m6A_bw_plus  = m6A_bw_plus
        self.m6A_bw_minus = m6A_bw_minus
        self.m6A_binding = m6A_binding
        self.mode        = mode
        self.input_len   = 800

        self.val_chr  = ['chr2', 'chr3', 'chr4']
        self.test_chr = ['chr1', 'chr8', 'chr9']

        peak_window = self._get_window(peak_bed, total_num)
        if mode == 'train':
            self.peak = peak_window[0]
        elif mode == 'val':
            self.peak = peak_window[1]
        elif mode == 'test':
            self.peak = peak_window[2]
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def __len__(self):
        return len(self.peak)

    def __getitem__(self, idx):
        start, end, chrom, direction = self.peak[idx]
        features = self._read_bigwig(self.bw_root, self.RBP_name,
                                     chrom, start, end, direction)
        RNA_seq = self._get_seq(self.chrom_root, chrom, start, end, direction)
        seq = self._get_onehot(RNA_seq)

        if self.m6A_binding == 1:
            m6a = self._read_m6A(chrom, start, end, direction)
            seq = torch.cat((seq, m6a), dim=0)

        return seq, features

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_window(self, file_root, total_num):
        trains, vals, tests = [], [], []
        with open(file_root, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc='Processing peaks', unit='peak'):
            fields = line.split('\t')
            chrom     = fields[0]
            direction = fields[5].replace('\n', '')
            start     = int(fields[1])
            end       = int(fields[2])
            peak_length = end - start
            for bp in range(start - self.input_len // 2,
                            end + self.input_len,
                            int(peak_length / 10) + 1):
                peak_start = bp - self.input_len + 1
                peak_end   = bp + 1
                item = (peak_start, peak_end, chrom, direction)
                if chrom in self.val_chr:
                    vals.append(item)
                elif chrom in self.test_chr:
                    tests.append(item)
                else:
                    trains.append(item)

        return (
            self._sample(sorted(set(trains)), int(total_num / 10 * 7)),
            self._sample(sorted(set(vals)),   int(total_num / 10 * 2)),
            self._sample(sorted(set(tests)),  int(total_num / 10)),
        )

    def _sample(self, seq, n):
        return random.sample(seq, min(len(seq), n))

    def _get_seq(self, chrom_root, chrom, start, end, direction):
        with gzip.open(f'{chrom_root}/{chrom}.fa.gz', 'r') as f:
            raw = f.read().decode('utf-8')
            raw = raw[raw.find('\n'):].replace('\n', '').lower()
        if direction == '-':
            comp = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
            return ''.join(comp[b] for b in raw[start:end])[::-1]
        return raw[start:end].replace('t', 'u')

    def _get_onehot(self, rna_seq):
        en = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
        idx = np.array([en.get(nt, 4) for nt in rna_seq], dtype=int)
        emb = np.zeros((len(rna_seq), 5))
        emb[np.arange(len(rna_seq)), idx] = 1
        seq = torch.from_numpy(emb.T)
        return seq[:4, :]

    def _read_bigwig(self, bw_root, rbp, chrom, start, end, direction):
        strand = 'plus' if direction == '+' else 'minus'
        bw = pbw.open(f'{bw_root}/{rbp}_{strand}.bw')
        signals = np.array(bw.values(chrom, start, end), dtype=np.float32)
        bw.close()
        if direction == '-':
            signals = signals[::-1]
        signals = np.nan_to_num(signals, nan=0.0)
        signals[signals < 0] = 0
        return torch.from_numpy(signals.reshape(1, -1))

    def _read_m6A(self, chrom, start, end, direction):
        path = self.m6A_bw_plus if direction == '+' else self.m6A_bw_minus
        bw = pbw.open(path)
        signals = np.array(bw.values(chrom, start, end), dtype=np.float32)
        bw.close()
        if direction == '-':
            signals = signals[::-1]
        signals = np.nan_to_num(signals, nan=0.0)
        signals[signals < 0] = 0
        return torch.from_numpy(signals.reshape(1, -1))
