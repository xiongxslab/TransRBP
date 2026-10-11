"""
One-time preprocessing for fine-tuning: write one HDF5 file per RBP.

Replicates the exact windowing strategy from dataset_windowneg.py:
  - Slide 800bp windows over each BED peak; step = int(peak_length/10)+1
  - Split: train=all chrom except chr1,2,3,4,8,9 / val=chr2,3,4 / test=chr1,8,9
  - random.sample(seed=43) downsample to 14k/4k/2k (70/20/10 of 20k)

Output per RBP (e.g. RBM15_20k.h5):
    train/inputs  (N_train, 5, 800)  float16
    train/labels  (N_train, 1, 800)  float16
    val/inputs    (N_val,   5, 800)  float16
    val/labels    (N_val,   1, 800)  float16
    test/inputs   (N_test,  5, 800)  float16
    test/labels   (N_test,  1, 800)  float16
    attrs: rbp_name, total_windows

Usage:
    python preprocess_finetune_h5.py \\
        --rbp_list   rbp_list.txt \\
        --bed_root   /path/to/RBP_bed \\
        --bw_root    /path/to/bamCompare_bw \\
        --chrom_root hg38/dna_sequence \\
        --out_dir    h5_cache/finetune \\
        --total_windows 20000
"""

import argparse
import gzip
import os
import random

import h5py
import numpy as np
import pyBigWig as pbw
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--rbp_list',      required=True)
parser.add_argument('--bed_root',      required=True)
parser.add_argument('--bw_root',       required=True)
parser.add_argument('--chrom_root',    required=True)
parser.add_argument('--out_dir',       required=True)
parser.add_argument('--total_windows', type=int, default=20000)
args = parser.parse_args()

INPUT_LEN = 800
VAL_CHR   = {'chr2', 'chr3', 'chr4'}
TEST_CHR  = {'chr1', 'chr8', 'chr9'}


# ------------------------------------------------------------------
# Window generation (mirrors dataset_windowneg.py exactly)
# ------------------------------------------------------------------

def get_window(bed_path, total_num):
    trains, vals, tests = [], [], []

    with open(bed_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f'  sliding {os.path.basename(bed_path)}', leave=False):
        fields = line.split('\t')
        if len(fields) < 6:
            continue
        chrom     = fields[0]
        direction = fields[5].replace('\n', '')
        start     = int(fields[1])
        end       = int(fields[2])
        peak_len  = end - start

        for bp in range(start - INPUT_LEN // 2,
                        end + INPUT_LEN,
                        int(peak_len / 10) + 1):
            item = (bp - INPUT_LEN + 1, bp + 1, chrom, direction)
            if chrom in VAL_CHR:
                vals.append(item)
            elif chrom in TEST_CHR:
                tests.append(item)
            else:
                trains.append(item)

    n_train = int(total_num / 10 * 7)
    n_val   = int(total_num / 10 * 2)
    n_test  = int(total_num / 10)

    return (sample_choice(sorted(set(trains)), n_train),
            sample_choice(sorted(set(vals)),   n_val),
            sample_choice(sorted(set(tests)),  n_test))


def sample_choice(seq, sample_size):
    random.seed(43)
    if len(seq) <= sample_size:
        return random.sample(seq, len(seq))
    return random.sample(seq, sample_size)


# ------------------------------------------------------------------
# Sequence / BigWig helpers
# ------------------------------------------------------------------

def load_chrom_seqs(chrom_root, chroms):
    seqs = {}
    for chrom in tqdm(chroms, desc='Loading chromosomes'):
        with gzip.open(f'{chrom_root}/{chrom}.fa.gz', 'r') as f:
            raw = f.read().decode('utf-8')
            raw = raw[raw.find('\n'):]
            seqs[chrom] = raw.replace('\n', '').lower()
    return seqs


def get_seq(chrom_seqs, chrom, start, end, direction):
    seq = chrom_seqs[chrom]
    if direction == '-':
        complement = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
        return ''.join(complement.get(b, 'n') for b in seq[start:end])[::-1]
    return seq[start:end].replace('t', 'u')


def get_onehot(rna_seq):
    en_dict = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
    en_seq  = np.array([en_dict.get(nt, 4) for nt in rna_seq], dtype=int)
    emb     = np.zeros((len(rna_seq), 5), dtype=np.float16)
    emb[np.arange(len(rna_seq)), en_seq] = 1
    return emb[:, :4].T   # (4, L)


def read_bw(bw_handle, chrom, start, end, direction):
    signals = np.array(bw_handle.values(chrom, start, end), dtype=np.float32)
    if direction == '-':
        signals = signals[::-1]
    signals = np.nan_to_num(signals)
    signals[signals < 0] = 0
    return signals.astype(np.float16).reshape(1, -1)   # (1, L)


# ------------------------------------------------------------------
# Per-RBP processing
# ------------------------------------------------------------------

def process_rbp(rbp, bed_root, bw_root, chrom_seqs, out_dir, total_windows):
    bed_path = os.path.join(bed_root, f'{rbp}.bed')
    out_path = os.path.join(out_dir, f'{rbp}_20k.h5')

    if os.path.exists(out_path):
        print(f'  {rbp}: HDF5 already exists, skipping.')
        return

    print(f'\n===== {rbp} =====')
    train_wins, val_wins, test_wins = get_window(bed_path, total_windows)
    print(f'  Windows: {len(train_wins)} train / {len(val_wins)} val / {len(test_wins)} test')

    # Open BigWig handles for this RBP + m6A
    bw = {
        'rbp_plus':  pbw.open(f'{bw_root}/{rbp}_plus.bw'),
        'rbp_minus': pbw.open(f'{bw_root}/{rbp}_minus.bw'),
        'm6a_plus':  pbw.open(f'{bw_root}/m6A_plus.bw'),
        'm6a_minus': pbw.open(f'{bw_root}/m6A_minus.bw'),
    }

    splits = [('train', train_wins), ('val', val_wins), ('test', test_wins)]

    with h5py.File(out_path, 'w') as h5:
        h5.attrs['rbp_name']      = rbp
        h5.attrs['rbp_list']      = [rbp]   # list form for MultiRBPDatasetH5 compatibility
        h5.attrs['total_windows'] = total_windows

        for split_name, windows in splits:
            N = len(windows)
            ds_in  = h5.create_dataset(f'{split_name}/inputs',
                                        shape=(N, 5, INPUT_LEN), dtype='float16',
                                        chunks=(max(1, min(64, N)), 5, INPUT_LEN), compression='lzf')
            ds_out = h5.create_dataset(f'{split_name}/labels',
                                        shape=(N, 1, INPUT_LEN), dtype='float16',
                                        chunks=(max(1, min(64, N)), 1, INPUT_LEN), compression='lzf')

            for i, (start, end, chrom, direction) in enumerate(
                    tqdm(windows, desc=f'  Writing {split_name}')):

                rna_seq = get_seq(chrom_seqs, chrom, start, end, direction)
                onehot  = get_onehot(rna_seq)   # (4, L)

                m6a_handle = bw['m6a_plus'] if direction == '+' else bw['m6a_minus']
                m6a        = read_bw(m6a_handle, chrom, start, end, direction)   # (1, L)
                inp        = np.concatenate([onehot, m6a], axis=0)               # (5, L)

                rbp_handle = bw['rbp_plus'] if direction == '+' else bw['rbp_minus']
                label      = read_bw(rbp_handle, chrom, start, end, direction)   # (1, L)

                ds_in[i]  = inp
                ds_out[i] = label

    for handle in bw.values():
        handle.close()

    print(f'  Saved {out_path}')


def main():
    with open(args.rbp_list) as f:
        rbp_list = [line.strip() for line in f if line.strip()]
    print(f'Loaded {len(rbp_list)} RBPs: {rbp_list}')

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect all chromosomes needed across all RBPs (scan BED files)
    chroms_needed = set()
    for rbp in rbp_list:
        bed_path = os.path.join(args.bed_root, f'{rbp}.bed')
        with open(bed_path) as f:
            for line in f:
                fields = line.split('\t')
                if len(fields) >= 1:
                    chroms_needed.add(fields[0])
    chroms_needed -= {''}
    print(f'Chromosomes to load: {sorted(chroms_needed)}')

    # Load all chromosomes once
    chrom_seqs = load_chrom_seqs(args.chrom_root, sorted(chroms_needed))

    # Process each RBP
    for rbp in rbp_list:
        process_rbp(rbp, args.bed_root, args.bw_root, chrom_seqs,
                    args.out_dir, args.total_windows)

    print('\nAll RBPs processed.')


if __name__ == '__main__':
    main()
