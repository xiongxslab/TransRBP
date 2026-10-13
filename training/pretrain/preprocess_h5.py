"""
One-time preprocessing: materialise all windows to an HDF5 file.

Keeps chromosome sequences and all BigWig handles in memory so each
window only costs a dict lookup + a slice — no file open/close overhead.

Output HDF5 layout:
    train/inputs  (N_train, 5, 800)  float16
    train/labels  (N_train, 32, 800) float16
    val/inputs    (N_val,   5, 800)  float16
    val/labels    (N_val,   32, 800) float16
    test/inputs   (N_test,  5, 800)  float16
    test/labels   (N_test,  32, 800) float16
    attrs: rbp_list, m6A_binding

Estimated disk size (float16, lzf compression):
    inputs  ~70k × 5  × 800 × 2 bytes ≈  560 MB
    labels  ~70k × 32 × 800 × 2 bytes ≈ 3.6 GB

Usage:
    python preprocess_h5.py \\
        --rbp_list   rbp_list.txt \\
        --bed_root   /path/to/RBP_bed \\
        --bw_root    /path/to/bamCompare_bw \\
        --chrom_root hg38/dna_sequence \\
        --output     h5_cache/32rbp_m6a.h5 \\
        --total_windows 100000 \\
        --m6A_binding 1
"""

import argparse
import gzip
import os

import h5py
import numpy as np
import pyBigWig as pbw
from tqdm import tqdm

from training.pretrain.dataset_multitask import MultiRBPDataset

parser = argparse.ArgumentParser()
parser.add_argument('--rbp_list',      required=True)
parser.add_argument('--bed_root',      required=True)
parser.add_argument('--bw_root',       required=True)
parser.add_argument('--chrom_root',    required=True)
parser.add_argument('--output',        required=True, help='Path for output .h5 file')
parser.add_argument('--total_windows', type=int, default=100000)
parser.add_argument('--m6A_binding',   type=int, default=1)
parser.add_argument('--m6A_bw_plus',   default=None, help='m6A BigWig, + strand (default {bw_root}/m6A_plus.bw)')
parser.add_argument('--m6A_bw_minus',  default=None, help='m6A BigWig, - strand (default {bw_root}/m6A_minus.bw)')
args = parser.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_chrom_seqs(chrom_root, chroms):
    """Read all needed chromosomes into a dict once."""
    seqs = {}
    for chrom in tqdm(chroms, desc='Loading chromosomes'):
        with gzip.open(f'{chrom_root}/{chrom}.fa.gz', 'r') as f:
            raw = f.read().decode('utf-8')
            raw = raw[raw.find('\n'):]
            seqs[chrom] = raw.replace('\n', '').lower()
    return seqs


def open_bw_handles(bw_root, rbp_list, m6A_binding):
    """Open all BigWig file handles once and keep them open."""
    handles = {}
    names = rbp_list + (['m6A'] if m6A_binding else [])
    for name in tqdm(names, desc='Opening BigWig handles'):
        for strand in ('plus', 'minus'):
            key = f'{name}_{strand}'
            path = {'m6A_plus': args.m6A_bw_plus, 'm6A_minus': args.m6A_bw_minus}.get(key) or f'{bw_root}/{key}.bw'
            handles[key] = pbw.open(path)
    return handles


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


def read_bw(handles, name, chrom, start, end, direction):
    strand  = 'plus' if direction == '+' else 'minus'
    signals = np.array(handles[f'{name}_{strand}'].values(chrom, start, end),
                       dtype=np.float32)
    if direction == '-':
        signals = signals[::-1]
    signals = np.nan_to_num(signals)
    signals[signals < 0] = 0
    return signals.astype(np.float16).reshape(1, -1)   # (1, L)


# ------------------------------------------------------------------
# Main processing
# ------------------------------------------------------------------

def process_split(split_name, windows, rbp_list, chrom_seqs, bw_handles,
                  m6A_binding, h5file):
    N    = len(windows)
    L    = 800
    n_in = 5 if m6A_binding else 4
    n_out = len(rbp_list)

    ds_in  = h5file.create_dataset(
        f'{split_name}/inputs',
        shape=(N, n_in, L), dtype='float16',
        chunks=(max(1, min(64, N)), n_in, L), compression='lzf')

    ds_out = h5file.create_dataset(
        f'{split_name}/labels',
        shape=(N, n_out, L), dtype='float16',
        chunks=(max(1, min(64, N)), n_out, L), compression='lzf')

    for i, (start, end, chrom, direction) in enumerate(
            tqdm(windows, desc=f'Writing {split_name}')):

        rna_seq = get_seq(chrom_seqs, chrom, start, end, direction)
        seq     = get_onehot(rna_seq)                              # (4, L)

        if m6A_binding:
            m6a = read_bw(bw_handles, 'm6A', chrom, start, end, direction)
            inp = np.concatenate([seq, m6a], axis=0)              # (5, L)
        else:
            inp = seq                                               # (4, L)

        label = np.concatenate([
            read_bw(bw_handles, rbp, chrom, start, end, direction)
            for rbp in rbp_list
        ], axis=0)                                                 # (32, L)

        ds_in[i]  = inp
        ds_out[i] = label


def main():
    with open(args.rbp_list) as f:
        rbp_list = [line.strip() for line in f if line.strip()]
    print(f'Loaded {len(rbp_list)} RBPs')

    # Compute windows once using the existing dataset logic
    print('Computing window pools...')
    tmp = MultiRBPDataset(rbp_list, args.bed_root, mode='train',
                          bw_root=args.bw_root, chrom_root=args.chrom_root,
                          total_num=args.total_windows, m6A_binding=args.m6A_binding)
    train_wins, val_wins, test_wins = tmp._all_windows
    del tmp   # free memory

    print(f'Windows: {len(train_wins)} train / {len(val_wins)} val / {len(test_wins)} test')

    # Collect unique chromosomes across all splits
    all_wins = train_wins + val_wins + test_wins
    chroms   = sorted(set(w[2] for w in all_wins))
    print(f'Unique chromosomes: {chroms}')

    chrom_seqs  = load_chrom_seqs(args.chrom_root, chroms)
    bw_handles  = open_bw_handles(args.bw_root, rbp_list, args.m6A_binding)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f'Writing {args.output} ...')
    with h5py.File(args.output, 'w') as h5:
        # Store metadata so the dataset class can recover rbp_list from the file
        h5.attrs['rbp_list']     = rbp_list
        h5.attrs['m6A_binding']  = args.m6A_binding
        h5.attrs['total_windows'] = args.total_windows

        process_split('train', train_wins, rbp_list, chrom_seqs, bw_handles,
                      args.m6A_binding, h5)
        process_split('val',   val_wins,   rbp_list, chrom_seqs, bw_handles,
                      args.m6A_binding, h5)
        process_split('test',  test_wins,  rbp_list, chrom_seqs, bw_handles,
                      args.m6A_binding, h5)

    for bw in bw_handles.values():
        bw.close()

    print('Done.')


if __name__ == '__main__':
    main()
