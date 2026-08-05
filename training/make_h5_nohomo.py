"""
Pre-materialise no-homology BED windows into a per-RBP HDF5 file.

Input BED files (pre-split, 800 bp windows, no sequence homology between splits):
    {bed_root}/{RBP}_{train,val,test}.bed

Download these from Zenodo (data/ folder) or generate your own splits.

Output HDF5 layout:
    {output_dir}/{RBP}.h5
        train/inputs  (N, 5, 800)  float16
        train/labels  (N, 1, 800)  float16
        val/inputs    ...
        val/labels    ...
        test/inputs   ...
        test/labels   ...

Example:
    python training/make_h5_nohomo.py \\
        --rbp FMR1 \\
        --bed_root  data/ \\
        --bw_root   /path/to/bamCompare_1_bw \\
        --chrom_root /path/to/hg38/dna_sequence \\
        --m6A_track  m6A_GLORI_K562proxy \\
        --output_dir ./h5_cache

Run all 32 RBPs in parallel:
    RBPs=(AKAP1 APOBEC3C CPSF6 DDX3X DDX43 DDX55 DDX6 EIF4E FMR1 FTO \\
          FXR1 FXR2 GEMIN5 GRWD1 IGF2BP1 LIN28B METAP2 MTPAP NOLC1 PABPC4 \\
          PUM1 PUM2 RBM15 SLTM SRSF1 TRA2A UCHL5 UPF1 YBX3 YWHAG ZNF622 ZNF800)
    for RBP in "${RBPs[@]}"; do
        python training/make_h5_nohomo.py --rbp $RBP --bed_root data/ ... &
    done
    wait
"""

import argparse
import gzip
import os

import h5py
import numpy as np
import pyBigWig as pbw
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--rbp',          required=True,
                    help='RBP name, e.g. FMR1')
parser.add_argument('--bed_root',     required=True,
                    help='Dir containing {RBP}_{train,val,test}.bed')
parser.add_argument('--bw_root',      required=True,
                    help='Dir with {RBP}_plus.bw / {RBP}_minus.bw BigWig files')
parser.add_argument('--chrom_root',   required=True,
                    help='Dir with {chrom}.fa.gz per-chromosome FASTA files')
parser.add_argument('--output_dir',   required=True,
                    help='Output directory for HDF5 files')
parser.add_argument('--m6A_track',    default='m6A_GLORI_K562proxy',
                    help='Name prefix for m6A BigWig files in bw_root')
parser.add_argument('--m6A_binding',  type=int, default=1,
                    help='1 = 5-channel input (seq + m6A); 0 = 4-channel seq only')
args = parser.parse_args()


def parse_bed(path):
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split('\t')
            windows.append((fields[0], int(fields[1]), int(fields[2]), fields[5]))
    return windows


def load_chrom_seqs(chrom_root, chroms):
    seqs = {}
    for chrom in tqdm(sorted(chroms), desc='Loading chromosomes'):
        with gzip.open(f'{chrom_root}/{chrom}.fa.gz', 'r') as f:
            raw = f.read().decode('utf-8')
            raw = raw[raw.find('\n'):].replace('\n', '').lower()
        seqs[chrom] = raw
    return seqs


def open_bw_handles(bw_root, tracks):
    handles = {}
    for name in tqdm(tracks, desc='Opening BigWig handles'):
        for strand in ('plus', 'minus'):
            key  = f'{name}_{strand}'
            path = f'{bw_root}/{key}.bw'
            handles[key] = pbw.open(path)
    return handles


def get_seq(chrom_seqs, chrom, start, end, direction):
    seq = chrom_seqs[chrom]
    if direction == '-':
        comp = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
        return ''.join(comp.get(b, 'n') for b in seq[start:end])[::-1]
    return seq[start:end].replace('t', 'u')


def get_onehot(rna_seq):
    en  = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
    idx = np.array([en.get(nt, 4) for nt in rna_seq], dtype=int)
    emb = np.zeros((len(rna_seq), 5), dtype=np.float16)
    emb[np.arange(len(rna_seq)), idx] = 1
    return emb[:, :4].T    # (4, L)


def read_bw(handles, name, chrom, start, end, direction):
    strand  = 'plus' if direction == '+' else 'minus'
    signals = np.array(handles[f'{name}_{strand}'].values(chrom, start, end),
                       dtype=np.float32)
    if direction == '-':
        signals = signals[::-1]
    signals = np.nan_to_num(signals, nan=0.0)
    signals[signals < 0] = 0
    return signals.astype(np.float16).reshape(1, -1)    # (1, L)


def write_split(h5file, split, windows, chrom_seqs, bw_handles):
    N    = len(windows)
    n_in = 5 if args.m6A_binding else 4
    L    = 800

    ds_in  = h5file.create_dataset(
        f'{split}/inputs', shape=(N, n_in, L), dtype='float16',
        chunks=(64, n_in, L), compression='lzf')
    ds_out = h5file.create_dataset(
        f'{split}/labels', shape=(N, 1, L), dtype='float16',
        chunks=(64, 1, L), compression='lzf')

    for i, (chrom, start, end, strand) in enumerate(
            tqdm(windows, desc=f'  {split} ({N})')):
        seq = get_onehot(get_seq(chrom_seqs, chrom, start, end, strand))  # (4, L)

        if args.m6A_binding:
            m6a = read_bw(bw_handles, args.m6A_track, chrom, start, end, strand)
            inp = np.concatenate([seq, m6a], axis=0)    # (5, L)
        else:
            inp = seq                                    # (4, L)

        label   = read_bw(bw_handles, args.rbp, chrom, start, end, strand)
        ds_in[i]  = inp
        ds_out[i] = label


def main():
    print(f'\n=== make_h5_nohomo: {args.rbp}  m6A_track={args.m6A_track} ===')

    splits = {}
    for s in ('train', 'val', 'test'):
        path = os.path.join(args.bed_root, f'{args.rbp}_{s}.bed')
        wins = parse_bed(path)
        splits[s] = wins
        print(f'  {s}: {len(wins)} windows  ({path})')

    all_windows = splits['train'] + splits['val'] + splits['test']
    chroms      = sorted(set(w[0] for w in all_windows))
    chrom_seqs  = load_chrom_seqs(args.chrom_root, chroms)

    tracks = [args.rbp]
    if args.m6A_binding:
        tracks.append(args.m6A_track)
    bw_handles = open_bw_handles(args.bw_root, tracks)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.rbp}.h5')
    print(f'  Writing → {out_path}')

    with h5py.File(out_path, 'w') as h5:
        h5.attrs['rbp']         = args.rbp
        h5.attrs['m6A_track']   = args.m6A_track
        h5.attrs['m6A_binding'] = args.m6A_binding
        for split, windows in splits.items():
            write_split(h5, split, windows, chrom_seqs, bw_handles)

    for bw in bw_handles.values():
        bw.close()

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'\nDone. {out_path}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
