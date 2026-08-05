"""
Run TransRBP inference on a set of genomic regions.

For each peak in the input BED file the model predicts the RBP binding signal
over the 800 bp window. No ground-truth RBP signal is required.

Inputs:
    BED6 file of regions (chrom, start, end, name, score, strand)
    hg38 per-chromosome FASTA directory
    m6A MeRIP-seq BigWig (plus and minus strand) — available on Zenodo
    Trained model .pth

Output CSV columns:
    peak_region       chr:start-end_strand
    mean_pred         mean predicted binding signal over 800 bp
    sum_pred          sum of predicted binding signal over 800 bp
    max_pred          maximum predicted value over 800 bp

The full per-position prediction vectors are also optionally saved as an HDF5
file (--output_h5) with layout  {chr:start-end_strand} → (800,) float32.

Example:
    python evaluation/predict.py \\
        --rbp_name   FMR1 \\
        --peak_bed   data/FMR1_test.bed \\
        --model_path models/FMR1.pth \\
        --chrom_root /path/to/hg38/dna_sequence \\
        --m6A_bw_plus  /path/to/m6A_plus.bw \\
        --m6A_bw_minus /path/to/m6A_minus.bw \\
        --output_csv FMR1_predictions.csv \\
        --output_h5  FMR1_predictions.h5
"""

import argparse
import csv
import gzip
import os

import h5py
import numpy as np
import pyBigWig as pbw
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.RBPResTransModels import RBPModel

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='TransRBP: predict binding signal from BED regions')
parser.add_argument('--rbp_name',     required=True,  help='RBP name (used for output labels)')
parser.add_argument('--peak_bed',     required=True,
                    help='BED6 file of regions to predict (chrom, start, end, name, score, strand). '
                         'Each row defines an 800 bp window; coordinates are used directly.')
parser.add_argument('--model_path',   required=True,  help='Path to .pth file')
parser.add_argument('--chrom_root',   required=True,
                    help='Dir with {chrom}.fa.gz per-chromosome FASTA files')
parser.add_argument('--m6A_bw_plus',  required=True,
                    help='m6A MeRIP-seq BigWig (plus strand) — from Zenodo')
parser.add_argument('--m6A_bw_minus', required=True,
                    help='m6A MeRIP-seq BigWig (minus strand) — from Zenodo')
parser.add_argument('--output_csv',   required=True,  help='Output CSV with per-peak summary')
parser.add_argument('--output_h5',    default=None,
                    help='Optional HDF5 to save full 800 bp prediction vectors')
parser.add_argument('--device',       default='cuda:0')
parser.add_argument('--batch_size',   type=int, default=64)
parser.add_argument('--num_workers',  type=int, default=0,
                    help='DataLoader workers (default 0; set >0 on Linux for speed)')
args = parser.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class BEDPredictionDataset(Dataset):
    """
    Reads BED regions and builds 5-channel model input:
        channels 0-3 : one-hot RNA sequence (A/U/C/G)
        channel 4    : m6A MeRIP-seq signal
    Window coordinates are taken directly from the BED start/end fields.
    """

    def __init__(self, peak_bed, chrom_root, m6A_bw_plus, m6A_bw_minus):
        self.chrom_root   = chrom_root
        self.m6A_bw_plus  = m6A_bw_plus
        self.m6A_bw_minus = m6A_bw_minus
        self.peaks        = self._load_peaks(peak_bed)

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, idx):
        chrom, start, end, direction = self.peaks[idx]
        rna = self._get_seq(chrom, start, end, direction)
        seq = self._get_onehot(rna)                             # (4, 800)
        m6a = self._read_m6A(chrom, start, end, direction)      # (1, 800)
        x   = torch.cat([seq, m6a], dim=0).float()             # (5, 800)
        region = f'{chrom}:{start}-{end}_{direction}'
        return x, region

    def _load_peaks(self, bed_path):
        peaks = []
        with open(bed_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                fields = line.split('\t')
                chrom  = fields[0]
                start  = int(fields[1])
                end    = int(fields[2])
                strand = fields[5] if len(fields) > 5 else '+'
                peaks.append((chrom, start, end, strand))
        return peaks

    def _get_seq(self, chrom, start, end, direction):
        with gzip.open(f'{self.chrom_root}/{chrom}.fa.gz', 'r') as f:
            raw = f.read().decode('utf-8')
            raw = raw[raw.find('\n'):].replace('\n', '').lower()
        if direction == '-':
            comp = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
            return ''.join(comp.get(b, 'n') for b in raw[start:end])[::-1]
        return raw[start:end].replace('t', 'u')

    def _get_onehot(self, rna_seq):
        en  = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
        idx = np.array([en.get(nt, 4) for nt in rna_seq], dtype=int)
        emb = np.zeros((len(rna_seq), 5), dtype=np.float32)
        emb[np.arange(len(rna_seq)), idx] = 1
        return torch.from_numpy(emb.T[:4, :])                   # (4, 800)

    def _read_m6A(self, chrom, start, end, direction):
        path = self.m6A_bw_plus if direction == '+' else self.m6A_bw_minus
        bw   = pbw.open(path)
        sig  = np.array(bw.values(chrom, start, end), dtype=np.float32)
        bw.close()
        if direction == '-':
            sig = sig[::-1]
        sig = np.nan_to_num(sig, nan=0.0)
        sig[sig < 0] = 0
        return torch.from_numpy(sig.reshape(1, -1))             # (1, 800)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(args.device)

    model = RBPModel(features=5, record_attn=False)
    state = torch.load(args.model_path, map_location='cpu', weights_only=True)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if device.type == 'cuda':
        model = torch.compile(model)

    dataset = BEDPredictionDataset(
        peak_bed     = args.peak_bed,
        chrom_root   = args.chrom_root,
        m6A_bw_plus  = args.m6A_bw_plus,
        m6A_bw_minus = args.m6A_bw_minus,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    print(f'{args.rbp_name}: {len(dataset)} regions')

    csv_rows = []
    h5f = h5py.File(args.output_h5, 'w') if args.output_h5 else None

    for x, regions in tqdm(loader, desc='Predicting'):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()       # (B, 1, 800)

        for i, region in enumerate(regions):
            track = pred[i, 0]                  # (800,)
            csv_rows.append([
                region,
                args.rbp_name,
                float(track.mean()),
                float(track.sum()),
                float(track.max()),
            ])
            if h5f is not None:
                h5f.create_dataset(region, data=track)

    if h5f is not None:
        h5f.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['peak_region', 'RBP_name', 'mean_pred', 'sum_pred', 'max_pred'])
        w.writerows(csv_rows)

    print(f'Saved {len(csv_rows)} rows → {args.output_csv}')
    if args.output_h5:
        print(f'Full tracks → {args.output_h5}')


if __name__ == '__main__':
    main()
