"""
In silico demethylation: compare model predictions with and without m6A signal.

For each peak in the input BED file the model is run twice:
  - original : m6A channel intact
  - KO       : m6A channel zeroed (complete KO) or zeroed only within --m6A_peak_bed sites

pred_diff = sum(p_KO - p_original) over 800 bp  [negative → reduced binding after KO]

Output CSV columns:
    peak_region          chr:start-end_strand
    pred_diff            sum of per-position difference (KO − original)
    mean_pred_KO         mean prediction under KO input
    mean_pred_original   mean prediction under original input
    KLD_pred             KL divergence between KO and original prediction profiles
    RBP_name

Example (complete KO — zero all m6A signal):
    python demethylation/insilico_demethylation.py \\
        --rbp_name   FMR1 \\
        --peak_bed   data/FMR1_test.bed \\
        --model_path models/FMR1.pth \\
        --chrom_root /path/to/hg38/dna_sequence \\
        --m6A_bw_plus  /path/to/m6A_plus.bw \\
        --m6A_bw_minus /path/to/m6A_minus.bw \\
        --output_csv FMR1_demethylation.csv

Example (partial KO — zero only within supplied m6A sites):
    python demethylation/insilico_demethylation.py \\
        --rbp_name     FMR1 \\
        --peak_bed     data/FMR1_test.bed \\
        --model_path   models/FMR1.pth \\
        --chrom_root   /path/to/hg38/dna_sequence \\
        --m6A_bw_plus  /path/to/m6A_plus.bw \\
        --m6A_bw_minus /path/to/m6A_minus.bw \\
        --m6A_peak_bed /path/to/m6A_sites.bed \\
        --output_csv   FMR1_demethylation_partial.csv
"""

import argparse
import csv
import gzip
import os

import numpy as np
import pyBigWig as pbw
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.RBPResTransModels import RBPModel

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='In silico m6A demethylation')
parser.add_argument('--rbp_name',     required=True,  help='RBP name')
parser.add_argument('--peak_bed',     required=True,
                    help='BED6 file of RBP peaks to analyse')
parser.add_argument('--model_path',   required=True,
                    help='Path to .pth file (5-channel TransRBP model)')
parser.add_argument('--chrom_root',   required=True,
                    help='Dir with {chrom}.fa.gz per-chromosome FASTA files')
parser.add_argument('--m6A_bw_plus',  required=True,
                    help='m6A MeRIP BigWig path (plus strand)')
parser.add_argument('--m6A_bw_minus', required=True,
                    help='m6A MeRIP BigWig path (minus strand)')
parser.add_argument('--m6A_peak_bed', default=None,
                    help='Optional BED of m6A sites. If provided, only these '
                         'positions are zeroed (partial KO). Default: zero all '
                         'm6A signal (complete KO).')
parser.add_argument('--output_csv',   required=True,  help='Path for output CSV')
parser.add_argument('--device',       default='cuda:0')
parser.add_argument('--batch_size',   type=int, default=64)
args = parser.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class DemethylationDataset(Dataset):
    """
    Returns (seq_original, seq_ko, peak_region_str) per peak.
    seq shape: (5, 800) — channels 0-3 one-hot RNA, channel 4 m6A signal.
    seq_ko has channel 4 zeroed (completely or within m6A peak sites).
    """

    def __init__(self, peak_bed, chrom_root, m6A_bw_plus, m6A_bw_minus,
                 m6A_peak_bed=None):
        self.chrom_root   = chrom_root
        self.m6A_bw_plus  = m6A_bw_plus
        self.m6A_bw_minus = m6A_bw_minus
        self.input_len    = 800

        self.peaks = self._load_peaks(peak_bed)
        self.m6A_sites = self._load_m6A_sites(m6A_peak_bed) if m6A_peak_bed else None

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, idx):
        chrom, peak_start, peak_end, direction = self.peaks[idx]
        win_start = peak_start - 400
        win_end   = peak_start + 400

        rna_seq = self._get_seq(chrom, win_start, win_end, direction)
        seq     = self._get_onehot(rna_seq)             # (4, 800) float32

        m6a_signal = self._read_m6A(chrom, win_start, win_end, direction)  # (1, 800)

        seq_original = torch.cat([seq, m6a_signal], dim=0)  # (5, 800)

        m6a_ko = m6a_signal.clone()
        if self.m6A_sites is None:
            m6a_ko[:] = 0.0                              # complete KO
        else:
            for (s_chrom, s_start, s_end) in self.m6A_sites.get(chrom, []):
                overlap_start = max(s_start, win_start)
                overlap_end   = min(s_end,   win_end)
                if overlap_start < overlap_end:
                    rel_s = overlap_start - win_start
                    rel_e = overlap_end   - win_start
                    m6a_ko[0, rel_s:rel_e] = 0.0         # partial KO

        seq_ko = torch.cat([seq, m6a_ko], dim=0)        # (5, 800)

        region = f'{chrom}:{win_start}-{win_end}_{direction}'
        return seq_original.float(), seq_ko.float(), region

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_peaks(self, bed_path):
        peaks = []
        with open(bed_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split('\t')
                peaks.append((fields[0], int(fields[1]), int(fields[2]),
                               fields[5]))
        return peaks

    def _load_m6A_sites(self, bed_path):
        sites = {}
        with open(bed_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split('\t')
                chrom = fields[0]
                sites.setdefault(chrom, []).append(
                    (chrom, int(fields[1]), int(fields[2])))
        return sites

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
        return torch.from_numpy(emb.T[:4, :])  # (4, 800)

    def _read_m6A(self, chrom, start, end, direction):
        path = self.m6A_bw_plus if direction == '+' else self.m6A_bw_minus
        bw   = pbw.open(path)
        sig  = np.array(bw.values(chrom, start, end), dtype=np.float32)
        bw.close()
        if direction == '-':
            sig = sig[::-1]
        sig = np.nan_to_num(sig, nan=0.0)
        sig[sig < 0] = 0
        return torch.from_numpy(sig.reshape(1, -1))     # (1, 800)


# ── KL divergence ─────────────────────────────────────────────────────────────

def _kl_divergence(p, q):
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


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

    dataset    = DemethylationDataset(
        peak_bed     = args.peak_bed,
        chrom_root   = args.chrom_root,
        m6A_bw_plus  = args.m6A_bw_plus,
        m6A_bw_minus = args.m6A_bw_minus,
        m6A_peak_bed = args.m6A_peak_bed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    ko_mode = 'partial' if args.m6A_peak_bed else 'complete'
    print(f'{args.rbp_name}: {len(dataset)} peaks  [{ko_mode} KO]')

    rows = []
    header = ['peak_region', 'pred_diff', 'mean_pred_KO',
              'mean_pred_original', 'KLD_pred', 'RBP_name']

    for seq_orig, seq_ko, regions in tqdm(loader, desc='Inference'):
        seq_orig = seq_orig.to(device)
        seq_ko   = seq_ko.to(device)

        with torch.no_grad():
            pred_orig = model(seq_orig).cpu().numpy()  # (B, 1, 800)
            pred_ko   = model(seq_ko).cpu().numpy()

        for i, region in enumerate(regions):
            p_orig = pred_orig[i, 0]   # (800,)
            p_ko   = pred_ko[i, 0]

            pred_diff        = float(np.sum(p_ko - p_orig))
            mean_pred_ko     = float(p_ko.mean())
            mean_pred_orig   = float(p_orig.mean())
            kld              = _kl_divergence(p_ko, p_orig)

            rows.append([region, pred_diff, mean_pred_ko,
                         mean_pred_orig, kld, args.rbp_name])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f'Saved {len(rows)} rows → {args.output_csv}')


if __name__ == '__main__':
    main()
