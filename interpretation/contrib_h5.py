"""
Compute Integrated Gradient (IG) or Saliency attribution scores for the full
5-channel TransRBP model (RNA one-hot + m6A).

Attribution is computed with respect to all five input channels, so the output
contains both the per-nucleotide sequence contribution (channels 0-3, A/U/C/G)
and the m6A-channel contribution (channel 4). This is the attribution used for
the motif analysis (Fig. 2, Extended Data Figs. 4-6) and the aggregated
sequence / m6A contribution profiles (Extended Data Fig. 3g).

Input sequences can be supplied in two ways (mutually exclusive):
  --peak_bed   BED6 file — an 800 bp window is extracted from the reference
               genome and the m6A channel is read from the MeRIP-seq BigWigs
               (--m6A_bw_plus / --m6A_bw_minus). By default the window is
               [peak_start - 400, peak_start + 400) (--window start, as used in
               the paper); --window center uses the peak midpoint instead.
  --fasta_file FASTA file — sequences are padded with N / trimmed to 800 nt.
               No genomic m6A signal is available, so channel 4 is zero.

Attribution details (identical to the analysis in the paper):
  target    : softmax-weighted sum of the predicted 800 bp binding profile
  IG        : captum IntegratedGradients, multiply_by_inputs=False, all-zero
              baseline (all 5 channels), 50 steps
  correction: attr - mean(attr over the 5 channels), then x input
              (so channel 4 = corrected gradient x m6A signal)

Output HDF5 layout:
    {RBPname}/
      Contribution_Score/
        contrib_score_Batch_0    # (B, 5, 800)  channels 0-3 sequence, 4 m6A
        ...
      Inputs/
        input_Batch_0            # (B, 5, 800)  one-hot sequence + m6A signal
        regions_Batch_0          # region strings (chr:start-end_strand or seq ID)
        ...
Batches follow the order of the input BED / FASTA.

Example (BED input):
    python interpretation/contrib_h5.py \\
        --RBPname      FMR1 \\
        --RBPmodel     models/FMR1.pth \\
        --peak_bed     /path/to/FMR1_peaks.bed \\
        --chrom_root   hg38/dna_sequence \\
        --m6A_bw_plus  m6A_bw/m6A_plus.bw \\
        --m6A_bw_minus m6A_bw/m6A_minus.bw \\
        --out_h5_fname FMR1_contrib.h5

Example (FASTA input):
    python interpretation/contrib_h5.py \\
        --RBPname      FMR1 \\
        --RBPmodel     models/FMR1.pth \\
        --fasta_file   sequences.fa \\
        --out_h5_fname FMR1_contrib.h5
"""

import argparse
import gzip
import os

import h5py
import numpy as np
import pyBigWig as pbw
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency
from torch.utils.data import Dataset, DataLoader

from model.RBPResTransModels import RBPModel

torch.set_float32_matmul_precision('high')


def onehot(rna_seq):
    en  = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
    idx = np.array([en.get(nt, 4) for nt in rna_seq], dtype=int)
    emb = np.zeros((len(rna_seq), 5))
    emb[np.arange(len(rna_seq)), idx] = 1
    return torch.from_numpy(emb.T[:4, :])                     # (4, L) float64


# ── Datasets ──────────────────────────────────────────────────────────────────

class BEDSequenceDataset(Dataset):
    """
    Reads BED peaks and builds the 5-channel input over an 800 bp window.
    Returns (input_tensor [5, 800], region_string).
    """

    def __init__(self, peak_bed, chrom_root, m6A_bw_plus, m6A_bw_minus,
                 window='start', seq_length=800):
        self.chrom_root   = chrom_root
        self.m6A_bw_plus  = m6A_bw_plus
        self.m6A_bw_minus = m6A_bw_minus
        self.window       = window
        self.seq_length   = seq_length
        self.half         = seq_length // 2
        self.peaks        = self._load_peaks(peak_bed)
        self._chrom_cache = {}

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, idx):
        chrom, start, end, strand = self.peaks[idx]
        anchor    = start if self.window == 'start' else (start + end) // 2
        win_start = anchor - self.half
        win_end   = anchor + self.half

        seq = onehot(self._get_seq(chrom, win_start, win_end, strand))   # (4, 800)
        m6a = self._read_m6A(chrom, win_start, win_end, strand)          # (1, 800)
        region = f'{chrom}:{win_start}-{win_end}_{strand}'
        return torch.cat([seq, m6a], dim=0), region

    def _load_peaks(self, bed_path):
        peaks = []
        with open(bed_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                fields = line.split('\t')
                strand = fields[5] if len(fields) > 5 else '+'
                peaks.append((fields[0], int(fields[1]), int(fields[2]), strand))
        return peaks

    def _get_seq(self, chrom, start, end, direction):
        if chrom not in self._chrom_cache:
            with gzip.open(f'{self.chrom_root}/{chrom}.fa.gz', 'r') as f:
                raw = f.read().decode('utf-8')
            self._chrom_cache = {chrom: raw[raw.find('\n'):].replace('\n', '').lower()}
        sub = self._chrom_cache[chrom][start:end]
        if direction == '-':
            comp = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
            return ''.join(comp.get(b, 'n') for b in sub)[::-1]
        return sub.replace('t', 'u')

    def _read_m6A(self, chrom, start, end, direction):
        path = self.m6A_bw_plus if direction == '+' else self.m6A_bw_minus
        bw   = pbw.open(path)
        sig  = np.array(bw.values(chrom, start, end), dtype=np.float64)
        bw.close()
        if direction == '-':
            sig = sig[::-1]
        sig = np.nan_to_num(sig, nan=0.0)
        sig[sig < 0] = 0
        return torch.from_numpy(sig.reshape(1, -1).copy())     # (1, 800) float64


class FastaSequenceDataset(Dataset):
    """
    Reads a FASTA file. Each sequence is padded/trimmed to seq_length.
    Returns (input_tensor [5, 800] with zero m6A channel, sequence_id).
    """

    def __init__(self, fasta_file, seq_length=800):
        self.seq_length = seq_length
        self.entries    = self._load_fasta(fasta_file)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        seq_id, seq = self.entries[idx]
        seq = self._pad_trim(seq.lower().replace('t', 'u'))
        x   = onehot(seq)
        m6a = torch.zeros(1, x.shape[1], dtype=x.dtype)
        return torch.cat([x, m6a], dim=0), seq_id

    def _load_fasta(self, fasta_file):
        entries = []
        current_id, current_seq = None, []
        with open(fasta_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        entries.append((current_id, ''.join(current_seq)))
                    current_id  = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        if current_id is not None:
            entries.append((current_id, ''.join(current_seq)))
        return entries

    def _pad_trim(self, seq):
        if len(seq) >= self.seq_length:
            return seq[:self.seq_length]
        return seq + 'n' * (self.seq_length - len(seq))


# ── Model wrapper for IG ──────────────────────────────────────────────────────

class WeightedSumWrapper(nn.Module):
    """
    Reduces the (B, 1, 800) prediction of the 5-channel RBPModel to a scalar per
    sample: softmax-weighted sum of the predicted profile (weights detached).
    """

    def __init__(self, rbp_model):
        super().__init__()
        self.model = rbp_model

    def forward(self, x):
        pred     = self.model(x)                        # (B, 1, 800)
        pred_sq  = torch.squeeze(pred, dim=1)           # (B, 800)
        weights  = F.softmax(pred_sq.detach(), dim=1)
        return torch.sum(pred_sq * weights, dim=1)      # (B,)


# ── Core attribution class ────────────────────────────────────────────────────

class ContribH5:

    def __init__(self, filename, RBPname, RBPmodel_path,
                 peak_bed=None, chrom_root=None,
                 m6A_bw_plus=None, m6A_bw_minus=None, window='start',
                 fasta_file=None,
                 contrib_function='IG', record_global=False,
                 batch_size=64, device='cuda:0'):

        if (peak_bed is None) == (fasta_file is None):
            raise ValueError('Provide exactly one of --peak_bed or --fasta_file.')
        if peak_bed is not None and None in (chrom_root, m6A_bw_plus, m6A_bw_minus):
            raise ValueError('--peak_bed requires --chrom_root, --m6A_bw_plus and --m6A_bw_minus.')

        self.RBPname           = RBPname
        self.contrib_function  = contrib_function
        self.record_global     = record_global
        self.device            = device
        self.batch_size        = batch_size

        # Load model
        base_model = RBPModel(features=5, record_attn=False)
        state      = torch.load(RBPmodel_path, map_location='cpu', weights_only=True)
        state      = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        base_model.load_state_dict(state)
        base_model.to(device)
        base_model.eval()
        self.wrapped_model = WeightedSumWrapper(base_model).to(device)
        print('Model loaded.')

        # Build dataset
        if peak_bed is not None:
            dataset = BEDSequenceDataset(peak_bed, chrom_root, m6A_bw_plus,
                                         m6A_bw_minus, window=window)
            print(f'BED input: {len(dataset)} peaks  (window anchored at peak {window})')
        else:
            dataset = FastaSequenceDataset(fasta_file)
            print(f'FASTA input: {len(dataset)} sequences  (m6A channel = 0)')

        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0,
        )

        # HDF5 output
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        self.h5   = h5py.File(filename, 'w')
        grp       = self.h5.create_group(RBPname)
        self.g_cs = grp.create_group('Contribution_Score')
        self.g_in = grp.create_group('Inputs')

    def run(self):
        if self.contrib_function == 'IG':
            attributor = IntegratedGradients(self.wrapped_model,
                                             multiply_by_inputs=False)
        elif self.contrib_function == 'SA':
            attributor = Saliency(self.wrapped_model)
        else:
            raise ValueError(f'Unknown contrib_function: {self.contrib_function}')

        n_batches = len(self.dataloader)
        cs_buf, in_buf, reg_buf = {}, {}, {}

        for idx, (x, regions) in enumerate(self.dataloader):
            print(f'Batch {idx+1}/{n_batches}  shape={tuple(x.shape)}')
            x = x.to(self.device)

            if self.contrib_function == 'IG':
                attr = attributor.attribute(x, internal_batch_size=self.batch_size)
            else:
                attr = attributor.attribute(x, abs=False)

            # Koo lab correction: subtract per-position mean across the 5 input channels
            attr = attr - torch.mean(attr, dim=1, keepdim=True)
            if not self.record_global:
                attr = attr * x                     # element-wise x input

            cs_buf[idx]  = attr.detach().cpu().numpy()
            in_buf[idx]  = x.detach().cpu().numpy()
            reg_buf[idx] = regions

            # flush every 2 batches (or last batch)
            if (idx + 1) % 2 == 0 or idx == n_batches - 1:
                self._flush(cs_buf, in_buf, reg_buf)

        self.h5.close()
        print('Done.')

    def _flush(self, cs_buf, in_buf, reg_buf):
        dt = h5py.string_dtype()
        for idx in sorted(cs_buf):
            self.g_cs.create_dataset(f'contrib_score_Batch_{idx}',
                                     data=cs_buf[idx])
            self.g_in.create_dataset(f'input_Batch_{idx}',
                                     data=in_buf[idx])
            self.g_in.create_dataset(f'regions_Batch_{idx}',
                                     data=np.array(reg_buf[idx], dtype=object),
                                     dtype=dt)
        cs_buf.clear()
        in_buf.clear()
        reg_buf.clear()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='TransRBP full-model (sequence + m6A) attribution')

    parser.add_argument('--RBPname',    required=True,
                        help='RBP name (used as HDF5 group key)')
    parser.add_argument('--RBPmodel',   required=True,
                        help='Path to .pth model file (5-channel TransRBP)')
    parser.add_argument('--out_h5_fname', required=True,
                        help='Output HDF5 file path')

    # Input — one of these two is required
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--peak_bed',  default=None,
                     help='BED6 file of peaks (requires --chrom_root, --m6A_bw_plus, --m6A_bw_minus).')
    grp.add_argument('--fasta_file', default=None,
                     help='FASTA file of sequences (800 nt; shorter sequences padded with N). '
                          'm6A channel is set to zero.')

    parser.add_argument('--chrom_root',   default=None,
                        help='Dir with {chrom}.fa.gz — required with --peak_bed')
    parser.add_argument('--m6A_bw_plus',  default=None,
                        help='m6A MeRIP-seq BigWig (plus strand) — required with --peak_bed')
    parser.add_argument('--m6A_bw_minus', default=None,
                        help='m6A MeRIP-seq BigWig (minus strand) — required with --peak_bed')
    parser.add_argument('--window', default='start', choices=['start', 'center'],
                        help='Anchor of the 800 bp window: peak start (paper) or peak midpoint')
    parser.add_argument('--contrib_function', default='IG',
                        choices=['IG', 'SA'],
                        help='Attribution method: IG (Integrated Gradients) or SA (Saliency)')
    parser.add_argument('--record_global_contrib', action='store_true',
                        help='If set, save raw gradients (not multiplied by input)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device',     default='cuda:0')

    args = parser.parse_args()

    contrib = ContribH5(
        filename         = args.out_h5_fname,
        RBPname          = args.RBPname,
        RBPmodel_path    = args.RBPmodel,
        peak_bed         = args.peak_bed,
        chrom_root       = args.chrom_root,
        m6A_bw_plus      = args.m6A_bw_plus,
        m6A_bw_minus     = args.m6A_bw_minus,
        window           = args.window,
        fasta_file       = args.fasta_file,
        contrib_function = args.contrib_function,
        record_global    = args.record_global_contrib,
        batch_size       = args.batch_size,
        device           = args.device,
    )
    contrib.run()


if __name__ == '__main__':
    main()
