"""
Compute Integrated Gradient (IG) or Saliency attribution scores for TransRBP.

Input sequences can be supplied in two ways (mutually exclusive):
  --peak_bed   BED6 file — 800 bp window is extracted centered on each peak's
               midpoint from the reference genome. Recommended for ENCODE peaks.
  --fasta_file FASTA file — sequences must be exactly 800 nt (padded with N if
               shorter, trimmed if longer).

The model loaded from --RBPmodel is the 5-channel (seq + m6A) TransRBP.
For IG purposes the m6A channel is set to zero; only the four sequence
channels contribute to the attribution.

Output HDF5 layout:
    {RBPname}/
      Contribution_Score/
        contrib_score_Batch_0    # (B, 4, 800)  IG score per nucleotide
        contrib_score_Batch_1
        ...
      Inputs/
        input_Batch_0            # (B, 4, 800)  one-hot encoded sequence
        regions_Batch_0          # region strings (chr:start-end_strand or seq ID)
        ...

Example (BED input — recommended):
    python interpretation/contrib_h5.py \\
        --RBPname    FMR1 \\
        --RBPmodel   models/FMR1.pth \\
        --peak_bed   data/FMR1_test.bed \\
        --chrom_root /path/to/hg38/dna_sequence \\
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency
from torch.utils.data import Dataset, DataLoader

from model.RBPResTransModels import RBPModel

torch.set_float32_matmul_precision('high')


# ── Datasets ──────────────────────────────────────────────────────────────────

class BEDSequenceDataset(Dataset):
    """
    Reads BED peaks and extracts an 800 bp window centered on each peak midpoint.
    Returns (one_hot_tensor [4, 800], region_string).
    """

    def __init__(self, peak_bed, chrom_root, seq_length=800):
        self.chrom_root = chrom_root
        self.seq_length = seq_length
        self.half       = seq_length // 2
        self.peaks      = self._load_peaks(peak_bed)

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, idx):
        chrom, start, end, strand = self.peaks[idx]
        center    = (start + end) // 2
        win_start = center - self.half
        win_end   = center + self.half

        rna    = self._get_seq(chrom, win_start, win_end, strand)
        tensor = self._onehot(rna)                    # (4, 800)
        region = f'{chrom}:{win_start}-{win_end}_{strand}'
        return tensor, region

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
        with gzip.open(f'{self.chrom_root}/{chrom}.fa.gz', 'r') as f:
            raw = f.read().decode('utf-8')
            raw = raw[raw.find('\n'):].replace('\n', '').lower()
        # pad if near chromosome boundary
        raw_len = len(raw)
        if start < 0 or end > raw_len:
            pad_l = max(0, -start)
            pad_r = max(0, end - raw_len)
            raw   = 'n' * pad_l + raw[max(0, start):min(end, raw_len)] + 'n' * pad_r
            start, end = 0, self.seq_length
        sub = raw[start:end]
        if direction == '-':
            comp = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
            return ''.join(comp.get(b, 'n') for b in sub)[::-1]
        return sub.replace('t', 'u')

    def _onehot(self, rna_seq):
        en  = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
        idx = np.array([en.get(nt, 4) for nt in rna_seq], dtype=int)
        emb = np.zeros((len(rna_seq), 5), dtype=np.float32)
        emb[np.arange(len(rna_seq)), idx] = 1
        return torch.from_numpy(emb.T[:4, :])        # (4, 800) drop N channel


class FastaSequenceDataset(Dataset):
    """
    Reads a FASTA file. Each sequence is padded/trimmed to seq_length.
    Returns (one_hot_tensor [4, 800], sequence_id).
    """

    def __init__(self, fasta_file, seq_length=800):
        self.seq_length = seq_length
        self.entries    = self._load_fasta(fasta_file)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        seq_id, seq = self.entries[idx]
        seq    = self._pad_trim(seq.lower().replace('t', 'u'))
        tensor = self._onehot(seq)
        return tensor, seq_id

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

    def _onehot(self, rna_seq):
        en  = {'a': 0, 'u': 1, 'c': 2, 'g': 3, 'n': 4}
        idx = np.array([en.get(nt, 4) for nt in rna_seq], dtype=int)
        emb = np.zeros((len(rna_seq), 5), dtype=np.float32)
        emb[np.arange(len(rna_seq)), idx] = 1
        return torch.from_numpy(emb.T[:4, :])


# ── Model wrapper for IG ──────────────────────────────────────────────────────

class SeqOnlyWrapper(nn.Module):
    """
    Wraps the 5-channel RBPModel so IG can differentiate w.r.t. the
    4-channel sequence input only. The m6A channel (channel 4) is fixed to zero.
    Returns a scalar per sample (softmax-weighted sum of predictions).
    """

    def __init__(self, rbp_model, device):
        super().__init__()
        self.model  = rbp_model
        self.device = device

    def forward(self, seq_input):
        # seq_input: (B, 4, 800)
        m6a = torch.zeros(seq_input.shape[0], 1, seq_input.shape[2],
                          device=seq_input.device, dtype=seq_input.dtype)
        x    = torch.cat([seq_input, m6a], dim=1)    # (B, 5, 800)
        pred = self.model(x)                          # (B, 1, 800)
        pred_sq  = torch.squeeze(pred, dim=1)         # (B, 800)
        weights  = F.softmax(pred_sq.detach(), dim=1)
        return torch.sum(pred_sq * weights, dim=1)    # (B,)


# ── Core attribution class ────────────────────────────────────────────────────

class ContribH5:

    def __init__(self, filename, RBPname, RBPmodel_path,
                 peak_bed=None, chrom_root=None,
                 fasta_file=None,
                 contrib_function='IG', record_global=False,
                 batch_size=64, device='cuda:0'):

        if (peak_bed is None) == (fasta_file is None):
            raise ValueError('Provide exactly one of --peak_bed or --fasta_file.')
        if peak_bed is not None and chrom_root is None:
            raise ValueError('--chrom_root is required when --peak_bed is used.')

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
        self.wrapped_model = SeqOnlyWrapper(base_model, device).to(device)
        print('Model loaded.')

        # Build dataset
        if peak_bed is not None:
            dataset = BEDSequenceDataset(peak_bed, chrom_root)
            print(f'BED input: {len(dataset)} peaks')
        else:
            dataset = FastaSequenceDataset(fasta_file)
            print(f'FASTA input: {len(dataset)} sequences')

        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0,
        )

        # HDF5 output
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

        for idx, (seq_tensor, regions) in enumerate(self.dataloader):
            print(f'Batch {idx+1}/{n_batches}  shape={seq_tensor.shape}')
            seq_tensor = seq_tensor.to(self.device)
            seq_tensor.requires_grad_()

            if self.contrib_function == 'IG':
                attr = attributor.attribute(seq_tensor,
                                            internal_batch_size=self.batch_size)
            else:
                attr = attributor.attribute(seq_tensor, abs=False)

            # Koo lab correction: subtract per-position mean across bases
            attr = attr - torch.mean(attr, dim=1, keepdim=True)
            if not self.record_global:
                attr = attr * seq_tensor           # element-wise × one-hot input

            cs_buf[idx]  = attr.detach().cpu().numpy()
            in_buf[idx]  = seq_tensor.detach().cpu().numpy()
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
        description='TransRBP Integrated Gradients attribution')

    parser.add_argument('--RBPname',    required=True,
                        help='RBP name (used as HDF5 group key)')
    parser.add_argument('--RBPmodel',   required=True,
                        help='Path to .pth model file')
    parser.add_argument('--out_h5_fname', required=True,
                        help='Output HDF5 file path')

    # Input — one of these two is required
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--peak_bed',  default=None,
                     help='BED6 file of peaks. 800 bp window centered on peak '
                          'midpoint is extracted from the genome (requires --chrom_root).')
    grp.add_argument('--fasta_file', default=None,
                     help='FASTA file of sequences (exactly 800 nt each; '
                          'shorter sequences padded with N).')

    parser.add_argument('--chrom_root', default=None,
                        help='Dir with {chrom}.fa.gz — required when --peak_bed is used')
    parser.add_argument('--contrib_function', default='IG',
                        choices=['IG', 'SA'],
                        help='Attribution method: IG (Integrated Gradients) or SA (Saliency)')
    parser.add_argument('--record_global_contrib', action='store_true',
                        help='If set, save raw gradients (not multiplied by input)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device',     default='cuda:0')

    args = parser.parse_args()

    contrib = ContribH5(
        filename        = args.out_h5_fname,
        RBPname         = args.RBPname,
        RBPmodel_path   = args.RBPmodel,
        peak_bed        = args.peak_bed,
        chrom_root      = args.chrom_root,
        fasta_file      = args.fasta_file,
        contrib_function = args.contrib_function,
        record_global   = args.record_global_contrib,
        batch_size      = args.batch_size,
        device          = args.device,
    )
    contrib.run()


if __name__ == '__main__':
    main()
