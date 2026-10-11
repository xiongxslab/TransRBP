"""
In silico mutagenesis of the ClinVar variants: variant impact (KLD), RBP-specific
cutoffs and the association of variant impact with distance to m6A.

Input: clinvar_rbp_variants.tsv from disease/prepare_variants.py.

  1. Each variant is placed at position 400 of an 800 nt window and the
     reference / alternative windows are scored with the RBP's TransRBP model:
     sequence-only (--features 4, default) or full model (--features 5, with the
     m6A BigWigs). KLD = KL(alt || ref) of the normalised 800 nt binding profiles.
  2. RBP-specific cutoff: 90th percentile of KLD of 100,000 background mutations
     (random positions within 800 bp of the RBP's eCLIP peaks, random alternative
     base at the window centre). Variants with KLD >= cutoff are "RBP-dependent".
  3. Spearman correlation between KLD and distance to the nearest m6A peak per
     RBP x disease (variants within 800 bp of an m6A peak, >= 10 variants);
     P-values are Benjamini-Hochberg adjusted across the tested pairs.
     m6A-dependent variants = variants within 800 bp of m6A for pairs with
     SCC < -0.4 and adjusted P < 0.01.

Output (out_dir):
    variant_impact.csv            all variants with KLD, cutoff and Annotation
    m6A_dependent_variants.csv    variants of the m6A-dependent RBP x disease pairs
    m6A_distance_correlation.csv  Spearman SCC, P and BH-adjusted P per RBP x disease
    background_scores.csv         background mutation scores

Models: one {RBP}.pth per RBP in --models_dir. Sequence-only models are trained
with training/make_h5_nohomo.py / train_nohomo.py --m6A_binding 0; the full model
is the released model.

Example:
    python disease/score_variants.py \\
        --variants         disease_out/clinvar_rbp_variants.tsv \\
        --models_dir       models_seq_only/ \\
        --reference_genome hg38/all/hg38.fa \\
        --peak_dir         RBP_bed/ \\
        --out_dir          disease_out/
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import pyBigWig
import torch
from pyfaidx import Fasta
from scipy.stats import spearmanr

from model.RBPResTransModels import RBPModel

torch.set_float32_matmul_precision('high')

HALF = 400                                    # 800 nt window, variant at index 400
COMP = str.maketrans('ACGU', 'UGCA')
TABLE_COLS = ['RBPname', 'Chr', 'Position', 'Strand', 'Ref', 'Alt', 'Mutation_Type', 'Related_Tissue',
              'Disease', 'Distance2m6A', 'Distance_Class', 'KLD', 'cutoff', 'Annotation']


def kl_divergence(p, q):
    """KL(p || q) per sample for (B, 1, L) profiles; called with p = alt, q = ref."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / np.sum(p, axis=2, keepdims=True)
    q = q / np.sum(q, axis=2, keepdims=True)
    return np.atleast_1d(np.squeeze(np.sum(p * np.log(p / q), axis=2)))


def onehot(seq):
    mapping = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    x = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq):
        if nt in mapping:
            x[mapping[nt], i] = 1
    return x


class Scorer:

    def __init__(self, genome, features=4, m6A_bw=None, device='cuda:0', batch_size=64):
        self.genome, self.features, self.m6A_bw = genome, features, m6A_bw
        self.device, self.batch_size = torch.device(device), batch_size

    def load(self, model_path):
        model = RBPModel(features=self.features, record_attn=False)
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
        self.model = model.to(self.device).eval()

    def window(self, chrom, pos0, strand):
        """[pos0-400, pos0+400) on the given strand; returns (RNA sequence, index of pos0)."""
        seq = self.genome[chrom][pos0 - HALF:pos0 + HALF]
        seq = (seq.reverse.complement.seq if strand == '-' else seq.seq).upper().replace('T', 'U')
        return seq, (HALF if strand == '+' else HALF - 1)

    def m6A(self, chrom, pos0, strand):
        bw  = pyBigWig.open(self.m6A_bw[0] if strand == '+' else self.m6A_bw[1])
        sig = np.nan_to_num(np.array(bw.values(chrom, pos0 - HALF, pos0 + HALF), dtype=np.float32), nan=0.0)
        bw.close()
        sig[sig < 0] = 0.0
        return sig[::-1].copy() if strand == '-' else sig

    def score(self, records):
        """records: list of (chrom, pos0, strand, ref_seq, alt_seq); returns KL(alt || ref)."""
        kls = []
        for b in range(0, len(records), self.batch_size):
            batch = records[b:b + self.batch_size]
            ref = np.stack([onehot(r[3]) for r in batch])
            alt = np.stack([onehot(r[4]) for r in batch])
            if self.features == 5:
                m6a = np.stack([self.m6A(r[0], r[1], r[2]) for r in batch])[:, None, :]
                ref, alt = np.concatenate([ref, m6a], 1), np.concatenate([alt, m6a], 1)
            with torch.no_grad():
                out_ref = self.model(torch.from_numpy(ref).to(self.device)).cpu().numpy()
                out_alt = self.model(torch.from_numpy(alt).to(self.device)).cpu().numpy()
            kls.extend(kl_divergence(out_alt, out_ref))
        return np.array(kls)

    def variants(self, df):
        records = []
        for chrom, pos, strand, alt in zip(df['Chr'], df['Position'], df['Strand'], df['Alt']):
            pos0 = int(pos) - 1
            seq, idx = self.window(chrom, pos0, strand)
            alt = alt.replace('T', 'U')
            alt = alt.translate(COMP) if strand == '-' else alt
            records.append((chrom, pos0, strand, seq, seq[:idx] + alt + seq[idx + 1:]))
        return self.score(records)

    def background(self, peak_bed, n, rng):
        intervals = []
        with open(peak_bed) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 6:
                    intervals.append((p[0], int(p[1]) - 800, int(p[2]) + 801, p[5]))
        sizes = np.array([e - s for _, s, e, _ in intervals])
        bounds = np.cumsum(sizes)
        total = int(bounds[-1])
        picks = rng.sample(range(total), n) if total > n else range(total)
        records, rows = [], []
        for k in picks:
            i = int(np.searchsorted(bounds, k, side='right'))
            chrom, s, _, strand = intervals[i]
            pos0 = s + k - (int(bounds[i - 1]) if i else 0)
            seq, idx = self.window(chrom, pos0, strand)
            alt = rng.choice([b for b in 'AUCG' if b != seq[idx]])
            records.append((chrom, pos0, strand, seq, seq[:idx] + alt + seq[idx + 1:]))
            rows.append((f'{chrom} {pos0}', strand, f'{seq[idx]}>{alt}'))
        out = pd.DataFrame(rows, columns=['Mutation_Position', 'Strand', 'Mutation'])
        out['KL'] = self.score(records)
        return out


def bh_adjust(p):
    """Benjamini-Hochberg adjusted P-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def m6A_distance_tests(table):
    rows = []
    near = table[table['Distance2m6A'] <= 800]
    for (rbp, disease), g in near.groupby(['RBPname', 'Disease']):
        if len(g) >= 10:
            scc, p = spearmanr(g['KLD'], g['Distance2m6A'])
            rows.append((rbp, disease, len(g), scc, p))
    corr = pd.DataFrame(rows, columns=['RBPname', 'Disease', 'n', 'SCC', 'P'])
    corr['P_BH'] = bh_adjust(corr['P'])
    return corr.sort_values('P')


def build_tables(variants, kld, background):
    table = variants.copy()
    table['KLD'] = kld
    cutoff = background.groupby('RBPname')['KL'].quantile(0.9)
    table['cutoff'] = table['RBPname'].map(cutoff)
    table['Annotation'] = np.where(table['KLD'] >= table['cutoff'], 'RBP-dependent', 'N.S.')
    table = table[TABLE_COLS]

    corr = m6A_distance_tests(table)
    sig = corr[(corr['SCC'] < -0.4) & (corr['P_BH'] < 0.01)]
    keys = set(zip(sig['RBPname'], sig['Disease']))
    s6 = table[[k in keys for k in zip(table['RBPname'], table['Disease'])] & (table['Distance2m6A'] < 800)]
    return table, s6, corr


def main():
    parser = argparse.ArgumentParser(description='ClinVar variant impact (KLD), cutoffs and association with m6A distance')
    parser.add_argument('--variants', required=True, help='clinvar_rbp_variants.tsv from prepare_variants.py')
    parser.add_argument('--models_dir', required=True, help='Dir with {RBP}.pth (sequence-only models)')
    parser.add_argument('--reference_genome', required=True, help='hg38.fa (with .fai)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--peak_dir', required=True, help='Dir with {RBP}.bed eCLIP peaks (background sampling)')
    parser.add_argument('--n_background', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--features', type=int, default=4, choices=[4, 5],
                        help='4 = sequence-only model (paper); 5 = full model (needs --m6A_bw_plus/minus)')
    parser.add_argument('--m6A_bw_plus', default=None)
    parser.add_argument('--m6A_bw_minus', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bs', type=int, default=64)
    args = parser.parse_args()
    if args.features == 5 and None in (args.m6A_bw_plus, args.m6A_bw_minus):
        parser.error('--features 5 requires --m6A_bw_plus and --m6A_bw_minus')

    os.makedirs(args.out_dir, exist_ok=True)
    variants = pd.read_csv(args.variants, sep='\t', keep_default_na=False, na_values=[''])
    scorer = Scorer(Fasta(args.reference_genome), args.features,
                    (args.m6A_bw_plus, args.m6A_bw_minus), args.device, args.bs)
    rng = random.Random(args.seed)

    kld = np.zeros(len(variants))
    background = []
    for rbp in pd.unique(variants['RBPname']):
        scorer.load(os.path.join(args.models_dir, f'{rbp}.pth'))
        mask = (variants['RBPname'] == rbp).values
        kld[mask] = scorer.variants(variants[mask])
        bg = scorer.background(os.path.join(args.peak_dir, f'{rbp}.bed'), args.n_background, rng)
        bg.insert(0, 'RBPname', rbp)
        background.append(bg)
        print(f'{rbp}: {mask.sum()} variants scored', flush=True)
    background = pd.concat(background, ignore_index=True)
    background.to_csv(os.path.join(args.out_dir, 'background_scores.csv'), index=False)

    table, s6, corr = build_tables(variants, kld, background)
    table.to_csv(os.path.join(args.out_dir, 'variant_impact.csv'), index=False)
    s6.to_csv(os.path.join(args.out_dir, 'm6A_dependent_variants.csv'), index=False)
    corr.to_csv(os.path.join(args.out_dir, 'm6A_distance_correlation.csv'), index=False)
    print(f'Variant impact: {len(table):,} variants ({(table["Annotation"] == "RBP-dependent").sum():,} RBP-dependent)')
    print(f'm6A-dependent variants: {len(s6):,}; m6A-distance pairs (|SCC| > 0.4, BH-adjusted P < 0.01):')
    print(corr[(corr['SCC'].abs() > 0.4) & (corr['P_BH'] < 0.01)].to_string(index=False))


if __name__ == '__main__':
    main()
