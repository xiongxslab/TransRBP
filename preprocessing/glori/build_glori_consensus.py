"""
Step 1 of the GLORI m6A track: replicate-consensus m6A sites for one cell line.

Input: the two GLORI replicate site tables from GEO GSE210563 (Liu et al. 2023,
Nat Biotechnol), e.g.
    HEK293T : GSM6432590_293T-mRNA-1_35bp_m2.totalm6A.FDR.csv.gz
              GSM6432591_293T-mRNA-2_35bp_m2.totalm6A.FDR.csv.gz
    HeLa    : GSM6432595_Hela-1_35bp_m2.totalm6A.FDR.csv.gz
              GSM6432596_Hela-2_35bp_m2.totalm6A.FDR.csv.gz

A site is kept if it is called in both replicates (inner join on Chr, Sites, Strand).
Mean_ratio is the mean of the two replicates' NormeRatio. Site coordinates are
kept as provided by GEO (1-based).

Output columns:
    Chr, Site, Strand, Gene, Ratio_rep1, Ratio_rep2, Mean_ratio,
    Cov_rep1, Cov_rep2, Padj_rep1, Padj_rep2

Example:
    python preprocessing/glori/build_glori_consensus.py \\
        --rep1 GSM6432595_Hela-1_35bp_m2.totalm6A.FDR.csv.gz \\
        --rep2 GSM6432596_Hela-2_35bp_m2.totalm6A.FDR.csv.gz \\
        --output GLORI_HeLa_consensus.tsv
"""

import argparse

import pandas as pd

parser = argparse.ArgumentParser(description='GLORI replicate consensus')
parser.add_argument('--rep1',   required=True, help='GEO GLORI site table, replicate 1')
parser.add_argument('--rep2',   required=True, help='GEO GLORI site table, replicate 2')
parser.add_argument('--output', required=True, help='Output consensus TSV')
args = parser.parse_args()

KEY = ['Chr', 'Sites', 'Strand']


def load(path):
    df = pd.read_csv(path, sep='\t', float_precision='round_trip')
    return df[KEY + ['Gene', 'AGcov', 'NormeRatio', 'P_adjust']]


r1 = load(args.rep1)
r2 = load(args.rep2)
m = pd.merge(r1, r2, on=KEY, how='inner', suffixes=('_1', '_2'))

out = pd.DataFrame({
    'Chr':        m['Chr'],
    'Site':       m['Sites'],
    'Strand':     m['Strand'],
    'Gene':       m['Gene_1'],
    'Ratio_rep1': m['NormeRatio_1'],
    'Ratio_rep2': m['NormeRatio_2'],
    'Mean_ratio': (m['NormeRatio_1'] + m['NormeRatio_2']) / 2,
    'Cov_rep1':   m['AGcov_1'],
    'Cov_rep2':   m['AGcov_2'],
    'Padj_rep1':  m['P_adjust_1'],
    'Padj_rep2':  m['P_adjust_2'],
})
out = out.sort_values(['Chr', 'Site']).reset_index(drop=True)
for col in ('Ratio_rep1', 'Ratio_rep2', 'Mean_ratio'):
    out[col] = out[col].map('{:.4f}'.format)
for col in ('Padj_rep1', 'Padj_rep2'):
    out[col] = out[col].map('{:.2e}'.format)
out.to_csv(args.output, sep='\t', index=False)
print(f'rep1 {len(r1):,}  rep2 {len(r2):,}  consensus {len(out):,}  -> {args.output}')
