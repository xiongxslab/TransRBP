"""
Step 2 of the GLORI m6A track: union of the HeLa and 293T GLORI consensus m6A sites.

Inputs (from build_glori_consensus.py):
    GLORI_HeLa_consensus.tsv
    GLORI_293T_consensus.tsv

Output:
    GLORI_union_consensus.tsv
        columns: Chr, Site, Strand, Gene, Mean_ratio, Source
        Source = 'HeLa' | '293T' | 'Both'

Example:
    python preprocessing/glori/build_GLORI_union.py \\
        --hela GLORI_HeLa_consensus.tsv --t293 GLORI_293T_consensus.tsv \\
        --output GLORI_union_consensus.tsv
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Union of HeLa and 293T GLORI consensus sites')
parser.add_argument('--hela',   required=True, help='GLORI_HeLa_consensus.tsv')
parser.add_argument('--t293',   required=True, help='GLORI_293T_consensus.tsv')
parser.add_argument('--output', required=True, help='GLORI_union_consensus.tsv')
args = parser.parse_args()

hela  = pd.read_csv(args.hela,  sep='\t')
t293  = pd.read_csv(args.t293,  sep='\t')

hela  = hela[['Chr', 'Site', 'Strand', 'Gene', 'Mean_ratio']].copy()
t293  = t293[['Chr', 'Site', 'Strand', 'Gene', 'Mean_ratio']].copy()

hela['Site']  = hela['Site'].astype(int)
t293['Site']  = t293['Site'].astype(int)

# Outer merge on genomic key
merged = pd.merge(
    hela.rename(columns={'Mean_ratio': 'ratio_HeLa', 'Gene': 'Gene_HeLa'}),
    t293.rename(columns={'Mean_ratio': 'ratio_293T', 'Gene': 'Gene_293T'}),
    on=['Chr', 'Site', 'Strand'],
    how='outer',
)

def assign_source(row):
    h = pd.notna(row['ratio_HeLa'])
    t = pd.notna(row['ratio_293T'])
    if h and t:
        return 'Both'
    elif h:
        return 'HeLa'
    else:
        return '293T'

merged['Source']     = merged.apply(assign_source, axis=1)
merged['Gene']       = merged['Gene_HeLa'].combine_first(merged['Gene_293T'])
merged['Mean_ratio'] = merged[['ratio_HeLa', 'ratio_293T']].mean(axis=1, skipna=True)

out = merged[['Chr', 'Site', 'Strand', 'Gene', 'Mean_ratio', 'Source']].copy()
out = out.sort_values(['Chr', 'Site']).reset_index(drop=True)

out_path = args.output
out.to_csv(out_path, sep='\t', index=False)

print(f"HeLa sites  : {len(hela):>7,}")
print(f"293T sites  : {len(t293):>7,}")
print(f"Union total : {len(out):>7,}")
print(f"  Both      : {(out['Source']=='Both').sum():>7,}")
print(f"  HeLa only : {(out['Source']=='HeLa').sum():>7,}")
print(f"  293T only : {(out['Source']=='293T').sum():>7,}")
print(f"\nSaved → {out_path}")
