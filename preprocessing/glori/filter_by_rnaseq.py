"""
Step 3 of the GLORI m6A track: keep GLORI union sites expressed in K562.

The K562 MeRIP-seq RNA-seq input (CPM, both strands; MeRIP_control_merged.bw from
preprocessing/merip_bigwig.sh) is used as a proxy for K562 expression. A site is
kept if the input signal at bigWig position Site is > --threshold (default 0.5).

Example:
    python preprocessing/glori/filter_by_rnaseq.py \
        --input  GLORI_union_consensus.tsv \
        --bw     MeRIP_control_merged.bw \
        --output GLORI_K562proxy_filtered.tsv
"""

import argparse
import os
import pandas as pd
import pyBigWig as pbw

parser = argparse.ArgumentParser()
parser.add_argument('--input',     required=True, help='GLORI_union_consensus.tsv')
parser.add_argument('--bw',        required=True,
                    help='Merged RNA-seq BigWig for K562 expression filter')
parser.add_argument('--output',    required=True, help='GLORI_K562proxy_filtered.tsv')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Minimum RNA-seq signal to consider a site expressed')
args = parser.parse_args()

print(f"Loading {args.input} ...")
df = pd.read_csv(args.input, sep='\t')
df['Site'] = df['Site'].astype(int)
print(f"  Total sites: {len(df):,}")

print(f"Opening BigWig: {args.bw}")
bw = pbw.open(args.bw)

# Vectorised-style loop — pyBigWig requires per-query calls
keep = []
for i, row in enumerate(df.itertuples(index=False)):
    if i % 50000 == 0:
        print(f"  Processing {i:,} / {len(df):,} ...", flush=True)
    try:
        vals = bw.values(row.Chr, row.Site, row.Site + 1)
        sig = 0.0 if (vals is None or vals[0] is None) else float(vals[0])
    except RuntimeError:
        sig = 0.0
    keep.append(sig > args.threshold)

bw.close()

filtered = df[keep].reset_index(drop=True)
n_kept = len(filtered)
pct    = 100.0 * n_kept / len(df)
print(f"\nThreshold = {args.threshold}")
print(f"Retained  : {n_kept:,} / {len(df):,} sites ({pct:.1f}%)")
print(f"  Both    : {(filtered['Source']=='Both').sum():,}")
print(f"  HeLa    : {(filtered['Source']=='HeLa').sum():,}")
print(f"  293T    : {(filtered['Source']=='293T').sum():,}")

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
filtered.to_csv(args.output, sep='\t', index=False)
print(f"\nSaved → {args.output}")
