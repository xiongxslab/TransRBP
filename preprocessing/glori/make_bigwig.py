"""
Step 4 of the GLORI m6A track: write strand-specific sparse BigWig files.

Output BigWig signal = GLORI methylation ratio (Mean_ratio, 0-1) at each site,
written at bigWig interval [Site, Site+1). All other positions are absent (NaN),
which the dataset reads as 0.

Example:
    python preprocessing/glori/make_bigwig.py \
        --input       GLORI_K562proxy_filtered.tsv \
        --bw_root     /path/to/bamCompare_bw \
        --chrom_sizes hg38/annotation/hg38.chrom.sizes
    # -> {bw_root}/m6A_GLORI_K562proxy_plus.bw, _minus.bw
"""

import argparse
import os
import pandas as pd
import pyBigWig as pbw

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True,
                    help='GLORI_K562proxy_filtered.tsv')
parser.add_argument('--bw_root', required=True,
                    help='Directory where output BigWig files will be written')
parser.add_argument('--chrom_sizes', required=True,
                    help='hg38.chrom.sizes')
parser.add_argument('--track_name',
                    default='m6A_GLORI_K562proxy',
                    help='Output track name; files become {track_name}_plus.bw / _minus.bw')
args = parser.parse_args()

# ── Load chromosome sizes ──────────────────────────────────────────────────
sizes = {}
with open(args.chrom_sizes) as f:
    for line in f:
        ch, sz = line.strip().split()
        sizes[ch] = int(sz)
chrom_order = {ch: i for i, ch in enumerate(sizes)}

# ── Load filtered sites ────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
df = pd.read_csv(args.input, sep='\t')
df['Site'] = df['Site'].astype(int)
df = df[df['Chr'].isin(sizes)].copy()           # drop non-standard contigs
df['chrom_idx'] = df['Chr'].map(chrom_order)
df = df.sort_values(['chrom_idx', 'Site'])       # must be sorted for pyBigWig
print(f"  Sites after chrom filter: {len(df):,}")

os.makedirs(args.bw_root, exist_ok=True)

for strand, suffix in [('+', 'plus'), ('-', 'minus')]:
    sub = df[df['Strand'] == strand]
    out_path = os.path.join(args.bw_root, f'{args.track_name}_{suffix}.bw')

    bw_out = pbw.open(out_path, 'w')
    bw_out.addHeader(list(sizes.items()))

    n_written = 0
    for chrom, grp in sub.groupby('Chr', sort=False):
        starts = grp['Site'].tolist()
        ends   = [s + 1 for s in starts]
        vals   = grp['Mean_ratio'].astype(float).tolist()
        chroms = [chrom] * len(grp)
        bw_out.addEntries(chroms, starts, ends=ends, values=vals)
        n_written += len(grp)

    bw_out.close()
    print(f"[{suffix:5s}] {n_written:>7,} sites → {out_path}")

print("\nDone. Spot-check with:")
print(f"  python -c \"import pyBigWig as p; b=p.open('{args.bw_root}/{args.track_name}_plus.bw'); print(b.header())\"")
