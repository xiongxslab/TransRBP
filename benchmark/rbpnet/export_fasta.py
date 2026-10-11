"""
Export the RBPNet benchmark train/test windows as FASTA for BLASTN homology filtering.

Writes {out_dir}/{RBP}_train.fa and {out_dir}/{RBP}_test.fa with headers
    >{RBP}|{split}|{index}|{chrom}:{start}-{end}({strand})
which homology.sh consumes to produce {RBP}_test_homology.ids.

Example:
    python benchmark/rbpnet/export_fasta.py \\
        --rbp rbp_list.txt \\
        --peak_root /path/to/RBP_bed \\
        --bw_root   /path/to/bamCompare_bw \\
        --chrom_root hg38/dna_sequence \\
        --out_dir   rbpnet_fasta
    sh benchmark/rbpnet/homology.sh rbp_list.txt rbpnet_fasta
"""

import argparse
import os

from benchmark.rbpnet.dataset import RBPDataset

parser = argparse.ArgumentParser(description='Export RBPNet benchmark windows as FASTA')
parser.add_argument('--rbp',        required=True, help='Text file, one RBP name per line')
parser.add_argument('--peak_root',  required=True, help='Dir containing {RBP}.bed')
parser.add_argument('--bw_root',    required=True, help='Dir with {RBP}_plus/minus.bw')
parser.add_argument('--chrom_root', required=True, help='Dir with {chrom}.fa.gz')
parser.add_argument('--out_dir',    required=True, help='Output FASTA directory')
parser.add_argument('--input_len',  type=int, default=300)
parser.add_argument('--seed',       type=int, default=43)
args = parser.parse_args()


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.rbp) as f:
        rbp_list = [line.strip() for line in f if line.strip()]
    for rbp in rbp_list:
        for mode in ('train', 'test'):
            ds = RBPDataset(rbp, args.peak_root, args.bw_root, args.chrom_root,
                            mode=mode, input_len=args.input_len, seed=args.seed)
            ds.export_fasta(os.path.join(args.out_dir, f'{rbp}_{mode}.fa'))


if __name__ == '__main__':
    main()
