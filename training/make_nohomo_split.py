"""
Generate the no-homology train/val/test BED splits used for the 32 m6A-associated RBPs.

For each RBP:
  1. Sliding 800 bp windows are generated across every eCLIP peak
     (step = peak_length/10 + 1, from start-400 to end+800), and assigned by
     chromosome: val = chr2/3/4, test = chr1/8/9, train = all others.
  2. Windows are randomly sampled (fixed seed): 14,000 train / 4,000 val / 2,500 test.
  3. Test windows are BLASTN-searched against the training windows
     (makeblastdb + blastn, e-value <= 1e-3). A test window is called homologous
     if any hit has identity >= 80% over >= 50% of the query length.
  4. Homologous test windows are removed and 2,000 test windows are re-sampled.

The splits deposited on Zenodo (data/) were generated with --seed 42.

Requires BLAST+ (makeblastdb, blastn) on PATH; installed by environment.yml
(bioconda blast=2.17.0, the version that reproduces the deposited splits).

Input:
    {peak_dir}/{RBP}.bed      BED6 eCLIP peaks: ENCODE K562 replicated peaks on chr1-22
                              (preprocessing/get_encode_peaks.py)
    {genome_dir}/{chrom}.fa.gz

Output:
    {out_dir}/{RBP}_{train,val,test}.bed    BED6, name = {RBP}|{split}|{index}

Example:
    python training/make_nohomo_split.py \\
        --rbp-list   rbp_list.txt \\
        --peak-dir   /path/to/RBP_bed \\
        --genome-dir hg38/dna_sequence \\
        --out-dir    data/ \\
        --seed 42
"""

import argparse
import gzip
import os
import random
import subprocess
import tempfile


def sample_choice(seq, sample_size, seed):
    random.seed(seed)
    if len(seq) <= sample_size:
        return seq
    return random.sample(seq, sample_size)


# total_num: per-RBP window budget (20,000), as in the original chromosome-split
# training data, so that every RBP has the same dataset size regardless of its
# number of peaks. Training and validation take 70% and 20% (14,000 / 4,000).
# The test split draws total_num / 8 = 2,500 candidates instead of 10% (2,000),
# so that 2,000 test windows remain after homologous windows are removed;
# the final splits are 14,000 / 4,000 / 2,000 = 20,000 windows, and the 2,000
# non-homologous test windows form the final test set.
def get_window(bed_file, seed, input_len=800, total_num=20000):
    val_chr  = {'chr2', 'chr3', 'chr4'}
    test_chr = {'chr1', 'chr8', 'chr9'}
    trains, vals, tests = [], [], []

    with open(bed_file) as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip().split('\t')
            chrom, start, end, strand = fields[0], int(fields[1]), int(fields[2]), fields[5]
            peak_length = end - start
            for bp in range(start - input_len // 2, end + input_len, int(peak_length / 10) + 1):
                item = (bp - input_len + 1, bp + 1, chrom, strand)
                if chrom in val_chr:
                    vals.append(item)
                elif chrom in test_chr:
                    tests.append(item)
                else:
                    trains.append(item)

    return (
        sample_choice(sorted(set(trains)), int(total_num / 10 * 7), seed),
        sample_choice(sorted(set(vals)),   int(total_num / 10 * 2), seed),
        sample_choice(sorted(set(tests)),  int(total_num / 8),      seed),
    )


def load_chrom(genome_dir, chrom, cache):
    if chrom not in cache:
        with gzip.open(os.path.join(genome_dir, f'{chrom}.fa.gz'), 'rt') as f:
            seq = ''.join(line.strip() for line in f if not line.startswith('>'))
        cache[chrom] = seq.upper()
    return cache[chrom]


def get_sequence(genome_dir, chrom, start, end, strand, cache):
    seq = load_chrom(genome_dir, chrom, cache)[start:end]
    if strand == '-':
        seq = seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]
    return seq


def fasta_id(rbp, mode, i, chrom, start, end, strand):
    return f'{rbp}|{mode}|{i}|{chrom}:{start}-{end}({strand})'


def write_fasta(windows, rbp, mode, genome_dir, outfile):
    cache = {}
    with open(outfile, 'w') as out:
        for i, (start, end, chrom, strand) in enumerate(windows):
            seq = get_sequence(genome_dir, chrom, start, end, strand, cache)
            out.write(f'>{fasta_id(rbp, mode, i, chrom, start, end, strand)}\n{seq}\n')


def find_homology(rbp, train_windows, test_windows, genome_dir, temp_dir, threads,
                  min_pident=80, min_qcov=0.5, evalue=1e-3):
    train_fa  = os.path.join(temp_dir, f'{rbp}_train.fa')
    test_fa   = os.path.join(temp_dir, f'{rbp}_test.fa')
    db_prefix = os.path.join(temp_dir, f'{rbp}_train_db')
    blast_out = os.path.join(temp_dir, f'{rbp}_blast.tsv')

    write_fasta(train_windows, rbp, 'train', genome_dir, train_fa)
    write_fasta(test_windows,  rbp, 'test',  genome_dir, test_fa)

    subprocess.run(['makeblastdb', '-in', train_fa, '-dbtype', 'nucl', '-out', db_prefix],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['blastn', '-query', test_fa, '-db', db_prefix, '-out', blast_out,
                    '-outfmt', '6 qseqid sseqid pident length qlen slen evalue bitscore',
                    '-evalue', str(evalue), '-num_threads', str(threads)],
                   check=True)

    homologous_ids = set()
    with open(blast_out) as f:
        for line in f:
            fields = line.rstrip().split('\t')
            pident, align_len, qlen = float(fields[2]), float(fields[3]), float(fields[4])
            if pident >= min_pident and align_len / qlen >= min_qcov:
                homologous_ids.add(fields[0])
    return homologous_ids


def remove_homology(rbp, test_windows, homologous_ids, seed, n_final=2000):
    kept = [item for i, item in enumerate(test_windows)
            if fasta_id(rbp, 'test', i, item[2], item[0], item[1], item[3]) not in homologous_ids]
    return sample_choice(sorted(set(kept)), n_final, seed)


def write_bed(windows, rbp, mode, outfile):
    with open(outfile, 'w') as out:
        for i, (start, end, chrom, strand) in enumerate(windows):
            out.write(f'{chrom}\t{start}\t{end}\t{rbp}|{mode}|{i}\t0\t{strand}\n')


def main():
    parser = argparse.ArgumentParser(description='No-homology train/val/test split (BLASTN filter)')
    parser.add_argument('--rbp-list',   required=True, help='Text file, one RBP name per line')
    parser.add_argument('--peak-dir',   required=True, help='Dir containing {RBP}.bed eCLIP peaks')
    parser.add_argument('--genome-dir', required=True, help='Dir containing {chrom}.fa.gz')
    parser.add_argument('--out-dir',    required=True, help='Output dir for {RBP}_{split}.bed')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Sampling seed (42 reproduces the Zenodo splits)')
    parser.add_argument('--threads',    type=int, default=32, help='blastn threads')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.rbp_list) as f:
        rbp_list = [line.strip() for line in f if line.strip()]

    for rbp in rbp_list:
        print('=' * 60)
        print(rbp)
        train_peak, val_peak, test_peak = get_window(
            os.path.join(args.peak_dir, f'{rbp}.bed'), seed=args.seed)

        with tempfile.TemporaryDirectory() as temp_dir:
            homologous_ids = find_homology(rbp, train_peak, test_peak,
                                           args.genome_dir, temp_dir, args.threads)
        test_peak_final = remove_homology(rbp, test_peak, homologous_ids, args.seed)
        print(f'{rbp}: test windows before = {len(test_peak)}, '
              f'homologous = {len(homologous_ids)}, final = {len(test_peak_final)}')

        write_bed(train_peak,      rbp, 'train', os.path.join(args.out_dir, f'{rbp}_train.bed'))
        write_bed(val_peak,        rbp, 'val',   os.path.join(args.out_dir, f'{rbp}_val.bed'))
        write_bed(test_peak_final, rbp, 'test',  os.path.join(args.out_dir, f'{rbp}_test.bed'))


if __name__ == '__main__':
    main()
