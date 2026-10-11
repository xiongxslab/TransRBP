"""
Motif discovery from full-model IG attributions: top k-mers -> clustered PWMs
(Fig. 2, Extended Data Figs. 4-6).

For each RBP (one HDF5 file from interpretation/contrib_h5.py):
  1. Every k-mer window (default 5 nt) of every sequence is scored by the sum of
     its sequence-channel contribution scores (channels 0-3, A/U/C/G; the m6A
     channel is not used). The top-N windows (default 10,000) are decoded to
     k-mers (argmax over the contribution matrix at each position).
  2. K-mers are processed in rank order. Each not-yet-assigned k-mer becomes a
     base sequence and collects all remaining k-mers within one mismatch
     (centre) or shifted by one nucleotide (left / right); these are stacked into
     a 7 x 4 position count matrix (columns A, C, G, U).
  3. Base sequences are ranked by the number of supporting k-mers and the top-3
     PWMs are kept.

Output (out_dir):
    {RBP}/kmers_top{N}.fasta         ranked k-mers
    {RBP}/pwm/{base_seq}_pwm.npy     count matrix (7 x 4, A C G U) for every base sequence
    {RBP}/sequence_support.csv       Base_Sequence, N_Supporting (sorted)
    top3/{RBP}_{rank}_{base_seq}_pwm.npy
    top3/pwm_data.csv                top-3 PWMs of all RBPs, flattened column-wise
                                     (A0..A6, C0..C6, G0..G6, U0..U6) + label {RBP}_#{rank}

Example:
    python interpretation/motif_clustering.py \\
        --rbp_list rbp_list.txt --h5_dir contrib_h5/ --out_dir motif_out/
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd

NUCLEOTIDES = ['A', 'U', 'C', 'G']          # contribution channel order


# ── 1. top-N k-mers ───────────────────────────────────────────────────────────

def top_kmers(h5_path, rbp, n_mer=5, top_n=10000):
    scores, windows = [], []
    with h5py.File(h5_path, 'r') as f:
        group = f[f'{rbp}/Contribution_Score']
        for batch_index in range(len(group)):
            for matrix in f[f'{rbp}/Contribution_Score/contrib_score_Batch_{batch_index}']:
                seq_matrix = matrix[:4, :]                  # drop the m6A channel
                for col in range(seq_matrix.shape[1] - (n_mer - 1)):
                    scores.append(np.sum(seq_matrix[:, col:col + n_mer]))
                windows.append(seq_matrix)
    order = np.argsort(-np.array(scores), kind='stable')[:int(top_n)]   # descending, ties in input order
    n_win = windows[0].shape[1] - (n_mer - 1)
    kmers = []
    for k in order:
        window = windows[k // n_win][:, k % n_win:k % n_win + n_mer]
        kmers.append(''.join(NUCLEOTIDES[np.argmax(window[:, c])] for c in range(n_mer)))
    return kmers


# ── 2. 1-mismatch / 1-shift aggregation ───────────────────────────────────────

def aggregate(base_seq, sequences):
    groups = {'left': [], 'center': [], 'right': []}
    for seq in sequences:
        # centre is checked first so each k-mer is assigned to one bucket only
        if seq == base_seq or sum(c1 != c2 for c1, c2 in zip(base_seq, seq)) <= 1:
            groups['center'].append(seq)
        elif base_seq[:-1] == seq[1:]:
            groups['left'].append(seq)
        elif base_seq[1:] == seq[:-1]:
            groups['right'].append(seq)
    return groups


def to_pwm(groups):
    pwm = np.zeros((7, 4))
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    offset = {'left': 0, 'center': 1, 'right': 2}
    for pos, sequences in groups.items():
        for seq in sequences:
            for i, base in enumerate(seq[:5]):
                pwm[i + offset[pos], base_idx[base]] += 1
    return pwm


def cluster_kmers(kmers, pwm_dir):
    os.makedirs(pwm_dir, exist_ok=True)
    processed, support = set(), []
    for base_seq in kmers:
        if base_seq in processed:
            continue
        groups = aggregate(base_seq, [s for s in kmers if s not in processed])
        for members in groups.values():
            processed.update(members)
        np.save(os.path.join(pwm_dir, f'{base_seq}_pwm.npy'), to_pwm(groups))
        support.append({'Base_Sequence': base_seq,
                        'N_Supporting': sum(len(m) for m in groups.values())})
    support_df = pd.DataFrame(support)
    support_df.sort_values(by='N_Supporting', ascending=False, inplace=True)
    return support_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Top k-mers -> clustered PWMs from IG attributions')
    parser.add_argument('--rbp_list', required=True, help='Text file, one RBP name per line')
    parser.add_argument('--h5_dir',   required=True, help='Dir with {RBP}.h5 from contrib_h5.py')
    parser.add_argument('--out_dir',  required=True)
    parser.add_argument('--n_mer',    type=int, default=5)
    parser.add_argument('--top_N',    type=int, default=10000)
    parser.add_argument('--top_k',    type=int, default=3, help='PWMs kept per RBP')
    args = parser.parse_args()

    with open(args.rbp_list) as f:
        rbps = [line.strip() for line in f if line.strip()]
    top_dir = os.path.join(args.out_dir, 'top3')
    os.makedirs(top_dir, exist_ok=True)

    rows = []
    for rbp in rbps:
        rbp_dir = os.path.join(args.out_dir, rbp)
        os.makedirs(rbp_dir, exist_ok=True)
        kmers = top_kmers(os.path.join(args.h5_dir, f'{rbp}.h5'), rbp, args.n_mer, args.top_N)
        with open(os.path.join(rbp_dir, f'kmers_top{args.top_N}.fasta'), 'w') as f:
            f.writelines(f'>seq_{i}\n{k}\n' for i, k in enumerate(kmers))

        support = cluster_kmers(kmers, os.path.join(rbp_dir, 'pwm'))
        support.to_csv(os.path.join(rbp_dir, 'sequence_support.csv'), index=False)

        for rank, seq in enumerate(support['Base_Sequence'].head(args.top_k), start=1):
            pwm = np.load(os.path.join(rbp_dir, 'pwm', f'{seq}_pwm.npy'))
            np.save(os.path.join(top_dir, f'{rbp}_{rank}_{seq}_pwm.npy'), pwm)
            row = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'U']).unstack().reset_index(drop=True)
            row['label'] = f'{rbp}_#{rank}'
            rows.append(row)
        print(f'{rbp}: {len(support)} PWMs; top {args.top_k}: '
              f'{", ".join(support["Base_Sequence"].head(args.top_k))}')

    pd.concat(rows, axis=1).T.reset_index(drop=True).to_csv(os.path.join(top_dir, 'pwm_data.csv'), index=False)
    print(f'Top-{args.top_k} PWMs -> {top_dir}/pwm_data.csv')


if __name__ == '__main__':
    main()
