"""
Standalone evaluation for the pretrained 32-RBP multi-task model.

Loads the saved checkpoint, runs inference on the test split of the HDF5 file,
and computes all four metrics (flat_pcc, flat_scc, sample_pcc, sample_scc).

Usage:
    python eval_pretrain.py \
        --model_path models_pretrain/pretrained_32rbp_m6a.pth \
        --h5_path    h5_cache/32rbp_m6a_100k.h5 \
        --output_csv test_metrics_eval.csv \
        --device     cuda:1
"""

import argparse
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from training.pretrain.dataset_h5_multitask import MultiRBPDatasetH5
from model.pretrain.RBPMultiTaskModel import RBPMultiTaskModel

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True,
                    help='Path to pretrained_32rbp_m6a.pth checkpoint')
parser.add_argument('--h5_path', required=True,
                    help='Path to HDF5 file (must contain test split)')
parser.add_argument('--output_csv', default=None,
                    help='Path to save per-RBP metrics CSV')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--num_tasks', type=int, default=32)
parser.add_argument('--features', type=int, default=5)
args = parser.parse_args()


# ------------------------------------------------------------------
# Vectorized metric functions (identical to main_pretrain.py with nanmean fix)
# ------------------------------------------------------------------

def _pcc_rows(X, Y):
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    num   = (X * Y).sum(axis=1)
    denom = np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom == 0, np.nan, num / denom)


def _rank_rows(X):
    idx   = np.argsort(X, axis=1, kind='quicksort')
    ranks = np.empty_like(X)
    ranks[np.arange(X.shape[0])[:, None], idx] = np.arange(X.shape[1], dtype=X.dtype)
    return ranks


def _scc_rows(X, Y, chunk=4000):
    K   = X.shape[0]
    out = np.empty(K, dtype=np.float32)
    for lo in range(0, K, chunk):
        hi = min(lo + chunk, K)
        out[lo:hi] = _pcc_rows(_rank_rows(X[lo:hi]), _rank_rows(Y[lo:hi]))
    return out


def compute_metrics(outputs, labels, rbp_list, compute_scc=True):
    N, T, L = outputs.shape
    outputs  = outputs.astype(np.float32)
    labels   = labels.astype(np.float32)

    # Method 1: flattened
    pred_flat  = outputs.transpose(1, 0, 2).reshape(T, N * L)
    label_flat = labels.transpose(1, 0, 2).reshape(T, N * L)
    flat_pcc_arr = _pcc_rows(pred_flat, label_flat)
    flat_scc_arr = _scc_rows(pred_flat, label_flat, chunk=1) if compute_scc \
                   else np.full(T, np.nan)

    # Method 2: per-sample
    pred_samp    = outputs.reshape(N * T, L)
    label_samp   = labels.reshape(N * T, L)
    samp_pcc_all = _pcc_rows(pred_samp, label_samp)
    samp_scc_all = _scc_rows(pred_samp, label_samp) if compute_scc \
                   else np.full(N * T, np.nan)

    samp_pcc_arr = np.nanmean(samp_pcc_all.reshape(N, T), axis=0)
    samp_scc_arr = np.nanmean(samp_scc_all.reshape(N, T), axis=0)

    flat_pcc   = {rbp: float(flat_pcc_arr[t]) for t, rbp in enumerate(rbp_list)}
    flat_scc   = {rbp: float(flat_scc_arr[t]) for t, rbp in enumerate(rbp_list)}
    sample_pcc = {rbp: float(samp_pcc_arr[t]) for t, rbp in enumerate(rbp_list)}
    sample_scc = {rbp: float(samp_scc_arr[t]) for t, rbp in enumerate(rbp_list)}

    return {
        'flat_pcc'       : flat_pcc,
        'flat_scc'       : flat_scc,
        'sample_pcc'     : sample_pcc,
        'sample_scc'     : sample_scc,
        'mean_flat_pcc'  : float(np.nanmean(flat_pcc_arr)),
        'mean_flat_scc'  : float(np.nanmean(flat_scc_arr)),
        'mean_sample_pcc': float(np.nanmean(samp_pcc_arr)),
        'mean_sample_scc': float(np.nanmean(samp_scc_arr)),
    }


def print_metrics(m, rbp_list):
    print(f'\nTest metrics:')
    print(f'  mean_flat_pcc={m["mean_flat_pcc"]:.4f}  '
          f'mean_flat_scc={m["mean_flat_scc"]:.4f}  '
          f'mean_sample_pcc={m["mean_sample_pcc"]:.4f}  '
          f'mean_sample_scc={m["mean_sample_scc"]:.4f}')
    print(f'  {"RBP":<14} {"flat_pcc":>10} {"flat_scc":>10} {"sample_pcc":>12} {"sample_scc":>12}')
    for rbp in rbp_list:
        print(f'  {rbp:<14} '
              f'{m["flat_pcc"][rbp]:>10.4f} '
              f'{m["flat_scc"][rbp]:>10.4f} '
              f'{m["sample_pcc"][rbp]:>12.4f} '
              f'{m["sample_scc"][rbp]:>12.4f}')


def main():
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))

    device = torch.device(args.device)

    # Dataset
    test_set    = MultiRBPDatasetH5(args.h5_path, mode='test')
    rbp_list    = test_set.rbp_list
    num_tasks   = len(rbp_list)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False,
                             num_workers=16, persistent_workers=True)
    print(f'Test samples: {len(test_set)}  batches: {len(test_loader)}')
    print(f'RBPs ({num_tasks}): {rbp_list}')

    # Model — strip _orig_mod. prefix from torch.compile checkpoint
    model = RBPMultiTaskModel(num_tasks=num_tasks, features=args.features)
    state = torch.load(args.model_path, map_location=device)
    if any(k.startswith('_orig_mod.') for k in state):
        print('Stripping _orig_mod. prefix from compiled-model checkpoint...')
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print('Model loaded.')

    # Inference
    all_pred  = []
    all_label = []
    with torch.no_grad():
        for batch_seq, batch_label in test_loader:
            batch_seq   = batch_seq.to(device)
            batch_label = batch_label.to(device).float()
            outputs     = model(batch_seq).float()
            all_pred.append(outputs.cpu().numpy())
            all_label.append(batch_label.cpu().numpy())

    all_pred  = np.concatenate(all_pred,  axis=0)   # (N, T, L)
    all_label = np.concatenate(all_label, axis=0)   # (N, T, L)
    print(f'Collected predictions: {all_pred.shape}')

    # Metrics (with SCC)
    print('Computing metrics (including SCC — this may take a few minutes)...')
    m = compute_metrics(all_pred, all_label, rbp_list, compute_scc=True)
    print_metrics(m, rbp_list)

    # Save CSV
    if args.output_csv is not None:
        import csv
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rbp', 'flat_pcc', 'flat_scc', 'sample_pcc', 'sample_scc'])
            for rbp in rbp_list:
                w.writerow([rbp,
                             m['flat_pcc'][rbp],
                             m['flat_scc'][rbp],
                             m['sample_pcc'][rbp],
                             m['sample_scc'][rbp]])
            w.writerow(['mean',
                        m['mean_flat_pcc'],
                        m['mean_flat_scc'],
                        m['mean_sample_pcc'],
                        m['mean_sample_scc']])
        print(f'Metrics saved to {args.output_csv}')


if __name__ == '__main__':
    main()
