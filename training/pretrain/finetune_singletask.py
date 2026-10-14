"""
Fine-tune the pretrained 32-RBP model for a single RBP.

Loads pretrained_32rbp_m6a.pth, replaces the 128->32 output conv with 128->1,
and trains on the single-RBP HDF5 prepared by preprocess_finetune_h5.py.

Freeze modes (--freeze_mode):
  head_only    (default) : only task_head[3:] (new Conv1d 128->1 + BN) are trained
  decoder_head           : also unfreeze decoder_res
  full                   : all parameters trainable

Usage:
    python finetune_singletask.py \\
        --pretrain_path models_pretrain/pretrained_32rbp_m6a.pth \\
        --rbp           RBM15 \\
        --h5_path       h5_cache/finetune/RBM15_20k.h5 \\
        --freeze_mode   head_only \\
        --device        cuda:1
"""

import argparse
import copy
import os
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.pretrain.RBPMultiTaskModel import RBPMultiTaskModel


class SingleRBPH5Dataset(Dataset):
    """Reads a single-RBP HDF5 file (no rbp_list attribute required)."""

    def __init__(self, h5_path, mode='train'):
        self.h5_path = h5_path
        self.mode    = mode
        self._h5     = None
        with h5py.File(h5_path, 'r') as f:
            self._len = f[f'{mode}/inputs'].shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        seq   = torch.from_numpy(self._h5[f'{self.mode}/inputs'][idx].astype(np.float32))
        label = torch.from_numpy(self._h5[f'{self.mode}/labels'][idx].astype(np.float32))
        return seq, label

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', required=True,
                    help='Path to pretrained_32rbp_m6a.pth')
parser.add_argument('--rbp', required=True,
                    help='RBP name (used for logging and saving)')
parser.add_argument('--h5_path', required=True,
                    help='HDF5 file for this RBP from preprocess_finetune_h5.py')
parser.add_argument('--freeze_mode', default='head_only',
                    choices=['head_only', 'decoder_head', 'full'],
                    help='Which layers to freeze (see module docstring)')
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--random_seed', type=int, default=43)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--tol_epoch', type=int, default=10)
parser.add_argument('--whether_grad_norm_clip', type=int, default=1)
parser.add_argument('--grad_norm_clip', type=float, default=5.0)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save_model_dir', type=str, default=None)
parser.add_argument('--tb_dir', type=str, default=None)
parser.add_argument('--output_csv', type=str, default=None)
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def check_mkdir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)


# ------------------------------------------------------------------
# Vectorized metrics (with nanmean fix for per-sample aggregation)
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


def compute_metrics(outputs, labels, compute_scc=False):
    """outputs, labels: (N, 1, L) numpy float32. Returns scalar dict."""
    N, T, L = outputs.shape   # T=1 for single-task
    outputs  = outputs.astype(np.float32)
    labels   = labels.astype(np.float32)

    pred_flat  = outputs.transpose(1, 0, 2).reshape(T, N * L)
    label_flat = labels.transpose(1, 0, 2).reshape(T, N * L)
    flat_pcc   = float(_pcc_rows(pred_flat, label_flat)[0])
    flat_scc   = float(_scc_rows(pred_flat, label_flat, chunk=1)[0]) if compute_scc \
                 else float('nan')

    pred_samp    = outputs.reshape(N * T, L)
    label_samp   = labels.reshape(N * T, L)
    samp_pcc_all = _pcc_rows(pred_samp, label_samp)
    samp_scc_all = _scc_rows(pred_samp, label_samp) if compute_scc \
                   else np.full(N * T, np.nan)

    sample_pcc = float(np.nanmean(samp_pcc_all))
    sample_scc = float(np.nanmean(samp_scc_all))

    return {'flat_pcc': flat_pcc, 'flat_scc': flat_scc,
            'sample_pcc': sample_pcc, 'sample_scc': sample_scc}


def print_metrics(m, split):
    print(f'  [{split}] flat_pcc={m["flat_pcc"]:.4f}  flat_scc={m["flat_scc"]:.4f}  '
          f'sample_pcc={m["sample_pcc"]:.4f}  sample_scc={m["sample_scc"]:.4f}')


def get_max(lst):
    """Best value and its epoch index. NaN (e.g. from constant predictions or labels) counts as the lowest
    score, so it never blocks later epochs; among equal values the latest epoch wins."""
    val, idx = max((v if np.isfinite(v) else -np.inf, i) for i, v in enumerate(lst))
    return val, idx


# ------------------------------------------------------------------
# Model building
# ------------------------------------------------------------------

def build_model(pretrain_path, freeze_mode, device):
    state = torch.load(pretrain_path, map_location='cpu')
    if any(k.startswith('_orig_mod.') for k in state):
        print('Stripping _orig_mod. prefix from compiled-model checkpoint...')
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    num_tasks = state['task_head.3.weight'].shape[0]          # 32 for the paper's pretraining
    backbone = RBPMultiTaskModel(num_tasks=num_tasks, features=5)
    backbone.load_state_dict(state)
    print(f'Loaded pretrained weights from {pretrain_path}')

    # Replace output conv: 128->32 becomes 128->1
    backbone.task_head[3] = nn.Conv1d(128, 1, kernel_size=1)
    backbone.task_head[4] = nn.BatchNorm1d(1)
    print('Replaced task_head[3:5] with Conv1d(128->1) + BN(1)')

    # Apply freezing
    if freeze_mode == 'head_only':
        # Freeze everything except task_head[3:] (new output layer)
        for name, param in backbone.named_parameters():
            param.requires_grad = name.startswith('task_head.3') or \
                                   name.startswith('task_head.4') or \
                                   name.startswith('task_head.5')
        trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
        print(f'freeze_mode=head_only: training only {trainable}')

    elif freeze_mode == 'decoder_head':
        # Freeze pre_conv, encoder_res, attn; unfreeze decoder_res + task_head
        for name, param in backbone.named_parameters():
            param.requires_grad = (name.startswith('decoder_res') or
                                   name.startswith('task_head'))
        n_train = sum(p.requires_grad for _, p in backbone.named_parameters())
        print(f'freeze_mode=decoder_head: {n_train} trainable param tensors')

    else:  # full
        for param in backbone.parameters():
            param.requires_grad = True
        print('freeze_mode=full: all parameters trainable')

    backbone.to(device)
    return backbone


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    setup_seed(args.random_seed)
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))

    device = torch.device(args.device)

    # Datasets
    train_set = SingleRBPH5Dataset(args.h5_path, mode='train')
    val_set   = SingleRBPH5Dataset(args.h5_path, mode='val')
    test_set  = SingleRBPH5Dataset(args.h5_path, mode='test')

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                              num_workers=8, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=args.bs, shuffle=False,
                              num_workers=4, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=args.bs, shuffle=False,
                              num_workers=4, persistent_workers=True)

    print(f'\nRBP: {args.rbp}')
    print(f'Train: {len(train_set)} samples ({len(train_loader)} batches)')
    print(f'Val:   {len(val_set)} samples ({len(val_loader)} batches)')
    print(f'Test:  {len(test_set)} samples ({len(test_loader)} batches)')

    model = build_model(args.pretrain_path, args.freeze_mode, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    loss_fn   = torch.nn.MSELoss(reduction='mean')

    if args.tb_dir:
        check_mkdir(args.tb_dir)
        writer = SummaryWriter(args.tb_dir)
    else:
        writer = None

    val_pcc_record = []
    best_model_wts = None
    tick_start     = time.time()

    for epoch in range(10000):
        print(f'\n------- epoch {epoch} [{args.rbp}] -------')
        tick_epoch = time.time()

        # ---- Train ----
        model.train()
        all_train_pred  = []
        all_train_label = []
        train_loss      = 0.0

        for i, (batch_seq, batch_label) in enumerate(train_loader):
            batch_seq   = batch_seq.to(device)
            batch_label = batch_label.to(device).float()
            outputs     = model(batch_seq).float()
            loss        = loss_fn(outputs, batch_label)

            optimizer.zero_grad()
            loss.backward()
            if args.whether_grad_norm_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()

            train_loss += loss.item() * len(batch_label)

            if i % max(1, len(train_loader) // 20) == 0:
                print(f'  batch {i}/{len(train_loader)}  loss={loss.item():.5f}', flush=True)

            all_train_pred.append(outputs.detach().cpu().numpy())
            all_train_label.append(batch_label.cpu().numpy())

        all_train_pred  = np.concatenate(all_train_pred,  axis=0)
        all_train_label = np.concatenate(all_train_label, axis=0)
        train_m = compute_metrics(all_train_pred, all_train_label, compute_scc=False)

        if writer:
            writer.add_scalar('Train/Loss', train_loss / len(train_set), epoch)
            writer.add_scalar('Train/FlatPCC',   train_m['flat_pcc'],   epoch)
            writer.add_scalar('Train/SamplePCC', train_m['sample_pcc'], epoch)
        print_metrics(train_m, 'Train')

        # ---- Validation ----
        model.eval()
        all_val_pred  = []
        all_val_label = []

        with torch.no_grad():
            for batch_seq, batch_label in val_loader:
                batch_seq   = batch_seq.to(device)
                batch_label = batch_label.to(device).float()
                outputs     = model(batch_seq).float()
                all_val_pred.append(outputs.cpu().numpy())
                all_val_label.append(batch_label.cpu().numpy())

        all_val_pred  = np.concatenate(all_val_pred,  axis=0)
        all_val_label = np.concatenate(all_val_label, axis=0)
        val_m = compute_metrics(all_val_pred, all_val_label, compute_scc=False)

        if writer:
            writer.add_scalar('Val/FlatPCC',   val_m['flat_pcc'],   epoch)
            writer.add_scalar('Val/SamplePCC', val_m['sample_pcc'], epoch)
        print_metrics(val_m, 'Val')

        val_pcc_record.append(val_m['flat_pcc'])

        # keep the best weights in memory; they are written to disk only with --save_model_dir
        max_, max_idx = get_max(val_pcc_record)
        if max_idx == len(val_pcc_record) - 1:
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        print(f'Epoch time: {time.time() - tick_epoch:.1f}s', flush=True)

        # Early stopping
        if len(val_pcc_record) >= args.tol_epoch:
            max_, max_idx = get_max(val_pcc_record)
            if max_idx < len(val_pcc_record) - args.tol_epoch:
                print(f'Early stop — best flat_pcc={max_:.4f} at epoch {max_idx}')
                break
            else:
                print(f'Best flat_pcc so far={max_:.4f} at epoch {max_idx}')
        if len(val_pcc_record) >= args.max_epoch:
            print('Reached max epoch, stopping.')
            break

    print(f'\nTraining finished after {len(val_pcc_record)} epochs.')
    max_, best_idx = get_max(val_pcc_record)
    print(f'Best val flat_pcc={max_:.4f} at epoch {best_idx}')

    # ---- Test (load best weights, compute SCC) ----
    test_model = RBPMultiTaskModel(num_tasks=32, features=5)
    test_model.task_head[3] = nn.Conv1d(128, 1, kernel_size=1)
    test_model.task_head[4] = nn.BatchNorm1d(1)
    test_model.load_state_dict(best_model_wts)
    test_model.eval()
    test_model.to(device)

    all_test_pred  = []
    all_test_label = []
    with torch.no_grad():
        for batch_seq, batch_label in test_loader:
            batch_seq   = batch_seq.to(device)
            batch_label = batch_label.to(device).float()
            outputs     = test_model(batch_seq).float()
            all_test_pred.append(outputs.cpu().numpy())
            all_test_label.append(batch_label.cpu().numpy())

    all_test_pred  = np.concatenate(all_test_pred,  axis=0)
    all_test_label = np.concatenate(all_test_label, axis=0)
    test_m = compute_metrics(all_test_pred, all_test_label, compute_scc=True)
    print_metrics(test_m, 'Test')

    # Save CSV
    if args.output_csv is not None:
        import csv
        check_mkdir(os.path.dirname(args.output_csv))
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rbp', 'flat_pcc', 'flat_scc', 'sample_pcc', 'sample_scc'])
            w.writerow([args.rbp,
                        test_m['flat_pcc'], test_m['flat_scc'],
                        test_m['sample_pcc'], test_m['sample_scc']])
        print(f'Test metrics saved to {args.output_csv}')

    # Save model
    if args.save_model_dir is not None:
        check_mkdir(args.save_model_dir)
        save_path = os.path.join(args.save_model_dir, f'finetune_{args.rbp}.pth')
        torch.save(best_model_wts, save_path)
        print(f'Model saved to {save_path}')

    print(f'Total time: {time.time() - tick_start:.1f}s')


if __name__ == '__main__':
    main()
