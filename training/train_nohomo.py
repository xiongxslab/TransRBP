"""
Train TransRBP on no-homology HDF5 data (GLORI m6A channel).

Data must first be generated with make_h5_nohomo.py.
All data is pre-loaded into RAM (fast GPU transfer, num_workers=0).

Example:
    python training/train_nohomo.py \\
        --rbp FMR1 \\
        --h5_path  ./h5_cache/FMR1.h5 \\
        --save_model_dir ./models \\
        --output_csv results.csv \\
        --device cuda:0
"""

import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader

from model.RBPResTransModels import RBPModel
from data.dataset_h5 import RBPDataset

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Train TransRBP on nohomo HDF5')
parser.add_argument('--rbp',             required=True,  help='RBP name')
parser.add_argument('--h5_path',         required=True,  help='Path to {RBP}.h5 from make_h5_nohomo.py')
parser.add_argument('--save_model_dir',  default=None,   help='Dir to save best .pth')
parser.add_argument('--output_csv',      default=None,   help='CSV to append test metrics')
parser.add_argument('--device',          default='cuda:0')
parser.add_argument('--bs',              type=int,   default=64)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--random_seed',     type=int,   default=43)
parser.add_argument('--max_epoch',       type=int,   default=100)
parser.add_argument('--tol_epoch',       type=int,   default=10)
parser.add_argument('--grad_norm_clip',  type=float, default=5.0)
parser.add_argument('--m6A_binding',     type=int,   default=1,
                    help='Must match what was used in make_h5_nohomo.py')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_pcc(pred, label):
    return scipy.stats.pearsonr(np.array(pred), np.array(label))


def compute_scc(pred, label):
    return scipy.stats.spearmanr(np.array(pred), np.array(label))


def get_max(lst):
    return max((v, i) for i, v in enumerate(lst))


def train():
    setup_seed(args.random_seed)

    train_loader = DataLoader(
        RBPDataset(args.rbp, 'train', args.h5_path),
        batch_size=args.bs, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(
        RBPDataset(args.rbp, 'val',   args.h5_path),
        batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(
        RBPDataset(args.rbp, 'test',  args.h5_path),
        batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True)

    features = 5 if args.m6A_binding else 4
    device   = torch.device(args.device)
    model    = RBPModel(features=features, record_attn=False)
    model.to(device)
    model    = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch)
    loss_fn   = torch.nn.MSELoss()

    val_pcc_record = []
    best_model_wts = None
    tick0 = time.time()

    for epoch in range(10000):
        print(f'------- epoch: {epoch} -------')
        model.train()
        train_pred, train_label = [], []
        for i, (x, y) in enumerate(train_loader):
            x   = x.to(device)
            y   = y.to(device).float()
            out = model(x).float()
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()
            train_pred.extend(torch.flatten(out).detach().cpu().tolist())
            train_label.extend(torch.flatten(y).cpu().tolist())
            if i % max(1, len(train_loader) // 20) == 0:
                print(f'  step {i}/{len(train_loader)}  loss: {loss.item():.4f}',
                      flush=True)

        pcc, _ = compute_pcc(train_pred, train_label)
        scc, _ = compute_scc(train_pred, train_label)
        print(f'Train  PCC={pcc:.4f}  SCC={scc:.4f}')

        model.eval()
        val_pred, val_label = [], []
        for x, y in val_loader:
            with torch.no_grad():
                x   = x.to(device)
                y   = y.to(device).float()
                out = model(x).float()
                val_pred.extend(torch.flatten(out).cpu().tolist())
                val_label.extend(torch.flatten(y).cpu().tolist())

        val_pcc, _ = compute_pcc(val_pred, val_label)
        val_scc, _ = compute_scc(val_pred, val_label)
        print(f'Val    PCC={val_pcc:.4f}  SCC={val_scc:.4f}')
        val_pcc_record.append(val_pcc)
        scheduler.step()

        if args.save_model_dir:
            os.makedirs(args.save_model_dir, exist_ok=True)
            best_pcc, _ = get_max(val_pcc_record)
            if val_pcc_record[-1] == best_pcc:
                best_model_wts = copy.deepcopy(model.state_dict())

        best_pcc, best_idx = get_max(val_pcc_record)
        n = len(val_pcc_record)
        if n >= args.max_epoch:
            print(f'Reached max_epoch. Best PCC={best_pcc:.4f} @ {best_idx}')
            break
        if n >= args.tol_epoch and best_idx < n - args.tol_epoch:
            print(f'Early stop. Best PCC={best_pcc:.4f} @ {best_idx}')
            break

    print(f'Training done in {time.time()-tick0:.1f}s')

    # Test
    test_model = RBPModel(features=features, record_attn=False)
    test_model.load_state_dict(best_model_wts)
    test_model.to(device)
    test_model.eval()
    test_model = torch.compile(test_model)

    test_pred, test_label = [], []
    for x, y in test_loader:
        with torch.no_grad():
            x   = x.to(device)
            y   = y.to(device).float()
            out = test_model(x).float()
            test_pred.extend(torch.flatten(out).cpu().tolist())
            test_label.extend(torch.flatten(y).cpu().tolist())

    test_pcc, _ = compute_pcc(test_pred, test_label)
    test_scc, _ = compute_scc(test_pred, test_label)
    print(f'Test   PCC={test_pcc:.4f}  SCC={test_scc:.4f}')

    if args.save_model_dir:
        out_path = os.path.join(args.save_model_dir, f'{args.rbp}.pth')
        torch.save(best_model_wts, out_path)
        print(f'Model saved → {out_path}')

    if args.output_csv:
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['RBP', 'best_val_PCC', 'best_val_epoch',
                             'test_PCC', 'test_SCC'])
            best_pcc, best_idx = get_max(val_pcc_record)
            w.writerow([args.rbp, f'{best_pcc:.4f}', best_idx,
                        f'{test_pcc:.4f}', f'{test_scc:.4f}'])


if __name__ == '__main__':
    train()
