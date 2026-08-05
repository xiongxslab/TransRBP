"""
Train a single-RBP TransRBP model (5-channel m6A or 4-channel seq-only).

Chromosome split: val = chr2/3/4, test = chr1/8/9, train = rest.
Early stopping on validation PCC. Best checkpoint saved to --save_model_dir/{RBP}.pth.

Example (m6A-binding model):
    python training/main.py \\
        --rbp FMR1 \\
        --peak_bed  data/FMR1.bed \\
        --bw_root   /path/to/bamCompare_1_bw \\
        --chrom_root /path/to/hg38/dna_sequence \\
        --m6A_bw_plus  /path/to/m6A_plus.bw \\
        --m6A_bw_minus /path/to/m6A_minus.bw \\
        --m6A_binding 1 \\
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
from data.dataset import RBPDataset

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Train TransRBP (m6A-aware)')
parser.add_argument('--rbp',           required=True,  help='RBP name')
parser.add_argument('--peak_bed',      required=True,  help='BED6 file of RBP peaks')
parser.add_argument('--bw_root',       required=True,  help='Dir with {RBP}_plus/minus.bw')
parser.add_argument('--chrom_root',    required=True,  help='Dir with {chrom}.fa.gz files')
parser.add_argument('--m6A_bw_plus',   default=None,   help='m6A BigWig plus strand')
parser.add_argument('--m6A_bw_minus',  default=None,   help='m6A BigWig minus strand')
parser.add_argument('--m6A_binding',   type=int, default=1,
                    help='1 = 5-channel (seq + m6A), 0 = 4-channel seq only')
parser.add_argument('--save_model_dir', default=None,  help='Dir to save best .pth')
parser.add_argument('--output_csv',    default=None,   help='CSV to append test metrics')
parser.add_argument('--device',        default='cuda:0')
parser.add_argument('--bs',            type=int,   default=64)
parser.add_argument('--num_workers',   type=int,   default=0,
                    help='DataLoader workers (default 0; set >0 on Linux for speed)')
parser.add_argument('--lr',            type=float, default=1e-5)
parser.add_argument('--random_seed',   type=int,   default=43)
parser.add_argument('--max_epoch',     type=int,   default=100)
parser.add_argument('--tol_epoch',     type=int,   default=10)
parser.add_argument('--grad_norm_clip', type=float, default=5.0)
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_pcc(pred, label):
    pcc, pval = scipy.stats.pearsonr(np.array(pred), np.array(label))
    return pcc, pval


def get_max(lst):
    return max((v, i) for i, v in enumerate(lst))


def loss_fn(output, label):
    return torch.nn.MSELoss()(output, label)


def train():
    setup_seed(args.random_seed)

    def make_loader(mode, shuffle):
        ds = RBPDataset(
            RBP_name=args.rbp,
            peak_bed=args.peak_bed,
            bw_root=args.bw_root,
            chrom_root=args.chrom_root,
            m6A_bw_plus=args.m6A_bw_plus,
            m6A_bw_minus=args.m6A_bw_minus,
            mode=mode,
            m6A_binding=args.m6A_binding,
        )
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader('train', True)
    val_loader   = make_loader('val',   False)
    test_loader  = make_loader('test',  False)

    features = 5 if args.m6A_binding else 4
    device   = torch.device(args.device)
    model    = RBPModel(features=features, record_attn=False)
    model.to(device)
    model    = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch)

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

        train_pcc, _ = compute_pcc(train_pred, train_label)
        print(f'Train  PCC={train_pcc:.4f}')

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
        print(f'Val    PCC={val_pcc:.4f}')
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
    test_model = RBPModel(features=5 if args.m6A_binding else 4, record_attn=False)
    test_model.load_state_dict(best_model_wts)
    test_model.to(device)
    test_model.eval()
    test_model = torch.compile(test_model)

    test_pred, test_label, sample_pccs = [], [], []
    for x, y in test_loader:
        with torch.no_grad():
            x   = x.to(device)
            y   = y.to(device).float()
            out = test_model(x).float()
        out_np = out.cpu().numpy()
        y_np   = y.cpu().numpy()
        test_pred.extend(out_np.flatten().tolist())
        test_label.extend(y_np.flatten().tolist())
        for j in range(out_np.shape[0]):
            pcc, _ = compute_pcc(out_np[j].flatten(), y_np[j].flatten())
            sample_pccs.append(pcc)

    flat_pcc, _ = compute_pcc(test_pred, test_label)
    mean_pcc    = float(np.nanmean(sample_pccs))
    print(f'Test   PCC={flat_pcc:.4f}  MeanPCC={mean_pcc:.4f}')

    if args.save_model_dir:
        out_path = os.path.join(args.save_model_dir, f'{args.rbp}.pth')
        torch.save(best_model_wts, out_path)
        print(f'Model saved → {out_path}')

    if args.output_csv:
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['RBP', 'm6A_binding', 'Mean_PCC', 'Concat_PCC'])
            w.writerow([args.rbp, args.m6A_binding, mean_pcc, flat_pcc])


if __name__ == '__main__':
    train()
