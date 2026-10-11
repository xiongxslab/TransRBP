import argparse
import numpy as np
import random
import time
import torch
import os
import copy


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from training.pretrain.dataset_multitask import MultiRBPDataset
from training.pretrain.dataset_h5_multitask import MultiRBPDatasetH5
from model.pretrain.RBPMultiTaskModel import RBPMultiTaskModel

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Pretrain multi-task RBP model on 32 BigWig tracks with m6A input')
parser.add_argument('--rbp_list', type=str, required=True,
                    help='Path to text file with one RBP name per line (32 RBPs)')
parser.add_argument('--bed_root', type=str, required=True,
                    help='Directory containing per-RBP BED files ({RBP}.bed)')
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--random_seed', type=int, default=43)
parser.add_argument('--tb_dir', type=str, required=True,
                    help='TensorBoard log directory')
parser.add_argument('--whether_grad_norm_clip', type=int, default=1)
parser.add_argument('--grad_norm_clip', type=float, default=5.0)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--tol_epoch', type=int, default=10)
parser.add_argument('--save_model_dir', type=str, default=None)
parser.add_argument('--m6A_binding', type=int, default=1,
                    help='1 to include m6A signal as 5th input channel, 0 for 4-ch only')
parser.add_argument('--bw_root', type=str,
                    default=None, help='needed only without --h5_path')
parser.add_argument('--chrom_root', type=str,
                    default=None, help='needed only without --h5_path')
parser.add_argument('--total_windows', type=int, default=100000,
                    help='Total window budget across all RBPs (divided equally per RBP)')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--output_csv', type=str, default=None,
                    help='Path to save per-RBP test metrics as CSV (optional)')
parser.add_argument('--h5_path', type=str, default=None,
                    help='Path to pre-computed HDF5 file from preprocess_h5.py. '
                         'When provided, skips on-the-fly BigWig/FASTA reading.')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def check_mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def loss_fn(output, label):
    return torch.nn.MSELoss(reduction='mean')(output, label)


def _pcc_rows(X, Y):
    """Vectorized Pearson r for every row pair simultaneously.
    X, Y: (K, M) float32 → (K,) float32
    Replaces K individual scipy.stats.pearsonr calls with one numpy operation.
    """
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    num   = (X * Y).sum(axis=1)
    denom = np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom == 0, np.nan, num / denom)


def _rank_rows(X):
    """Rank each row of X using argsort (faster than scipy.rankdata in a loop).
    X: (K, M) → (K, M) ranks as float32
    """
    idx   = np.argsort(X, axis=1, kind='quicksort')
    ranks = np.empty_like(X)
    ranks[np.arange(X.shape[0])[:, None], idx] = np.arange(X.shape[1], dtype=X.dtype)
    return ranks


def _scc_rows(X, Y, chunk=4000):
    """Vectorized Spearman r, chunked over rows to keep memory bounded.
    X, Y: (K, M) → (K,) — processes `chunk` rows at a time.
    chunk=4000 rows × 800 values × 4 bytes × 2 arrays ≈ 25 MB per chunk.
    """
    K   = X.shape[0]
    out = np.empty(K, dtype=np.float32)
    for lo in range(0, K, chunk):
        hi = min(lo + chunk, K)
        out[lo:hi] = _pcc_rows(
            _rank_rows(X[lo:hi]),
            _rank_rows(Y[lo:hi]),
        )
    return out


def compute_metrics(outputs, labels, rbp_list, compute_scc=True):
    """
    Vectorized PCC and SCC for all tasks at once — no per-sample Python loops.

    outputs : (N, T, L) numpy float32
    labels  : (N, T, L) numpy float32
    compute_scc : set False on training split to skip the expensive ranking step
                  (SCC on 70k×32 samples is slow; PCC alone is fast and sufficient
                  for monitoring training progress)

    Method 1 — flat:   one corr over all N×L values per task
    Method 2 — sample: per-sample corr on L values, averaged over N
    """
    N, T, L = outputs.shape
    outputs = outputs.astype(np.float32)
    labels  = labels.astype(np.float32)

    # ---- Method 1: flattened (T, N*L) ----
    pred_flat  = outputs.transpose(1, 0, 2).reshape(T, N * L)   # (T, N*L)
    label_flat = labels.transpose(1, 0, 2).reshape(T, N * L)    # (T, N*L)

    flat_pcc_arr = _pcc_rows(pred_flat, label_flat)              # (T,) — fast
    if compute_scc:
        # Rank N*L values per task; process one task at a time (chunk=1 row)
        # to avoid holding the full (T, N*L) ranked array in memory
        flat_scc_arr = _scc_rows(pred_flat, label_flat, chunk=1)  # (T,)
    else:
        flat_scc_arr = np.full(T, np.nan)

    # ---- Method 2: per-sample (N*T, L) ----
    pred_samp  = outputs.reshape(N * T, L)                       # (N*T, L)
    label_samp = labels.reshape(N * T, L)                        # (N*T, L)

    samp_pcc_all = _pcc_rows(pred_samp, label_samp)              # (N*T,) — fast
    if compute_scc:
        samp_scc_all = _scc_rows(pred_samp, label_samp)          # (N*T,) — chunked
    else:
        samp_scc_all = np.full(N * T, np.nan)

    samp_pcc_arr = samp_pcc_all.reshape(N, T).mean(axis=0)       # (T,)
    samp_scc_arr = samp_scc_all.reshape(N, T).mean(axis=0)       # (T,)

    # Pack into dicts
    flat_pcc   = {rbp: float(flat_pcc_arr[t])  for t, rbp in enumerate(rbp_list)}
    flat_scc   = {rbp: float(flat_scc_arr[t])  for t, rbp in enumerate(rbp_list)}
    sample_pcc = {rbp: float(samp_pcc_arr[t])  for t, rbp in enumerate(rbp_list)}
    sample_scc = {rbp: float(samp_scc_arr[t])  for t, rbp in enumerate(rbp_list)}

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


def log_metrics(writer, m, rbp_list, split, epoch):
    """Write all metrics to TensorBoard."""
    writer.add_scalar(f'{split}/MeanFlatPCC',   m['mean_flat_pcc'],   epoch)
    writer.add_scalar(f'{split}/MeanFlatSCC',   m['mean_flat_scc'],   epoch)
    writer.add_scalar(f'{split}/MeanSamplePCC', m['mean_sample_pcc'], epoch)
    writer.add_scalar(f'{split}/MeanSampleSCC', m['mean_sample_scc'], epoch)
    for rbp in rbp_list:
        writer.add_scalar(f'{split}/FlatPCC/{rbp}',   m['flat_pcc'][rbp],   epoch)
        writer.add_scalar(f'{split}/FlatSCC/{rbp}',   m['flat_scc'][rbp],   epoch)
        writer.add_scalar(f'{split}/SamplePCC/{rbp}', m['sample_pcc'][rbp], epoch)
        writer.add_scalar(f'{split}/SampleSCC/{rbp}', m['sample_scc'][rbp], epoch)


def print_metrics(m, rbp_list, split):
    """Print a summary table of all metrics."""
    print(f'\n{split} metrics:')
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


def get_max(lst):
    val, idx = max((v, i) for i, v in enumerate(lst))
    return val, idx


def main():
    setup_seed(args.random_seed)

    for arg in vars(args):
        print(arg, ':', getattr(args, arg), flush=True)

    if args.h5_path:
        # Fast path: load from pre-computed HDF5 (no BigWig/FASTA I/O at runtime)
        train_set = MultiRBPDatasetH5(args.h5_path, mode='train')
        val_set   = MultiRBPDatasetH5(args.h5_path, mode='val')
        test_set  = MultiRBPDatasetH5(args.h5_path, mode='test')
        rbp_list  = train_set.rbp_list   # read from HDF5 metadata
        features  = 5   # stored in HDF5; assume m6A always included when using h5
    else:
        # Slow path: read from BigWig/FASTA on-the-fly
        with open(args.rbp_list) as f:
            rbp_list = [line.strip() for line in f if line.strip()]
        features = 5 if args.m6A_binding else 4
        dataset_kwargs = dict(bw_root=args.bw_root, chrom_root=args.chrom_root,
                              total_num=args.total_windows, m6A_binding=args.m6A_binding)
        train_set = MultiRBPDataset(rbp_list, args.bed_root, mode='train', **dataset_kwargs)
        shared_windows = train_set._all_windows
        val_set   = MultiRBPDataset(rbp_list, args.bed_root, mode='val',
                                    precomputed_windows=shared_windows, **dataset_kwargs)
        test_set  = MultiRBPDataset(rbp_list, args.bed_root, mode='test',
                                    precomputed_windows=shared_windows, **dataset_kwargs)

    num_tasks = len(rbp_list)
    print(f'Pretraining on {num_tasks} RBPs, input channels={features}')

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                              num_workers=64, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=args.bs, shuffle=False,
                              num_workers=32, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=args.bs, shuffle=False,
                              num_workers=16, persistent_workers=True)

    print(f'Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}')

    # Model
    model  = RBPMultiTaskModel(num_tasks=num_tasks, features=features)
    device = torch.device(args.device)
    model.to(device)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    check_mkdir(args.tb_dir)
    writer = SummaryWriter(args.tb_dir)

    # Early stopping tracks mean_flat_pcc on validation set
    val_pcc_record = []
    best_model_wts = None
    tick_start     = time.time()

    for epoch in range(10000):
        print(f'\n------- epoch {epoch} -------')
        tick_epoch       = time.time()
        train_loadersize = len(train_loader)

        # ---- Train ----
        model.train()
        train_loss      = 0.0
        all_train_pred  = []
        all_train_label = []

        for i, (batch_seq, batch_label) in enumerate(train_loader):
            batch_seq   = batch_seq.to(device)
            batch_label = batch_label.to(device).float()

            outputs = model(batch_seq).float()
            loss    = loss_fn(outputs, batch_label)

            optimizer.zero_grad()
            loss.backward()
            if args.whether_grad_norm_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()

            train_loss += loss.item() * len(batch_label)

            if i % max(1, train_loadersize // 50) == 0:
                print(f'  batch {i}/{train_loadersize}  loss={loss.item():.5f}', flush=True)

            all_train_pred.append(outputs.detach().cpu().numpy())
            all_train_label.append(batch_label.cpu().numpy())

        all_train_pred  = np.concatenate(all_train_pred,  axis=0)
        all_train_label = np.concatenate(all_train_label, axis=0)

        train_loss_avg = train_loss / (train_loadersize * args.bs)
        train_m        = compute_metrics(all_train_pred, all_train_label, rbp_list,
                                          compute_scc=False)  # skip SCC on 70k train samples

        writer.add_scalar('Train/Loss', train_loss_avg, epoch)
        log_metrics(writer, train_m, rbp_list, 'Train', epoch)
        print_metrics(train_m, rbp_list, 'Train')

        # ---- Validation ----
        model.eval()
        val_loss       = 0.0
        all_val_pred   = []
        all_val_label  = []

        with torch.no_grad():
            for batch_seq, batch_label in val_loader:
                batch_seq   = batch_seq.to(device)
                batch_label = batch_label.to(device).float()
                outputs     = model(batch_seq).float()
                val_loss   += loss_fn(outputs, batch_label).item() * len(batch_label)
                all_val_pred.append(outputs.cpu().numpy())
                all_val_label.append(batch_label.cpu().numpy())

        all_val_pred  = np.concatenate(all_val_pred,  axis=0)
        all_val_label = np.concatenate(all_val_label, axis=0)

        val_loss_avg = val_loss / (len(val_loader) * args.bs)
        val_m        = compute_metrics(all_val_pred, all_val_label, rbp_list, compute_scc = False)

        writer.add_scalar('Val/Loss', val_loss_avg, epoch)
        log_metrics(writer, val_m, rbp_list, 'Val', epoch)
        print_metrics(val_m, rbp_list, 'Val')

        # Early stopping based on mean_flat_pcc
        mean_val_pcc = val_m['mean_flat_pcc']
        val_pcc_record.append(mean_val_pcc)

        if args.save_model_dir is not None:
            max_, _ = get_max(val_pcc_record)
            if val_pcc_record[-1] == max_:
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        print(f'Epoch time: {time.time() - tick_epoch:.1f}s', flush=True)

        # Early stop
        if len(val_pcc_record) >= args.tol_epoch:
            max_, max_idx = get_max(val_pcc_record)
            if max_idx < len(val_pcc_record) - args.tol_epoch:
                print(f'Early stop — best mean_flat_pcc={max_:.4f} at epoch {max_idx}')
                break
            else:
                print(f'Best mean_flat_pcc so far={max_:.4f} at epoch {max_idx}')
        if len(val_pcc_record) >= args.max_epoch:
            print('Reached max epoch, stopping.')
            break

    print(f'\nTraining finished after {len(val_pcc_record)} epochs.')
    max_, best_idx = get_max(val_pcc_record)
    print(f'Best mean_flat_pcc={max_:.4f} at epoch {best_idx}')

    # ---- Test ----
    test_model = RBPMultiTaskModel(num_tasks=num_tasks, features=features)
    test_model = torch.compile(test_model)
    test_model.load_state_dict(best_model_wts)
    test_model.eval()
    test_model.to(device)

    test_loss      = 0.0
    all_test_pred  = []
    all_test_label = []

    with torch.no_grad():
        for batch_seq, batch_label in test_loader:
            batch_seq   = batch_seq.to(device)
            batch_label = batch_label.to(device).float()
            outputs     = test_model(batch_seq).float()
            test_loss  += loss_fn(outputs, batch_label).item() * len(batch_label)
            all_test_pred.append(outputs.cpu().numpy())
            all_test_label.append(batch_label.cpu().numpy())

    all_test_pred  = np.concatenate(all_test_pred,  axis=0)
    all_test_label = np.concatenate(all_test_label, axis=0)

    test_loss_avg = test_loss / (len(test_loader) * args.bs)
    test_m        = compute_metrics(all_test_pred, all_test_label, rbp_list)

    print(f'\nTest loss={test_loss_avg:.5f}')
    print_metrics(test_m, rbp_list, 'Test')

    if args.output_csv is not None:
        import csv
        check_mkdir(os.path.dirname(args.output_csv))
        with open(args.output_csv, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['rbp', 'flat_pcc', 'flat_scc', 'sample_pcc', 'sample_scc'])
            for rbp in rbp_list:
                writer_csv.writerow([rbp,
                                     test_m['flat_pcc'][rbp],
                                     test_m['flat_scc'][rbp],
                                     test_m['sample_pcc'][rbp],
                                     test_m['sample_scc'][rbp]])
            writer_csv.writerow(['mean',
                                 test_m['mean_flat_pcc'],
                                 test_m['mean_flat_scc'],
                                 test_m['mean_sample_pcc'],
                                 test_m['mean_sample_scc']])
        print(f'Test metrics saved to {args.output_csv}')

    if args.save_model_dir is not None:
        check_mkdir(args.save_model_dir)
        save_path = os.path.join(args.save_model_dir, 'pretrained_32rbp_m6a.pth')
        torch.save(best_model_wts, save_path)
        print(f'Model saved to {save_path}')

    print(f'Total time: {time.time() - tick_start:.1f}s')


if __name__ == '__main__':
    main()
