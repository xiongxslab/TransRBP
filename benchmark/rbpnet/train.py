"""
Train and evaluate the RBPNet baseline (Fig. 1d-e benchmark) on the
homology-filtered chromosome split.

Pipeline:
    1. python benchmark/rbpnet/export_fasta.py --rbp rbp_list.txt ... --out_dir rbpnet_fasta
    2. sh benchmark/rbpnet/homology.sh rbp_list.txt rbpnet_fasta
    3. python benchmark/rbpnet/train.py --rbp rbp_list.txt ... --homology_id rbpnet_fasta

Example:
    python benchmark/rbpnet/train.py \\
        --rbp rbp_list.txt \\
        --peak_root /path/to/RBP_bed \\
        --bw_root   /path/to/bamCompare_bw \\
        --chrom_root hg38/dna_sequence \\
        --homology_id rbpnet_fasta \\
        --tb_dir ./rbpnet_tb --save_model_dir ./rbpnet_models
"""

import argparse
import numpy as np
import random
import time
import torch
import os
import scipy.stats
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from benchmark.rbpnet import dataset as dataset
from benchmark.rbpnet import models


torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='train the model')
parser.add_argument('--peak_root', type = str, required = True,
                    help = 'directory containing the input RBP peak BED files ({RBP}.bed)')
parser.add_argument('--bw_root', type = str, required = True,
                    help = 'directory containing {RBP}_plus.bw / {RBP}_minus.bw')
parser.add_argument('--chrom_root', type = str, required = True,
                    help = 'directory containing {chrom}.fa.gz per-chromosome FASTA files')
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--num_workers', type = int, default = 8)
parser.add_argument('--bs', type = int, default = 64,
                    help = 'the batch size of training and validation, int, such as 128, default is 640')
parser.add_argument('--lr', type = float, default = 1e-5,
                    help = 'learning rate, default is 1e-5')
parser.add_argument('--random_seed', type = int, default = 43,
                    help = 'the random seed, int, such as 43')
parser.add_argument('--tb_dir', type = str,
                    help = 'tensorboard directory')
parser.add_argument('--save_model_dir', type = str, default=None,
                    help = 'the directory to save model, will create directory with RBP in name.')
parser.add_argument('--homology_id', type = str, default=None,
                    help = 'the directory to the homology ID file used to remove test sequences homologous to the training set.')
parser.add_argument('--rbp', type = str, required = True,
                    help = 'text file with one RBP name per line')

# parameters about early stop
parser.add_argument('--max_epoch', type = int, default = 100,
                    help = 'the max epoch for training, int, such as 200, default is 200')
parser.add_argument('--tol_epoch', type = int, default = 10,
                    help = 'the tolerance epoch for early stop, int, such as 20, default is 20')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


def check_mkdir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

def loss_fn(output, label):
    true_counts = torch.squeeze(label,1).int()
    logits = torch.squeeze(output,2)
    total_counts = torch.sum(true_counts, dim=-1)
    
    loss_sum = 0
    for i in range(len(logits)):
        dist = torch.distributions.Multinomial(total_count=int(total_counts[i]), logits=logits[i])
        loss = -torch.mean(dist.log_prob((true_counts[i])))
        loss_sum = loss_sum + loss
    return loss_sum / len(logits)

def compute_pcc(pred_:list, label_:list):
    pcc, p_value = scipy.stats.pearsonr(np.array(pred_), np.array(label_))
    return pcc, p_value

def get_min(list_):
    loss, idx = min((loss, idx) for (idx, loss) in enumerate(list_))
    return loss, idx

def train_val_test(rbp_name):
    setup_seed(args.random_seed)
    
    tick1 = time.time()
    print('\n')
    print('------------')
    print('now begin !')
    print('rbp:%s'%(rbp_name))
    print('------------')
    

    roots = dict(peak_root=args.peak_root, bw_root=args.bw_root, chrom_root=args.chrom_root)
    train_loader = DataLoader(dataset.RBPDataset(rbp_name, **roots), batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset.RBPDataset(rbp_name, **roots, mode='val'), batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    print(len(val_loader))

    homology_ids = None
    if args.homology_id is not None:
        homology_ids = os.path.join(args.homology_id, f"{rbp_name}_test_homology.ids")
    lengths = [300]
    test_loader_lists = []
    for seqlen in lengths:
        test_loader_lists.append((seqlen, DataLoader(dataset.RBPDataset(rbp_name, **roots, mode='test', input_len=seqlen, remove_homology_ids=homology_ids), batch_size=args.bs, shuffle=False, num_workers=args.num_workers)))


    model = models.rbpmodel()

    model.eval()
    device = torch.device(args.device)
    model.to(device)
    # model = torch.nn.DataParallel(model)

    check_mkdir(args.tb_dir)
    tb_ind = os.path.join(args.tb_dir, rbp_name)
    writer = SummaryWriter(tb_ind)
    check_mkdir(tb_ind)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr}
        ], lr=args.lr)

    
    epochs = 10000
    
    # begin training
    epoch_time_list = []
    val_pcc_record = []
    val_loss_record = []
    
    
    model = torch.compile(model)
    for epoch in range(epochs):
        print('-------epoch:%s-------' %(epoch))
        train_loadersize = len(train_loader)
        model.train()
        train_loss = 0
        
        tick_epoch = time.time()
        epoch_time_list.append(tick_epoch)

        train_pred = []
        train_label = []
        
        for i1, batch in enumerate(train_loader):
            batch_input, batch_label = batch
            batch_input = batch_input.to(device)
            batch_label = batch_label.to(device).to(torch.float32)
            outputs = model(batch_input).to(torch.float32)
            loss = loss_fn(outputs, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(batch_label)
            if i1 % max(1, train_loadersize//50) == 0:
                print('now pass number:', i1, '/ '+str(train_loadersize), flush = True)
                print('this batch loss:', loss.item()*len(batch_label), flush = True)
                
            
            train_pred.extend(torch.flatten(outputs).cpu().tolist())
            train_label.extend(torch.flatten(batch_label).cpu().tolist())
        
        pcc, p_value = compute_pcc(train_pred, train_label)
        train_loss_avg = train_loss / (train_loadersize * args.bs)
        
        print('Training/Loss:', train_loss_avg)
        print('Training/PearsonCorrelationCoefficient:', pcc)
        print('Training/P value:', p_value)
        writer.add_scalar('Training/Loss', train_loss_avg, epoch)
        writer.add_scalar('Training/PearsonCorrelationCoefficient:', pcc, epoch)
        

        # begin validation
        model.eval()
        val_loss = 0
        total_val_pred = []
        total_val_label = []
        
        val_loadersize = len(val_loader)
        for index_, batch in enumerate(val_loader):
            with torch.no_grad():
                batch_input, batch_label = batch
                batch_input = batch_input.to(device)
                batch_label = batch_label.to(device).to(torch.float32)
                outputs = model(batch_input).to(torch.float32)
                loss = loss_fn(outputs, batch_label)
                val_loss += loss.item()*len(batch_label)
                    
                total_val_pred.extend(torch.flatten(outputs).cpu().tolist())
                total_val_label.extend(torch.flatten(batch_label).cpu().tolist())
            
                 
        pcc, p_value = compute_pcc(total_val_pred, total_val_label)
        val_loss_avg = val_loss / (val_loadersize * args.bs)
        val_loss_record.append(val_loss_avg)
        
        print('Validation/Loss:', val_loss_avg)
        print('Validation/PearsonCorrelationCoefficient:', pcc)
        print('Validation/P value:', p_value)
        writer.add_scalar('Validation/Loss', val_loss_avg, epoch)
        writer.add_scalar('Validation/PearsonCorrelationCoefficient:', pcc, epoch)
        
        val_pcc_record.append(pcc)


        tick_trainval_1 = time.time()
        print('this epcoh cost time:', tick_trainval_1 - epoch_time_list[-1], flush = True)
        
        
        if args.save_model_dir != None:
            check_mkdir(args.save_model_dir)
            
            
        min_, min_index = get_min(val_loss_record)
        best_loss = min_
        if val_loss_record[-1] == best_loss:
            best_model_wts = copy.deepcopy(model.state_dict()) 
            wait = 0
        else:
            wait += 1
            if wait >= 6:
                lr = optimizer.param_groups[0]['lr']
                lr_new = lr * 0.5
                print(f'Patience reached. Adjusting lr from {lr:.8f} to {lr_new:.8f}.')
                optimizer.param_groups[0]['lr'] = lr_new
                wait = 0
        
        # write code about early stop
        if len(val_loss_record) < args.tol_epoch:
            min_, min_index = get_min(val_loss_record)
            best_model_wts = copy.deepcopy(model.state_dict())
            print('now best loss is:', min_, 'epoch is:', min_index)
            print('now best pcc is:', val_pcc_record[min_index])
            print('not enough epoch to stop, continue training.')
        elif len(val_loss_record) >= args.tol_epoch and len(val_loss_record) < args.max_epoch:
            min_, min_index = get_min(val_loss_record)
            if min_index < len(val_loss_record) - args.tol_epoch:
                print('early stop, best loss is:', min_, 'epoch is:', min_index)
                print('best pcc is:', val_pcc_record[min_index])
                break
            else:
                print('now best loss is:', min_, 'epoch is:', min_index)
                print('now best pcc is:', val_pcc_record[min_index])
                print('not enough epoch to stop, continue training.')
        elif len(val_loss_record) >= args.max_epoch:
            min_, min_index = get_min(val_loss_record)
            print('now best loss is:', min_, 'epoch is:', min_index)
            print('now best pcc is:', val_pcc_record[min_index])
            print('reach max epoch, stop training.')
            break
        
    print('finish training after %s epochs'%(len(val_loss_record)))
    min_, best_index = get_min(val_loss_record)
    print('for RBP %s, best pcc is:'%(rbp_name), val_pcc_record[min_index], 'epoch is:', best_index)

    
    # begin test
    test_model = models.rbpmodel()
    # test_model = torch.nn.DataParallel(test_model)
    test_model = torch.compile(test_model)
    test_model.load_state_dict(best_model_wts)
    test_model.eval()
    test_model.to(device)
    
    test_loss = 0
    total_test_pred = []
    total_test_label = []
    
    for seqlen, loader in test_loader_lists:
        print('Testing/InputLength:', seqlen)
        test_loadersize = len(loader)
        for index_, batch in enumerate(loader):
            with torch.no_grad():
                batch_input, batch_label = batch
                batch_input = batch_input.to(device)
                batch_label = batch_label.to(device).to(torch.float32)
                outputs = test_model(batch_input).to(torch.float32)
                loss = loss_fn(outputs, batch_label)
                test_loss += loss.item()*len(batch_label)
                        
                total_test_pred.extend(torch.flatten(outputs).cpu().tolist())
                total_test_label.extend(torch.flatten(batch_label).cpu().tolist())

                        
        pcc, p_value = compute_pcc(total_test_pred, total_test_label)
        test_loss_avg = test_loss / (test_loadersize * args.bs)
            
        print('Testing/Loss:', test_loss_avg)
        print('Testing/PearsonCorrelationCoefficient:', pcc)
    

    tick_final = time.time()
    print('time cost for RBP %s: '%(rbp_name), tick_final - tick1)
    
    
    if args.save_model_dir != None:
        torch.save(best_model_wts, os.path.join(args.save_model_dir, rbp_name+'.pth'))


def main():
    tick_all_start = time.time()
    for arg in vars(args):
        print(arg, ':', getattr(args, arg), flush=True)

    rbp_list = []
    with open(args.rbp, 'r') as file:
        for line in file:
            rbp_list.append(line.strip())
    for rbp in rbp_list:
        train_val_test(rbp)

    
    tick_all_finish = time.time()
    print('all time cost:', tick_all_finish - tick_all_start, flush=True)

if __name__ == '__main__':
    main()
