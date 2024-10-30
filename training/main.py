import argparse
import numpy as np
import random
import time
import torch
import os
import pandas as pd
import scipy
import copy
import sys
# sys.path.append('./TransRBP')
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from TransRBP.data import dataset
from TransRBP.model import RBPResTransModels

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='TransRBP')
parser.add_argument('--bs', type = int, default = 64,
                    help = 'the batch size for data loading')
parser.add_argument('--lr', type = float, default = 1e-3,
                    help = 'the learning rate for the optimizer')
parser.add_argument('--random_seed', type = int, default = 43,
                    help = 'random seed for training')
parser.add_argument('--tb_dir', type = str,
                    help = 'the directory to save tensorboard logs')
parser.add_argument('--whether_grad_norm_clip', type = int, default = 1,
                    help='flag to specify whether gradient norm clipping is used'
                         '1 means enabled, 0 means disabled')
parser.add_argument('--grad_norm_clip', type = float, default = 5,
                    help = 'the value of gradient norm clip when `whether_grad_norm_clip` is set to 1')
parser.add_argument('--save_model_dir', type = str, default=None,
                    help = 'the directory to save the trained model')
parser.add_argument('--RBP', type = str, default=None,
                    help= 'RBP for which the model is trained')
parser.add_argument('--data_root', type = str, default=None,
                    help= 'root path to training data')

# parameters about early stop
parser.add_argument('--max_epoch', type = int, default = 100,
                    help = 'the max epochs for training')
parser.add_argument('--tol_epoch', type = int, default = 10,
                    help = 'the tolerance epochs for early stop')

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
    loss = torch.nn.MSELoss(reduction='mean')
    return loss(output, label)

def compute_pcc(pred_:list, label_:list):
    pcc, p_value = scipy.stats.pearsonr(np.array(pred_), np.array(label_))
    return pcc, p_value

def compute_scc(pred_:list, label_:list):
    pcc, p_value = scipy.stats.spearmanr(np.array(pred_), np.array(label_))
    return pcc, p_value

def get_max(list_):
    val, idx = max((val, idx) for (idx, val) in enumerate(list_))
    return val, idx

def train_val_test(rbp_name):
    setup_seed(args.random_seed)
    torch.cuda.empty_cache()
        
    tick1 = time.time()
    print('\n')
    print('------------')
    print('now begin !')
    print('rbp:%s'%(rbp_name))
    print('------------')
    # print('\n')
    
    train_loader = DataLoader(dataset.RBPDataset(rbp_name, args.data_root), batch_size=args.bs, shuffle=True, num_workers=64, persistent_workers=True)
    val_loader = DataLoader(dataset.RBPDataset(rbp_name, args.data_root, mode='val'), batch_size=args.bs, shuffle=False, num_workers=32, persistent_workers=True)
    test_loader = DataLoader(dataset.RBPDataset(rbp_name, args.data_root, mode='test'), batch_size=args.bs, shuffle=False, num_workers=16, persistent_workers=True)

    model = RBPResTransModels.RBPModel()

    model.eval()
    device = torch.device("cuda")
    model.to(device)
    # model = torch.nn.DataParallel(model)

    check_mkdir(args.tb_dir)
    tb_ind = os.path.join(args.tb_dir, rbp_name)
    writer = SummaryWriter(tb_ind)
    check_mkdir(tb_ind)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr}
        ], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.max_epoch)

    
    epochs = 10000
    
    # begin training
    epoch_time_list = []
    val_pcc_record = []
    
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
            if args.whether_grad_norm_clip == 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()
            train_loss += loss.item()*len(batch_label)
            if i1 % (train_loadersize//50) == 0:
                print('now pass number:', i1, '/ '+str(train_loadersize), flush = True)
                print('this batch loss:', loss.item()*len(batch_label), flush = True)
                
            train_pred.extend(torch.flatten(outputs).cpu().tolist())
            train_label.extend(torch.flatten(batch_label).cpu().tolist())
        
        pcc, _ = compute_pcc(train_pred, train_label)
        scc, _ = compute_scc(train_pred, train_label)
        train_loss_avg = train_loss / (train_loadersize * args.bs)
        
        print('Training/Loss:', train_loss_avg)
        print('Training/PearsonCorrelationCoefficient:', pcc)
        print('Training/SpearmanCorrelationCoefficient:', scc)
        writer.add_scalar('Training/Loss', train_loss_avg, epoch)
        writer.add_scalar('Training/PearsonCorrelationCoefficient:', pcc, epoch)
        writer.add_scalar('Training/SpearmanCorrelationCoefficient:', scc, epoch)
        

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
            
        pcc, _ = compute_pcc(total_val_pred, total_val_label)
        scc, _ = compute_scc(total_val_pred, total_val_label)
        val_loss_avg = val_loss / (val_loadersize * args.bs)
        
        print('Validation/Loss:', val_loss_avg)
        print('Validation/PearsonCorrelationCoefficient:', pcc)
        print('Validation/SpearmanCorrelationCoefficient:', scc)
        writer.add_scalar('Validation/Loss', val_loss_avg, epoch)
        writer.add_scalar('Validation/PearsonCorrelationCoefficient:', pcc, epoch)
        writer.add_scalar('Validation/SpearmanCorrelationCoefficient:', scc, epoch)
        
        val_pcc_record.append(pcc)

        scheduler.step()


        tick_trainval_1 = time.time()
        print('this epcoh cost time:', tick_trainval_1 - epoch_time_list[-1], flush = True)
        # epoch_time_list.append(tick_trainval_1)
        
        
        if args.save_model_dir != None:
            # save_model_dir_ind = os.path.join(args.save_model_dir, rbp_name)
            check_mkdir(args.save_model_dir)
            
            max_, max_index = get_max(val_pcc_record)
            best_pcc = max_
            if val_pcc_record[-1] == best_pcc:
                best_model_wts = copy.deepcopy(model.state_dict()) 
        
        
        # write code about early stop
        if len(val_pcc_record) < args.tol_epoch:
            max_, max_index = get_max(val_pcc_record)
            # best_model_wts = copy.deepcopy(model.state_dict())
            print('now best pcc is:', max_, 'epoch is:', max_index)
            print('not enough epoch to stop, continue training.')
        elif len(val_pcc_record) >= args.tol_epoch and len(val_pcc_record) < args.max_epoch:
            max_, max_index = get_max(val_pcc_record)
            # print('now best pcc is:', max_, 'epoch is:', max_index)
            if max_index < len(val_pcc_record) - args.tol_epoch:
                print('early stop, best pcc is:', max_, 'epoch is:', max_index)
                break
            else:
                print('now best pcc is:', max_, 'epoch is:', max_index)
                print('not enough epoch to stop, continue training.')
        elif len(val_pcc_record) >= args.max_epoch:
            max_, max_index = get_max(val_pcc_record)
            print('now best pcc is:', max_, 'epoch is:', max_index)
            print('reach max epoch, stop training.')
            break
        
    print('finish training after %s epochs'%(len(val_pcc_record)))
    max_, best_index = get_max(val_pcc_record)
    print('for RBP %s, best pcc is:'%(rbp_name), max_, 'epoch is:', best_index)

    
    # begin test
    
    # test_model = RBPResTransModels.RBPModel()  
    test_model = RBPResTransModels.RBPModel()    
    # test_model = torch.nn.DataParallel(test_model)
    test_model = torch.compile(test_model)
    test_model.load_state_dict(best_model_wts)
    test_model.eval()
    test_model.to(device)
    
    test_loss = 0
    total_test_pred = []
    total_test_label = []
    
    test_loadersize = len(test_loader)
    for index_, batch in enumerate(test_loader):
        with torch.no_grad():
            batch_input, batch_label = batch
            batch_input = batch_input.to(device)
            batch_label = batch_label.to(device).to(torch.float32)
            outputs = test_model(batch_input).to(torch.float32)
            loss = loss_fn(outputs, batch_label)
            test_loss += loss.item()*len(batch_label)
                       
            total_test_pred.extend(torch.flatten(outputs).cpu().tolist())
            total_test_label.extend(torch.flatten(batch_label).cpu().tolist())
      
    pcc, _ = compute_pcc(total_test_pred, total_test_label)
    scc, _ = compute_scc(total_test_pred, total_test_label)
    test_loss_avg = test_loss / (test_loadersize * args.bs)
    
        
    print('Testing/Loss:', test_loss_avg)
    print('Testing/PearsonCorrelationCoefficient:', pcc)
    print('Testing/SpearmanCorrelationCoefficient:', scc)
    

    tick_final = time.time()
    print('time cost for RBP %s: '%(rbp_name), tick_final - tick1)
    
    if args.save_model_dir != None:
        torch.save(best_model_wts, os.path.join(args.save_model_dir, rbp_name+'.pth'))


def main():
    for arg in vars(args):
        print(arg, ':', getattr(args, arg), flush=True)

    train_val_test(args.RBP)


if __name__ == '__main__':
    main()