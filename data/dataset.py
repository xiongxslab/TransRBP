from torch.utils.data import Dataset
import torch
import pyBigWig as pbw
import numpy as np
import gzip
import random
from tqdm import tqdm
import h5py


class RBPDataset(Dataset):
    
    def __init__(self, RBP_name, data_root, mode = 'train', m6A_binding = 1):
        
        self.chrom_root = f'{data_root}/hg38'
    
        self.peak_file = f'{data_root}/RBP/{RBP_name}/bindingpeak.bed'
        
        self.bw_plus_file = f'{data_root}/RBP/{RBP_name}/bindingsignal_plus.bw'
        self.bw_minus_file = f'{data_root}/RBP/{RBP_name}/bindingsignal_minus.bw'

        self.m6A_bw_plus = f'{data_root}/m6A/MeRIP_signal_plus.bw'
        self.m6A_bw_minus = f'{data_root}/m6A/MeRIP_signal_minus.bw'
        
        
        self.RBP_name = RBP_name
        self.input_len = 800
        
        self.mode = mode

        self.m6A_binding = m6A_binding

        self.val_chr = ['chr2','chr3','chr4']
        self.test_chr = ['chr1','chr8','chr9']
        
        self.peak_window = self.get_window(self.peak_file, total_num=20000)
        
        if mode == 'train':
            self.peak = self.peak_window[0]
        elif mode == 'val':
            self.peak = self.peak_window[1]
        elif mode == 'test':
            self.peak = self.peak_window[2]
    
    
    def __getitem__(self, idx):
        start, end, chrom, direction = self.peak[idx]
        
        features = self.read_bigwig(self.bw_plus_file, self.bw_minus_file, chrom, start, end, direction)
        RNA_seq = self.get_seq_npy(self.chrom_root, chrom, start, end, direction)
        model_in = self.get_onehot(RNA_seq)
        
        if self.m6A_binding == 1:
                binding_features = self.read_bigwig(self.m6A_bw_plus, self.m6A_bw_minus, chrom, start, end, direction)
                model_in = torch.cat((model_in, binding_features), dim=0)
        
       
        output = model_in, features
        return output
        
    
    def __len__(self):
        return len(self.peak)
    
    def get_peakinfo(self, whether_whole = False):
        if whether_whole:
            return self.peak_window[0]+self.peak_window[1]+self.peak_window[2]
        else:
            return self.peak
    
    def save_to_hdf5(self, filedir):
        with h5py.File(filedir+'/'+self.RBP_name+'_'+self.mode+'.h5', 'w') as hf:
            for idx in tqdm(range(len(self))):
                seq_onehot, bigwig = self[idx]
                group = hf.create_group(f'sample_{idx}')
                group.create_dataset('input', data=seq_onehot.numpy())
                group.create_dataset('feature', data=bigwig.numpy())
    
    
    def get_window(self, file_root, total_num):
        
        trains = []
        vals = []
        tests = []
        
        with open(file_root, 'r') as f:
            lines = f.readlines()          
            for line in tqdm(lines, desc="Processing peaks", unit="peak"):
                fields = line.split('\t')
                chrom = fields[0]
                direction = fields[5].replace('\n', '')
                start = int(fields[1])
                end = int(fields[2])
                peak_length = end - start
                for bp in range(start-self.input_len//2, end+self.input_len, int(peak_length/10)+1):
                    peak_start, peak_end = (bp-self.input_len+1, bp+1)
                    
                    item = peak_start,peak_end,chrom,direction
                    
                    if chrom in self.val_chr:
                        vals.append(item)
                    elif chrom in self.test_chr:
                        tests.append(item)
                    else:
                        trains.append(item)

        return self.sample_choice(sorted(list(set(trains))), int(total_num/10*7)), self.sample_choice(sorted(list(set(vals))), int(total_num/10*2)), self.sample_choice(sorted(list(set(tests))), int(total_num/10))
        

    def sample_choice(self, seq, sample_size):
        random.seed(43)
        if len(seq) <= sample_size:
            return random.sample(seq, len(seq))
        else:
            return random.sample(seq, sample_size)
    
    
    def get_seq_npy(self, chrom_root, chr, start, end, direction):
        with gzip.open(f'{chrom_root}/{chr}.fa.gz', 'r') as f:
            seq = f.read().decode("utf-8")
            seq = seq[seq.find('\n'):]
            seq = seq.replace('\n', '').lower()
            
                        
            if direction == '-':
                complement_dict = {'a': 'u', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
                com_seq = ''   
                for base in seq[start:end]:
                    com_seq += complement_dict[base]
                com_seq = com_seq[::-1]
                
            if direction == '+':
                com_seq = seq[start:end].replace("t", "u")

            return com_seq


    def get_onehot(self, RNAseq):
        en_dict = {'a' : 0, 'u' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        en_seq = [en_dict[nt] for nt in RNAseq]
        np_seq = np.array(en_seq, dtype = int)

        seq_emb = np.zeros((len(RNAseq), 5))
        seq_emb[np.arange(len(RNAseq)), np_seq] = 1
        seq = np.transpose(seq_emb)
        seq = torch.from_numpy(seq)
        seq = seq[:4,:]
        
        return seq
    
    
    def read_bigwig(self, bw_plus, bw_minus, chr, start, end, direction):
        if direction == '+':
            bw=pbw.open(bw_plus)
            signals=bw.values(chr,start,end)
            arr=np.reshape(signals, (1, -1))
            features = np.nan_to_num(arr, nan=0)
            
        if direction == '-':
            bw=pbw.open(bw_minus)
            signals=bw.values(chr,start,end)
            signals=signals[::-1]   
            arr=np.reshape(signals, (1, -1))
            features = np.nan_to_num(arr, nan=0)

        features[features < 0] = 0
        features = torch.from_numpy(features)
        
        return features
