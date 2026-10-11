from torch.utils.data import Dataset
import torch
import pyBigWig as pbw
import numpy as np
import gzip
import random
from tqdm import tqdm


class RBPDataset(Dataset):
    """
    Sliding-window dataset for the RBPNet benchmark (sequence only, 300 nt input).

    Uses the same peak windowing and chromosome split as TransRBP
    (val = chr2/3/4, test = chr1/8/9, train = rest; 14,000 / 4,000 / 2,500 windows).
    If remove_homology_ids is given (output of homology.sh), test windows whose
    FASTA ID is listed are dropped and 2,000 test windows are re-sampled.

    Args:
        RBP_name   : RBP identifier
        peak_root  : directory containing the input RBP peak BED files ({RBP}.bed)
        bw_root    : directory containing {RBP}_plus.bw / {RBP}_minus.bw
        chrom_root : directory containing {chrom}.fa.gz per-chromosome FASTA files
    """

    def __init__(self, RBP_name, peak_root, bw_root, chrom_root, mode='train',
                 input_len=300, remove_homology_ids=None, seed=43):

        self.chrom_root = chrom_root
        self.peak_root = peak_root
        self.bw_root = bw_root
        self.seed = seed

        self.RBP_name = RBP_name
        self.input_len = input_len
        self.genome_cache = {}

        self.mode = mode
        self.val_chr = ['chr2','chr3','chr4']
        self.test_chr = ['chr1','chr8','chr9']

        self.peak_window = self.get_window(self.peak_root+'/'+self.RBP_name+'.bed', 20000)

        if mode == 'train':
            self.peak = self.peak_window[0]
        elif mode == 'val':
            self.peak = self.peak_window[1]
        elif mode == 'test':
            self.peak = self.peak_window[2]

        if remove_homology_ids is not None and mode == "test":
            with open(remove_homology_ids) as f:
                remove_ids = set(line.strip() for line in f)

            filtered_peak = []
            for i, item in enumerate(self.peak):
                start, end, chrom, strand = item
                fasta_id = f"{self.RBP_name}|test|{i}|{chrom}:{start}-{end}({strand})"

                if fasta_id not in remove_ids:
                    filtered_peak.append(item)

            print(f"{self.RBP_name}: test before = {len(self.peak)}, after = {len(filtered_peak)}")
            self.peak = self.sample_choice(sorted(list(set(filtered_peak))), 2000)


    def __getitem__(self, idx):

        start, end, chrom, direction= self.peak[idx]
        RNA_seq = self.get_seq_npy(self.chrom_root, chrom, start, end, direction)
        seq = self.get_onehot(RNA_seq)
        features = self.read_bigwig(self.bw_root, self.RBP_name, chrom, start, end, direction)
        output = seq, features
        return output


    def __len__(self):
        return len(self.peak)


    def get_peakinfo(self):
        return self.peak


    def export_fasta(self, out_fa):
        with open(out_fa, 'w') as f:
            for i in tqdm(range(len(self)), desc=f"Exporting {self.mode} FASTA"):
                start, end, chrom, strand = self.peak[i]

                seq = self.get_seq_npy_cache(
                    self.chrom_root,
                    chrom, start, end, strand
                )

                header = f">{self.RBP_name}|{self.mode}|{i}|{chrom}:{start}-{end}({strand})"
                f.write(header + "\n")
                f.write(seq.upper() + "\n")


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

        return self.sample_choice(sorted(list(set(trains))), int(total_num/10*7)), self.sample_choice(sorted(list(set(vals))), int(total_num/10*2)), self.sample_choice(sorted(list(set(tests))), int(total_num/8))


    def sample_choice(self, seq, sample_size):
        random.seed(self.seed)
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


    def get_seq_npy_cache(self, chrom_root, chr, start, end, direction):
            if chr not in self.genome_cache:
                with gzip.open(f'{chrom_root}/{chr}.fa.gz', 'r') as f:
                    seq = f.read().decode("utf-8")
                    seq = seq[seq.find('\n'):]
                    seq = seq.replace('\n', '').lower()
                    self.genome_cache[chr] = seq

            chrom_seq = self.genome_cache[chr]
            slice_seq = chrom_seq[start:end]

            if direction == '-':
                trans_dict = str.maketrans('acgtn', 'ugcan')
                com_seq = slice_seq.translate(trans_dict)[::-1]
            elif direction == '+':
                com_seq = slice_seq.replace("t", "u")
            else:
                com_seq = slice_seq

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


    def read_bigwig(self, bw_root, RBP, chr, start, end, direction):
        if direction == '+':
            bw=pbw.open(f'{bw_root}/{RBP}_plus.bw')
            signals=bw.values(chr,start,end)
            arr=np.reshape(signals, (1, -1))
            features = np.nan_to_num(arr, nan=0)

        if direction == '-':
            bw=pbw.open(f'{bw_root}/{RBP}_minus.bw')
            signals=bw.values(chr,start,end)
            signals=signals[::-1]
            arr=np.reshape(signals, (1, -1))
            features = np.nan_to_num(arr, nan=0)

        features[features < 0] = 0
        features = torch.from_numpy(features)

        return features
