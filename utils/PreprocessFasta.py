import torch
from Bio import SeqIO

class FastaDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file, seq_length=800):
        self.seq_length = seq_length
        self.sequences = self.read_fasta_with_biopython(fasta_file)
        self.one_hot_sequences = self.process_sequences()

    def read_fasta_with_biopython(self, fasta_file):
        sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
        return sequences

    def pad_sequence(self, seq):
        if len(seq) > self.seq_length:
            # Trim the sequence
            seq = seq[:self.seq_length]
        elif len(seq) < self.seq_length:
            # Pad the sequence with 'N'
            seq += 'N' * (self.seq_length - len(seq))
        return seq

    def one_hot_encode_sequence(self, seq):
        
        mapping = {'A': [1, 0, 0, 0],
                   'U': [0, 1, 0, 0],
                   'T': [0, 1, 0, 0],
                   'C': [0, 0, 1, 0],
                   'G': [0, 0, 0, 1],
                   'N': [0, 0, 0, 0]}
        seq = seq.upper()
        one_hot_seq = [mapping.get(base, [0, 0, 0, 0]) for base in seq]
        return one_hot_seq

    def process_sequences(self):
        processed_seqs = []
        for seq in self.sequences:
            seq = self.pad_sequence(seq)
            one_hot_seq = self.one_hot_encode_sequence(seq)
            processed_seqs.append(one_hot_seq)
        return processed_seqs

    def __len__(self):
        return len(self.one_hot_sequences)

    def __getitem__(self, idx):
        seq = self.one_hot_sequences[idx]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).permute(1, 0)  # Shape (4, seq_length)
        original_seq = self.sequences[idx]
        return seq_tensor, original_seq


