from pyfaidx import Fasta
from Bio.Seq import Seq
import numpy as np
import pyBigWig
import torch
import csv

def read_m6A(m6A_bw_plus, m6A_bw_minus, chromosome, start, end, strand):
    """
    m6A MeRIP-seq signal over genomic [start, end) on the given strand,
    oriented 5'->3' like the sequence (NaN -> 0, negatives clipped to 0).
    """
    bw = pyBigWig.open(m6A_bw_plus if strand == '+' else m6A_bw_minus)
    signal = np.nan_to_num(np.array(bw.values(chromosome, start, end), dtype=np.float32), nan=0.0)
    bw.close()
    signal[signal < 0] = 0.0
    if strand == '-':
        signal = signal[::-1].copy()
    return signal

def extract_sequences(reference_genome_path, input_table_path, m6A_bw_plus=None, m6A_bw_minus=None):
    """
    Builds reference / alternative 800 nt windows centred on each variant
    (variant at index 400 on the + strand). If m6A BigWigs are given, the m6A
    signal of the window is attached (identical for reference and alternative).
    """
    # Load the reference genome
    genome = Fasta(reference_genome_path)
    results = []

    # Read the input table
    with open(input_table_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for tokens in reader:
            # Skip empty lines and comments
            if not tokens or tokens[0].startswith('#'):
                continue
            if len(tokens) < 5:
                continue  # Incomplete data
            chromosome, position, strand, allele_A, allele_B = [t.strip() for t in tokens[:5]]
            if not position.lstrip('-').isdigit():
                continue  # skip header row
            position = int(position)

            # Convert position to 0-based index
            position0 = position - 1

            # Calculate start and end indices for 800 bp region
            start = position0 - 400
            end = position0 + 399  # last position of the window (inclusive)

            # Parts of the window beyond a chromosome end are padded with N (and zero m6A signal), so every
            # window is 800 nt and the variant stays at index 400 (+ strand) / 399 (- strand)
            seq_length = len(genome[chromosome])
            pad_left = max(0, -start)
            pad_right = max(0, end - (seq_length - 1))
            start_in, end_in = start + pad_left, end - pad_right   # part of the window inside the chromosome

            # Extract sequence region
            inside = genome[chromosome][start_in:end_in+1].seq if end_in >= start_in else ''
            seq_region = 'N' * pad_left + inside + 'N' * pad_right
            pos_in_seq = position0 - start  # Position within seq_region

            # Handle strand information
            if strand == '-':
                # Get reverse complement
                seq_region = str(Seq(seq_region).reverse_complement())
                # Adjust position in reverse complement
                pos_in_seq = len(seq_region) - pos_in_seq - 1
                base_at_pos = seq_region[pos_in_seq]
            else:
                base_at_pos = seq_region[pos_in_seq]

            # Check if allele_A matches the base at position
            if base_at_pos.upper() != allele_A.upper():
                print(f"Warning: Base at position {position} on {chromosome} does not match allele_A ({allele_A}). Found {base_at_pos} instead. Please check your input.")
                continue  # Skip if alleles do not match

            # Generate mutated sequence
            mutated_seq = seq_region[:pos_in_seq] + allele_B + seq_region[pos_in_seq+1:]

            # Collect results
            result = {
                'chromosome': chromosome,
                'position': position,
                'strand': strand,
                'allele_A': allele_A,
                'allele_B': allele_B,
                'original_sequence': seq_region,
                'mutated_sequence': mutated_seq
            }
            if m6A_bw_plus is not None:
                signal = read_m6A(m6A_bw_plus, m6A_bw_minus, chromosome, start_in, end_in + 1, strand)
                # zero padding on the padded side(s), in the 5'->3' orientation of the window
                before, after = (pad_left, pad_right) if strand == '+' else (pad_right, pad_left)
                result['m6A_signal'] = np.concatenate([np.zeros(before, np.float32), signal,
                                                       np.zeros(after, np.float32)])
            results.append(result)

    return results

def one_hot_encode_sequence(seq):
    """
    One-hot encodes a DNA sequence.
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'U': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    seq = seq.upper()
    one_hot_seq = [mapping.get(base, [0, 0, 0, 0]) for base in seq]
    return one_hot_seq

def prepare_data(results):
    """
    Prepares data by one-hot encoding sequences and converting them into tensors.
    Returns reference_tensors and alternative_tensors, shape (N, 800, 4), or
    (N, 800, 5) with the m6A signal as the last channel when it was extracted.
    """
    reference_sequences = []
    alternative_sequences = []

    for entry in results:
        ref_seq = entry['original_sequence']
        alt_seq = entry['mutated_sequence']

        # One-hot encode the sequences
        ref_one_hot = one_hot_encode_sequence(ref_seq)
        alt_one_hot = one_hot_encode_sequence(alt_seq)

        reference_sequences.append(ref_one_hot)
        alternative_sequences.append(alt_one_hot)

    # Convert lists to tensors
    reference_tensors = torch.tensor(reference_sequences, dtype=torch.float32)
    alternative_tensors = torch.tensor(alternative_sequences, dtype=torch.float32)

    if results and 'm6A_signal' in results[0]:
        m6A = torch.tensor(np.stack([entry['m6A_signal'] for entry in results]),
                           dtype=torch.float32).unsqueeze(-1)       # (N, 800, 1)
        reference_tensors   = torch.cat([reference_tensors,   m6A], dim=-1)
        alternative_tensors = torch.cat([alternative_tensors, m6A], dim=-1)

    return reference_tensors, alternative_tensors

def get_data_loader(reference_tensors, alternative_tensors, batch_size):
    """
    Creates a DataLoader for batching the data.
    """
    dataset = torch.utils.data.TensorDataset(reference_tensors, alternative_tensors)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader
