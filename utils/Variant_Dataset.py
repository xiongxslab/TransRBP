from pyfaidx import Fasta
from Bio.Seq import Seq
import torch
import csv

def extract_sequences(reference_genome_path, input_table_path):
  
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
            chromosome, position, strand, allele_A, allele_B = tokens[:5]
            position = int(position)

            # Convert position to 0-based index
            position0 = position - 1

            # Calculate start and end indices for 800 bp region
            start = position0 - 400
            end = position0 + 399  # pyfaidx is end-inclusive

            # Ensure indices are within chromosome bounds
            seq_length = len(genome[chromosome])
            start = max(start, 0)
            end = min(end, seq_length - 1)

            # Extract sequence region
            seq_region = genome[chromosome][start:end+1].seq
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
    Returns two tensors: reference_tensors and alternative_tensors.
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

    return reference_tensors, alternative_tensors

def get_data_loader(reference_tensors, alternative_tensors, batch_size):
    """
    Creates a DataLoader for batching the data.
    """
    dataset = torch.utils.data.TensorDataset(reference_tensors, alternative_tensors)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader
