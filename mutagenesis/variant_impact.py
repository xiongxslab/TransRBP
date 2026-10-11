"""
Score single-nucleotide variants by the change they cause in the predicted RBP
binding profile.

variant_score = KL(alt || ref), the KL divergence between the alternative- and
reference-allele 800 nt binding profiles (each normalised to sum to 1).

Model type (both are used for the variant analyses):
  sequence-only model (default, --features 4): the 4-channel TransRBP trained
      without the m6A channel.
  full model (--features 5): sequence + m6A; the m6A MeRIP-seq signal of the
      window (--m6A_bw_plus / --m6A_bw_minus) is identical for the reference and
      alternative input.

Example:
    python mutagenesis/variant_impact.py \\
        --input_csv        variants.csv \\
        --reference_genome hg38/all/hg38.fa \\
        --model_path       models_seq_only/FMR1.pth \\
        --output_tsv       FMR1_variants.tsv
"""

import argparse
import csv
import os

import numpy as np
import torch
from scipy import special

from model.RBPResTransModels import RBPModel
from mutagenesis.variant_dataset import extract_sequences, prepare_data, get_data_loader

torch.set_float32_matmul_precision('high')

def normalize_distribution(dist):
    # Ensure the distributions are normalized to prob distribution and make sure (0,1)
    epsilon = 1e-10
    dist += epsilon 
    # Adjust axis to sum across the sequence length for normalization
    sum_dist = np.sum(dist, axis=2, keepdims=True)
    normalized_dist = dist / sum_dist
    return normalized_dist

def compute_kl_divergence(p, q):
    # KL(p || q); called with p = alternative allele, q = reference allele
    p = normalize_distribution(p)
    q = normalize_distribution(q)

    # Compute KL divergence for each element in the batch
    kl_elementwise = special.rel_entr(p, q)
    kl_div = np.sum(kl_elementwise, axis=2)
    kl_div = np.squeeze(kl_div)

    return kl_div

def get_variant_scores(model, data_loader, device):

    model.eval()
    variant_scores = []

    with torch.no_grad():
        for reference_tensors, alternative_tensors in data_loader:
            # Move tensors to the specified device
            reference_tensors   = reference_tensors.to(device).permute(0, 2, 1)    # (B, C, 800)
            alternative_tensors = alternative_tensors.to(device).permute(0, 2, 1)  # (B, C, 800)

            # Get predictions for reference and alternative sequences
            reference_outputs   = model(reference_tensors).cpu().numpy()
            alternative_outputs = model(alternative_tensors).cpu().numpy()

            # Compute variant scores: KL(alt || ref)
            batch_variant_scores = compute_kl_divergence(alternative_outputs, reference_outputs)

            # Append batch scores to the list
            variant_scores.extend(np.atleast_1d(batch_variant_scores))

    return variant_scores

def main():
    parser = argparse.ArgumentParser(description='Compute variant scores using a deep learning model.')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV: chromosome, position (1-based), strand, allele_A (ref, on that strand), allele_B (alt).')
    parser.add_argument('--reference_genome', type=str, required=True, help='Path to the reference genome FASTA file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--output_tsv', type=str, required=True, help='Path to the output TSV file.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the computations on (e.g., "cpu" or "cuda").')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing data.')
    parser.add_argument('--features', type=int, default=4, choices=[4, 5],
                        help='4 = sequence-only model (default); 5 = full model (needs --m6A_bw_plus/minus)')
    parser.add_argument('--m6A_bw_plus', type=str, default=None, help='m6A MeRIP-seq BigWig (plus strand).')
    parser.add_argument('--m6A_bw_minus', type=str, default=None, help='m6A MeRIP-seq BigWig (minus strand).')

    args = parser.parse_args()
    if args.features == 5 and (args.m6A_bw_plus is None or args.m6A_bw_minus is None):
        parser.error('--features 5 (full model) requires --m6A_bw_plus and --m6A_bw_minus.')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Extract sequences
    if args.features == 5:
        results = extract_sequences(args.reference_genome, args.input_csv,
                                    args.m6A_bw_plus, args.m6A_bw_minus)
    else:
        results = extract_sequences(args.reference_genome, args.input_csv)

    if not results:
        print("No valid sequences found. Exiting.")
        raise SystemExit(1)

    # Prepare data
    reference_tensors, alternative_tensors = prepare_data(results)

    # Get data loader
    data_loader = get_data_loader(reference_tensors, alternative_tensors, args.batch_size)

    # Load the model
    print("-----Loading Model------\n")
    model = RBPModel(features=args.features, record_attn=False)
    state = torch.load(args.model_path, map_location='cpu', weights_only=True)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if device.type == 'cuda':
        model = torch.compile(model)

    print("Model loaded successfully.")

    # Get variant scores
    variant_scores = get_variant_scores(model, data_loader, device)

    # Add variant scores to results
    for idx, entry in enumerate(results):
        entry['variant_score'] = variant_scores[idx]

    # Write output TSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_tsv)), exist_ok=True)
    with open(args.output_tsv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write header
        header = ['chromosome', 'position', 'strand', 'ref', 'alt', 'variant_score']
        writer.writerow(header)

        for entry in results:
            row = [
                entry['chromosome'],
                entry['position'],
                entry['strand'],
                entry['allele_A'],
                entry['allele_B'],
                entry['variant_score']
            ]
            writer.writerow(row)

    print(f"Variant scores have been written to {args.output_tsv}")

if __name__ == "__main__":
    main()
