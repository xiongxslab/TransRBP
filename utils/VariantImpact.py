import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import csv
from scipy import special
from TransRBP.model import RBPResTransModels
from TransRBP.utils.Variant_Dataset import extract_sequences, prepare_data, get_data_loader

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
    # p is the reference allele distribution and q is the alternative allele
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
            reference_tensors = reference_tensors.to(device).permute(0, 2, 1)  # Shape: (batch_size, 4, 800)
            alternative_tensors = alternative_tensors.to(device).permute(0, 2, 1)

            # Get predictions for reference and alternative sequences
            reference_outputs = model(reference_tensors).cpu().numpy()
            alternative_outputs = model(alternative_tensors).cpu().numpy()

            # Compute variant scores
            batch_variant_scores = compute_kl_divergence(reference_outputs, alternative_outputs)

            # Append batch scores to the list
            variant_scores.extend(batch_variant_scores)

    return variant_scores 

def main():
    parser = argparse.ArgumentParser(description='Compute variant scores using a deep learning model.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input TSV file.')
    parser.add_argument('--reference_genome', type=str, required=True, help='Path to the reference genome FASTA file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--output_tsv', type=str, required=True, help='Path to the output TSV file.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the computations on (e.g., "cpu" or "cuda").')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing data.')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Extract sequences
    results = extract_sequences(args.reference_genome, args.input_csv)

    if not results:
        print("No valid sequences found. Exiting.")
        sys.exit(1)

    # Prepare data
    reference_tensors, alternative_tensors = prepare_data(results)

    # Get data loader
    data_loader = get_data_loader(reference_tensors, alternative_tensors, args.batch_size)

    # Load the model
    print("-----Loading Model------\n")
    model = RBPResTransModels.RBPModel(record_attn=False)   
    model = torch.compile(model)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    if not isinstance(model, nn.Module):
        sys.exit("Model loading failed, exiting the program.")

    print("Model Loading Successfull!")

    # Get variant scores
    variant_scores = get_variant_scores(model, data_loader, device)

    # Add variant scores to results
    for idx, entry in enumerate(results):
        entry['variant_score'] = variant_scores[idx]

    # Write output TSV
    with open(args.output_tsv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write header
        header = ['chromosome', 'position', 'strand', 'ref', 'alt', 'variant_score']
        writer.writerow(header)

        for entry in results:
        
            #print(entry)
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
