import sys
import torch
import numpy as np
import torch.nn
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from torch.utils.data import DataLoader
from captum.attr import NoiseTunnel
import h5py
import torch.nn.functional as F
import torch.nn as nn
import argparse
from TransRBP.utils import PreprocessFasta
from TransRBP.model import RBPResTransModels




class ContribH5:
    def __init__(self, filename, RBPname, RBPmodel_root, fasta_file, record_global = False,  contrib_function = 'IG', mode='w', device ='cuda:0'):
        if any(arg is None for arg in (filename, RBPname, RBPmodel_root, fasta_file)):
            raise ValueError("All arguments (filename, RBPname, RBPmodel_root, fasta_file) must be provided.")
        self.filename = filename
        self.file = h5py.File(filename, mode)
        self.contrib_function = contrib_function
        self.device = device
        self.record_global_contrib = record_global
        self.RBPname = RBPname
        self.RBPmodel_root = RBPmodel_root
        self.RBPmodel = self.get_RBP_model()
        self.group1 = self.file.create_group(f'{self.RBPname}')
        self.group2 = self.group1.create_group("Contribution_Score")
        self.group3 = self.group1.create_group("Inputs")
        self.fasta_file = fasta_file
        self.contrib_score = self.get_contrib_score()

    torch.set_float32_matmul_precision('high')

    def __del__(self):
        self.file.close()

    def refined_model(self, input):
        pred = self.RBPmodel(input)
        pred_squeezed = torch.squeeze(pred, dim=1)
        weights = F.softmax(pred_squeezed.detach(), dim=1)
        weight_sum = torch.sum((pred_squeezed * weights), dim=1)
        return weight_sum

    def get_contrib_model(self):
        if self.contrib_function == "IG":
            print("Using Integrated Gradient as the contribution score calculating algorithm.")
            IG = IntegratedGradients(self.refined_model, multiply_by_inputs=False)
            return IG

        elif self.contrib_function == "SA":
            print("Using Saliency as the contribution score calculating algorithm.")
            SA = Saliency(self.refined_model)
            return SA
        
        else:
            raise ValueError("Algorithm not supported")

    def get_contrib_score(self):
        print('\n')
        print("-----Getting Contribution Scores-----")

        desired_seq_length = 800  # Set this to the required sequence length
        data = PreprocessFasta.FastaDataset(self.fasta_file, seq_length=desired_seq_length)
        dataLoader = DataLoader(data, batch_size=64, shuffle=False, num_workers=3, pin_memory=False, collate_fn=self.custom_collate_fn)

        contrib_dict = {}
        input_dict = {}
        seq_dict = {}
        batchsize = len(dataLoader)
        contrib_model = self.get_contrib_model()

        for idx, (batch_input, original_seqs) in enumerate(dataLoader):
            print('\n')
            print(f"RBP name: {self.RBPname}")
            print(f"-----Processing batch {idx + 1} of {batchsize}-----")
            print('\n')

            print(f"Shape of input: {batch_input.shape}")
            batch_input = batch_input.to(self.device)
            batch_input.requires_grad_()

            if self.contrib_function == "IG":
                attr = contrib_model.attribute(batch_input, internal_batch_size=64)  # Shape (bs, 4, seqL)
                
            elif self.contrib_function == "SA":
                attr = contrib_model.attribute(batch_input, abs=False)

            # Gradient Correction
            attr -= torch.mean(attr, dim=1, keepdim=True)

            if not self.record_global_contrib:
                attr = attr * batch_input
          

            contrib_dict[idx] = attr
            input_dict[idx] = batch_input
            seq_dict[idx] = original_seqs

            if batchsize == 1:
                self.save_contrib_scores_chunk(contrib_dict)
                self.save_input_chunk(input_dict, seq_dict)
                print(f"Batch 1 input and contribution data saved to h5 file")

            else:
                if ((idx + 1) % 2 == 0) or (idx == batchsize - 1) and idx>1:
                    self.save_contrib_scores_chunk(contrib_dict)
                    self.save_input_chunk(input_dict, seq_dict)
                    print(f"Batches {idx} and {idx+1} input and contribution data saved to h5 file")

    def custom_collate_fn(self, batch):
        seq_tensors = [item[0] for item in batch]
        original_seqs = [item[1] for item in batch]
        seq_tensors = torch.stack(seq_tensors, dim=0)
        return seq_tensors, original_seqs

    def save_contrib_scores_chunk(self, contrib_dict):
        for idx, score_tensor in contrib_dict.items():
            self.group2.create_dataset(f'contrib_score_Batch_{idx}', data=score_tensor.detach().cpu().numpy())
        contrib_dict.clear()

    def save_input_chunk(self, input_dict, seq_dict):
        for idx in input_dict.keys():
            batch_input = input_dict[idx]
            original_seqs = seq_dict[idx]
            self.group3.create_dataset(f'input_Batch_{idx}', data=batch_input.detach().cpu().numpy())
            dt = h5py.special_dtype(vlen=str)
            self.group3.create_dataset(f'sequences_Batch_{idx}', data=np.array(original_seqs, dtype=object), dtype=dt)
        input_dict.clear()
        seq_dict.clear()

    def get_RBP_model(self):
        print('\n')
        print("-----Loading the RBP model-----")

        model = RBPResTransModels.RBPModel(record_attn=False)   
        model = torch.compile(model)
        model.load_state_dict(torch.load(self.RBPmodel_root))
        model.to(self.device)
        model.eval()
        RBP_model = model
        if isinstance(RBP_model, nn.Module):
            print("Model loading successfull!")
            return RBP_model.to(self.device)
        else: 
            raise ValueError("Model loading unsuccessfull.")

def main():

    parser = argparse.ArgumentParser(description="Contribution score attribution, given model and input data")

    # Adding arguments
    parser.add_argument("--out_h5_fname", type=str, help="Path to the output contribution scores stored in h5 file.", default=None)
    parser.add_argument("--RBPname", type=str, help="RBP name for contribution score calculation.", default=None)
    parser.add_argument("--RBPmodel", type=str, help="File path to the RBP model pth file.", default=None)
    parser.add_argument("--fasta_file", type=str, help="Path to the input fasta file containing sequences.", default=None)
    parser.add_argument("--contrib_function", type=str, help="Contribution score calculation algorithm, currently available: IG/SA", default="IG")
    parser.add_argument("--device", type=str, help="cuda option", default="cuda:0")
    parser.add_argument("--record_global_contrib", action='store_true', help="whether to record contribution score for 4 bases or not", default=False)

    # Parse the arguments
    args = parser.parse_args()

    if args.RBPname is None:
        parser.error("--RBPname is a required argument. Please provide the correct name.")
    if args.out_h5_fname is None:
        parser.error("--out_h5_fname is a required argument. Please provide the correct output file path.")
    if args.RBPmodel is None:
        parser.error("--RBPmodel is a required argument. Please provide the correct file path to the RBP model.")
    if args.fasta_file is None:
        parser.error("--fasta_file is a required argument. Please provide the correct path to the input fasta file.")

    # Create instance with command line arguments
    contrib_h5 = ContribH5(
        filename=args.out_h5_fname,
        RBPname=args.RBPname,
        RBPmodel_root=args.RBPmodel,
        fasta_file=args.fasta_file,
        record_global=args.record_global_contrib,
        contrib_function=args.contrib_function,
        device=args.device
    )

if __name__ == '__main__':
    main()