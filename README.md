# TransRBP

This is the official implementation of our paper:
### In-silico modeling of RBP binding disentangles m6A-RBP interaction and reveals genetic disease mechanisms
Jianche Liu#, Xinlu Zhu#, Yang Yin#, Zhoutong Xu, Jialin He, Xushen Xiong*

## Introduction
TransRBP, a novel Transformer-based deep learning framework for modeling RBP binding profiles at a base-resolution fromwith RNA sequences.
![overview](https://github.com/IAMZhuXinlu/TransRBP/blob/main/overview.png)

## Requirements and Installation

To use TransRBP, you need the following requisites:
- Python 3.9
- PyTorch 2.0.1+cu117
- Other required Python libraries (requirements.txt)

You can run the following command to install all the required packages for TransRBP.

```bash
conda create -n transrbp python=3.9
conda activate transrbp
```

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

```bash
pip install -r requirements.txt
```

## Data directory

The input data is expected to be organized as follows. Here we shows the data of RBP DDX43 as the example.

```docs
root
|
├── hg38
|   ├── chr1.fa.gz
|   ├── chr2.fa.gz
|   ├── chr3.fa.gz
|   ├── chr4.fa.gz
|   ├── chr5.fa.gz
|   ├── chr6.fa.gz
|   ├── chr7.fa.gz
|   ├── chr8.fa.gz
|   ├── chr9.fa.gz
|   ├── chr10.fa.gz
|   ├── chr11.fa.gz
|   ├── chr12.fa.gz
|   ├── chr13.fa.gz
|   ├── chr14.fa.gz
|   ├── chr15.fa.gz
|   ├── chr16.fa.gz
|   ├── chr17.fa.gz
|   ├── chr18.fa.gz
|   ├── chr19.fa.gz
|   ├── chr20.fa.gz
|   ├── chr21.fa.gz
|   └── chr22.fa.gz
|── m6A
|   ├── MeRIP_signal_minus.bw
|   └── MeRIP_signal_plus.bw
|  
└── RBP
    ├── DDX43
    |   ├── bindingpeak.bed
    |   ├── bindingsignal_minus.bw
    |   └── bindingsignal_plus.bw
    └── *...
```

## Training

To train TransRBP from scratch, you can run the following command:

```docs
python -m TransRBP.training.main [options]

Options:
-h --help   show the help message and exit.
--seed  Random seed for training. Default: 43.
--bs    The batch size for data loading. Default: 64.
--lr    The learning rate for the optimizer. Default: 1e-3.
--RBP   RBP for which the model is trained.
--whether_grad_norm_clip    Flag to specify whether gradient norm clipping is used. Default: 1.
--grad_norm_clip    The value of gradient norm clip. Default: 5.
--tb_dir    The directory to save tensorboard logs.
--save_model_dir    The directory to save the trained model.
--data_root Root path to training data.
--max_epoch The max epochs for training. Default: 100.
--tol_epoch The tolerance epochs for early stop. Default: 10

```

To monitor the training process, you can use tensorboard at http://localhost:6006 by default:

```bash
tensorboard --logdir tb_dir/RBP
```

## Usage
### Contribution Score

Contribution scores calculation `ContribH5` takes input sequence in Fasta file and returns the h5 file storing the calculated contribution scores of the sequences and the corresponding inputs.

```docs
python -m TransRBP.utils.ContribH5 [options]

Options:
--out_h5_fname: Path to the output H5 file where the contribution scores and input sequences will be stored. If not specified, no file will be saved.
--RBPname: Name of the RBP for which contribution scores are being calculated.
--RBPmodel: File path to the trained RBP model (.pth file).
--fasta_file: Path to the input fasta file containing RNA sequences to be analyzed.
--contrib_function: Specifies the algorithm used for contribution score calculation. Options are "IG" (Integrated Gradients) or "SA" (Saliency Map). Default is "IG".
--device: Specifies the computing device to use, e.g., "cuda:0" for CUDA on GPU 0. Default is "cuda:0".
--record_global_contrib: Whether record the contribution scores on the 4 bases or just on the exact base in the input. Default is False.

```

### In-silico Mutation Scoring

The `VariantImpact` accepts a csv file as a positional argument containing variant information, where each line specifies a variant in the following format:
`<chr>  <pos>   <strand>    <ref_allele>    <alt_allele>`

For example, the csv file in the structure:
```docs
chr1,122891,+,A,G
chr7,92456384,+,A,C  
```

```docs
python -m TransRBP.utils.VariantImpact [options]

Options:
--input_csv: Path to the input CSV file that contains the variants to be scored. 
--reference_genome: Path to the reference genome FASTA file.
--model_path: Path to the trained RBP model file (.pth file).
--output_tsv: Path to the output TSV file where the scored variants will be saved.
--device: Specifies the computing device for model inference. Default is "cuda:0". Options include "cpu" or "cuda" for GPU acceleration.
--batch_size: Batch size for processing variants during model inference. Default is 32.

```


## Citation
If you want to use TransRBP in your research, please cite:

```doc

```


## License
This project is licensed under MIT License.

## Contact
For any questions or inquiries, please contact xiongxs@zju.edu.cn.
