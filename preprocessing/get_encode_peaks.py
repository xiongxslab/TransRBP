"""
Download the K562 eCLIP peaks of each RBP from ENCODE (Supplementary Table 1).

For each experiment the released GRCh38 replicated peak file (bed narrowPeak,
biological replicates 1 and 2) is downloaded; peaks on chr1-chr22 are kept and
written as BED6 ({out_dir}/{RBP}.bed). These are the peak files used for the
no-homology splits (training/make_nohomo_split.py), the RBPNet benchmark and the
ClinVar analysis (disease/).

Example:
    python preprocessing/get_encode_peaks.py \\
        --accessions preprocessing/encode_eclip_accessions.tsv \\
        --rbp_list rbp_list.txt --out_dir RBP_bed/
"""

import argparse
import gzip
import io
import json
import os
import re
import urllib.request

ENCODE = 'https://www.encodeproject.org'
AUTOSOMES = re.compile(r'^chr([1-9]|1[0-9]|2[0-2])$')

parser = argparse.ArgumentParser(description='Download ENCODE K562 eCLIP replicated peaks')
parser.add_argument('--accessions', required=True, help='TSV with columns RBP, ENCODE_experiment')
parser.add_argument('--rbp_list',   required=True, help='Text file, one RBP name per line')
parser.add_argument('--out_dir',    required=True)
args = parser.parse_args()

with open(args.accessions) as f:
    next(f)
    experiment = dict(line.rstrip('\n').split('\t')[:2] for line in f if line.strip())
with open(args.rbp_list) as f:
    rbps = [line.strip() for line in f if line.strip()]
os.makedirs(args.out_dir, exist_ok=True)

for rbp in rbps:
    req = urllib.request.Request(f'{ENCODE}/experiments/{experiment[rbp]}/?format=json&frame=embedded',
                                 headers={'Accept': 'application/json'})
    files = json.load(urllib.request.urlopen(req, timeout=120))['files']
    peaks = [f for f in files if f.get('file_format') == 'bed' and f['output_type'] == 'peaks'
             and f.get('assembly') == 'GRCh38' and sorted(f.get('biological_replicates', [])) == [1, 2]
             and f.get('status') == 'released']
    if len(peaks) != 1:
        raise RuntimeError(f'{rbp}: expected one replicated peak file, found {[p["accession"] for p in peaks]}')
    acc = peaks[0]['accession']
    raw = urllib.request.urlopen(f'{ENCODE}/files/{acc}/@@download/{acc}.bed.gz', timeout=300).read()
    n = 0
    with gzip.open(io.BytesIO(raw), 'rt') as fin, open(os.path.join(args.out_dir, f'{rbp}.bed'), 'w') as fout:
        for line in fin:
            fields = line.rstrip('\n').split('\t')
            if AUTOSOMES.match(fields[0]):
                fout.write('\t'.join(fields[:6]) + '\n')
                n += 1
    print(f'{rbp}: {experiment[rbp]} {acc} -> {n} peaks')
