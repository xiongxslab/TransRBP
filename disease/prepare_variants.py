"""
ClinVar disease variants near RBP binding sites, with their distance to m6A
(the variant table without the KLD columns; score with disease/score_variants.py).

ClinVar: GRCh38 VCF release 2024-07-08 (clinvar_20240708.vcf.gz), downloaded
11 July 2024 from https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/.

Steps:
  1. Parse the VCF INFO fields (CLNDN, molecular consequence, CLNVC, ORIGIN, CLNSIG).
  2. Select records of 26 brain / heart / lung disease traits by keyword
     (case-insensitive match on the whole record).
  3. Keep single-nucleotide records (REF and ALT of length 1).
  4. Variants are placed on the '+' strand (0-based BED).
  5. For each RBP keep variants within 800 bp of its eCLIP peaks on the same
     strand (bedtools slop -b 800 + intersect -s; i.e. plus-strand peaks), and
     drop duplicate variant-RBP records.
  6. Distance from each variant to the nearest plus-strand MeRIP-seq m6A peak
     (|variant start - peak start|; classes <50bp, 50-200bp, 200-800bp, >800bp).
     --m6A_peaks: BED6 of K562 MeRIP-seq peaks (MACS2) with the strand of the
     overlapping gene annotation, generated as described in Methods.
  7. Related tissue: muscular dystrophy and atrophy -> Muscle,
     small vessel disease -> Heart.

Output: one TSV, rows ordered by --rbp_list, columns
    RBPname, Chr, Position, Strand, Ref, Alt, Mutation_Type, Related_Tissue,
    Disease, Distance2m6A, Distance_Class
(Position is 1-based; Ref/Alt in RNA alphabet). Score it with
disease/score_variants.py to obtain KLD, the RBP-specific cutoff and the annotation.

Example:
    python disease/prepare_variants.py \\
        --vcf        clinvar_20240708.vcf.gz \\
        --rbp_list   rbp_list.txt \\
        --peak_dir   RBP_bed/ \\
        --chrom_sizes hg38/annotation/hg38.chrom.sizes \\
        --m6A_peaks  MeRIP_peaks_stranded.bed \\
        --out_dir    disease_out/
"""

import argparse
import bisect
import collections
import gzip
import io
import os
import re

import pandas as pd

# (tissue, disease, keyword pattern) — keywords as used for the paper
TRAITS = [
    ('Brain', 'AD',                                    'alzhei'),
    ('Brain', 'SCZ',                                   'schizophrenia'),
    ('Heart', 'Cardiomyopathy',                        'Cardiomyopathy'),
    ('Heart', 'Cardiac_arrhythmia',                    'Cardiac_arrhythmia'),
    ('Brain', 'PD',                                    'Parkinson'),
    ('Brain', 'Depression',                            'Depression|depressive_disorder'),
    ('Brain', 'atrophy',                               'Brain_atrophy'),
    ('Brain', 'small_vessel_disease',                  'Brain_small_vessel_disease'),
    ('Brain', 'stem_glioma',                           'Brainstem_glioma'),
    ('Brain', 'Neurodevelopmental_disorder',           'Neurodevelopmental_disorder'),
    ('Heart', 'Congenital_heart_disease',              'Congenital_heart'),
    ('Heart', 'Coronary_heart_disease',                'Coronary_heart_disease'),
    ('Heart', 'Hypertension',                          'Hypertension'),
    ('Heart', 'Heart_failure',                         'Heart_failure'),
    ('Brain', 'Muscular_dystrophy',                    'Muscular_dystrophy'),
    ('Brain', 'Neurodegeneration',                     'Neurodegeneration'),
    ('Brain', 'Autism',                                'Autism'),
    ('Brain', 'Anxiety',                               'Anxiety'),
    ('Brain', 'ALS',                                   'Amyotrophic_Lateral_Sclerosis'),
    ('Heart', 'Progressive_familial_heart_block',      'Progressive_familial_heart_block'),
    ('Heart', 'Hypoplastic_left_heart_syndrome',       'Hypoplastic_left_heart_syndrome'),
    ('Heart', 'Conotruncal_heart_malformations',       'Conotruncal_heart_malformations'),
    ('Heart', 'Conduction_disorder_of_the_heart',      'Conduction_disorder_of_the_heart'),
    ('Lung',  'Chronic_obstructive_pulmonary_disease', 'Chronic_obstructive'),
    ('Lung',  'Asthma',                                'Asthma'),
    ('Lung',  'Pulmonary_Fibrosis',                    'Pulmonary_fibrosis'),
]
TISSUE_REMAP = {'Muscular_dystrophy': 'Muscle', 'atrophy': 'Muscle', 'small_vessel_disease': 'Heart'}


# ── 1. parse ClinVar ──────────────────────────────────────────────────────────

def parse_vcf(vcf):
    pats = [re.compile(p) for p in (r'CLNDN=(\S+?);', r'MC=\S+?\|(\S+?);', r'CLNVC=(\S+?);',
                                    r'ORIGIN=(\d+)', r'CLNSIG=(\S+?);')]
    lines = ['CHROM\tPOS\tID\tREF\tALT\tCLNDN\tmutation_type\tCLNVC\tORIGIN\tCLNSIG']
    with gzip.open(vcf, 'rt') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                continue
            sp = line.split('\t')
            info = [(m.group(1) if (m := p.search(line)) else '') for p in pats]
            lines.append('\t'.join(sp[:5] + info))
    return lines


# ── 2-4. traits, SNVs, BED ────────────────────────────────────────────────────

def select_snvs(parsed):
    rows = ['CHROM\tPOS\tID\tREF\tALT\tCLNDN\tmutation_type\tCLNVC\tORIGIN\tCLNSIG\tTissue\tDisease']
    # trait files were combined in case-insensitive '{Tissue}.{Disease}' order
    for tissue, disease, pattern in sorted(TRAITS, key=lambda t: f'{t[0]}.{t[1]}'.upper()):
        regex = re.compile(pattern, re.IGNORECASE)
        for line in parsed:
            if not regex.search(line):
                continue
            fields = line.split()                                   # whitespace split (as awk)
            fields += [''] * (10 - len(fields))
            if len(fields[3]) == 1 and len(fields[4]) == 1:
                rows.append('\t'.join(fields[:10] + [tissue, disease]))
    return rows


def to_bed(rows):
    clinvar_df = pd.read_csv(io.StringIO('\n'.join(rows) + '\n'), sep='\t')
    bed = pd.DataFrame()
    bed['chrom']  = 'chr' + clinvar_df['CHROM']
    bed['start']  = clinvar_df['POS'] - 1
    bed['end']    = clinvar_df['POS']
    bed['name']   = clinvar_df['Tissue']
    bed['score']  = '.'
    bed['strand'] = '+'
    bed['info1']  = clinvar_df['Disease']
    bed['info2']  = clinvar_df['mutation_type']
    bed['info3']  = clinvar_df['REF']
    bed['info4']  = clinvar_df['ALT']
    text = bed.to_csv(sep='\t', index=False, header=False)
    return [line.split('\t') for line in text.rstrip('\n').split('\n')]


# ── 5. variants near RBP peaks ────────────────────────────────────────────────

def intersect_peaks(bed, rbp_list, peak_dir, chrom_sizes):
    sizes = {}
    with open(chrom_sizes) as f:
        for line in f:
            ch, sz = line.split()[:2]
            sizes[ch] = int(sz)
    per_rbp = {}
    for rbp in rbp_list:
        peaks = collections.defaultdict(list)
        with open(os.path.join(peak_dir, f'{rbp}.bed')) as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                start = max(0, int(p[1]) - 800)                     # bedtools slop -b 800
                end   = min(sizes[p[0]], int(p[2]) + 800)
                peaks[(p[0], p[5])].append((start, end, [p[0], str(start), str(end)] + p[3:]))
        out = []
        for v in bed:                                               # intersect -wa -wb -s
            s, e = int(v[1]), int(v[2])
            for ps, pe, peak_fields in peaks.get((v[0], v[5]), ()):
                if ps < e and s < pe:
                    # fields 1-10 of the A+B record, whitespace-split as in the original awk step
                    out.append('\t'.join(v + peak_fields).split()[:10] + [rbp])
        per_rbp[rbp] = out
    combined = [row for rbp in sorted(per_rbp) for row in per_rbp[rbp]]
    seen, unique = set(), []
    for row in combined:                                            # de-duplicate, order kept
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


# ── 6. nearest m6A peak ───────────────────────────────────────────────────────

def nearest_m6A(unique, m6A_peaks):
    cols = ['Chromosome', 'Start', 'End', 'Tissue', 'Score', 'Strand', 'Disease',
            'Mutation_Type', 'REF', 'ALT', 'RBP']
    text = '\n'.join('\t'.join(r) for r in unique) + '\n'
    clinvar_df = pd.read_csv(io.StringIO(text), sep='\t', header=None)
    clinvar_df.columns = cols

    merip_df = pd.read_csv(m6A_peaks, sep='\t', header=None)
    merip_df.columns = ['Chromosome', 'Start', 'End', 'Peak', 'Score', 'Strand']
    merip_df = merip_df[merip_df['Strand'] == '+']
    m6a_positions = {chrom: sorted(merip_df[merip_df['Chromosome'] == chrom]['Start'].tolist())
                     for chrom in merip_df['Chromosome'].unique()}

    def find_nearest(chrom, pos):
        if chrom not in m6a_positions:
            return float('inf'), '>800bp'
        pos_list = m6a_positions[chrom]
        idx = bisect.bisect_left(pos_list, int(pos))
        nearest = float('inf')
        if idx < len(pos_list):
            nearest = abs(pos_list[idx] - pos)
        if idx > 0:
            nearest = min(nearest, abs(pos_list[idx - 1] - pos))
        if nearest < 50:
            return nearest, '<50bp'
        if nearest < 200:
            return nearest, '50-200bp'
        if nearest < 800:
            return nearest, '200-800bp'
        return nearest, '>800bp'

    clinvar_df['Distance2m6A'], clinvar_df['Distance_class'] = zip(
        *clinvar_df.apply(lambda row: find_nearest(row['Chromosome'], row['Start']), axis=1))
    return clinvar_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ClinVar variants near RBP peaks with distance to m6A')
    parser.add_argument('--vcf',         required=True, help='clinvar_20240708.vcf.gz')
    parser.add_argument('--rbp_list',    required=True, help='Text file, one RBP per line (output order)')
    parser.add_argument('--peak_dir',    required=True, help='Dir with {RBP}.bed eCLIP peaks')
    parser.add_argument('--chrom_sizes', required=True)
    parser.add_argument('--m6A_peaks',   required=True,
                        help='BED6 of strand-annotated K562 MeRIP-seq peaks (see Methods)')
    parser.add_argument('--out_dir',     required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.rbp_list) as f:
        rbp_list = [line.strip() for line in f if line.strip()]

    parsed = parse_vcf(args.vcf)
    print(f'ClinVar records: {len(parsed) - 1:,}')
    bed = to_bed(select_snvs(parsed))
    print(f'Disease-trait SNV records: {len(bed):,}')
    unique = intersect_peaks(bed, rbp_list, args.peak_dir, args.chrom_sizes)
    print(f'Variant-RBP pairs within 800 bp of peaks: {len(unique):,}')
    df = nearest_m6A(unique, args.m6A_peaks)

    rna = lambda s: s.replace('T', 'U') if s == 'T' else s
    table = pd.DataFrame({
        'RBPname':        df['RBP'],
        'Chr':            df['Chromosome'],
        'Position':       df['End'],
        'Strand':         df['Strand'],
        'Ref':            df['REF'].astype(str).map(rna),
        'Alt':            df['ALT'].astype(str).map(rna),
        'Mutation_Type':  df['Mutation_Type'],
        'Related_Tissue': [TISSUE_REMAP.get(d, t) for d, t in zip(df['Disease'], df['Tissue'])],
        'Disease':        df['Disease'],
        'Distance2m6A':   df['Distance2m6A'],
        'Distance_Class': df['Distance_class'],
    })
    order = {rbp: i for i, rbp in enumerate(rbp_list)}
    table = table.iloc[sorted(range(len(table)), key=lambda i: order[table['RBPname'].iat[i]])]
    out = os.path.join(args.out_dir, 'clinvar_rbp_variants.tsv')
    table.to_csv(out, sep='\t', index=False)
    print(f'{len(table):,} variant-RBP records -> {out}')


if __name__ == '__main__':
    main()
