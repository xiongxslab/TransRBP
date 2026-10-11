#!/bin/bash
# =============================================================================
# GLORI single-nucleotide m6A track and GLORI-model inputs (Extended Data Fig. 3f).
#
#   1. build_glori_consensus.py  replicate consensus per cell line (HEK293T, HeLa)
#   2. build_GLORI_union.py      union of HeLa and HEK293T sites
#   3. filter_by_rnaseq.py       keep sites expressed in K562 (MeRIP input CPM > 0.5)
#   4. make_bigwig.py            strand-specific sparse BigWig of the GLORI m6A ratio
#   5. make_h5_nohomo.py         per-RBP HDF5 inputs with the GLORI m6A channel
#
# Inputs: GEO GSE210563 site tables (GSM6432590/91 HEK293T, GSM6432595/96 HeLa),
#         MeRIP_control_merged.bw (preprocessing/merip_bigwig.sh), hg38.chrom.sizes,
#         no-homology split BEDs (data/), RBP BigWigs (bw_root).
#
# Usage (from the repo root):
#   bash preprocessing/glori/run_preparation.sh <GSE210563_RAW_dir> <MeRIP_control_merged.bw> \
#        <hg38.chrom.sizes> <glori_out_dir> <bw_root> [rbp_list.txt] [h5_out_dir]
# Output: <bw_root>/m6A_GLORI_K562proxy_plus.bw, _minus.bw (+ <h5_out_dir>/{RBP}.h5)
# =============================================================================
set -euo pipefail
RAW=$1
CONTROL_BW=$2
CHROM_SIZES=$3
OUT=$4
BW_ROOT=$5
RBP_LIST=${6:-}
H5_OUT=${7:-h5_cache_GLORI}
M6A_TRACK=m6A_GLORI_K562proxy
mkdir -p "$OUT" "$BW_ROOT"

python preprocessing/glori/build_glori_consensus.py \
    --rep1 "$RAW/GSM6432590_293T-mRNA-1_35bp_m2.totalm6A.FDR.csv.gz" \
    --rep2 "$RAW/GSM6432591_293T-mRNA-2_35bp_m2.totalm6A.FDR.csv.gz" \
    --output "$OUT/GLORI_293T_consensus.tsv"
python preprocessing/glori/build_glori_consensus.py \
    --rep1 "$RAW/GSM6432595_Hela-1_35bp_m2.totalm6A.FDR.csv.gz" \
    --rep2 "$RAW/GSM6432596_Hela-2_35bp_m2.totalm6A.FDR.csv.gz" \
    --output "$OUT/GLORI_HeLa_consensus.tsv"
python preprocessing/glori/build_GLORI_union.py \
    --hela "$OUT/GLORI_HeLa_consensus.tsv" --t293 "$OUT/GLORI_293T_consensus.tsv" \
    --output "$OUT/GLORI_union_consensus.tsv"
python preprocessing/glori/filter_by_rnaseq.py \
    --input "$OUT/GLORI_union_consensus.tsv" --bw "$CONTROL_BW" \
    --output "$OUT/GLORI_K562proxy_filtered.tsv" --threshold 0.5
python preprocessing/glori/make_bigwig.py \
    --input "$OUT/GLORI_K562proxy_filtered.tsv" --bw_root "$BW_ROOT" \
    --chrom_sizes "$CHROM_SIZES" --track_name "$M6A_TRACK"

if [ -n "$RBP_LIST" ]; then
    while IFS= read -r RBP; do
        [ -z "$RBP" ] && continue
        python training/make_h5_nohomo.py --rbp "$RBP" --bed_root data/ --bw_root "$BW_ROOT" \
            --chrom_root hg38/dna_sequence --m6A_track "$M6A_TRACK" --output_dir "$H5_OUT"
    done < "$RBP_LIST"
fi
