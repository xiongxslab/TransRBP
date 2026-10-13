#!/bin/bash
# =============================================================================
# Multi-task pretraining on 32 m6A-associated RBPs, then per-RBP fine-tuning
# (Extended Data Fig. 3a-b). Hyperparameters are those used in the paper.
#
# Usage (from the repo root):
#   bash training/pretrain/run_pretrain_finetune.sh \
#        rbp_list.txt /path/to/RBP_bed /path/to/bamCompare_bw hg38/dna_sequence out_dir cuda:0 \
#        m6A_bw/m6A_plus.bw m6A_bw/m6A_minus.bw
#
# bw_root must contain {RBP}_plus.bw / {RBP}_minus.bw for every RBP. The last two (optional) arguments are
# the m6A BigWigs (the m6A input channel); default: bw_root/m6A_plus.bw and bw_root/m6A_minus.bw.
# =============================================================================
set -euo pipefail

RBP_LIST=$1
BED_ROOT=$2
BW_ROOT=$3
CHROM_ROOT=$4
OUT=$5
DEVICE=${6:-cuda:0}
M6A_PLUS=${7:-$BW_ROOT/m6A_plus.bw}
M6A_MINUS=${8:-$BW_ROOT/m6A_minus.bw}

mkdir -p "$OUT/h5_cache/finetune" "$OUT/models_pretrain" "$OUT/models_finetune" "$OUT/metrics"
H5=$OUT/h5_cache/32rbp_m6a_100k.h5
PRETRAIN=$OUT/models_pretrain/pretrained_32rbp_m6a.pth

# ---- Step 1: pre-compute pretraining windows -> HDF5 (100k windows total) ----
if [ ! -f "$H5" ]; then
    python -u training/pretrain/preprocess_h5.py \
        --rbp_list      "$RBP_LIST" \
        --bed_root      "$BED_ROOT" \
        --bw_root       "$BW_ROOT" \
        --chrom_root    "$CHROM_ROOT" \
        --output        "$H5" \
        --total_windows 100000 \
        --m6A_binding   1 \
        --m6A_bw_plus   "$M6A_PLUS" \
        --m6A_bw_minus  "$M6A_MINUS"
fi

# ---- Step 2: multi-task pretraining ----
python -u training/pretrain/main_pretrain.py \
    --h5_path       "$H5" \
    --rbp_list      "$RBP_LIST" \
    --bed_root      "$BED_ROOT" \
    --bs            64 \
    --lr            1e-3 \
    --random_seed   43 \
    --whether_grad_norm_clip 1 \
    --grad_norm_clip 5 \
    --max_epoch     100 \
    --tol_epoch     10 \
    --device        "$DEVICE" \
    --tb_dir        "$OUT/metrics/tb_pretrain" \
    --save_model_dir "$OUT/models_pretrain" \
    --output_csv    "$OUT/metrics/pretrain_test_metrics.csv"

python -u training/pretrain/eval_pretrain.py \
    --model_path "$PRETRAIN" \
    --h5_path    "$H5" \
    --output_csv "$OUT/metrics/pretrain_test_metrics_eval.csv" \
    --device     "$DEVICE"

# ---- Step 3: per-RBP fine-tuning data (20k windows per RBP) ----
python -u training/pretrain/preprocess_finetune_h5.py \
    --rbp_list   "$RBP_LIST" \
    --bed_root   "$BED_ROOT" \
    --bw_root    "$BW_ROOT" \
    --chrom_root "$CHROM_ROOT" \
    --out_dir    "$OUT/h5_cache/finetune" \
    --total_windows 20000 \
    --m6A_bw_plus  "$M6A_PLUS" \
    --m6A_bw_minus "$M6A_MINUS"

# ---- Step 4: fine-tune the task head for each RBP ----
while IFS= read -r rbp; do
    [ -z "$rbp" ] && continue
    python -u training/pretrain/finetune_singletask.py \
        --pretrain_path "$PRETRAIN" \
        --rbp           "$rbp" \
        --h5_path       "$OUT/h5_cache/finetune/${rbp}_20k.h5" \
        --freeze_mode   head_only \
        --bs            64 \
        --lr            1e-4 \
        --random_seed   43 \
        --whether_grad_norm_clip 1 \
        --grad_norm_clip 5 \
        --max_epoch     50 \
        --tol_epoch     10 \
        --device        "$DEVICE" \
        --save_model_dir "$OUT/models_finetune/$rbp" \
        --tb_dir         "$OUT/metrics/tb_finetune/$rbp" \
        --output_csv     "$OUT/metrics/finetune_test_metrics_${rbp}.csv"
done < "$RBP_LIST"
