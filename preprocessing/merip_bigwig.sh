#!/bin/bash
# =============================================================================
# m6A input channel from K562 MeRIP-seq (GSE137752) — produces the Zenodo
# m6A_bw/m6A_plus.bw and m6A_minus.bw used by the released models.
#
# Steps (as used for the paper):
#   1. Merge MeRIP IP replicates and RNA-seq input (control) replicates
#   2. Split by alignment strand                        samtools view -F 16 / -f 16
#   3. Sort + index
#   4. IP over input, per strand:
#        bamCompare --binSize 1 --operation log2 --scaleFactorsMethod readCount
#                   (pseudocount = deepTools default, 1; no blacklist)
#      Negative log2 values are clipped to 0 when the BigWig is read.
#   5. Input-only CPM coverage (both strands), used as the K562 expression
#      filter for the GLORI track (preprocessing/glori/):
#        bamCoverage --binSize 1 --normalizeUsing CPM
#
# Usage:
#   bash preprocessing/merip_bigwig.sh <out_dir> <IP_rep1.bam,IP_rep2.bam,...> <input_rep1.bam,input_rep2.bam,...>
#
# Output:
#   <out_dir>/m6A_plus.bw  <out_dir>/m6A_minus.bw  <out_dir>/MeRIP_control_merged.bw
#
# Alignment: STAR (v2.7.11) to hg38. Requires samtools and deepTools (v3.5.4).
# =============================================================================
set -euo pipefail

if [ "$#" -ne 3 ]; then
    sed -n '2,25p' "$0"; exit 1
fi

OUT_DIR=$1
IFS=',' read -r -a IP_BAMS    <<< "$2"
IFS=',' read -r -a INPUT_BAMS <<< "$3"
THREADS=${THREADS:-8}

TMP="$OUT_DIR/tmp_merip"
mkdir -p "$OUT_DIR" "$TMP"

# Step 1: merge replicates
samtools merge -f "$TMP/ip.bam"    "${IP_BAMS[@]}"
samtools merge -f "$TMP/input.bam" "${INPUT_BAMS[@]}"

# Steps 2-3: strand split, sort, index
for lib in ip input; do
    samtools sort -o "$TMP/${lib}_sorted.bam" "$TMP/$lib.bam"
    samtools index "$TMP/${lib}_sorted.bam"
    samtools view -F 16 -b -o "$TMP/${lib}_plus.bam"  "$TMP/$lib.bam"
    samtools view -f 16 -b -o "$TMP/${lib}_minus.bam" "$TMP/$lib.bam"
    for strand in plus minus; do
        samtools sort -o "$TMP/${lib}_${strand}_sorted.bam" "$TMP/${lib}_${strand}.bam"
        samtools index "$TMP/${lib}_${strand}_sorted.bam"
    done
done

# Step 4: log2(IP / input) per strand -> m6A channel
for strand in plus minus; do
    bamCompare \
        -b1 "$TMP/ip_${strand}_sorted.bam" \
        -b2 "$TMP/input_${strand}_sorted.bam" \
        -o  "$OUT_DIR/m6A_${strand}.bw" \
        --binSize 1 \
        --operation log2 \
        --scaleFactorsMethod readCount \
        --numberOfProcessors "$THREADS" \
        -of bigwig
done

# Step 5: input CPM coverage (expression proxy for the GLORI filter)
bamCoverage \
    --bam "$TMP/input_sorted.bam" \
    -o "$OUT_DIR/MeRIP_control_merged.bw" \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors "$THREADS" \
    -of bigwig

rm -rf "$TMP"
echo "Done: $OUT_DIR/m6A_plus.bw  $OUT_DIR/m6A_minus.bw  $OUT_DIR/MeRIP_control_merged.bw"
