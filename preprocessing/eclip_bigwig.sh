#!/bin/bash
# =============================================================================
# Strand-specific eCLIP signal BigWig for one RBP (training labels).
#
# Steps (as used for the paper):
#   1. Paired-end libraries: keep read 2 only           samtools view -f 128
#   2. Merge the two eCLIP replicates (and SMInput files, if more than one)
#   3. Split by alignment strand                        samtools view -F 16 / -f 16
#   4. Sort + index
#   5. IP over SMInput, per strand:
#        bamCompare --binSize 1 -bl <blacklist> --operation log2
#                   --scaleFactorsMethod readCount        (pseudocount = deepTools default, 1)
#      Negative log2 values are clipped to 0 when the BigWig is read for training.
#
# Usage:
#   bash preprocessing/eclip_bigwig.sh <RBP> <PE|SE> <blacklist.bed> <out_dir> \
#        <IP_rep1.bam> <IP_rep2.bam> <SMInput.bam> [SMInput_2.bam]
#
# Output:
#   <out_dir>/<RBP>_plus.bw  <out_dir>/<RBP>_minus.bw     (use as --bw_root)
#
# Requires samtools (v1.5+) and deepTools (v3.5.4).
# =============================================================================
set -euo pipefail

if [ "$#" -lt 7 ]; then
    sed -n '2,24p' "$0"; exit 1
fi

RBP=$1
LAYOUT=$2
BLACKLIST=$3
OUT_DIR=$4
IP1=$5
IP2=$6
shift 6
SMI=("$@")
THREADS=${THREADS:-8}

TMP="$OUT_DIR/tmp_${RBP}"
mkdir -p "$OUT_DIR" "$TMP"

# Step 1: read 2 only for paired-end data
select_reads() {   # in.bam out.bam
    if [ "$LAYOUT" = "PE" ]; then
        samtools view -f 128 -b -o "$2" "$1"
    else
        cp "$1" "$2"
    fi
}

select_reads "$IP1" "$TMP/ip_1.bam"
select_reads "$IP2" "$TMP/ip_2.bam"
SMI_SEL=()
for i in "${!SMI[@]}"; do
    select_reads "${SMI[$i]}" "$TMP/smi_$i.bam"
    SMI_SEL+=("$TMP/smi_$i.bam")
done

# Step 2: merge replicates
samtools merge -f "$TMP/ip.bam" "$TMP/ip_1.bam" "$TMP/ip_2.bam"
if [ "${#SMI_SEL[@]}" -gt 1 ]; then
    samtools merge -f "$TMP/smi.bam" "${SMI_SEL[@]}"
else
    mv "${SMI_SEL[0]}" "$TMP/smi.bam"
fi

# Steps 3-4: strand split, sort, index
for lib in ip smi; do
    samtools view -F 16 -b -o "$TMP/${lib}_plus.bam"  "$TMP/$lib.bam"
    samtools view -f 16 -b -o "$TMP/${lib}_minus.bam" "$TMP/$lib.bam"
    for strand in plus minus; do
        samtools sort -o "$TMP/${lib}_${strand}_sorted.bam" "$TMP/${lib}_${strand}.bam"
        samtools index "$TMP/${lib}_${strand}_sorted.bam"
    done
done

# Step 5: log2(IP / SMInput) per strand
for strand in plus minus; do
    bamCompare \
        -b1 "$TMP/ip_${strand}_sorted.bam" \
        -b2 "$TMP/smi_${strand}_sorted.bam" \
        -o  "$OUT_DIR/${RBP}_${strand}.bw" \
        --binSize 1 \
        -bl "$BLACKLIST" \
        --operation log2 \
        --scaleFactorsMethod readCount \
        --numberOfProcessors "$THREADS" \
        -of bigwig
done

rm -rf "$TMP"
echo "Done: $OUT_DIR/${RBP}_plus.bw  $OUT_DIR/${RBP}_minus.bw"
