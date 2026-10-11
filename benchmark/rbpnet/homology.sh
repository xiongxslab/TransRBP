#!/bin/sh

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <RBP_LIST> <FASTA_DIR>"
    exit 1
fi

LIST_FILE="$1"
FASTA_DIR="$2"
DB_DIR="${FASTA_DIR}/blast_db"

mkdir -p "$DB_DIR"

while IFS= read -r rbp; do

    [ -z "$rbp" ] && continue

    echo "=================================================="
    echo "${rbp}"

    TRAIN_FA="${FASTA_DIR}/${rbp}_train.fa"
    TEST_FA="${FASTA_DIR}/${rbp}_test.fa"
    CLEAN_TEST_FA="${FASTA_DIR}/${rbp}_test.nohomo.fa"
    HOMOLOGY_ID="${FASTA_DIR}/${rbp}_test_homology.ids"
    DB_PREFIX="${DB_DIR}/${rbp}_train_db"
    BLAST_OUT="${FASTA_DIR}/${rbp}_blast_results.txt"

    makeblastdb \
        -in "$TRAIN_FA" \
        -dbtype nucl \
        -out "$DB_PREFIX"

    blastn \
        -query "$TEST_FA" \
        -db "$DB_PREFIX" \
        -out "$BLAST_OUT" \
        -outfmt "6 qseqid sseqid pident length qlen slen evalue bitscore" \
        -evalue 1e-3 \
        -num_threads 32

    Homology_seqs=$(
        awk '$3 >= 80 && ($4/$5) >= 0.5 {print $1}' "$BLAST_OUT" \
        | sort -u \
        | wc -l
    )

    awk '$3 >= 80 && ($4 / $5) >= 0.5 {print $1}' "$BLAST_OUT" \
        | sort -u \
        > "$HOMOLOGY_ID"

    awk '
        NR==FNR {
            remove[$1]=1
            next
        }
        /^>/ {
            id=substr($0, 2)
            keep = !(id in remove)
        }
        keep {
            print
        }
    ' "$HOMOLOGY_ID" "$TEST_FA" > "$CLEAN_TEST_FA"

    echo "${rbp}: homologous=${Homology_seqs}"

done < "$LIST_FILE"
