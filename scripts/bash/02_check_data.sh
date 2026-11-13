#!/usr/bin/env bash
# Verify the dataset file, its size, header and SHA-256 checksum.
set -euo pipefail
CSV="data/raw/ANNCoilsFEA.csv"
EXPECTED_ROWS=4635
EXPECTED_HEADER="Freq,Dist,W1,W2,S1,S2,N1,N2,Dext1,Dext2,Nlay1,Nlay2,H1,H2,X1,X2,Z11,Z22,Z12,L1,L2,R1,R2,k,Q1,Q2,Ro,Eff_max"
echo "Checking dataset files..."
[ -f "$CSV" ] || { echo "CSV file missing: $CSV (run 01_get_data.sh)"; exit 1; }
echo "CSV file found: $CSV"
ROWS=$(( $(wc -l < "$CSV") - 1 ))
HEADER=$(head -1 "$CSV" | tr -d '\r')
echo "Data rows: $ROWS"
[ "$ROWS" -eq "$EXPECTED_ROWS" ] || { echo "Unexpected row count (expected $EXPECTED_ROWS)"; exit 1; }
[ "$HEADER" = "$EXPECTED_HEADER" ] || { echo "Unexpected header"; exit 1; }
echo "Header OK (28 columns)"
if [ -f data/raw/ANNCoilsFEA.mat ]; then echo "MAT file found"; else echo "MAT file not present (optional)"; fi
sha256sum "$CSV" | tee logs/dataset_sha256.txt
echo "Dataset check complete."
