#!/usr/bin/env bash
# Download the ANNcoilsDataset (Rodriguez-Fuentes et al., IEEE TPEL 2026) into data/raw/.
set -euo pipefail
DEST="data/raw"
if [ -f "$DEST/ANNCoilsFEA.csv" ]; then
    echo "Dataset already present: $DEST/ANNCoilsFEA.csv"; exit 0
fi
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/alvarordf/ANNcoilsDataset.git "$TMP/ANNcoilsDataset"
cp "$TMP"/ANNcoilsDataset/ANNCoilsFEA.* "$DEST"/
rm -rf "$TMP"
echo "Dataset copied to $DEST"
