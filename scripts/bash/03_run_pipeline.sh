#!/usr/bin/env bash
# Execute the full analysis notebook headlessly and export an HTML report.
# Runtime: roughly 15–40 minutes depending on CPU (Gaussian-process fits dominate).
set -euo pipefail
NB="wpt_complete_analysis.ipynb"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/run_pipeline_${STAMP}.log"
echo "Running $NB ... (log: $LOG)"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$NB" 2>&1 | tee "$LOG"
jupyter nbconvert --to html "$NB" --output wpt_complete_analysis.html 2>&1 | tee -a "$LOG"
echo "Done. Outputs in results_python/ and paper/"
