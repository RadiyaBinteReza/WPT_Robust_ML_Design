#!/usr/bin/env bash
# Create the project folder structure and a Python virtual environment.
# Usage (from the repository root):  ./scripts/bash/00_setup_project.sh
set -euo pipefail
mkdir -p data/raw data/processed data/external src \
         results_python/{figures,tables,models,optimization,robustness} \
         paper/{manuscript,tables,figures} logs
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt
python -m ipykernel install --user --name wpt-robust-ml --display-name "WPT Robust ML" >/dev/null
echo "Setup complete. Activate with: source .venv/bin/activate"
