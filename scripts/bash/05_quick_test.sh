#!/usr/bin/env bash
# Reduced pipeline run (~1 min) + unit tests. Run from the repository root.
set -euo pipefail
python3 -m pytest
python3 scripts/quick_test.py "$@" | tee logs/quick_test.log
