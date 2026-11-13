# Convenience commands. Run from the repository root.
PY ?= python3

.PHONY: help setup data check test quick full summary clean
help:      ## list commands
	@grep -E '^[a-z]+:.*##' Makefile | sed 's/:.*##/\t/'
setup:     ## create .venv and install dependencies
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
data:      ## download the dataset (skipped if present)
	bash scripts/bash/01_get_data.sh
check:     ## verify dataset rows, header and checksum
	bash scripts/bash/02_check_data.sh
test:      ## run unit tests (~5 s)
	$(PY) -m pytest
quick:     ## reduced pipeline run (~1 min)
	$(PY) scripts/quick_test.py
full:      ## execute the full notebook and export HTML (15-40 min)
	bash scripts/bash/03_run_pipeline.sh
summary:   ## print headline numbers of the last full run
	bash scripts/bash/04_summarize_outputs.sh
clean:     ## remove caches
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
