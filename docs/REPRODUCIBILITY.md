# Reproducibility

## Tested environment

| Item | Version |
|---|---|
| OS | Ubuntu 24.04 (Linux) |
| Python | 3.12.3 |
| numpy / pandas / scipy | 2.5.4 / 3.0.6 / 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib / seaborn | 3.11.2 / 0.13.2 |
| joblib / tabulate | 1.6.0 / 0.10.0 |

The exact pins are in `requirements-lock.txt`. Older versions within the ranges of `requirements.txt` should give the same conclusions. Individual metrics may differ in the 4th decimal because of BLAS and optimiser differences.

## Randomness

All random elements use **seed 42**: the data split, every model, the GPR tuning subset, cross-validation folds, permutation importance, candidate sampling (seed 2026), differential evolution and Monte Carlo (seeds per design). Two independent executions of the notebook produced identical numbers.

## Runtime and memory (single CPU core)

| Notebook part | Approx. time |
|---|---|
| 1–13 (data, baseline, features, EDA) | 2 min |
| 14 (144-model benchmark) | 7 min |
| 15 (5-fold CV) | 13 min |
| 16–17 (final models, theory-assisted surrogate) | 3 min |
| 18 (explainability) | 5 min |
| 19 (inverse design) | 7 min |
| 20–23 (robustness, designs, export) | 4 min |
| **Total** | **≈ 40 min** (≈ 15 min on a multi-core laptop) |

Peak memory is about 2–3 GB, mostly the fitted Gaussian processes. The notebook frees the 144 benchmark models before inverse design (Part 19A).

## Expected outputs

After `bash scripts/bash/03_run_pipeline.sh`:

* the notebook is executed in place, and `wpt_complete_analysis.html` is regenerated;
* 27 figures appear in `results_python/figures/`, plus the tables and `key_results.json`;
* `bash scripts/bash/04_summarize_outputs.sh` prints, for example, `Eff_max  Gaussian Process  set B+  test R2=0.99658  MAPE=1.23%`.

## Quick check instead of the full run

`python scripts/quick_test.py` reproduces the core findings on an 800-row training subset in about a minute. It checks the circuit relations, feasibility, descriptor quality, a GPR with raw vs physics features, and the composed surrogate. It writes `logs/quick_test_report.json`. Its numbers are lower than the full run because it uses a quarter of the training data.

## Known messages and issues

* **`ConvergenceWarning: ... close to the specified upper bound 1000`** from the Gaussian process is expected. ARD has switched an irrelevant input off by making its length-scale very long. The tests and quick test silence it.
* **Windows:** run the `.sh` scripts in Git Bash or WSL, or use the Python and Jupyter commands directly.
* **Working directory:** run everything from the repository root, since paths are relative to it.
* **Trained models** (`results_python/models/*.joblib`, several hundred MB) are not committed. The notebook re-creates them.
