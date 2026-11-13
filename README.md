# Physics-Guided Machine Learning and Robust Inverse Design of Multilayer PCB Coils for 6.78 MHz Wireless Power Transfer

A reproducible Python pipeline that learns fast **surrogate models** of coupled multilayer PCB coils from 4,635 Ansys HFSS finite-element simulations, **explains** what drives coupling, resistance and efficiency, and uses the surrogate for **feasibility-constrained inverse design** and **Monte Carlo robustness analysis** under manufacturing tolerances.

MProfStuds Data Science research project, The University of Auckland, 2026.
Author: **Radiya Binte Reza** · Supervisor: **Dr Aiguo Patrick Hu** · Industry partner: Resolinc Research

---

## Contents

1. [Headline results](#1-headline-results)
2. [Repository structure](#2-repository-structure)
3. [Quick start (5 minutes)](#3-quick-start-5-minutes)
4. [Environment](#4-environment)
5. [Dataset](#5-dataset)
6. [Running the pipeline](#6-running-the-pipeline)
7. [How the analysis was done](#7-how-the-analysis-was-done)
8. [Outputs](#8-outputs)
9. [Tests and continuous integration](#9-tests-and-continuous-integration)
10. [Limitations](#10-limitations)
11. [Citation and licence](#11-citation-and-licence)

---

## 1. Headline results

| Result | Value |
|---|---|
| Best model for all six outputs (L1, L2, R1, R2, k, η_max) | Gaussian-process regression (ARD kernel) |
| Mean test error over the six outputs | **1.45 % MAPE** (original ANN study: ~1.5 %) |
| Efficiency, theory-assisted surrogate (test set) | **R² = 0.9974**, MAPE 1.05 % (54 % lower RMSE than the SVR baseline) |
| Effect of analytical physics features (median over 8 model families) | Validation RMSE −34 % (η), −38 % (k), −63 % (L) |
| 5-fold cross-validation, physics vs raw features | Better in **5/5 folds** for every model tested (p ≤ 0.009) |
| Unconstrained random design search (audit) | only **24.8 %** of candidates were physically buildable |
| Best recommended design | η = 0.911, robust (5th-percentile) η = 0.897, k = 0.111 |
| Main tolerance driver | implant-coil outer dimension: **86 %** of tolerance-induced variance |

Full tables: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## 2. Repository structure

```
WPT_Robust_ML_Design/
├── README.md                     ← this file
├── LICENSE                       ← MIT (code); dataset is CC-BY-4.0 (see data/README.md)
├── CITATION.cff                  ← how to cite this repository
├── Makefile                      ← one-word commands: make setup / test / quick / full
├── requirements.txt              ← pip dependencies (minimum versions)
├── requirements-lock.txt         ← exact versions used to produce the results
├── environment.yml               ← conda alternative
├── pytest.ini
│
├── wpt_complete_analysis.ipynb   ← THE FULL PIPELINE (Parts 1–23), executed, with outputs
├── wpt_complete_analysis.html    ← the same notebook as a static web page
│
├── src/                          ← reusable code imported by the notebook and the tests
│   ├── wpt_physics.py            ←   analytical EM descriptors, feasibility, circuit relations
│   └── gpr_model.py              ←   Gaussian process with subset-tuned hyperparameters
│
├── scripts/
│   ├── quick_test.py             ← ~1-minute reduced run that checks everything works
│   └── bash/                     ← setup, data download/check, full run, summary
│
├── tests/                        ← pytest unit tests (run in ~5 s)
│
├── data/
│   ├── README.md                 ← dataset description, columns, licence, checksum
│   ├── raw/ANNCoilsFEA.csv       ← the 4,635 HFSS simulations (CC-BY-4.0)
│   ├── processed/                ← (empty; reserved)
│   └── external/                 ← (empty; reserved)
│
├── results_python/               ← every figure and table produced by the notebook
│   ├── figures/                  ←   27 figures
│   ├── tables/                   ←   result tables (CSV)
│   ├── optimization/             ←   inverse-design candidates and Pareto fronts
│   ├── robustness/               ←   Monte Carlo results
│   ├── models/                   ←   trained models (git-ignored; re-created by the notebook)
│   └── key_results.json          ←   headline numbers in machine-readable form
│
├── docs/
│   ├── METHODOLOGY.md            ← detailed, step-by-step description of how everything was done
│   ├── RESULTS.md                ← key result tables with interpretation
│   ├── REPRODUCIBILITY.md        ← environment, seeds, runtimes, expected outputs, known issues
│   ├── CHANGELOG.md              ← what changed relative to the initial analysis, and why
│   └── project_notes.md          ← decisions and open questions
│
├── logs/                         ← run logs (quick test, data checksum, notebook execution)
└── .github/workflows/tests.yml   ← GitHub Actions: runs the tests and the quick test on every push
```

---

## 3. Quick start (5 minutes)

```bash
git clone https://github.com/<your-username>/WPT_Robust_ML_Design.git
cd WPT_Robust_ML_Design

python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/quick_test.py        # ~1 minute: 6 stages, every check should print PASS
pytest                              # ~5 seconds: 14 unit tests
```

Expected end of the quick test:

```
  PASS  composed surrogate R2 > 0.97
  PASS  all composed efficiencies lie in [0, 1]

ALL CHECKS PASSED in 23s  (summary: logs/quick_test_report.json)
```

With `make` available, the same is `make setup && make quick && make test`.

---

## 4. Environment

* **Python 3.10–3.12** (results produced with Python 3.12.3).
* Core packages: NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, joblib, tabulate; Jupyter/nbconvert for running the notebook; pytest for the tests.
* `requirements.txt` gives minimum versions; `requirements-lock.txt` pins the exact versions used for the reported numbers.

**pip**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements-lock.txt
```

**conda**
```bash
conda env create -f environment.yml
conda activate wpt-robust-ml
```

**Hardware.** Any laptop works. The full notebook needs about **3 GB RAM** and takes **15–40 minutes**, mostly Gaussian-process fitting and 5-fold cross-validation. No GPU is used.

---

## 5. Dataset

`data/raw/ANNCoilsFEA.csv`: 4,635 Ansys HFSS 2025 R1 simulations of a coaxially aligned pair of multilayer square PCB coils at **6.78 MHz** and **15 mm** separation, from Rodriguez-Fuentes, Carrizosa and Ramos (IEEE TPEL, 2026), published under **CC-BY-4.0**.

| Group | Columns |
|---|---|
| Geometry inputs (10) | `W1 W2` trace width · `S1 S2` spacing · `N1 N2` turns/layer · `Dext1 Dext2` outer side · `Nlay1 Nlay2` layers |
| Fixed | `Freq` 6.78 MHz · `Dist` 15 mm · `H1 H2` copper code 2 · `X1 X2` PCB 1.6 mm |
| Outputs | `L1 L2` (µH) · `R1 R2` (Ω) · `k` · `Q1 Q2` · `Ro` (Ω) · `Eff_max` · raw `Z11 Z22 Z12` |

Full column descriptions, ranges, checksum and attribution: [`data/README.md`](data/README.md).
To re-download from the original source: `bash scripts/bash/01_get_data.sh`.

---

## 6. Running the pipeline

| Goal | Command | Time |
|---|---|---|
| Check the environment works | `python scripts/quick_test.py` | ~1 min |
| Unit tests | `pytest` | ~5 s |
| Verify the dataset (rows, header, SHA-256) | `bash scripts/bash/02_check_data.sh` | 1 s |
| **Reproduce everything** (executes the notebook, exports HTML) | `bash scripts/bash/03_run_pipeline.sh` | 15–40 min |
| Print headline numbers from the last run | `bash scripts/bash/04_summarize_outputs.sh` | 1 s |
| Interactive | `jupyter notebook wpt_complete_analysis.ipynb` | — |

Run all commands **from the repository root**: the notebook reads `data/raw/` and imports from `src/` relative to it.

---

## 7. How the analysis was done

The notebook is organised in 23 parts. Each stage is summarised below; the full step-by-step account, with every formula, threshold and the reason for each choice, is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

| Stage | Notebook parts | What was done |
|---|---|---|
| **1. Data preparation** | 1–5 | Load the CSV, parse complex impedances, check missing values and ranges, fixed 70/15/15 split (seed 42: 3,244 / 695 / 696 rows). |
| **2. Baseline** | 6–11 | 7 models predict η_max from raw geometry (A) and ratio features (B); SVR selected on validation; residual diagnostics. |
| **3. Physics checks and features** | 12 | Verify Q = ωL/R, η_max = x/(1+√(1+x))², Ro = R2√(1+x) on every row (error ~1e-14). Build 13 analytical descriptors (modified-Wheeler inductance, fill factor, AC-resistance and self-resonance proxies, coaxial-filament coupling): feature set **B+**. Audit of the original inverse design. |
| **4. Extended EDA** | 13 | Spearman correlation heatmap, Q–η relations, geometry of the top-10 % designs. |
| **5. Benchmark** | 14 | 6 outputs × 3 feature sets × 8 models = 144 models (L, R, k in log10). Selection by validation RMSE. Neural-network architecture sweep. |
| **6. Cross-validation** | 15 | 5-fold CV on train+validation with identical folds; paired t-tests of B and B+ against A. |
| **7. Final test** | 16 | Selected models refitted on train+validation and evaluated **once** on the test set; GPR uncertainty calibration. |
| **8. Theory-assisted surrogate** | 17 | Predict L1, L2, R1, R2, k and compute Q, η, Ro by circuit theory; compared with direct, hybrid and physics-guided-NN estimators. |
| **9. Explainability** | 18 | Permutation importance, **feasibility-constrained** partial dependence, design maps, counterfactual layer study, answers to 7 engineering questions. |
| **10. Inverse design** | 19 | 100,000 feasible geometries (inner opening ≥ 1 mm), applicability-domain filter, weighted score, differential-evolution refinement, Pareto fronts, nearest-FEA check. |
| **11. Robustness** | 20 | 1,000 Monte Carlo builds per design (W, S ±5 %; Dext ±2 %) plus GPR uncertainty; infeasible builds count as failures; robust efficiency = 5th percentile; stress test and sensitivity. |
| **12. Recommendations and export** | 21–23 | Four design categories, coil drawings, diagrams, `key_results.json`, paper tables and figures. |

---

## 8. Outputs

| Where | What |
|---|---|
| `results_python/figures/` | All figures (EDA, benchmark, CV, predictions, importance, partial dependence, Pareto, robustness, final designs) |
| `results_python/tables/` | Benchmark, CV, test performance, calibration, importance, layer study and more (CSV) |
| `results_python/optimization/` | Top-20 designs, Pareto fronts |
| `results_python/robustness/` | Robustness ranking, stress test, sensitivity |
| `results_python/key_results.json` | All headline numbers |

---

## 9. Tests and continuous integration

* `tests/test_physics.py`: circuit relations hold exactly on the data; η formula bounds; inner-opening formula; feasibility detection; mutual-inductance properties; descriptor correlations.
* `tests/test_models.py`: the subset-tuned GPR fits a known function, batched prediction equals unbatched, and a small real-data surrogate reaches R² > 0.95.
* `.github/workflows/tests.yml`: on every push, GitHub installs the environment, runs `pytest` and `python scripts/quick_test.py --fast`.

---

## 10. Limitations

* The operating point is fixed: 6.78 MHz, 15 mm gap, fixed copper and PCB thickness. There is no misalignment or tissue loading.
* Secondary FEA data only. The recommended designs are **not yet HFSS-verified**, and they lie at the upper bounds of the simulated W and Dext2 ranges, where the surrogate is least certain.
* Manufacturing tolerances are simulated, and GPR intervals are slightly over-confident (93 % coverage at a nominal 95 %).
* The results apply to multilayer square planar coils for implantable WPT only.

---

## 11. Citation and licence

**Code:** MIT licence (see `LICENSE`). **Dataset:** CC-BY-4.0, © the original authors. If you use the data, cite:

> A. Rodriguez-Fuentes, M. J. Carrizosa and R. Ramos, "Neural Network-Based Design of Wireless Power Transfer Systems for Implantable Medical Devices," *IEEE Transactions on Power Electronics*, vol. 41, no. 2, pp. 3011–3024, Feb. 2026, doi: 10.1109/TPEL.2025.3614366.

To cite this repository, use `CITATION.cff` (GitHub shows a "Cite this repository" button).
