# Methodology: how the analysis was done

This document describes every step of the pipeline in `wpt_complete_analysis.ipynb`: what was done, how, the settings used, and **why** each choice was made. Section numbers follow the notebook parts.

---

## Contents

- [0. Problem and approach in one paragraph](#0-problem-and-approach-in-one-paragraph)
- [1. Data and split (Parts 1–5)](#1-data-and-split-parts-15)
- [2. Baseline models (Parts 6–11)](#2-baseline-models-parts-611)
- [3. Physics verification and analytical features (Part 12)](#3-physics-verification-and-analytical-features-part-12)
- [4. Extended exploratory analysis (Part 13)](#4-extended-exploratory-analysis-part-13)
- [5. Multi-target benchmark (Part 14)](#5-multi-target-benchmark-part-14)
- [6. Cross-validation and significance (Part 15)](#6-cross-validation-and-significance-part-15)
- [7. Final test evaluation and uncertainty (Part 16)](#7-final-test-evaluation-and-uncertainty-part-16)
- [8. Theory-assisted surrogate (Part 17)](#8-theory-assisted-surrogate-part-17)
- [9. Explainability (Part 18)](#9-explainability-part-18)
- [10. Inverse design (Part 19)](#10-inverse-design-part-19)
- [11. Robustness (Part 20)](#11-robustness-part-20)
- [12. Final designs and export (Parts 21–23)](#12-final-designs-and-export-parts-2123)
- [13. Summary of every threshold and setting](#13-summary-of-every-threshold-and-setting)

---

## 0. Problem and approach in one paragraph

An implant is powered through two coupled PCB coils: a transmitter outside the body and a small receiver inside it. Their geometry (trace width W, spacing S, turns per layer N, outer side Dext, layers Nlay) determines the inductances L, resistances R, coupling k and therefore the maximum link efficiency η_max. Simulating each geometry in HFSS is slow, so we trained a **surrogate model** on 4,635 existing simulations. We then (i) added physics knowledge to the inputs and to the outputs, (ii) tested rigorously whether it helps, (iii) explained what the model learned, (iv) used it to search for good designs, but only among buildable ones, and (v) checked how those designs behave under manufacturing tolerances.

---

## 1. Data and split (Parts 1–5)

**Data.** `data/raw/ANNCoilsFEA.csv`, 4,635 rows × 28 columns (see `data/README.md`). There are no missing values. Frequency (6.78 MHz), gap (15 mm), copper code (2) and PCB thickness (1.6 mm) are constant.

**Complex impedances.** `Z11`, `Z22` and `Z12` are stored as strings such as `0.25+42.1i`; they are parsed into real and imaginary parts.

**Split.** A random permutation with `numpy.random.RandomState(42)` gives 70 % train (3,244), 15 % validation (695) and 15 % test (696).
- *Why fixed:* changing the seed until results look good biases the evaluation, so the seed is never changed.
- *Rule used throughout:* every choice (model, feature set, estimator) is made on **validation** data or by cross-validation on train+validation. The **test set is used once**, at the very end.

---

## 2. Baseline models (Parts 6–11)

The first analysis predicted η_max with 7 models (linear, polynomial, decision tree, random forest, gradient boosting, SVR and neural network), using two feature sets:

- **A, raw (10 features):** W1, W2, S1, S2, N1, N2, Dext1, Dext2, Nlay1, Nlay2.
- **B, ratios (18 features):** A plus Dext2/Dext1, W2/W1, S2/S1, N1·Nlay1, N2·Nlay2, N2/N1, Nlay2/Nlay1 and Dext1+Dext2.

The SVR on A had the best validation R² (test R² 0.9876, RMSE 0.0180, MAPE 2.60 %). Residual diagnostics showed no bias but heavy tails (Q–Q plot) and more scatter at low efficiency, where the data are sparse. Ratio features helped weak models but not the SVR, which was a first sign that generic ratios add little information.

---

## 3. Physics verification and analytical features (Part 12)

### 3.1 Circuit relations

On every row we verified:

```
Q_i     = ω L_i / R_i                 ω = 2π · 6.78 MHz
x       = k² Q1 Q2
η_max   = x / (1 + √(1 + x))²
Ro      = R2 √(1 + x)
```

The maximum relative error is about 1e-14, which is floating-point precision. **So the five quantities L1, L2, R1, R2 and k fully determine every other output.** This motivates the theory-assisted surrogate (Section 8).

### 3.2 Geometric feasibility

For a square spiral, the inner opening is

```
Din = Dext − 2·N·W − 2·(N − 1)·S
```

All 4,635 simulations have Din ≥ 1 mm on both coils. **Threshold Din ≥ 1 mm:** this is the minimum present in the data. A geometry below it is either impossible to build (negative Din means the turns do not fit) or outside anything the model has seen.

### 3.3 Audit of the original inverse design

The first inverse-design attempt sampled each variable independently. We regenerated it exactly (same seed) and found that only **24.8 %** of its 100,000 candidates were feasible. The baseline SVR predicted η < 0 for 16.9 % of all candidates, with a minimum of −0.83. Almost all of these were infeasible geometries, where the model was extrapolating. Inverse design was therefore rebuilt with constraints (Section 10).

### 3.4 Analytical descriptors (feature set B+, 31 features)

All descriptors are closed-form functions of geometry only, so no FEA output leaks into them. They are implemented in `src/wpt_physics.py`.

| Descriptor | Formula / idea | Why |
|---|---|---|
| `Din1`, `Din2` | inner opening (above) | how much of the coil area is copper |
| `fill1`, `fill2` | ρ = (Dext − Din)/(Dext + Din) | enters the Wheeler formula |
| `log_Lwh1`, `log_Lwh2` | modified Wheeler (Mohan et al., 1999): L = 2.34 μ0 n² d_avg / (1 + 2.75 ρ), n = N·Nlay, d_avg = (Dext + Din)/2 | standard planar-spiral inductance; layers are in series, so n = N·Nlay |
| `log_Rprox1`, `log_Rprox2` | total trace length 4·d_avg·N·Nlay divided by (W + 0.07 mm) | at 6.78 MHz the skin depth is ≈ 25 µm, so R scales with length ÷ conductor perimeter; 0.07 mm ≈ copper thickness (2 oz) |
| `log_srf1`, `log_srf2` | ω² · L_wh · C, with C a parallel-plate inter-layer capacitance (FR-4, εr = 4.4) | ≈ (f/f_SRF)²: flags coils close to self-resonance, whose resistance is inflated (explains R1 up to 906 Ω) |
| `log_kprox` | Neumann mutual inductance of two coaxial circular filaments at the mean radii and 15 mm (elliptic integrals), × turns, ÷ √(L_wh1·L_wh2) | closed-form coupling estimate |
| `x_prox`, `eta_prox` | k_prox² Q1_prox Q2_prox and η from the formula | proxy figure of merit |

**Validation of the descriptors against FEA:**
- Wheeler L vs FEA L: r = 0.997 (log–log).
- Coupling proxy vs FEA k: r = 0.988.
- Resistance proxy vs FEA R: r = 0.87–0.91 (the weakest).

**Choice made by testing:** for the equivalent filament radius we tried d_avg/2 × 1.0, × 1.128 and × 1.2; the factor 1.0 correlated best with FEA k.

**Assumption:** the dataset documentation states 1 oz copper in its text but 2 in its table and data. We assume 2 oz. The thickness is constant across all rows, so this only shifts the proxy by a near-constant factor.

---

## 4. Extended exploratory analysis (Part 13)

- **Spearman correlation heatmap.** Spearman rather than Pearson, because L and R span up to five decades and Pearson would be dominated by a few extreme values. Efficiency correlates most with k (0.89), Dext2 (0.73) and Q2 (0.72). L_i and R_i are almost perfectly rank-correlated (0.96–0.97).
- **Q vs efficiency plots.** These show k and Q acting together, as in x = k²Q1Q2.
- **Geometry of the top 10 % designs** (η ≥ 0.84): large coils, wide traces and few turns.

---

## 5. Multi-target benchmark (Part 14)

**Targets:** L1, L2, R1, R2, k and η_max. L, R and k are modelled as **log10**. *Why:* they span 1–5 decades. In linear space a model would concentrate on the few largest coils; in log space relative errors are weighted equally. η_max (0–0.9) is modelled directly.

**Feature sets:**
- **A** (10): raw geometry.
- **B** (18): A plus ratios.
- **B+** (31): B plus the analytical descriptors.

*Why three sets:* B vs A tests whether generic feature engineering helps; B+ vs B tests whether real physics helps.

**Models and settings** (scikit-learn):

| Model | Setting | Reason |
|---|---|---|
| Linear regression | standardised inputs | baseline: if ML cannot beat it, the problem does not need ML |
| Polynomial ridge | degree 2, α = 1 | degree 3 on 31 features would give ~6,000 terms and overfit |
| Decision tree | max depth 12, min leaf 3 | interpretable; limited depth avoids memorising |
| Random forest | 300 trees, max features √p, min leaf 2 | accuracy plateaus well before 300 trees |
| Gradient boosting | histogram GBM, 600 iterations, learning rate 0.05 | small learning rate with many iterations is more stable |
| SVR | RBF, C = 10, ε = 0.01, **standardised target** | standardising the target lets one setting work for all six outputs |
| Gaussian process | constant × ARD-RBF + white noise | see below |
| Neural network | MLP (64, 32), ReLU, Adam, early stopping | small network for 4,635 rows; early stopping prevents overfitting |

**Gaussian process detail (`src/gpr_model.py`).** ARD (automatic relevance determination) gives each input its own length-scale, so irrelevant inputs are switched off automatically: their length-scale grows to the upper bound of 1,000, and the resulting warning is expected. Full marginal-likelihood optimisation on 3,244 × 31 did not finish in 6 minutes on one CPU. We therefore tune the hyperparameters on a random **600-row subset** (L-BFGS-B, at most 60 iterations), then refit on all training rows with the kernel fixed. Subsets of 500 and 800 gave the same accuracy, so 600 is a safe middle value. Prediction is batched (4,000 rows) to limit memory.

**Selection:** for each target, the model and feature set with the lowest validation RMSE in modelling space. Result: GPR for all six targets (feature set B for L1 and R1, B+ for the others; the differences for L1 and R1 are in the fifth decimal).

**Neural-network sweep:** architectures [10], [20], [20,10], [50,20], [50,25,10], [64,32] and [128,64], each with A and B+.

---

## 6. Cross-validation and significance (Part 15)

- **Setup:** 5-fold CV on train+validation (3,939 rows) for η_max and k, with four models (SVR, gradient boosting, GPR, MLP) and all three feature sets.
- **Pairing:** the same folds are used for every model and feature set, so per-fold RMSEs can be compared with a **paired t-test**. Pairing removes the fold-difficulty variation and is more sensitive than an unpaired test.
- **Reporting:** with only 5 folds the test has limited power, so we also report how many folds improved.

---

## 7. Final test evaluation and uncertainty (Part 16)

- **Refit and test:** each selected model is refitted on train+validation and evaluated **once** on the 696-row test set. Metrics are RMSE, MAE, R² and MAPE in original units, and R² in modelling space.
- **Separate GPR for uncertainty:** a GPR on B+ features for η_max provides the predictive standard deviation σ.
- **Calibration check:** we count how many test points fall inside ±1σ, ±1.96σ and ±2.58σ.

---

## 8. Theory-assisted surrogate (Part 17)

Four η estimators were compared on **validation**:

1. **Direct:** the best model trained on η_max.
2. **Composed:** the best models for L1, L2, R1, R2 and k, then Q, x, η and Ro from the formulas in Section 3.1.
3. **Hybrid:** the mean of direct and composed.
4. **Physics-guided NN:** one MLP (128, 64) predicting the five standardised log10 quantities from B+, followed by the same fixed formulas.

The composed estimator won (validation RMSE 0.0079, against 0.0095 for direct), so it is used for all design work. It always gives 0 ≤ η ≤ 1 and is consistent with the predicted k and Q. The original dataset study also predicted these five quantities; our contribution is the head-to-head test against the alternatives.

**Consistency check:** on 20,000 random feasible geometries, we count direct predictions outside [0, 1] (0.13 %) and compare them with the composed predictions.

---

## 9. Explainability (Part 18)

- **Permutation importance.** Run on the test set, with 10 repeats, using the best **raw-feature (A)** model per target, so that importance is expressed in designer variables. With B+ models, the importance would be split across correlated derived features. Permutation importance is model-agnostic and, unlike forest impurity importance, unbiased towards variables with many distinct values.
- **Feasibility-constrained partial dependence (our modification).** A standard PDP sets a variable to a value for every row and so creates impossible coils (e.g. five turns on 10 mm). We average only over rows that remain feasible (Din ≥ 1 mm) and report the feasible fraction. Settings: a base sample of 300 rows, and 20 grid points for continuous variables or every level for discrete ones.
- **Design maps.** 2-D feasibility-constrained PD over (Dext1, Dext2) and (N1, N2).
- **Counterfactual layer study.** For each test design, Nlay is switched between 2 and 4 on one coil, and we record the median change in L, R, Q, k and η.

---

## 10. Inverse design (Part 19)

1. **Feasible sampling.** We draw 100,000 geometries uniformly within the dataset bounds (N and Nlay from their levels) and keep only those with Din ≥ 1 mm on both coils.
2. **Applicability domain.** A candidate is kept only if its nearest training-plus-validation point (standardised raw geometry) is no farther than the **99th percentile** of test-to-training nearest-neighbour distances. Test points at those distances were predicted accurately. The 99th rather than 95th percentile keeps rare but valid regions. In practice 97.9 % passed, so feasibility did most of the work.
3. **Score** (weights from the project brief):
   ```
   S = 0.45·η̂ + 0.25·k̂ − 0.20·(log10 R_tot)^ − 0.10·Size^
   ```
   The hats denote min–max normalisation over the feasible in-domain pool, R_tot = R1 + R2 and Size = Dext1 + Dext2. Normalisation puts different units on one scale. R_tot is normalised on a log scale because it spans five decades.
4. **Differential evolution** (`scipy.optimize.differential_evolution`; the Python analogue of MATLAB `ga`). It optimises the six continuous variables for each of the six best distinct (N1, N2, Nlay1, Nlay2) configurations among the top 300 random-search designs. Settings: population 12 × 6, 60 generations, vectorised, seed 42, penalty +1 for infeasible or out-of-domain points (the score range is about [−0.3, 0.7], so penalised points never win).
5. **Pareto fronts.** Two-objective fronts η–Size, η–R_tot, k–R_tot and η–Dext2, plus a four-objective non-dominated set (top 5,000 by score).
6. **Nearest-FEA check.** Each top design is compared with the closest simulation that has the same turns and layers.

---

## 11. Robustness (Part 20)

**Designs tested:** 31 in total, after removing duplicates. These are the top 20 by score, the top 5 by η, the 5 smallest designs with η ≥ 0.80, and all DE optima.

**Perturbations:** 1,000 Monte Carlo copies per design, with relative Gaussian scatter of W and S ±5 % and Dext ±2 % (1σ, from the brief).

**Differences from a naive analysis, and why:**

| Choice | Why |
|---|---|
| Infeasible copies count as failures (η = 0) | a board whose turns do not fit does not work |
| No clipping to the data range | clipping hides sensitivity for designs at the boundary; the out-of-range fraction is reported instead |
| η + N(0, σ_GPR) added to each copy | includes surrogate error, not only fabrication scatter |
| Robust efficiency = 5th percentile (P5) | "95 of 100 boards reach at least this"; the mean hides bad cases and the minimum depends on one sample; 1,000 copies leave about 50 below P5 |
| Pass threshold η ≥ 0.85 | the original 0.70 was passed by every design (no discrimination); 0.85 lies inside the top-design range |

**Further analyses:**
- **Stress test:** all tolerances scaled 0.5–4×.
- **One-at-a-time variance attribution:** for the most robust design.

---

## 12. Final designs and export (Parts 21–23)

**Four categories**, chosen from the Monte Carlo-evaluated set:
- **Highest η:** the largest nominal efficiency.
- **Smallest compact:** the smallest Dext1 + Dext2 with P5 ≥ 0.75. The threshold was relaxed from 0.80 because no compact design reached it (maximum 0.793).
- **Best robust:** the largest P5.
- **Best balanced:** the largest score.

**Exports:**
- Coil drawings to scale, and workflow and dataset diagrams.
- `results_python/key_results.json`.
- Paper tables (CSV + Markdown) and figures renamed Fig01–Fig11 in `paper/` (generated locally; not included in this repository).

---

## 13. Summary of every threshold and setting

| Item | Value | Reason |
|---|---|---|
| Split | 70/15/15, seed 42 | project brief; fixed seed avoids selection bias |
| Target scale | log10 for L, R, k | 1–5 decades; equal weight to relative error |
| Feasibility | Din ≥ 1 mm | minimum in all FEA designs |
| GPR tuning | 600 rows, 60 iterations | 500/800 gave equal accuracy; full fit intractable |
| Selection | min validation RMSE (model space) | test set used once |
| CV | 5-fold, paired t-test | identical folds allow pairing |
| Permutation repeats | 10 | stable mean and s.d. |
| PD | 300 base rows, 20 grid points | stable averages, moderate cost |
| Candidates | 100,000 feasible | dense enough for 10-D screening |
| Applicability domain | 99th percentile of test NN distance | trust where accuracy was verified |
| Score weights | 0.45 / 0.25 / 0.20 / 0.10 | project brief |
| DE | population 12×6, 60 generations, penalty 1 | standard for 6-D; penalty exceeds score range |
| Tolerances | W, S 5 %; Dext 2 % | project brief |
| Monte Carlo | 1,000 per design | ~50 samples below P5 |
| Robust efficiency | P5 incl. failures and σ_GPR | fabrication-yield interpretation |
| Pass threshold | η ≥ 0.85 | 0.70 did not discriminate |
| Compact category | P5 ≥ 0.75 | no compact design reached 0.80 |
