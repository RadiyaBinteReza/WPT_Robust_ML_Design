# Results

All numbers come from `wpt_complete_analysis.ipynb` and are stored in `results_python/` (headline values in `results_python/key_results.json`). 

## 1. Initial baseline (η_max, 7 models)

| Model | Test R² (A, raw) | Test MAPE (A) | Test R² (B, ratios) |
|---|---|---|---|
| **SVR** | **0.9876** | **2.60 %** | 0.9871 |
| Gradient boosting | 0.9779 | 3.28 % | 0.9785 |
| Neural network | 0.9737 | 3.78 % | 0.9658 |
| Polynomial ridge | 0.9629 | 4.46 % | 0.9717 |
| Random forest | 0.9124 | 7.40 % | 0.9323 |
| Decision tree | 0.8694 | 7.97 % | 0.8785 |
| Linear regression | 0.8030 | 11.17 % | 0.8381 |

## 2. Analytical descriptors vs FEA

| Proxy | FEA quantity | Pearson r (log–log) |
|---|---|---|
| Modified-Wheeler inductance | L1 / L2 | 0.997 / 0.997 |
| AC-resistance proxy | R1 / R2 | 0.872 / 0.911 |
| Coaxial-filament coupling | k | 0.988 |

## 3. Effect of feature sets (median validation-RMSE change over 8 model families)

| Comparison | η_max | L1 | L2 | R1 | R2 | k |
|---|---|---|---|---|---|---|
| B (ratios) vs A | −5.4 % | −5.8 % | +0.2 % | −2.5 % | +1.3 % | −7.0 % |
| **B+ (analytical EM) vs A** | **−33.9 %** | **−63.1 %** | **−63.9 %** | **−33.9 %** | **−15.8 %** | **−38.1 %** |

Linear regression on η_max improves from R² 0.778 (A) to 0.974 (B+). A 10-neuron network with B+ matches a 128–64 network without it.

## 4. 5-fold cross-validation (B+ vs A, paired over identical folds)

| Target | SVR | Gradient boosting | GPR | Neural network |
|---|---|---|---|---|
| η_max | −33.1 % (5/5, p = 0.0002) | −30.9 % (5/5, p = 0.0006) | −25.0 % (5/5, p = 0.0006) | −24.6 % (5/5, p = 0.0001) |
| k | −19.2 % (5/5, p = 0.0087) | −29.0 % (5/5, p = 0.0023) | −20.9 % (5/5, p = 0.0037) | −18.1 % (5/5, p = 0.0027) |

Ratio features (B vs A) were not significant, except a small gain for GPR on k (p = 0.025).

## 5. Final test-set performance (selected model: GPR for every target)

| Target | Feature set | R² (model space) | R² | MAPE |
|---|---|---|---|---|
| L1 | B | 0.99997 | 0.9997 | 0.50 % |
| L2 | B+ | 0.99994 | 0.9999 | 0.59 % |
| R1 | B | 0.99932 | 0.9947 | 2.55 % |
| R2 | B+ | 0.99846 | 0.9982 | 3.36 % |
| k | B+ | 0.99954 | 0.9997 | 0.47 % |
| η_max (direct) | B+ | 0.99658 | 0.9966 | 1.23 % |
| **Mean MAPE** | | | | **1.45 %** |

## 6. Efficiency estimators

| Estimator | Validation RMSE | Test RMSE | Test R² | Test MAPE |
|---|---|---|---|---|
| Direct GPR | 0.0095 | 0.0094 | 0.9966 | 1.23 % |
| **Composed (L, R, k → theory)** | **0.0079** | **0.0082** | **0.9974** | **1.05 %** |
| Hybrid | 0.0082 | 0.0084 | 0.9973 | 1.07 % |
| Physics-guided NN | 0.0157 | 0.0152 | 0.9912 | 2.03 % |
| Baseline SVR | 0.0192 | 0.0180 | 0.9876 | 2.60 % |

The composed surrogate also predicts Q1 (2.50 % MAPE), Q2 (3.13 %) and Ro (2.14 %). Its outputs always lie in [0, 1]; the direct model violated this for 0.13 % of 20,000 random designs.

**GPR uncertainty calibration (test):** observed coverage was 74.6 % / 93.1 % / 96.6 % for nominal 68 % / 95 % / 99 % intervals, i.e. slightly over-confident in the tails.

## 7. What the model learned (permutation importance, drop in test R²)

| Output | Most important variables |
|---|---|
| η_max | Dext2 (2.34), N2 (1.16), Dext1 (0.99), N1 (0.91) |
| k | Dext2 (1.77), Dext1 (0.58) |
| R1 / R2 | N1 (0.93), Nlay1 (0.71) / N2 (0.86), Nlay2 (0.67) |
| L1 / L2 | Dext1 (0.73), Nlay1 (0.65), N1 (0.51) / Dext2 (0.85), Nlay2 (0.69), N2 (0.47) |

* Efficiency peaks at **N1 = 2, N2 = 1**: more turns are not better.
* **Layers (2 → 4):** L +288 to +306 %, R +268 to +355 %, so Q changes by only −12 % to +4 % and η by less than 2 percentage points.
* **Self-quantities are local:** L_i and R_i depend only on coil i.

## 8. Inverse design

* **Audit of unconstrained sampling:** 24.8 % feasible; 16.9 % predicted η < 0; minimum −0.83.
* **Constrained search:** 100,000 feasible candidates, 97.9 % in the applicability domain. Predicted η ranges from 0.015 to 0.911 (dataset max 0.905).
* **Differential evolution:** best score 0.466 → 0.476.
* **Top designs:** wide traces, 1 receiver turn, 1–2 transmitter turns, 2 layers, Dext2 ≈ 20 mm. Their nearest FEA simulations reach η 0.82–0.90.

## 9. Robustness (1,000 Monte Carlo builds per design)

* **Threshold:** 26 of 31 designs stay above η = 0.85 in every build.
* **Fabrication vs model uncertainty:** fabrication-only SD ≈ 0.003, against σ_GPR ≈ 0.008.
* **Stress test:** at 4× tolerance, P5 is still 0.875–0.881, with no infeasible builds.
* **Sensitivity:** Dext2 explains **86 %** of the tolerance-induced variance (W2 7 %, Dext1 5 %, W1 2 %, spacing ≈ 0).

## 10. Recommended designs (mm; η predicted; P5 = robust efficiency)

| Category | W1 | W2 | S1 | S2 | N1 | N2 | Dext1 | Dext2 | Layers | η | P5 | k | R1+R2 (Ω) | Nearest FEA η |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Highest η / best robust | 2.93 | 2.60 | 2.44 | 1.09 | 2 | 1 | 39.4 | 19.6 | 2/2 | 0.911 | 0.897 | 0.111 | 0.152 | 0.884 |
| Best balanced | 4.00 | 3.00 | 1.31 | 0.66 | 1 | 1 | 28.8 | 20.0 | 2/2 | 0.901 | 0.887 | 0.104 | 0.060 | 0.830 |
| Smallest compact | 2.61 | 2.93 | 1.51 | 1.32 | 1 | 1 | 16.7 | 18.7 | 4/2 | 0.800 | 0.782 | 0.064 | 0.126 | 0.843 |

These are early-stage recommendations at the edge of the simulated range; HFSS confirmation is required.
