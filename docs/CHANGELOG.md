# Changelog — what changed relative to the original notebook and why

## Kept unchanged
Parts 1–11F of `wpt_complete_analysis.ipynb` (loading, complex-impedance parsing, ratio features, EDA,
70/15/15 split with seed 42, 7-model comparison on raw and ratio features, validation-based selection,
final SVR, residual diagnostics). One edit: `DATA_PATH` now also finds `data/raw/ANNCoilsFEA.csv`, so the
notebook runs inside the repository layout. Bug fix: Part 10 saved a figure to `FIGURE`, which was only
defined later in Part 11A (it worked only because of the order cells were run interactively); Part 1 now defines
`FIGURE = FIG` (same folder), so the notebook runs top to bottom.

## Replaced: original Parts 12–14 (inverse design, Pareto front, robustness)

| Issue in the original | Consequence | Fix in Parts 19–20 |
|---|---|---|
| Candidates sampled independently per variable | Only 24.8 % of the 100,000 candidates were buildable; the rest had traces that do not fit inside the coil (inner opening Din < 1 mm, often negative). The SVR extrapolated there (predicted η as low as −0.83). | Sample only geometries with Din ≥ 1 mm on both coils (the minimum in the FEA data), and additionally restrict to an applicability domain (nearest-FEA-neighbour distance ≤ 99th percentile of test distances). |
| Resistance = `N·Nlay·Dext` proxy | Score did not use the actual electrical resistance. | R1, R2 predicted by trained surrogates (test MAPE ≈ 2–3 %). |
| `Pred_k` set equal to predicted efficiency (no k model) | Efficiency counted twice in the score (0.55 + 0.20). | k predicted by its own surrogate (test R² > 0.999 in log space). |
| Random search only | Optimum limited by sampling density. | Differential evolution (analogue of MATLAB `ga`) refines the best discrete configurations. |
| Robustness: values clipped to the data range | Hides sensitivity of designs at the range boundary. | No clipping; out-of-range fraction reported. |
| Robustness: infeasible perturbed geometries ignored | Designs at the geometric limit looked robust. | Infeasible samples count as failures (η = 0). |
| Robustness: threshold η ≥ 0.70 | Every design had probability 1.0; no discrimination. | Threshold 0.85 (0.80 and 0.70 also reported); ranking by the 5th percentile. |
| Robustness: surrogate treated as exact | Model error ignored. | GPR predictive σ added to every Monte Carlo sample (uncertainty-aware). |
| Only `Eff_max` modelled | Outline asks for L1, L2, R1, R2, k, Eff_max. | Part 14 benchmarks all six outputs; Part 17 derives Q1, Q2, Ro and η by circuit theory. |

## Added
* `src/wpt_physics.py`: analytical descriptors (modified-Wheeler inductance, fill factor, AC-resistance and
  self-resonance proxies, coaxial-filament coupling), feasibility check, circuit relations.
* `src/gpr_model.py`: ARD Gaussian process with subset-tuned hyperparameters and batched prediction.
* Parts 12–23 (see notebook), bash automation in `scripts/bash/`.

## Hyperparameter notes
* SVR and the neural network now standardise the target (`TransformedTargetRegressor`), so one ε/learning
  rate works for all six outputs. L, R and k are modelled as log10.
* Random-forest trees reduced from 400 to 300 to keep the 144-fit benchmark tractable (no measurable change).
