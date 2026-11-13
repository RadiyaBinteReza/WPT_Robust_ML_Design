# Project notes

## Key decisions (and why)
- **Python instead of MATLAB.** The analysis was started in Python; every method has a MATLAB equivalent (see README).
- **Log-space targets** for L, R, k: they span 2–4 decades; linear-space errors would be dominated by a few large coils.
- **Selection on validation only; test used once** (Part 16). The same 70/15/15 split (seed 42) as the original notebook.
- **GPR hyperparameters tuned on a 600-sample subset**, then refitted on all training data with the kernel fixed:
  full marginal-likelihood optimisation on 3,244 × 31 was too slow on one CPU; the subset-tuned GP was the most
  accurate model for every output anyway.
- **Composed η estimator** (L, R, k → theory) chosen for design on validation RMSE; guarantees 0 ≤ η ≤ 1.
- **Feasibility constraint Din ≥ 1 mm**: the minimum present in the FEA data.
- **Applicability domain**: nearest-training-point distance ≤ 99th percentile of test distances (retained 97.9 %).
  This is a weak filter; a stricter one (e.g. 95th percentile or GPR σ threshold) is an option.
- **Robust efficiency = P5** of (η under tolerances + GPR noise), infeasible samples counted as η = 0.
- **Compact-design threshold relaxed to P5 ≥ 0.75** because no compact design reached 0.80.

## Open questions for the supervisor
1. Score weights (0.45/0.25/0.20/0.10) come from the brief. Should the implant size (Dext2) replace Dext1 + Dext2?
2. The optimum sits at the upper W and Dext2 bounds. Is running a few HFSS simulations of the recommended designs
   (and slightly beyond the bounds) possible? This is the most valuable next step for publication.
3. Target journal: given no hardware, IEEE Access / Engineering Applications of AI seem realistic; JESTPE if HFSS
   confirmation is added.

## Next steps
- [ ] HFSS confirmation of the 3 recommended designs (Table 2).
- [ ] Recalibrate GPR σ (coverage 93 % at nominal 95 %).
- [ ] Optional: active-learning loop (use GPR σ to select new FEA runs).
- [ ] Add the repository URL to the dissertation.
