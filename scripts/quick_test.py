#!/usr/bin/env python3
"""
Quick test run (about 1 minute on a laptop).

Checks that the environment, the dataset and the core methods work, using a
reduced version of the full pipeline:

  1. dataset loads with the expected shape and no missing values
  2. inductive-link circuit relations hold on every row (Q, eta_max, Ro)
  3. every FEA design satisfies the geometric-feasibility constraint
  4. the analytical physics descriptors correlate with FEA as expected
  5. a Gaussian process trained on a small subset predicts eta_max well,
     with raw (A) and physics-guided (B+) features
  6. the theory-assisted surrogate (predict L1, L2, R1, R2, k -> compute eta)
     works end to end

Usage:
    python scripts/quick_test.py            # default subset (800 training rows)
    python scripts/quick_test.py --fast     # smaller subset, for CI (~20 s)

A JSON summary is written to logs/quick_test_report.json.
The full analysis is in wpt_complete_analysis.ipynb (see README.md).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from wpt_physics import (RAW_FEATURES, RATIO_FEATURES, ANALYTIC_FEATURES, add_physics_features,
                         is_feasible, electrical_from_LRk)
from gpr_model import TunedGPR

# ARD length-scales of irrelevant inputs hit their upper bound (1000): this is the intended
# automatic feature down-weighting, not a fitting problem, so the warning is silenced.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA = ROOT / "data" / "raw" / "ANNCoilsFEA.csv"
SEED = 42


def step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(f"Quick test failed: {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="smaller subset (for CI)")
    args = ap.parse_args()
    n_train = 400 if args.fast else 800
    n_tune = 200 if args.fast else 300
    t_start = time.perf_counter()
    report = {"mode": "fast" if args.fast else "default", "n_train": n_train}

    # 1 ------------------------------------------------------------------
    step("1/6  Load dataset")
    check(DATA.exists(), f"dataset found at {DATA.relative_to(ROOT)}")
    df = pd.read_csv(DATA)
    check(df.shape == (4635, 28), f"shape is {df.shape} (expected (4635, 28))")
    check(int(df.isna().sum().sum()) == 0, "no missing values")
    d = add_physics_features(df)

    # 2 ------------------------------------------------------------------
    step("2/6  Verify inductive-link circuit relations")
    e = electrical_from_LRk(d.L1, d.L2, d.R1, d.R2, d.k)
    errs = {"Q1": float(np.max(np.abs(e.Q1 / d.Q1 - 1))),
            "eta_max": float(np.max(np.abs(e.Eff - d.Eff_max) / d.Eff_max)),
            "Ro": float(np.max(np.abs(e.Ro / d.Ro - 1)))}
    for k_, v in errs.items():
        check(v < 1e-9, f"{k_} relation: max relative error {v:.1e}")
    report["circuit_relation_max_error"] = errs

    # 3 ------------------------------------------------------------------
    step("3/6  Geometric feasibility (inner opening >= 1 mm)")
    frac = float(is_feasible(d).mean())
    check(frac == 1.0, f"{frac*100:.1f}% of FEA designs are feasible")

    # 4 ------------------------------------------------------------------
    step("4/6  Analytical physics descriptors vs FEA")
    r_L = float(np.corrcoef(np.log10(d.Lwh1), np.log10(d.L1))[0, 1])
    r_k = float(np.corrcoef(np.log10(d.kprox), np.log10(d.k))[0, 1])
    check(r_L > 0.99, f"Wheeler inductance vs FEA L1: r = {r_L:.4f}")
    check(r_k > 0.98, f"filament coupling vs FEA k:   r = {r_k:.4f}")
    report["proxy_correlation"] = {"wheeler_L1": r_L, "coupling_k": r_k}

    # 5 ------------------------------------------------------------------
    step(f"5/6  Gaussian process on a {n_train}-row subset (eta_max, sets A and B+)")
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(d)); rng.shuffle(idx)
    tr, te = idx[:n_train], idx[-696:]                      # test rows = same size as the real test set
    sets = {"A": RAW_FEATURES, "B+": RAW_FEATURES + RATIO_FEATURES + ANALYTIC_FEATURES}
    res = {}
    for name, feats in sets.items():
        t0 = time.perf_counter()
        gp = TunedGPR(n_tune=n_tune, random_state=SEED).fit(d.iloc[tr][feats], d.iloc[tr].Eff_max)
        mu, sd = gp.predict(d.iloc[te][feats], return_std=True)
        res[name] = {"R2": float(r2_score(d.iloc[te].Eff_max, mu)),
                     "MAPE_percent": float(mean_absolute_percentage_error(d.iloc[te].Eff_max, mu) * 100),
                     "median_sigma": float(np.median(sd)), "seconds": time.perf_counter() - t0}
        print(f"        set {name:3s}: R2 = {res[name]['R2']:.4f}, MAPE = {res[name]['MAPE_percent']:.2f}%, "
              f"{res[name]['seconds']:.1f}s", flush=True)
    check(res["A"]["R2"] > 0.95, "raw-feature GPR R2 > 0.95")
    check(res["B+"]["R2"] > 0.97, "physics-feature GPR R2 > 0.97")
    check(res["B+"]["R2"] >= res["A"]["R2"] - 0.002, "physics features do not hurt accuracy")
    report["gpr_eta_max"] = res

    # 6 ------------------------------------------------------------------
    step("6/6  Theory-assisted surrogate (L1, L2, R1, R2, k -> eta_max)")
    feats = sets["B+"]; preds = {}
    for t in ["L1", "L2", "R1", "R2", "k"]:
        gp = TunedGPR(n_tune=n_tune, random_state=SEED).fit(d.iloc[tr][feats], np.log10(d.iloc[tr][t]))
        preds[t] = 10 ** gp.predict(d.iloc[te][feats])
    comp = electrical_from_LRk(preds["L1"], preds["L2"], preds["R1"], preds["R2"], preds["k"])
    r2c = float(r2_score(d.iloc[te].Eff_max, comp.Eff))
    print(f"        composed eta_max: R2 = {r2c:.4f}", flush=True)
    check(r2c > 0.97, "composed surrogate R2 > 0.97")
    check(bool(((comp.Eff >= 0) & (comp.Eff <= 1)).all()), "all composed efficiencies lie in [0, 1]")
    report["composed_eta_R2"] = r2c

    report["total_seconds"] = time.perf_counter() - t_start
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "quick_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nALL CHECKS PASSED in {report['total_seconds']:.0f}s  "
          f"(summary: logs/quick_test_report.json)")
    print("Note: this is a reduced run. Full-pipeline numbers (3,244 training rows) are in docs/RESULTS.md.")


if __name__ == "__main__":
    main()
