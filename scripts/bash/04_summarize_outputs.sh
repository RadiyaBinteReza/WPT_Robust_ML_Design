#!/usr/bin/env bash
# List produced figures/tables and print headline numbers from key_results.json.
set -euo pipefail
echo "== Figures ==";  ls -1 results_python/figures | sed 's/^/  /'
echo "== Tables ==";   ls -1 results_python/tables  | sed 's/^/  /'
echo "== Paper figures =="; ls -1 paper/figures | sed 's/^/  /'
python3 - <<'PY'
import json
k = json.load(open("results_python/key_results.json"))
print("\n== Selected surrogate per target ==")
for t, v in k["selected_models"].items():
    ft = k["final_test"][t]
    print(f"  {t:8s} {v['model']:18s} set {v['feature_set']:3s}  test R2={ft['R2_model_space']:.5f}  MAPE={ft['MAPE_percent']:.2f}%")
print("\n== Final designs ==")
for cat, d in k["final_designs"].items():
    print(f"  {cat:20s} eta={d['Nominal_Eff']:.3f}  robust P5={d['Robust_Eff_P5']:.3f}  k={d['k']:.3f}")
PY
