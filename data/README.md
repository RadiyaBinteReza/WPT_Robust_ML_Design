# Dataset: ANNcoilsDataset (ANNCoilsFEA.csv)

| | |
|---|---|
| **File** | `raw/ANNCoilsFEA.csv` (4,635 rows × 28 columns, no missing values) |
| **Source** | https://github.com/alvarordf/ANNcoilsDataset |
| **Authors** | Á. Rodríguez Fuentes, M. Jiménez Carrizosa, R. Ramos Hortal |
| **Licence** | Creative Commons Attribution 4.0 (CC-BY-4.0): free reuse with attribution |
| **Simulator** | Ansys HFSS 2025 R1 |
| **SHA-256** | `761441264a3c1623d53538b412a483e8a3a44873fe808b9ef98417fa5e180d2b` |

Check the file with `bash scripts/bash/02_check_data.sh`. Re-download it with `bash scripts/bash/01_get_data.sh`. The original repository also provides `ANNCoilsFEA.mat`, which this project does not use.

## What one row is

One row is one HFSS simulation of a **coaxially aligned pair of multilayer square planar PCB coils**: a primary/transmitter (index 1) and a secondary/receiver for the implant (index 2).

## Columns

| Column | Meaning | Unit | Range in data |
|---|---|---|---|
| `Freq` | operating frequency (fixed) | MHz | 6.78 |
| `Dist` | coil separation (fixed) | mm | 15 |
| `W1`, `W2` | trace width, primary / secondary | mm | 0.5–4 / 0.5–3 |
| `S1`, `S2` | spacing between turns | mm | 0.5–3 / 0.5–2 |
| `N1`, `N2` | turns per layer | – | 1–5 |
| `Dext1`, `Dext2` | outer side length of the square spiral | mm | 10–40 / 5–20 |
| `Nlay1`, `Nlay2` | number of copper layers (connected in series) | – | {2, 4} |
| `H1`, `H2` | copper thickness code (fixed) | – | 2 (see note) |
| `X1`, `X2` | PCB thickness (fixed) | mm | 1.6 |
| `Z11`, `Z22`, `Z12` | complex impedance matrix entries, as text `a+bi` | Ω | – |
| `L1`, `L2` | self-inductance | µH | 0.026–142 / 0.012–7.4 |
| `R1`, `R2` | AC resistance | Ω | 0.017–906 / 0.013–5.7 |
| `k` | coupling factor | – | 0.009–0.23 |
| `Q1`, `Q2` | quality factor ωL/R | – | 6.7–265 / 13–172 |
| `Ro` | optimal load resistance R2·√(1+k²Q1Q2) | Ω | – |
| `Eff_max` | maximum theoretical link efficiency x/(1+√(1+x))², x = k²Q1Q2 | – | 0.033–0.905 |

`Q1`, `Q2`, `Ro` and `Eff_max` follow exactly from `L1`, `L2`, `R1`, `R2`, `k`. This is verified in the notebook (Part 12C) and in `tests/test_physics.py`.

## Notes

* **Copper thickness:** the source documentation states 1 oz in its text but lists 2 in its table, and the data contain 2. This project assumes 2 oz (≈ 70 µm). The value is constant across all rows, so it only scales one proxy feature.
* **Geometric feasibility:** every row has an inner opening Din = Dext − 2NW − 2(N−1)S ≥ 1 mm on both coils.

## Citation (required by the licence)

> A. Rodriguez-Fuentes, M. J. Carrizosa and R. Ramos, "Neural Network-Based Design of Wireless Power Transfer Systems for Implantable Medical Devices," *IEEE Transactions on Power Electronics*, vol. 41, no. 2, pp. 3011–3024, Feb. 2026, doi: 10.1109/TPEL.2025.3614366.
