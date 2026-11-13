"""
wpt_physics.py — physics-guided descriptors and analytical relations for
multilayer square planar PCB coils (ANNcoilsDataset, 6.78 MHz, 15 mm gap).

All lengths in mm, inductance in uH, resistance in ohm (dataset units).
"""
import numpy as np
import pandas as pd
from scipy.special import ellipk, ellipe

MU0 = 4e-7 * np.pi
FREQ_HZ = 6.78e6
OMEGA = 2 * np.pi * FREQ_HZ
DIST_MM = 15.0
PCB_THICK_MM = 1.6      # X1 = X2 (fixed in dataset)
EPS_R_FR4 = 4.4         # nominal FR-4 permittivity (used only inside a proxy)
DIN_MIN_MM = 1.0        # smallest inner opening present in the FEA data

RAW_FEATURES = ["W1", "W2", "S1", "S2", "N1", "N2", "Dext1", "Dext2", "Nlay1", "Nlay2"]

RATIO_FEATURES = ["size_ratio", "width_ratio", "spacing_ratio", "turn_product_1",
                  "turn_product_2", "turn_ratio", "layer_ratio", "total_coil_size"]

ANALYTIC_FEATURES = ["Din1", "Din2", "fill1", "fill2",
                     "log_Lwh1", "log_Lwh2", "log_Rprox1", "log_Rprox2",
                     "log_srf1", "log_srf2", "log_kprox", "x_prox", "eta_prox"]


def inner_dimension(D, N, W, S):
    """Inner side length of a square spiral: Din = Dext - 2NW - 2(N-1)S."""
    return D - 2 * N * W - 2 * (N - 1) * S


def mutual_coaxial_loops(a, b, d):
    """Mutual inductance (H) of two coaxial circular filaments, radii a, b and gap d (m)."""
    m = 4 * a * b / ((a + b) ** 2 + d ** 2)
    kk = np.sqrt(m)
    return MU0 * np.sqrt(a * b) * ((2 / kk - kk) * ellipk(m) - (2 / kk) * ellipe(m))


def eta_max_from_x(x):
    """Maximum link efficiency of a 2-coil inductive link, x = k^2 Q1 Q2."""
    x = np.maximum(x, 0)
    return x / (1 + np.sqrt(1 + x)) ** 2


def add_physics_features(d):
    """Return a copy of d with ratio features (Group B) and analytical descriptors (Group C)."""
    d = d.copy()
    # --- Group B: ratio / balance features (from the project outline)
    d["size_ratio"] = d.Dext2 / d.Dext1
    d["width_ratio"] = d.W2 / d.W1
    d["spacing_ratio"] = d.S2 / d.S1
    d["turn_product_1"] = d.N1 * d.Nlay1
    d["turn_product_2"] = d.N2 * d.Nlay2
    d["turn_ratio"] = d.N2 / d.N1
    d["layer_ratio"] = d.Nlay2 / d.Nlay1
    d["total_coil_size"] = d.Dext1 + d.Dext2

    # --- Group C: analytical electromagnetic descriptors
    for s in "12":
        N, W, S, D, NL = d["N" + s], d["W" + s], d["S" + s], d["Dext" + s], d["Nlay" + s]
        din = inner_dimension(D, N, W, S)
        davg = (D + din) / 2
        fill = (D - din) / (D + din)
        nt = N * NL                                   # series-connected turns across layers
        # Modified Wheeler inductance for a square spiral (Mohan et al., 1999), uH
        Lwh = 2.34 * MU0 * nt ** 2 * (davg * 1e-3) / (1 + 2.75 * fill) * 1e6
        length = 4 * davg * nt                        # total trace length, mm
        Rprox = length / (W + 0.07)                   # skin-effect-limited AC resistance proxy
        gap = PCB_THICK_MM / (NL - 1)                 # inter-layer dielectric gap, mm
        C = EPS_R_FR4 * 8.854e-12 * (length / NL * W * 1e-6) / (gap * 1e-3) * (NL - 1)
        srf = OMEGA ** 2 * Lwh * 1e-6 * C             # (f / f_self-resonance)^2 proxy
        d["Din" + s], d["Davg" + s], d["fill" + s] = din, davg, fill
        d["Lwh" + s], d["len" + s], d["Rprox" + s], d["srf" + s] = Lwh, length, Rprox, srf
        d["log_Lwh" + s] = np.log10(Lwh)
        d["log_Rprox" + s] = np.log10(Rprox)
        d["log_srf" + s] = np.log10(srf)

    # Coupling proxy: equivalent coaxial filaments at the mean radius
    a = d.Davg1 / 2 * 1e-3
    b = d.Davg2 / 2 * 1e-3
    M = mutual_coaxial_loops(a, b, DIST_MM * 1e-3) * d.turn_product_1 * d.turn_product_2 * 1e6
    d["kprox"] = M / np.sqrt(d.Lwh1 * d.Lwh2)
    d["log_kprox"] = np.log10(d["kprox"])
    Q1p = OMEGA * d.Lwh1 * 1e-6 / (d.Rprox1 * 1e-3)
    Q2p = OMEGA * d.Lwh2 * 1e-6 / (d.Rprox2 * 1e-3)
    d["x_prox"] = np.log10(d.kprox ** 2 * Q1p * Q2p)
    d["eta_prox"] = eta_max_from_x(d.kprox ** 2 * Q1p * Q2p)
    return d


def is_feasible(d, din_min=DIN_MIN_MM):
    """Geometric feasibility: the spiral must leave an inner opening >= din_min on both coils."""
    din1 = inner_dimension(d.Dext1, d.N1, d.W1, d.S1)
    din2 = inner_dimension(d.Dext2, d.N2, d.W2, d.S2)
    return (din1 >= din_min) & (din2 >= din_min)


def electrical_from_LRk(L1, L2, R1, R2, k):
    """Derive Q1, Q2, x = k^2Q1Q2, Eff_max and optimal load Ro from predicted L, R, k."""
    Q1 = OMEGA * L1 * 1e-6 / R1
    Q2 = OMEGA * L2 * 1e-6 / R2
    x = k ** 2 * Q1 * Q2
    eff = eta_max_from_x(x)
    Ro = R2 * np.sqrt(1 + x)
    return pd.DataFrame({"Q1": Q1, "Q2": Q2, "x": x, "Eff": eff, "Ro": Ro})


def pow10(z):
    """Inverse of log10 (named function so fitted models can be pickled)."""
    return np.power(10.0, z)
