"""Unit tests for src/wpt_physics.py (circuit relations, geometry, descriptors)."""
import numpy as np
import pandas as pd
import pytest

from wpt_physics import (RAW_FEATURES, ANALYTIC_FEATURES, inner_dimension, is_feasible, eta_max_from_x,
                         electrical_from_LRk, mutual_coaxial_loops, pow10, MU0)


def test_dataset_shape_and_columns(data):
    assert len(data) == 4635
    for c in RAW_FEATURES + ["L1", "L2", "R1", "R2", "k", "Q1", "Q2", "Ro", "Eff_max"]:
        assert c in data.columns


def test_fixed_operating_point(data):
    assert set(data.Freq.unique()) == {6.78}
    assert set(data.Dist.unique()) == {15}


def test_circuit_relations_hold_exactly(data):
    e = electrical_from_LRk(data.L1, data.L2, data.R1, data.R2, data.k)
    assert np.allclose(e.Q1, data.Q1, rtol=1e-9)
    assert np.allclose(e.Q2, data.Q2, rtol=1e-9)
    assert np.allclose(e.Eff, data.Eff_max, rtol=1e-9)
    assert np.allclose(e.Ro, data.Ro, rtol=1e-9)


def test_eta_max_formula_bounds_and_monotonic():
    x = np.logspace(-3, 5, 200)
    eta = eta_max_from_x(x)
    assert np.all((eta >= 0) & (eta < 1))
    assert np.all(np.diff(eta) > 0)
    assert eta_max_from_x(np.array([0.0]))[0] == 0.0


def test_inner_dimension_formula():
    # 3 turns, 2 mm traces, 1 mm gaps on a 20 mm coil: 20 - 2*3*2 - 2*2*1 = 4 mm
    assert inner_dimension(20.0, 3, 2.0, 1.0) == pytest.approx(4.0)


def test_all_fea_designs_are_feasible(data):
    assert is_feasible(data).all()


def test_infeasible_geometry_is_detected():
    g = pd.DataFrame([{"W1": 3, "W2": 1, "S1": 2, "S2": 1, "N1": 5, "N2": 1,
                       "Dext1": 10, "Dext2": 10, "Nlay1": 2, "Nlay2": 2}])
    assert not is_feasible(g).iloc[0]          # 5 wide turns cannot fit in 10 mm


def test_mutual_inductance_symmetric_and_decreasing():
    a, b = 0.01, 0.005
    assert mutual_coaxial_loops(a, b, 0.015) == pytest.approx(mutual_coaxial_loops(b, a, 0.015))
    assert mutual_coaxial_loops(a, b, 0.01) > mutual_coaxial_loops(a, b, 0.03) > 0


def test_analytical_descriptors_finite_and_informative(data):
    assert np.isfinite(data[ANALYTIC_FEATURES].values).all()
    assert np.corrcoef(np.log10(data.Lwh1), np.log10(data.L1))[0, 1] > 0.99
    assert np.corrcoef(np.log10(data.Lwh2), np.log10(data.L2))[0, 1] > 0.99
    assert np.corrcoef(np.log10(data.kprox), np.log10(data.k))[0, 1] > 0.98


def test_pow10_inverts_log10():
    z = np.array([-2.0, 0.0, 1.5])
    assert np.allclose(np.log10(pow10(z)), z)


def test_mu0_value():
    assert MU0 == pytest.approx(4e-7 * np.pi)
