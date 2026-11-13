"""Unit tests for src/gpr_model.py and a tiny end-to-end surrogate fit."""
import numpy as np
import pytest
from sklearn.metrics import r2_score

from gpr_model import TunedGPR
from wpt_physics import RAW_FEATURES, RATIO_FEATURES, ANALYTIC_FEATURES


def test_tuned_gpr_fits_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (300, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - 0.2 * X[:, 2]
    gp = TunedGPR(n_tune=150, random_state=0).fit(X[:250], y[:250])
    mu, sd = gp.predict(X[250:], return_std=True)
    assert mu.shape == (50,) and sd.shape == (50,)
    assert np.all(sd > 0)
    assert r2_score(y[250:], mu) > 0.98


def test_tuned_gpr_batched_prediction_matches_unbatched():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2)); y = X[:, 0] - X[:, 1]
    a = TunedGPR(n_tune=100, random_state=0, batch=10_000).fit(X, y).predict(X[:120])
    b = TunedGPR(n_tune=100, random_state=0, batch=7).fit(X, y).predict(X[:120])
    assert np.allclose(a, b)


def test_small_surrogate_on_real_data(data):
    feats = RAW_FEATURES + RATIO_FEATURES + ANALYTIC_FEATURES
    rng = np.random.RandomState(42); idx = np.arange(len(data)); rng.shuffle(idx)
    tr, te = idx[:500], idx[-300:]
    gp = TunedGPR(n_tune=200, random_state=42).fit(data.iloc[tr][feats], data.iloc[tr].Eff_max)
    assert r2_score(data.iloc[te].Eff_max, gp.predict(data.iloc[te][feats])) > 0.95
