"""Tests for the variance EWMA (exponentially weighted mean square of z, MacGregor and Harris 1993)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import calibrate  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd import kernels as K  # noqa: E402


def _reference(z, lam):
    """Plain-Python version of the definition in kernels.ewma_var_path."""
    e, m, out = 1.0, 0, []
    for x in z:
        if not np.isnan(x):
            e = lam * x * x + (1 - lam) * e
            m += 1
        out.append(0.0 if m == 0 else (e - 1) / np.sqrt(2 * lam / (2 - lam) * (1 - (1 - lam) ** (2 * m))))
    return np.array(out)


def test_matches_definition_with_missing_days():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(400) * np.r_[np.ones(200), np.full(200, 1.5)]
    z[[0, 5, 6, 150]] = np.nan
    for lam in (0.05, 0.1, 0.3):
        assert np.allclose(K.ewma_var_path(z, lam), _reference(z, lam), atol=1e-12)
    assert np.allclose(D.stat_path("EWMA-var", z), _reference(z, D.PARAMS["EWMA-var"]["lam"]), atol=1e-12)


def test_first_value_and_in_control_standardisation():
    # D_1 = (z_1^2 - 1) / sqrt(2) exactly; for iid N(0,1) every D_t has mean 0 and variance 1
    z = np.array([1.7, -0.3])
    assert abs(K.ewma_var_path(z, 0.1)[0] - (1.7 ** 2 - 1) / np.sqrt(2)) < 1e-12
    rng = np.random.default_rng(2)
    d = np.array([K.ewma_var_path(rng.standard_normal(60), 0.1)[[4, 59]] for _ in range(20000)])
    assert np.all(np.abs(d.mean(axis=0)) < 0.03)
    assert np.all(np.abs(d.var(axis=0) - 1) < 0.05)


def test_detects_increase_not_decrease():
    rng = np.random.default_rng(3)
    h = 3.0

    def arl(sd):
        rl = []
        for _ in range(2000):
            t = K.first_cross(K.ewma_var_path(sd * rng.standard_normal(2000), 0.1), h)
            rl.append(t + 1 if t >= 0 else 2000)
        return np.mean(rl)

    a1, a2, ah = arl(1.0), arl(np.sqrt(2.0)), arl(np.sqrt(0.5))
    assert a2 < a1 / 5          # doubling the variance is found much sooner
    assert ah > a1              # a fall in variance does not raise the one-sided statistic


def test_same_calibration_path_gives_target_on_fresh_seeds():
    # threshold from the calibration code (running scale included) on one set of seeds, checked on another
    reps = 3000
    h = np.quantile(calibrate.max_stats(0.0, "surprise", reps, 11, dets=("EWMA-var",))["EWMA-var"], 0.95)
    fa = (calibrate.max_stats(0.0, "surprise", reps, 12, dets=("EWMA-var",))["EWMA-var"] > h).mean()
    assert abs(fa - 0.05) < 3 * np.sqrt(0.05 * 0.95 / reps) + 0.005
