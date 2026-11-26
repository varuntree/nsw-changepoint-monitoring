"""Tests for variance FOCuS (focus_cpt gamma family on z^2). Guards the parametrisation: focus_cpt's gamma
`theta0` is the scale, so an in-control mean of 1 for z^2 ~ Gamma(1/2, scale 2) needs theta0 = 2."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import detectors as D  # noqa: E402


def _gamma_glr(y, shape, mean0):
    """Direct GLR statistic at the last time point: max over changepoints of the log-likelihood ratio of a
    Gamma(shape) segment with free scale against the in-control scale mean0 / shape."""
    th0 = mean0 / shape
    s = np.cumsum(y[::-1])
    m = np.arange(1, len(y) + 1)
    th = s / (shape * m)
    return float(np.max(-shape * m * np.log(th) - shape * m + shape * m * np.log(th0) + s / th0))


def test_matches_direct_gamma_glr_with_mean_one():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(400) * np.r_[np.ones(250), np.full(150, 1.4)]
    path = D.stat_path("FOCuS-var", z)
    y = z * z + 1e-12
    for t in (30, 150, 249, 320, 399):
        assert abs(path[t] - _gamma_glr(y[: t + 1], 0.5, 1.0)) < 1e-6 * max(1.0, path[t])


def test_in_control_statistic_does_not_drift():
    # GLR statistics grow like log log n in control; a wrong in-control mean makes them grow linearly in n
    rng = np.random.default_rng(2)
    med = np.median([D.stat_path("FOCuS-var", rng.standard_normal(4000))[[499, 3999]] for _ in range(30)], axis=0)
    assert med[1] < 10 and med[1] < 2 * med[0] + 3
