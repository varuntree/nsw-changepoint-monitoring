"""Validation tests (run: .venv/bin/python -m pytest -q tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import kernels as K  # noqa: E402


def _arl(path_fn, h, *a, reps=6000, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    rl = []
    for _ in range(reps):
        t = K.first_cross(path_fn(rng.standard_normal(n), *a), h)
        rl.append(t + 1 if t >= 0 else n)
    return float(np.mean(rl))


def test_cusum_arl0_matches_published():
    # two-sided tabular CUSUM, k = 0.5: ARL0 168 (h = 4) and 465 (h = 5), Montgomery (2019) Table 9.3
    assert abs(_arl(K.cusum_path, 4.0, 0.5) - 168) / 168 < 0.06
    assert abs(_arl(K.cusum_path, 5.0, 0.5) - 465) / 465 < 0.06


def test_ewma_arl0_close_to_published():
    # Lucas and Saccucci (1990): lambda 0.1, L 2.814 gives ARL0 500 with asymptotic limits; exact-variance
    # limits alarm a little earlier at the start, so allow 10%
    assert abs(_arl(K.ewma_path, 2.814, 0.1) - 500) / 500 < 0.10


def test_cpm_matches_scipy_mann_whitney():
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(3)
    x = np.round(rng.standard_normal(60), 1)          # ties on purpose
    p = K.cpm_mw_path(x, 20)
    n = len(x)
    _, cnt = np.unique(x, return_counts=True)
    ts = (cnt ** 3 - cnt).sum()
    best = 0.0
    for k in range(2, n - 1):
        u = mannwhitneyu(x[k:], x[:k], method="asymptotic", use_continuity=False).statistic
        n1, n2 = k, n - k
        var = n1 * n2 / 3 * ((n + 1) - ts / (n * (n - 1)))
        best = max(best, abs(2 * u - n1 * n2) / np.sqrt(var))
    assert abs(p[-1] - best) < 1e-9


def test_pelt_matches_ruptures():
    import ruptures as rpt
    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(150, 400))
        x = rng.standard_normal(n)
        for c in rng.integers(20, n - 20, size=int(rng.integers(0, 4))):
            x[c:] += rng.normal(0, 2)
        pen = 3 * np.log(n)
        a = list(K.pelt_l2(x, pen, 7))
        b = [v for v in rpt.Pelt(model="l2", min_size=7, jump=1).fit(x.reshape(-1, 1)).predict(pen=pen) if v < n]
        assert a == b


def test_offline_find_obvious_changes():
    rng = np.random.default_rng(5)
    y = np.r_[rng.standard_normal(200), rng.standard_normal(200) + 2, rng.standard_normal(200) - 1]
    ints = np.sort(rng.integers(0, 600, size=(2000, 2)), axis=1)
    for cps in (K.pelt_l2(y, 3 * np.log(600), 7), K.binseg_l2(y, 4.0, 7, 10), K.wbs_l2(y, 4.5, 7, ints)):
        assert any(abs(c - 200) <= 3 for c in cps) and any(abs(c - 400) <= 3 for c in cps)
