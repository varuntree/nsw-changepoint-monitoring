"""CPM kernel on fully tied starts (edge-case tests of the independent CPM code, 27 Oct 2026): our kernel raised
ZeroDivisionError when the first 20 or more observed values were equal. It must return 0 while every value is tied,
then agree with a brute-force tie-corrected Mann-Whitney statistic once values differ."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import kernels as K  # noqa: E402


def brute(z, startup=20):
    out = np.zeros(len(z))
    xs = []
    for t, x in enumerate(z):
        if np.isnan(x):
            out[t] = out[t - 1] if t else 0.0
            continue
        xs.append(x)
        n = len(xs)
        if n < startup:
            continue
        a = np.array(xs)
        _, cnt = np.unique(a, return_counts=True)
        corr = (n + 1) - (cnt ** 3 - cnt).sum() / (n * (n - 1))
        best = 0.0
        for k in range(2, n - 1):
            if corr <= 1e-9:
                break
            u = np.sign(a[k:][:, None] - a[:k][None, :]).sum()
            best = max(best, abs(u) / np.sqrt(k * (n - k) * corr / 3))
        out[t] = best
    return out


def test_all_equal_start_returns_zero():
    z = np.zeros(40)
    assert np.all(K.cpm_mw_path(z, 20) == 0.0)


def test_equal_then_shift_matches_brute_force():
    # the fixtures "equal" and "shift" of those tests: 25 equal values, then a level shift, with a missing day
    z = np.concatenate([np.full(25, 1.5), np.full(15, 3.0)])
    z[30] = np.nan
    got = K.cpm_mw_path(z, 20)
    assert np.all(got[:25] == 0.0) and got[-1] > 0
    assert np.allclose(got, brute(z), atol=1e-12)


def test_signed_zero_is_one_tie_group():
    z = np.concatenate([np.arange(1.0, 19.0), [0.0, -0.0]])
    assert np.allclose(K.cpm_mw_path(z, 20), brute(z), atol=1e-12)


def test_random_tied_series_unchanged_by_guard():
    rng = np.random.default_rng(5)
    for _ in range(20):
        z = np.round(rng.standard_normal(int(rng.integers(30, 200))), 1)
        z[rng.random(len(z)) < 0.2] = np.nan
        assert np.allclose(K.cpm_mw_path(z, 20), brute(z), atol=1e-10)
