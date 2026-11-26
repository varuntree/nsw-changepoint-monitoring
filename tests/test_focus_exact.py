"""Exact FOCuS and FOCuS-AR against brute-force searches over every start (Eq. 2.4 of the thesis), and the
relation to focus_cpt 0.1.11, which leaves out the first one or two starts of each series it is given."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import detectors as D  # noqa: E402


def brute_gauss(y, smin=0):
    S = np.concatenate([[0.0], np.cumsum(y)])
    out = []
    for t in range(1, len(y) + 1):
        s = np.arange(smin, t)
        out.append(np.max((S[t] - S[s]) ** 2 / (t - s)) if len(s) else -1.0)
    return np.array(out)


def brute_ar(x, phi, smin=0):
    n = len(x)
    w = np.empty(n)
    w[0] = np.sqrt(1 - phi * phi) * x[0]
    w[1:] = x[1:] - phi * x[:-1]
    out = np.full(n, -1.0)
    for t in range(n):
        for s in range(smin, t + 1):
            a = np.r_[np.sqrt(1 - phi * phi), np.full(t, 1 - phi)] if s == 0 else np.r_[1.0, np.full(t - s, 1 - phi)]
            out[t] = max(out[t], (a @ w[s:t + 1]) ** 2 / (a @ a))
    return out


def test_toy_series():
    assert np.allclose(D.stat_path("FOCuS", np.array([3.0, 3, 3])), [9, 18, 27])
    assert np.allclose(D.stat_path("FOCuS", np.ones(12)), np.arange(1, 13))
    assert np.allclose(D.stat_path("FOCuS-AR", np.array([3.0, 3, 3]), 0.0), [9, 18, 27])


def test_gaussian_matches_brute_force_and_package_never_exceeds_it():
    rng = np.random.default_rng(1)
    for _ in range(60):
        y = rng.standard_normal(150) + 0.4 * (rng.random() < 0.5)
        ex = D.stat_path("FOCuS", y)
        assert np.allclose(ex, brute_gauss(y))
        pkg, _ = D._focus_pkg(y, kind="gaussian")
        assert np.all(pkg <= ex + 1e-9)
    # the maximising start is a valid change start
    y = np.r_[rng.standard_normal(50), rng.standard_normal(30) + 2]
    assert abs(D.change_start("FOCuS", y, 79) - 50) <= 3


def test_ar_matches_brute_force_and_package_is_the_restricted_search():
    rng = np.random.default_rng(2)
    for phi in (0.0, 0.3, 0.6):
        for _ in range(15):
            x = rng.standard_normal(70) + 0.3
            ex = D.stat_path("FOCuS-AR", x, phi)
            assert np.allclose(ex, brute_ar(x, phi))
            pkg, _ = D._focus_pkg(x, phi, kind="arp")
            assert np.allclose(pkg[2:], brute_ar(x, phi, smin=2)[2:])


def test_missing_days_are_skipped():
    z = np.array([1.0, np.nan, 2.0, 3.0])
    assert np.allclose(D.stat_path("FOCuS", z), [1, 1, 4.5, 12.5])
    assert D.change_start("FOCuS", z, 3) == 2


def test_frozen_rule_lets_one_alarm_catch_two_episodes():
    import pandas as pd
    from cpd import score
    # 21 Feb 2022 (shutdown onset) is 3 days after 18 Feb 2022 (Omicron end): one alarm on 20 Feb is within +-7 of both
    a = pd.DatetimeIndex(["2022-02-20"])
    assert score.episodes_caught(a, None, after=7, before=7, credit="all")[0] == 2
    assert score.episodes_caught(a, None, after=7, before=7)[0] == 1


def _gamma_brute(y, shape=0.5, mean0=1.0):
    th0 = mean0 / shape
    out = []
    for t in range(1, len(y) + 1):
        s = np.cumsum(y[:t][::-1])
        m = np.arange(1, t + 1)
        th = s / (shape * m)
        out.append(float(np.max(-shape * m * np.log(th) - shape * m + shape * m * np.log(th0) + s / th0)))
    return np.array(out)


def test_all_three_focus_statistics_against_brute_force_including_changes_at_the_start():
    """The statistic the code uses equals a direct O(n^2) GLR (to 1e-6) for Gaussian FOCuS,
    FOCuS-AR at phi = 0.4 and variance FOCuS, on 50 random series and on series whose change starts at the first,
    second or third observation. The package's shortfall is printed (pytest -s) for Appendix D."""
    rng = np.random.default_rng(24)
    series = [rng.standard_normal(120) * (1 + 0.5 * (i % 2)) + 0.5 * (i % 3 == 0) for i in range(50)]
    for start in (0, 1, 2):
        y = rng.standard_normal(60)
        y[start:] += 1.5
        series.append(y)
    short = {"FOCuS": 0.0, "FOCuS-AR": 0.0, "FOCuS-var": 0.0}
    for y in series:
        g = D.stat_path("FOCuS", y)
        assert np.allclose(g, brute_gauss(y), atol=1e-6)
        a = D.stat_path("FOCuS-AR", y, 0.4)
        assert np.allclose(a, brute_ar(y, 0.4), atol=1e-6)
        v = D.stat_path("FOCuS-var", y)
        assert np.allclose(v, _gamma_brute(y * y + 1e-12), atol=1e-6 * max(1.0, v.max()))
        short["FOCuS"] = max(short["FOCuS"], float(np.max(g - D._focus_pkg(y, 0.0, "gaussian")[0])))
        short["FOCuS-AR"] = max(short["FOCuS-AR"], float(np.max(a - np.maximum(D._focus_pkg(y, 0.4, "arp")[0], 0))))
    print("\nlargest shortfall of focus_cpt 0.1.11 against the exact statistic:", {k: round(x, 2) for k, x in short.items()})
    # a change from the very first observation: the exact statistic sees it at once, the package does not
    y = np.r_[np.full(5, 3.0)]
    assert D.stat_path("FOCuS", y)[0] == 9.0 and D._focus_pkg(y, 0.0, "gaussian")[0][0] == 0.0
