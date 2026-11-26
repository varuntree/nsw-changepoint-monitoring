"""Implementation check for FOCuS (a row of repository table checks).

focus_cpt 0.1.11 does not consider the first observation (sometimes the first two) of the series it is given as a
change start, so its statistic can fall below the exact maximum of Eq. (2.4) over every start. The monitor gives a
new series at the start of every segment (after each 28-day relearn), so those opening days were always excluded.
Since 24 Oct 2026 the code uses the exact direct search (cpd/kernels.focus_exact, focus_ar_exact; unit tests in
tests/test_focus_exact.py). This script records the discrepancy:
  focus_check.csv        50 random N(0,1) series of 365 days (25 with a 1-SD drop at day 100-250), Gaussian
                         FOCuS and FOCuS-AR (phi = 0.4 on AR(1) data): share of days on which the package equals the
                         exact statistic, the largest shortfall, the share of days where the exact statistic is
                         above 10 and the package differs, and the agreement of exact with a brute-force loop
  focus_check_opal.csv   every Opal monitor with FOCuS, FOCuS-AR or capped FOCuS (all adjustments, both rules) run
                         with the package and with the exact statistic, same thresholds: alarms, alarms in common,
                         direction-aware and frozen-rule episodes
  focus_check_example.csv  the Holt-Winters + capped FOCuS (AR(1)) segment from 15 Apr 2020: both statistics
                         day by day for 10 days, with the threshold
Thresholds: results/calib/h_table.json (exact); /tmp is not used."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import calibrate, monitor, score  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd.kernels import ar1  # noqa: E402

R = HERE / "results"
OP = R / "opal"


def brute(y):
    S = np.concatenate([[0.0], np.cumsum(y)])
    return np.array([np.max((S[t] - S[:t]) ** 2 / (t - np.arange(t))) for t in range(1, len(y) + 1)])


def random_check():
    rng = np.random.default_rng(20261024)
    rows = []
    for i in range(50):
        for det, phi in (("FOCuS", 0.0), ("FOCuS-AR", 0.4)):
            y = ar1(365, phi, rng.standard_normal(365)) if phi else rng.standard_normal(365)
            if i % 2:
                y[int(rng.integers(100, 251)):] -= 1.0
            ex = D.stat_path(det, y, phi)
            pk, _ = D._focus_pkg(y, phi, "gaussian" if det == "FOCuS" else "arp")
            pk = np.where(pk < 0, 0.0, pk)
            diff = ~np.isclose(pk, ex)
            rows.append(dict(series=i, detector=det, shift=bool(i % 2), share_equal=float((~diff).mean()),
                             max_shortfall=float(np.max(ex - pk)), share_differ_above10=float((diff & (ex > 10)).mean()),
                             exact_equals_bruteforce=bool(np.allclose(ex, brute(y))) if det == "FOCuS" else np.nan))
    return pd.DataFrame(rows)


def opal_job(args):
    m, det, rule, kind, r, t0, dates = args
    tabs = calibrate.load(R / "calib" / "h_table.json", kind)
    out = {}
    for pkg in (True, False):
        D.USE_PACKAGE = pkg
        res = monitor.run(r, kind, det, rule, tabs, t0, method=m)
        D.USE_PACKAGE = False
        ad = dates[res.alarms]
        out[pkg] = (set(ad), score.episodes_caught(ad, np.array(res.dirs))[0],
                    score.episodes_caught(ad, None, 7, 7, credit="all")[0])
    return dict(method=m, detector=det, rule=rule, alarms_package=len(out[True][0]), alarms_exact=len(out[False][0]),
                alarms_in_common=len(out[True][0] & out[False][0]), dir_package=out[True][1], dir_exact=out[False][1],
                frozen_package=out[True][2], frozen_exact=out[False][2])


def main():
    rc = random_check()
    rc.to_csv(R / "calib" / "focus_check.csv", index=False)
    print(rc.groupby("detector")[["share_equal", "max_shortfall", "share_differ_above10"]].agg(["mean", "max"]).round(3))
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    kinds = {m: ("level" if m in ("raw", "STL", "MSTL", "REG") else "surprise") for m in adj.columns}
    jobs = [(m, d, ru, kinds[m], adj[m].to_numpy(), t0, dates) for m in adj.columns for d in ("FOCuS", "FOCuS-AR", "FOCuS-r")
            for ru in ("indep", "ar1") if not (d == "FOCuS-AR" and ru == "indep")]
    with ProcessPoolExecutor(10) as ex:
        op = pd.DataFrame(list(ex.map(opal_job, jobs)))
    op.to_csv(OP / "focus_check_opal.csv", index=False)
    print(op[(op.dir_package != op.dir_exact) | (op.frozen_package != op.frozen_exact)].to_string(index=False))
    # the 15 Apr 2020 segment, Holt-Winters + capped FOCuS, AR(1)
    tabs = calibrate.load(R / "calib" / "h_table.json", "surprise")
    r = adj["HW"].to_numpy()
    D.USE_PACKAGE = True
    res = monitor.run(r, "surprise", "FOCuS-r", "ar1", tabs, t0, method="HW")
    D.USE_PACKAGE = False
    s0 = min(res.seg_starts, key=lambda s: abs((dates[s] - pd.Timestamp("2020-04-15")).days))
    blind = res.blind.copy()
    blind[s0:] = False                     # the state of the monitor at the start of that segment
    dl = monitor.delevel(r, False)
    ph = monitor.phi_hat(dl, blind, s0)
    z = np.clip(r[s0:] / monitor.running_scale(dl, blind, s0, len(r)), -D.CLIP, D.CLIP)[:10]
    ex = D.stat_path("FOCuS", z)
    pk, _ = D._focus_pkg(z, 0.0, "gaussian")
    pd.DataFrame(dict(date=dates[s0:s0 + 10].date, package=pk, exact=ex, threshold=tabs["FOCuS-r"](ph))).to_csv(
        R / "calib" / "focus_check_example.csv", index=False)
    print(pd.read_csv(R / "calib" / "focus_check_example.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
