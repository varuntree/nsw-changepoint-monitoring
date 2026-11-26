"""Computing time beyond results/runtime.csv (which is left as it was):
  1. the variance EWMA and the variance FOCuS per call (variance FOCuS re-timed after the gamma parametrisation
     fix of 23 Oct 2026; its row in runtime.csv predates the fix), timed exactly as in runtime.py (single
     thread, after JIT compilation, median of 51 repeats of a batch of calls), for n = 10, 365, 1,095 and 2,463
     -> results/runtime_spread.csv (EWMA-var rows also in results/runtime_ewma_var.csv, read by tables.py)
  2. a full Opal run per past-only adjustment: the seasonal adjustment of the whole daily series (2,463 days)
     and one monitor run over the monitoring period (1 Mar 2020 to 28 Sep 2026) with CUSUM and with FOCuS
     (AR(1) rule, 28-day relearn), wall-clock seconds, median of up to 5 repeats -> results/runtime_monitor.csv
Constants reflect the implementation (numba kernels, the compiled focus_cpt package, statsmodels for MSTL), so
only growth with n and the order of magnitude are comparable between methods.
Usage: python scripts/runtime_extra.py [spread|monitor|focus]   (default: all)"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import adjust as A  # noqa: E402
from cpd import calibrate, monitor  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

REPS = 51


def timeit(f, reps=REPS, batch_s=2e-3):
    for _ in range(3):
        f()
    t = time.perf_counter()
    f()
    one = max(time.perf_counter() - t, 1e-7)
    inner = max(1, int(batch_s / one))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(inner):
            f()
        ts.append((time.perf_counter() - t) / inner)
    return float(np.median(ts)) * 1000


def wall(f, reps=5, budget_s=120):
    ts = []
    t_all = time.perf_counter()
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t)
        if time.perf_counter() - t_all > budget_s:
            break
    return float(np.median(ts)), len(ts), out


def spread():
    rng = np.random.default_rng(1)
    rows = []
    for n in (10, 365, 1095, 2463):
        z = rng.standard_normal(n)
        for d in ("EWMA-var", "FOCuS-var"):
            rows.append(dict(method=d, n=n, ms=timeit(lambda: D.stat_path(d, z, 0.4))))
    df = pd.DataFrame(rows)
    df["ns_per_obs"] = 1e6 * df["ms"] / df["n"]
    df["reps"] = REPS
    df.to_csv(HERE / "results" / "runtime_spread.csv", index=False)
    df[df["method"] == "EWMA-var"].to_csv(HERE / "results" / "runtime_ewma_var.csv", index=False)   # read by tables.py
    print(df.round(4).to_string(index=False))


def focus_timing():
    """The published focus_cpt algorithm (gaussian and arp at phi = 0.4, via focus_offline) against our
    exact direct search, per call and per observation, n = 365, 1,095, 2,463, 10^4 (both) and 10^5 (package only);
    51 repeats (5 at 10^5) of a batch, as runtime.py -> results/runtime_focus.csv. Also results/calib/calib_time.csv:
    single-thread time for 1,000 calibration years at phi = 0.4 per detector (cpd.calibrate.max_stats), scaled to a
    full table (20,000 years x 19 phi x 2 kinds of series)."""
    import focus_cpt as fc
    from cpd import calibrate
    rng = np.random.default_rng(1)
    rows = []
    for n in (365, 1095, 2463, 10_000, 100_000):
        z = rng.standard_normal(n)
        reps = 5 if n >= 100_000 else REPS
        fns = {"focus_cpt gaussian": lambda: fc.focus_offline(z, threshold=np.inf, type="univariate", family="gaussian", theta0=0.0),
               "focus_cpt arp (phi 0.4)": lambda: fc.focus_offline(z, threshold=np.inf, type="arp", family="arp", rho=[0.4], mu0_arp=0.0)}
        if n <= 10_000:
            fns["exact FOCuS (ours)"] = lambda: D.stat_path("FOCuS", z, 0.4)
            fns["exact FOCuS-AR (ours)"] = lambda: D.stat_path("FOCuS-AR", z, 0.4)
        for name, f in fns.items():
            rows.append(dict(method=name, n=n, ms=timeit(f, reps=reps), reps=reps))
    df = pd.DataFrame(rows)
    df["us_per_obs"] = 1e3 * df["ms"] / df["n"]
    df.to_csv(HERE / "results" / "runtime_focus.csv", index=False)
    print(df.round(4).to_string(index=False))
    ct = []
    for det in ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "FOCuS-r"):
        t = time.perf_counter()
        calibrate.max_stats(0.4, "surprise", 1000, 1, dets=(det,))
        sec = time.perf_counter() - t
        ct.append(dict(det=det, seconds_per_1000_years=sec, full_table_hours_single_thread=sec * 20 * 19 * 2 / 3600))
    pd.DataFrame(ct).to_csv(HERE / "results" / "calib" / "calib_time.csv", index=False)
    print(pd.DataFrame(ct).round(3).to_string(index=False))


def monitor_runs():
    x = opal_daily("train")
    xv, dates = x.to_numpy(), x.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    tabs = calibrate.load(HERE / "results" / "calib" / "h_table.json", "surprise")
    rows = []
    for m in ("A", "B", "HW", "YM", "MSTL_p", "REG_p"):
        sec, k, (r, _) = wall(lambda: A.adjust("HW" if m == "YM" else m, xv, dates))
        rows.append(dict(adj=m, step="adjust", det="", seconds=sec, repeats=k, n_days=len(xv)))
        for det in ("CUSUM", "FOCuS"):
            monitor.run(r, "surprise", det, "ar1", tabs, t0, method=m)          # compile / warm up
            sec, k, res = wall(lambda: monitor.run(r, "surprise", det, "ar1", tabs, t0, method=m))
            rows.append(dict(adj=m, step="monitor", det=det, seconds=sec, repeats=k, n_days=len(xv) - t0,
                             n_alarms=len(res.alarms)))
        print(rows[-3:], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "results" / "runtime_monitor.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("spread", "both"):
        spread()
    if what in ("monitor", "both"):
        monitor_runs()
    if what in ("focus", "both"):
        focus_timing()
