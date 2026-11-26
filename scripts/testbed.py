"""Real-data test bed: known changes inserted into real Opal train tap-ons (2023-01-01 .. 2026-09-28).

For each eligible start day tau (every second eligible day) and each insertion setting, the series is
modified from tau onwards in RAW counts, re-adjusted (past-only methods only change from tau; the
retrospective STL reference is refitted), and monitored from tau-90 (fresh start) to tau+60. The same
window is monitored on the unmodified series (paired control). Settings:
  drop_p    permanent level drop of p% (2.5, 5, 10, 20, 40)
  temp_p    14-day drop of p% (5, 10, 20, 40), then back to normal
  spread_v  variance of the log series around a hindsight baseline multiplied by v (1.5, 2)
Eligible days exclude: the industrial-action period (18 Sep 2024 - 19 Feb 2025), publisher faults
+-30 d, the Metro opening +-60 d, incomplete days, 20 Dec - 15 Jan, scored events +-30 d, and
(screened variant) +-30 d around changes found by a strict PELT on the hindsight STL residual.
Adjustments: A, B, HW, YM (past-only), REG_p and its retrospective twin REG (RQ3 pair), STL (retrospective).
Missing days stay missing (the adjustments skip them; STL interpolates internally).
Output: results/testbed/pairs.csv (one row per tau x setting x adjustment x detector x rule).
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import adjust as A  # noqa: E402
from cpd import calibrate, monitor  # noqa: E402
from cpd import kernels as K  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

OUT = HERE / "results" / "testbed"
OUT.mkdir(parents=True, exist_ok=True)
TABLE = HERE / "results" / "calib" / "h_table.json"
ADJS = ("A", "B", "HW", "YM", "STL", "REG", "REG_p")
DETS = ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "FOCuS-r")
RULES = ("indep", "ar1")
SETTINGS = [("drop", 2.5), ("drop", 5), ("drop", 10), ("drop", 20), ("drop", 40),
            ("temp", 5), ("temp", 10), ("temp", 20), ("temp", 40), ("spread", 1.5), ("spread", 2.0)]
PRE, POST = 90, 60


def eligible(x: pd.Series):
    d = x.index
    ok = (d >= "2023-01-01") & (d <= pd.Timestamp("2026-09-28") - pd.Timedelta(days=POST))
    def excl(a, b, pad=0):
        nonlocal ok
        ok &= ~((d >= pd.Timestamp(a) - pd.Timedelta(days=pad)) & (d <= pd.Timestamp(b) + pd.Timedelta(days=pad)))
    excl("2024-09-18", "2025-02-19")
    for a, b in [("2025-01-28", "2025-02-05"), ("2026-01-28", "2026-02-12"), ("2026-06-18", "2026-07-02")]:
        excl(a, b, 30)
    excl("2024-08-19", "2024-08-19", 60)
    excl("2025-07-31", "2025-08-01", 7)
    ok &= ~(((d.month == 12) & (d.day >= 20)) | ((d.month == 1) & (d.day <= 15)))
    sc = pd.read_csv(HERE / "data" / "opal" / "events_scored_FROZEN.csv", parse_dates=["date"])
    for b in sc["date"]:
        excl(b, b, 30)
    ok &= x.notna().to_numpy()
    base = np.flatnonzero(ok)
    # hindsight screen: strict PELT on the STL residual (long-run scaled), +-30 d
    stl = A.adj_STL_retro(x.to_numpy(), d)
    res = stl - pd.Series(stl).rolling(29, center=True, min_periods=7).median().to_numpy()
    sd = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res)))
    zz = np.nan_to_num((stl - np.nanmedian(stl)) / (sd * np.sqrt(1.4 / 0.6)))
    cps = K.pelt_l2(zz, 8 * np.log(len(zz)), 14)
    screen = ok.copy()
    for c in cps:
        screen &= ~((np.arange(len(d)) >= c - 30) & (np.arange(len(d)) <= c + 30))
    return base[::2], set(np.flatnonzero(screen)), cps


def insert(x, tau, kind, size, stl_base):
    y = x.copy()
    if kind == "drop":
        y[tau:] *= (1 - size / 100)
    elif kind == "temp":
        y[tau:tau + 14] *= (1 - size / 100)
    else:
        lx = np.log(y[tau:])
        b = stl_base[tau:]
        y[tau:] = np.exp(b + np.sqrt(size) * (lx - b))
    return y


def adjust_one(adj, x, dates, tau=0, base=None):
    if adj == "STL":
        return A.adj_STL_retro(x, dates), "level"
    if adj == "REG":
        return A.adj_REG_retro(x, dates), "level"
    if adj == "REG_p":
        return A.adj_REG_past(x, dates, t_from=tau, base=base), "surprise"
    if adj in ("HW", "YM"):
        return A.adj_HW(x, dates), "surprise"
    return A.adjust(adj, x, dates)


BASE_REGP = None


def job(args):
    global BASE_REGP
    tau, x, dates, stl_base, screened, base_regp = args
    BASE_REGP = base_regp
    tabs = {k: calibrate.load(TABLE, k) for k in ("surprise", "level")}
    rows = []
    t0, t1 = tau - PRE, min(len(x), tau + POST)
    base_r = {adj: (BASE_REGP, "surprise") if adj == "REG_p" else adjust_one(adj, x, dates) for adj in ADJS}
    variants = [("none", 0.0)] + SETTINGS
    for kind, size in variants:
        xx = x if kind == "none" else insert(x, tau, kind, size, stl_base)
        for adj in ADJS:
            r, k = base_r[adj] if kind == "none" else adjust_one(adj, xx, dates, tau, base_r[adj][0])
            for det in DETS:
                for rule in RULES:
                    if det == "FOCuS-AR" and rule == "indep":
                        continue
                    res = monitor.run(r, k, det, rule, tabs[k], t0, t1, method=adj)
                    al = np.array(res.alarms, int)
                    pre = al[(al >= t0) & (al < tau)]
                    post = al[(al >= tau) & (al < t1)]
                    rows.append(dict(tau=int(tau), date=str(dates[tau].date()), screened=tau in screened,
                                     setting=f"{kind}{size:g}", adj=adj, det=det, rule=rule,
                                     pre_alarms=len(pre), first_post=int(post[0] - tau) if len(post) else -1,
                                     n_post=len(post), early14=int(((al >= tau - 14) & (al < tau)).any())))
    return rows


def main():
    x = opal_daily("train")
    dates = x.index
    taus, screened, cps = eligible(x)
    print("eligible starts:", len(taus), "screened:", len(screened & set(taus)), "hindsight cps:", len(cps), flush=True)
    xv = x.to_numpy().copy()
    xi = pd.Series(xv).interpolate(limit_direction="both").to_numpy()
    stl = A.adj_STL_retro(xi, dates)
    from statsmodels.tsa.seasonal import STL
    fit = STL(np.log(xi), period=7, robust=True).fit()
    stl_base = np.asarray(fit.trend) + np.asarray(fit.seasonal)
    base_regp = A.adj_REG_past(xv, dates)
    jobs = [(int(t), xv, dates, stl_base, screened, base_regp) for t in taus]
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for i, rr in enumerate(ex.map(job, jobs, chunksize=2)):
            rows += rr
            if i % 25 == 0:
                print(i, "/", len(jobs), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "pairs.csv", index=False)


if __name__ == "__main__":
    main()
