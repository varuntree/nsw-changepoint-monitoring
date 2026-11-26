"""Live application back-test: NSW electricity demand (AEMO TOTALDEMAND, region NSW1), daily mean MW.

Step 1 (unchanged): the Opal pipeline exactly as used on Opal (same adjustments, detectors, thresholds,
scale, relearn), on log daily demand. Step 2 (adapted): the past-only regression adjustment with
temperature added to the forecast of normal (cooling and heating degree days at Sydney and Penrith;
temperature is a driver measured at the same time, not a calendar). Monitoring 2020-01-01 .. last
complete day. Scored against the frozen electricity event list (COVID 2020, Greater Sydney 2021); all
other events are explanations. A paired insertion test (permanent drops of 2.5, 5, 10%) on 2023-2026
measures detection directly.
Outputs: results/electricity/{scores.csv, alarms.csv, testbed.csv, adjusted.csv}
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

SRC = HERE / "data" / "electricity"
OUT = HERE / "results" / "electricity"
OUT.mkdir(parents=True, exist_ok=True)
TABLE = HERE / "results" / "calib" / "h_table.json"
ADJS = ("A", "B", "HW", "YM", "REG_p", "REGT_p")
DETS = ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "FOCuS-r")
RULES = ("indep", "ar1")
T_START = "2020-01-01"


def load():
    d = pd.read_csv(SRC / "nsw_demand_daily.csv", parse_dates=["date_nem"]).set_index("date_nem")
    d = d[d["complete"]]
    x = d["mean_mw"].asfreq("D")
    return x


def temps(index):
    """Temperature terms for the forecast of normal, at Sydney and Penrith: cooling and heating degree days
    (base 24 and 18 C on the daily mean), hot-day and cold-night terms (max above 28 C, min below 15 C), and
    yesterday's cooling, heating and hot-day terms (heat builds up in buildings)."""
    cols = []
    for site in ("sydney", "penrith"):
        t = pd.read_csv(SRC / f"sydney_temp_daily_{site}.csv")
        t = t.set_index(pd.to_datetime(t.iloc[:, 0])).reindex(index)
        hot = np.maximum(t["t_max"] - 28, 0)
        cold = np.maximum(15 - t["t_min"], 0)
        cols += [t["cdd24"], t["hdd18"], hot, cold, t["cdd24"].shift(1), t["hdd18"].shift(1), hot.shift(1)]
    return np.column_stack([np.asarray(c, float) for c in cols])


def adj_regt_past(x, dates, Z, window=365, refit=7, min_obs=56, t_from=0, base=None):
    """Past-only regression: weekday + yearly Fourier (once a year of history) + temperature terms Z,
    refitted weekly on the previous 365 days by robust regression; intercept follows the last 28 days."""
    import statsmodels.api as sm
    lx = np.log(np.asarray(x, float))
    X0 = A._reg_design(dates)
    X = np.column_stack([X0, Z])
    n = len(lx)
    r = np.full(n, np.nan) if base is None else np.array(base, float).copy()
    params = None
    k0 = X0.shape[1]
    for t in range(max(0, t_from - (t_from % refit)), n):
        if (t % refit == 0) or params is None:
            lo = max(0, t - window)
            ok = ~np.isnan(lx[lo:t]) & ~np.isnan(X[lo:t]).any(axis=1)
            if ok.sum() >= min_obs:
                cols = list(range(7)) + (list(range(7, k0)) if (t - lo) >= 365 else []) + list(range(k0, X.shape[1]))
                fit = sm.RLM(lx[lo:t][ok], X[lo:t][ok][:, cols], M=sm.robust.norms.HuberT()).fit()
                params = np.zeros(X.shape[1])
                params[cols] = fit.params
                rec = slice(max(lo, t - 28), t)
                okr = ~np.isnan(lx[rec]) & ~np.isnan(X[rec]).any(axis=1)
                if okr.sum() >= 7:
                    params[0] += np.median(lx[rec][okr] - X[rec][okr] @ params)
        if params is not None and not np.isnan(lx[t]) and not np.isnan(X[t]).any():
            r[t] = lx[t] - X[t] @ params
    return r


def adjust_all(x, Z, t_from=0, base=None):
    v = x.to_numpy()
    d = x.index
    out = {}
    for a in ADJS:
        if a == "REGT_p":
            out[a] = adj_regt_past(v, d, Z, t_from=t_from, base=None if base is None else base[a])
        elif a == "REG_p":
            out[a] = A.adj_REG_past(v, d, t_from=t_from, base=None if base is None else base[a])
        elif a in ("HW", "YM"):
            out[a] = A.adj_HW(v, d)
        else:
            out[a] = A.adjust(a, v, d)[0]
    return out


def scored_score(alarms, dates, t0, dirs=None):
    """Reuse the Opal scorer with the electricity frozen list (swap the file in)."""
    from cpd import score as S
    sc_old = S.EV
    try:
        S.EV = SRC
        import shutil
        # score.scored() reads events_scored_FROZEN.csv and events_verified.csv from S.EV
        tmp1 = SRC / "events_scored_FROZEN.csv"
        tmp2 = SRC / "events_verified.csv"
        if not tmp1.exists():
            shutil.copy(SRC / "events_electricity_scored_FROZEN.csv", tmp1)
        if not tmp2.exists():
            e = pd.read_csv(SRC / "events_electricity.csv", parse_dates=["start", "end"])
            e["end"] = e["end"].fillna(e["start"])
            # explanations are dated events of at most 31 days; long trends and recurring-calendar rows
            # (rooftop solar, daylight saving, 'recurring' holidays) would explain every day and are left out
            e = e[((e["end"] - e["start"]).dt.days <= 31) & ~e["event"].str.contains("Recurring|recurring", na=False)]
            e.loc[e["kind"].isin(["weather", "disruption"]) & ~e["event"].str.contains("COVID", na=False), "kind"] = "policy"
            e.to_csv(tmp2, index=False)
        return S.score(alarms, dates, t0, n_shift=1000, dirs=dirs)
    finally:
        S.EV = sc_old


def one(args):
    a, det, rule, r, dates, t0 = args
    tabs = calibrate.load(TABLE, "surprise")
    res = monitor.run(r, "surprise", det, rule, tabs, t0, method=a)
    sc = scored_score(res.alarms, dates, t0, res.dirs)
    sc.update(adj=a, det=det, rule=rule)
    al = pd.DataFrame({"adj": a, "det": det, "rule": rule, "alarm": dates[res.alarms] if res.alarms else []})
    return sc, al


def tb_job(args):
    tau, v, dates, Z, base, size = args
    tabs = calibrate.load(TABLE, "surprise")
    x = pd.Series(v, index=dates)
    y = x.copy()
    y.iloc[tau:] *= (1 - size / 100)
    adj_y = adjust_all(y, Z, t_from=tau, base=base)
    rows = []
    for a in ADJS:
        for det in ("CUSUM", "FOCuS", "FOCuS-AR", "CUSUM-r", "FOCuS-r"):
            for which, rr in (("none", base[a]), ("drop", adj_y[a])):
                res = monitor.run(rr, "surprise", det, "ar1", tabs, tau - 90, min(len(v), tau + 60), method=a)
                al = np.array(res.alarms, int)
                post = al[al >= tau]
                rows.append(dict(tau=tau, size=size, adj=a, det=det, which=which,
                                 first_post=int(post[0] - tau) if len(post) else -1,
                                 pre=int(((al >= tau - 90) & (al < tau)).sum())))
    return rows


def main():
    x = load()
    dates = x.index
    Z = temps(dates)
    base = adjust_all(x, Z)
    pd.DataFrame(base, index=dates).to_csv(OUT / "adjusted.csv")
    t0 = int(np.searchsorted(dates, pd.Timestamp(T_START)))
    jobs = [(a, d, ru, base[a], dates, t0) for a in ADJS for d in DETS for ru in RULES if not (d == "FOCuS-AR" and ru == "indep")]
    scores, alarms = [], []
    with ProcessPoolExecutor(6) as ex:
        for sc, al in ex.map(one, jobs):
            scores.append(sc)
            alarms.append(al)
    s = pd.DataFrame(scores)
    s["first_alarm_rel"] = s["first_alarm_rel"].astype(str)
    s.to_csv(OUT / "scores.csv", index=False)
    pd.concat(alarms).to_csv(OUT / "alarms.csv", index=False)
    # paired insertion test 2023-2026, start days every 7 days, away from Dec-Jan
    taus = [i for i, d in enumerate(dates) if d >= pd.Timestamp("2023-01-01") and i + 60 < len(dates)
            and not ((d.month == 12 and d.day >= 15) or (d.month == 1 and d.day <= 15))][::7]
    tjobs = [(t, x.to_numpy(), dates, Z, base, s_) for t in taus for s_ in (2.5, 5.0, 10.0)]
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for rr in ex.map(tb_job, tjobs, chunksize=2):
            rows += rr
    pd.DataFrame(rows).to_csv(OUT / "testbed.csv", index=False)


if __name__ == "__main__":
    main()
