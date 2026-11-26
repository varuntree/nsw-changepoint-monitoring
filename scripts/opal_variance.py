"""Variance detectors on the real Opal series: do spread and mean alarms fall on
different dates?

Monitors: past-only forecasts A, B, Holt-Winters (HW) and the yearly memory (YM) x variance CUSUM, variance EWMA
and variance FOCuS x the independent and AR(1) threshold rules, run exactly as opal_case.one (monitor.run on the
adjusted series of results/opal/adjusted.csv, kind "surprise", from 1 March 2020, 28-day relearn).
Variance alarms have no direction, so they are scored with the direction-free rules only: the original rule
(+-7 days of a boundary, nearest-boundary crediting; and `frozen`, the rule as frozen on 22 Oct, where one alarm may
catch several episodes) and the 0..+7-day rule without direction; score.score(dirs=None) gives the latter plus
the alarm classes and burden. Chance: circular shift (seed 0, 2,000 shifts) as score.score. These 24 monitors are
NOT part of the best-of-136 family; only per-monitor p-values are reported.
Start estimates: detectors.change_start supports variance FOCuS; for variance CUSUM and EWMA it returns the alarm
day itself, so start_est = alarm date for them.

Outputs (results/opal/):
  variance_scores.csv   per monitor: score.score columns (0..+7, no direction), plus original-rule episodes and
                        both rules' chance medians and p-values
  variance_alarms.csv   every alarm: monitor, date, start_est, class (score.classify), and the nearest mean alarm
                        (CUSUM, EWMA, FOCuS, same forecast and rule) with its signed day difference
  variance_vs_mean.csv  for HW and YM: per scored boundary, the first variance alarm and the first mean alarm
                        (CUSUM and FOCuS, same forecast and rule) in [boundary - 7, boundary + 28] and the
                        difference in days (variance minus mean); "who_first" rows counting the boundaries where
                        the variance or the mean detector (earlier of CUSUM and FOCuS) alarmed first; and summary rows: share of variance alarms
                        within +-1 day of a mean alarm of the same forecast and rule, and the share in each class
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import calibrate, monitor, score  # noqa: E402
from opal_extra import _shift_test  # noqa: E402

OP = HERE / "results" / "opal"
TABLE = HERE / "results" / "calib" / "h_table.json"
ADJS = ["A", "B", "HW", "YM"]
VDETS = ["CUSUM-var", "EWMA-var", "FOCuS-var"]
MDETS = ["CUSUM", "EWMA", "FOCuS"]
RULES = ["indep", "ar1"]


def one(args):
    m, det, rule, r, t0, dates = args
    tabs = calibrate.load(TABLE, "surprise")
    res = monitor.run(r, "surprise", det, rule, tabs, t0, method=m)
    sc = score.score(res.alarms, dates, t0, n_shift=2000, dirs=None)
    sc.pop("first_alarm_rel", None)
    mon = dates[t0:]
    rel = np.array(res.alarms, int) - t0
    s = score.scored()
    (o7, n7, p7, _), (opm, npm, ppm, eps), (ofz, nfz, pfz, _) = _shift_test(
        (rel, None, mon, [(7, 0, False, s), (7, 7, False, s), (7, 7, False, s, "all")], 2000))
    assert o7 == sc["episodes_caught"]
    sc.update(method=m, detector=det, rule=rule, r0_7_nodir=o7, r0_7_nodir_null=n7, r0_7_nodir_p=p7,
              pm7_nodir=opm, pm7_nodir_null=npm, pm7_nodir_p=ppm, frozen=ofz, frozen_null=nfz, frozen_p=pfz, episodes_original=";".join(eps),
              blind_days_per_year=res.blind.sum() / ((len(r) - t0) / 365.25),
              h_first=res.hs[0] if res.hs else np.nan, phi_first=res.phis[0] if res.phis else np.nan)
    al = pd.DataFrame({"method": m, "detector": det, "rule": rule, "alarm": dates[res.alarms] if res.alarms else [],
                       "start_est": dates[np.clip(res.starts, 0, len(dates) - 1)] if res.alarms else []})
    return sc, al


def main():
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    jobs = [(m, d, r, adj[m].to_numpy(), t0, dates) for m in ADJS for d in VDETS for r in RULES]
    with ProcessPoolExecutor(10) as ex:
        out = list(ex.map(one, jobs))
    scores = pd.DataFrame([o[0] for o in out])
    va = pd.concat([o[1] for o in out], ignore_index=True)
    va = va[va.alarm >= "2020-03-01"]
    scores.to_csv(OP / "variance_scores.csv", index=False)

    # classes and nearest mean alarm
    cls = score.day_classes(dates)
    va["class"] = score.classify([dates.get_loc(a) for a in va.alarm], cls)
    ma = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm"])
    ma = ma[ma.detector.isin(MDETS) & (ma.alarm >= "2020-03-01")]
    near, diff, which = [], [], []
    for _, a in va.iterrows():
        g = ma[(ma.method == a.method) & (ma.rule == a.rule)]
        if len(g):
            dd = (g.alarm - a.alarm).dt.days.to_numpy()
            k = int(np.argmin(np.abs(dd)))
            diff.append(-dd[k])          # variance minus mean, days
            which.append(g.detector.iloc[k])
            near.append(abs(dd[k]) <= 1)
        else:
            diff.append(np.nan), which.append(""), near.append(False)
    va["nearest_mean_detector"], va["days_after_nearest_mean"], va["within_1d_of_mean"] = which, diff, near
    va.to_csv(OP / "variance_alarms.csv", index=False)

    # per boundary: first variance vs first mean alarm, HW and YM
    sc = score.scored()
    rows = []
    for m in ("HW", "YM"):
        for rule in RULES:
            for _, b in sc.iterrows():
                lo, hi = b.date - pd.Timedelta(days=7), b.date + pd.Timedelta(days=28)
                row = dict(kind="boundary", method=m, rule=rule, episode=b.episode, boundary=b.boundary, date=b.date.date())
                for det in VDETS + ["CUSUM", "FOCuS"]:
                    src = va if det in VDETS else ma
                    g = src[(src.method == m) & (src.detector == det) & (src.rule == rule) & (src.alarm >= lo) & (src.alarm <= hi)]
                    row[f"first_{det}"] = g.alarm.min().date() if len(g) else None
                    row[f"days_{det}"] = (g.alarm.min() - b.date).days if len(g) else np.nan
                for v in VDETS:
                    for mm in ("CUSUM", "FOCuS"):
                        row[f"diff_{v}_minus_{mm}"] = row[f"days_{v}"] - row[f"days_{mm}"]
                rows.append(row)
    bd = pd.DataFrame(rows)
    for (m, rule), g in bd.groupby(["method", "rule"]):
        mean_first = g[["days_CUSUM", "days_FOCuS"]].min(axis=1)
        for v in VDETS:
            dv = g[f"days_{v}"]
            rows.append(dict(kind="who_first", method=m, detector=v, rule=rule, boundaries=len(g),
                             both=int((dv.notna() & mean_first.notna()).sum()),
                             variance_earlier=int((dv.notna() & (mean_first.isna() | (dv < mean_first))).sum()),
                             same_day=int((dv.notna() & (dv == mean_first)).sum()),
                             mean_earlier=int((mean_first.notna() & (dv.isna() | (mean_first < dv))).sum()),
                             variance_only=int((dv.notna() & mean_first.isna()).sum()),
                             mean_only=int((dv.isna() & mean_first.notna()).sum())))
    for (m, det, rule), g in va.groupby(["method", "detector", "rule"]):
        if m not in ("HW", "YM"):
            continue
        row = dict(kind="summary", method=m, detector=det, rule=rule, n_alarms=len(g),
                   share_within_1d_of_mean=float(g.within_1d_of_mean.mean()) if len(g) else np.nan)
        for c in score.ORDER + ["unexplained"]:
            row[f"share_{c}"] = float((g["class"] == c).mean()) if len(g) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(OP / "variance_vs_mean.csv", index=False)
    pd.set_option("display.width", 250)
    print(scores[["method", "detector", "rule", "n_alarms", "non_event_per_year", "r0_7_nodir", "r0_7_nodir_p",
                  "pm7_nodir", "pm7_nodir_p", "blind_days_per_year"]].round(3).to_string(index=False))
    print(va.groupby(["method", "detector", "rule"]).agg(n=("alarm", "size"), within1=("within_1d_of_mean", "mean")).round(2))


if __name__ == "__main__":
    main()
