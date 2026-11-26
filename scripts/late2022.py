"""The extreme Opal days of 2022 and what the main monitors did on them.

Extreme day: the past-only surprise of the same-weekday difference (A), the median of four (B), Holt-Winters (HW) or
the yearly memory (YM) lies more than 5 robust SD (1.4826 MAD over 2022) from its 2022 median. Each day is joined to
the verified event list. For the late-2022 clusters (11 Sep, 21-25 Nov, and the same-weekday echo 28 Nov-2 Dec) the
script reports, for the monitors of Table 6.1 and for all 150, the alarms 0-7 days after the cluster starts, and, for
a monitor that did not alarm, whether it was in its 28-day relearn after an earlier alarm.
Outputs: results/opal/extreme_2022_days.csv, results/opal/extreme_2022_monitors.csv, results/opal/extreme_2022_summary.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd.data import opal_daily  # noqa: E402

R = HERE / "results" / "opal"
MAIN = [("A", "CUSUM", "ar1"), ("HW", "EWMA", "ar1"), ("HW", "FOCuS", "ar1"), ("YM", "FOCuS", "ar1"),
        ("YM", "CUSUM-r", "ar1"), ("A", "EWMA-r", "ar1"), ("A", "FOCuS-r", "ar1"), ("B", "EWMA", "indep"),
        ("YM", "CUSUM", "indep"), ("STL", "FOCuS-AR", "ar1"), ("A", "CPM", "indep")]
CLUSTERS = [("11 Sep 2022 (candidate data fault)", "2022-09-11", "2022-09-11"),
            ("21-25 Nov 2022 (fare-free week, readers off)", "2022-11-21", "2022-11-25"),
            ("28 Nov-2 Dec 2022 (same-weekday echo of the fare-free week)", "2022-11-28", "2022-12-02")]
RELEARN = 28


def main():
    x = opal_daily("train")
    adj = pd.read_csv(R / "adjusted.csv", parse_dates=["date"]).set_index("date")
    ev = pd.read_csv(HERE / "data" / "opal" / "events_verified.csv", parse_dates=["start", "end"])
    zs = {}
    for m in ("A", "B", "HW", "YM"):
        v = adj[m]
        base = v.loc["2022-01-01":"2022-12-31"]
        med = base.median()
        zs[m] = ((v - med) / (1.4826 * (base - med).abs().median())).loc["2022-01-01":"2022-12-31"]
    z = pd.DataFrame(zs)
    ext = z[(z.abs() > 5).any(axis=1)].copy()
    ext["taps"] = x.reindex(ext.index)
    ext["weekday"] = ext.index.day_name()

    def explain(d):
        e = ev[(ev.start <= d) & (ev.end >= d)]
        return "; ".join(f"{k}: {t}" for k, t in zip(e.kind, e.event)) if len(e) else ""
    ext["event_list"] = [explain(d) for d in ext.index]
    ext.index.name = "date"
    ext.round(2).to_csv(R / "extreme_2022_days.csv")

    al = pd.read_csv(R / "alarms.csv", parse_dates=["alarm"])
    rows, summ = [], []
    for lab, a, b in CLUSTERS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        win = (al.alarm >= a) & (al.alarm <= a + pd.Timedelta(days=7))
        for m, d, r in MAIN:
            g = al[(al.method == m) & (al.detector == d) & (al.rule == r)]
            hit = g[win.loc[g.index]]
            prev = g[(g.alarm < a) & (g.alarm >= a - pd.Timedelta(days=RELEARN))]
            rows.append(dict(cluster=lab, method=m, detector=d, rule=r,
                             alarms=";".join(hit.alarm.dt.strftime("%Y-%m-%d")),
                             directions=";".join(map(str, hit.direction)),
                             blind_from_alarm=prev.alarm.max().strftime("%Y-%m-%d") if len(prev) and not len(hit) else ""))
        y = al[win]
        unc = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR"]           # uncapped linear-sum detectors (CPM is reported apart)
        day1 = al[al.alarm == a]
        summ.append(dict(cluster=lab, monitors=al.groupby(["method", "detector", "rule"]).ngroups,
                         uncapped_monitors=al[al.detector.isin(unc)].groupby(["method", "detector", "rule"]).ngroups,
                         first_day_alarms=len(day1), first_day_alarms_uncapped=int(day1.detector.isin(unc).sum()),
                         first_day_alarms_cpm=int((day1.detector == "CPM").sum()),
                         monitors_alarming=y.groupby(["method", "detector", "rule"]).ngroups,
                         downward=int((y.direction == -1).sum()), upward=int((y.direction == 1).sum())))
    pd.DataFrame(rows).to_csv(R / "extreme_2022_monitors.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(R / "extreme_2022_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(ext.round(1).to_string())
    print(pd.DataFrame(rows).to_string(index=False))
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
