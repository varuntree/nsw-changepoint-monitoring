"""Opal case study under one design.

Every seasonal adjustment x detector x threshold rule runs through the same monitor (cpd/monitor.py)
on daily NSW train tap-ons, monitoring from 1 Mar 2020 to 28 Sep 2026, and is scored by cpd/score.py
against the frozen event list. Writes results/opal/adjusted.csv, alarms.csv, scores.csv.
Usage: python scripts/opal_case.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import adjust as A  # noqa: E402
from cpd import calibrate, monitor, score  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

OUT = HERE / "results" / "opal"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["raw", "A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG"]
DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
RULES = ["indep", "ar1"]
T_START = "2020-03-01"
TABLE = HERE / "results" / "calib" / "h_table.json"


def adjusted() -> tuple[pd.DataFrame, dict]:
    f = OUT / "adjusted.csv"
    x = opal_daily("train")
    if f.exists():
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    else:
        cols = {}
        for m in METHODS:
            r, kind = A.adjust(m, x.to_numpy(), x.index)
            cols[m] = r
            print("adjusted", m, flush=True)
        df = pd.DataFrame(cols, index=x.index)
        df.index.name = "date"
        df.to_csv(f)
    kinds = {m: ("level" if (m in A.RETRO or m == "raw") else "surprise") for m in METHODS}
    return df, kinds


def one(args):
    m, det, rule, r, kind, t0, dates = args
    tables = calibrate.load(TABLE, kind)
    res = monitor.run(r, kind, det, rule, tables, t0, method=m)
    sc = score.score(res.alarms, dates, t0, n_shift=1000, dirs=res.dirs)
    sc.update({"method": m, "kind": kind, "detector": det, "rule": rule,
               "phi_first": res.phis[0] if res.phis else np.nan, "h_first": res.hs[0] if res.hs else np.nan,
               "blind_days_per_year": res.blind.sum() / ((len(r) - t0) / 365.25)})
    al = pd.DataFrame({"method": m, "detector": det, "rule": rule, "alarm": dates[res.alarms] if res.alarms else [],
                       "direction": res.dirs if res.alarms else [],
                       "start_est": dates[np.clip(res.starts, 0, len(dates) - 1)] if res.alarms else []})
    return sc, al


def main():
    df, kinds = adjusted()
    dates = df.index
    t0 = int(np.searchsorted(dates, pd.Timestamp(T_START)))
    jobs = [(m, d, ru, df[m].to_numpy(), kinds[m], t0, dates) for m in METHODS for d in DETS for ru in RULES
            if not (d == "FOCuS-AR" and ru == "indep")]
    scores, alarms = [], []
    with ProcessPoolExecutor(6) as ex:
        for sc, al in ex.map(one, jobs):
            scores.append(sc)
            alarms.append(al)
            print(sc["method"], sc["detector"], sc["rule"], sc["n_alarms"], round(sc["share_real"] or 0, 3),
                  sc["episodes_caught"], flush=True)
    s = pd.DataFrame(scores)
    s["first_alarm_rel"] = s["first_alarm_rel"].apply(json.dumps)
    s.to_csv(OUT / "scores.csv", index=False)
    pd.concat(alarms).to_csv(OUT / "alarms.csv", index=False)


if __name__ == "__main__":
    main()
