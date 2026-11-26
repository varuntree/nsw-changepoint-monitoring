"""Quick check of the Opal case study: re-score every monitor's alarms and compare with the full run.

Reads results/opal/alarms.csv (every alarm of every adjustment x detector x threshold rule, written by
scripts/opal_case.py), scores each monitor again with cpd/score.py against the frozen event list in data/opal/
(same scorer, same 1,000 circular shifts and seed as the full run), writes results/opal/scorecard.csv and checks
every scoring column against results/opal/scores.csv. Prints the headline rows (the Opal results table).
Columns that come from the monitor itself (phi_first, h_first, blind_days_per_year) are not re-derived here;
`make results` rebuilds them.
Usage: python scripts/scorecard.py     (a few minutes; exit code 1 if anything differs)
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import adjust as A  # noqa: E402
from cpd import score  # noqa: E402
from cpd.data import END, START  # noqa: E402

OP = HERE / "results" / "opal"
METHODS = ["raw", "A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG"]
DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
RULES = ["indep", "ar1"]
T_START = "2020-03-01"
N_SHIFT = 1000  # as in scripts/opal_case.py
KEYS = ["method", "detector", "rule"]
HEADLINE = [("A", "CUSUM", "indep"), ("A", "CUSUM", "ar1"), ("A", "EWMA", "ar1"), ("A", "FOCuS", "ar1"),
            ("A", "FOCuS-AR", "ar1"), ("A", "CPM", "ar1"), ("A", "CUSUM-r", "ar1"), ("A", "FOCuS-r", "ar1"),
            ("HW", "CUSUM", "ar1"), ("HW", "EWMA", "ar1"), ("HW", "FOCuS", "ar1"), ("YM", "CUSUM", "ar1"),
            ("YM", "FOCuS", "ar1"), ("YM", "FOCuS-r", "ar1"), ("MSTL", "CUSUM", "ar1"), ("raw", "CUSUM", "ar1")]


def one(args):
    m, det, rule, idx, dirs, dates, t0 = args
    sc = score.score(idx, dates, t0, n_shift=N_SHIFT, dirs=dirs if len(idx) else None)
    sc.update({"method": m, "kind": "level" if (m in A.RETRO or m == "raw") else "surprise", "detector": det,
               "rule": rule})
    return sc


def main():
    dates = pd.date_range(START, END, freq="D")
    t0 = int(np.searchsorted(dates, pd.Timestamp(T_START)))
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm"])
    pos = pd.Series(np.arange(len(dates)), index=dates)
    jobs = []
    for m in METHODS:
        for d in DETS:
            for r in RULES:
                if d == "FOCuS-AR" and r == "indep":
                    continue
                a = al[(al.method == m) & (al.detector == d) & (al.rule == r)]
                idx = [int(i) for i in pos.loc[a["alarm"]].to_numpy()]
                jobs.append((m, d, r, idx, np.asarray(a["direction"].to_numpy(), int), dates, t0))
    rows = []
    with ProcessPoolExecutor(min(os.cpu_count() or 2, 10)) as ex:
        for sc in ex.map(one, jobs, chunksize=2):
            rows.append(sc)
    new = pd.DataFrame(rows)
    new["first_alarm_rel"] = new["first_alarm_rel"].apply(json.dumps)
    new = new[KEYS + ["kind"] + [c for c in new.columns if c not in KEYS + ["kind"]]]
    new.to_csv(OP / "scorecard.csv", index=False)

    # compare with the full run
    old = pd.read_csv(OP / "scores.csv")
    new = pd.read_csv(OP / "scorecard.csv")
    both = old.merge(new, on=KEYS, how="outer", suffixes=("_full", "_quick"), indicator=True)
    bad = []
    if (both["_merge"] != "both").any():
        bad.append(f"{(both['_merge'] != 'both').sum()} monitors are in only one of the two files")
    cols = [c for c in new.columns if c not in KEYS and c in old.columns]
    for c in cols:
        a, b = both[f"{c}_full"], both[f"{c}_quick"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            ok = (a.astype(str) == b.astype(str)).to_numpy()
        if not ok.all():
            k = int((~ok).sum())
            ex = both.loc[~ok, KEYS + [f"{c}_full", f"{c}_quick"]].head(3).to_string(index=False)
            bad.append(f"{c}: {k} rows differ, e.g.\n{ex}")

    # headline table
    s = new.merge(old[KEYS + ["blind_days_per_year"]], on=KEYS, how="left")
    print("Opal case study, 1 Mar 2020 - 28 Sep 2026 (re-scored from results/opal/alarms.csv)")
    print(f"{'adj.':<8}{'detector':<10}{'thr.':<7}{'alarms':>7}{'near':>6}{'%near':>7}{'(chance)':>9}"
          f"{'ep.':>5}{'p(ep.)':>8}{'non-event/yr':>14}{'unexpl./yr':>12}{'blind d/yr':>12}")
    for m, d, r in HEADLINE:
        q = s[(s.method == m) & (s.detector == d) & (s.rule == r)].iloc[0]
        pct = lambda x: "--" if pd.isna(x) else f"{100 * x:.0f}"  # noqa: E731
        print(f"{m:<8}{d:<10}{'AR(1)' if r == 'ar1' else 'indep.':<7}{int(q.n_alarms):>7}{int(q.near_real):>6}"
              f"{pct(q.share_real):>7}{'(' + pct(q.null_share_real) + ')':>9}{int(q.episodes_caught):>5}"
              f"{q.p_episodes:>8.3f}{q.non_event_per_year:>14.1f}{q.unexplained_per_year:>12.1f}"
              f"{q.blind_days_per_year:>12.0f}")
    print(f"\nwrote {OP / 'scorecard.csv'} ({len(new)} monitors, {len(cols)} scoring columns compared)")
    if bad:
        print("MISMATCH with results/opal/scores.csv:\n" + "\n".join(bad))
        sys.exit(1)
    print("scorecard matches results/opal/scores.csv")


if __name__ == "__main__":
    main()
