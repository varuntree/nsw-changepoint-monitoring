"""Offline segmentation on the electricity forecast errors.

PELT, Binary Segmentation and WBS were also run on electricity with the group's shared code; those six CSVs and our
re-run of the same command go in data/independent/offline/electricity/ (not in this repository; see data/LICENSES.md).
This script recomputes the same thing with our code: the Holt-Winters (HW) and temperature-regression (REGT_p) forecast errors of
results/electricity/adjusted.csv from the first day both exist, scaled as scripts/opal_offline.py (MAD-based long-run
scale), PELT 3 log n, BinSeg sqrt(3 log n), WBS 1.3 sqrt(2 log n) with 5,000 intervals (seed 0, drawn for HW then
REGT_p), minimum segment 7 days; change dates from 1 Mar 2020 scored with our scorer on the frozen electricity
list (direction-aware check, 1,000 shifts) and also under the main rule (+-7 days, any direction).
Outputs: results/electricity/offline_changes.csv, offline_scores.csv, offline_boundaries.csv (nearest change to each
scored boundary), results/extra/crosscheck_offline_elec.csv (the independent files and their re-run against ours)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import kernels as K  # noqa: E402
from cpd import score as SC  # noqa: E402
from electricity import SRC, scored_score  # noqa: E402
from opal_offline import scaled  # noqa: E402

R = HERE / "results"
SIH = HERE / "data" / "independent" / "offline" / "electricity"
SERIES = ("HW", "REGT_p")
METHODS = ("pelt", "binseg", "wbs")
NAME = {"pelt": "PELT", "binseg": "BinSeg", "wbs": "WBS"}
COLS = ["n_alarms", "alarms_per_year", "near_real", "share_real", "null_share_real", "p_share_real", "episodes_caught",
        "null_episodes", "p_episodes", "unexplained_per_year", "non_event_per_year"]


def run():
    adj = pd.read_csv(R / "electricity" / "adjusted.csv", parse_dates=["date_nem"]).set_index("date_nem")
    adj = adj.loc[adj[list(SERIES)].notna().all(axis=1).idxmax():]
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    ev = pd.read_csv(SRC / "events_electricity_scored_FROZEN.csv", parse_dates=["date"])
    scores, changes = [], []
    for meth in METHODS:
        rng = np.random.default_rng(0)
        for a in SERIES:
            y, phi = scaled(adj[a])
            n = len(y)
            ints = np.sort(rng.integers(0, n, size=(5000, 2)), axis=1)
            cps = {"pelt": lambda: K.pelt_l2(y, 3 * np.log(n), 7), "binseg": lambda: K.binseg_l2(y, np.sqrt(3 * np.log(n)), 7, 10 ** 6),
                   "wbs": lambda: K.wbs_l2(y, 1.3 * np.sqrt(2 * np.log(n)), 7, ints)}[meth]()
            allc = sorted(int(c) for c in cps)
            bnd = [0] + allc + [n]
            dirs = {c: (1 if y[c:bnd[k + 2]].mean() > y[bnd[k]:c].mean() else -1) for k, c in enumerate(allc)}
            idx = [c for c in allc if c >= t0]
            sc = scored_score(idx, dates, t0, dirs=[dirs[c] for c in idx])
            cds = pd.DatetimeIndex([dates[c] for c in idx])
            old_ev = SC.EV
            SC.EV = SRC
            try:
                fz, _ = SC.episodes_caught(cds, None, after=7, before=7, credit="all") if len(cds) else (0, {})
            finally:
                SC.EV = old_ev
            scores.append({**{k: sc.get(k, np.nan) for k in COLS}, "adjustment": a, "detector": NAME[meth], "phi": phi,
                           "frozen": int(fz)})
            changes += [dict(detector=NAME[meth], adjustment=a, change_date=dates[c].date(), direction=dirs[c]) for c in idx]
    sc, ch = pd.DataFrame(scores), pd.DataFrame(changes)
    # nearest change date (any direction) to each scored boundary, per method and series
    rows = []
    ch["change_date"] = pd.to_datetime(ch.change_date)
    for (meth, a), g in ch.groupby(["detector", "adjustment"], sort=False):
        for _, b in ev.iterrows():
            off = (g.change_date - b.date).dt.days
            j = off.abs().idxmin() if len(g) else None
            want = -1 if str(b.boundary).lower() in ("onset", "start") else 1
            rows.append(dict(detector=meth, adjustment=a, episode=b.episode, boundary=b.boundary, boundary_date=b.date.date(),
                             nearest_change=g.change_date[j].date() if j is not None else None,
                             days_from_boundary=int(off[j]) if j is not None else None,
                             direction=int(g.direction[j]) if j is not None else None, boundary_direction=want,
                             within_7_any=bool(abs(off[j]) <= 7) if j is not None else False,
                             within_0_7_same_direction=bool(((off >= 0) & (off <= 7) & (g.direction == want)).any())))
    return sc, ch, pd.DataFrame(rows)


def main():
    sc, ch, bd = run()
    sc.to_csv(R / "electricity" / "offline_scores.csv", index=False)
    out = ch.copy()
    out["change_date"] = out.change_date.dt.date
    out.to_csv(R / "electricity" / "offline_changes.csv", index=False)
    bd.to_csv(R / "electricity" / "offline_boundaries.csv", index=False)
    if not SIH.exists():
        print("data/independent/ not present: results/extra/crosscheck_offline_elec.csv kept as shipped")
        print(sc.round(4).to_string(index=False))
        return
    rows = []
    for who, d in (("independent", SIH), ("shared code re-run", SIH / "rerun")):
        for meth in METHODS:
            hc = pd.read_csv(d / f"{meth}_changes.csv", parse_dates=["change_date"])
            hs = pd.read_csv(d / f"{meth}_scores.csv")
            for a in SERIES:
                mine = ch[(ch.detector == NAME[meth]) & (ch.adjustment == a)]
                theirs = hc[hc.adjustment == a]
                same_dates = list(mine.change_date) == list(theirs.change_date)
                same_dirs = same_dates and list(mine.direction) == list(theirs.direction)
                r1 = sc[(sc.detector == NAME[meth]) & (sc.adjustment == a)].iloc[0]
                r2 = hs[hs.adjustment == a].iloc[0]
                diffs = {c: float(r2[c] - r1[c]) for c in COLS if not (pd.isna(r1[c]) and pd.isna(r2[c])) and not np.isclose(r1[c], r2[c])}
                rows.append(dict(source=who, detector=NAME[meth], adjustment=a, n_changes_ours=len(mine), n_changes_indep=len(theirs),
                                 dates_identical=same_dates, directions_identical=same_dirs, scores_identical=not diffs,
                                 differing_fields=";".join(f"{k} indep-ours {v:+.4f}" for k, v in diffs.items())))
    x = pd.DataFrame(rows)
    x["explanation"] = np.where(x.differing_fields.str.contains("unexplained_per_year"),
                                "one change date (28 Apr 2026) is the day after Monday 27 Apr 2026, NSW's additional ANZAC Day "
                                "public holiday for 2026 (holidays >= the release with the NSW 2026-2027 rule, ours 0.107); an "
                                "older holidays package has no such day, so the alarm counts as unexplained there. Ours is right.", "")
    x.to_csv(R / "extra" / "crosscheck_offline_elec.csv", index=False)
    pd.set_option("display.width", 220)
    print(sc.round(4).to_string(index=False))
    print(x.drop(columns="explanation").to_string(index=False))
    print(bd.to_string(index=False))


if __name__ == "__main__":
    main()
