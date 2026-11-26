"""Section 6.2 of the thesis: what the retrospective MSTL does with Christmas and with holidays whose dates move.

Reads   results/opal/adjusted.csv (columns STL and MSTL: the log series minus the fitted seasonal components),
        results/opal/offline.csv and offline_changes.csv (offline change dates on each adjusted series).
Prints  (1) for each year, the mean of the adjusted series over 25 December to 1 January minus its median over
            the 21 days ending 8 days before 25 December and the 21 days starting 40 days after it;
        (2) for each Good Friday in the monitored period, the adjusted value minus the median of the days
            2 to 10 days either side, and the mean over the years;
        (3) the number of offline change dates on each adjusted series.
Usage   python scripts/mstl_holiday_check.py      (make mstl-check)
"""
from pathlib import Path

import holidays
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
a = pd.read_csv(f"{REPO}/results/opal/adjusted.csv", parse_dates=["date"]).set_index("date")
rows = []
for y in range(2020, 2026):
    x0 = pd.Timestamp(f"{y}-12-25")
    win = pd.date_range(x0, x0 + pd.Timedelta(days=7))
    ref = list(pd.date_range(x0 - pd.Timedelta(days=28), x0 - pd.Timedelta(days=8))) + \
        list(pd.date_range(x0 + pd.Timedelta(days=40), x0 + pd.Timedelta(days=60)))
    rows.append({"year": y, **{m: a.loc[win, m].mean() - a.loc[ref, m].median() for m in ("STL", "MSTL")}})
x = pd.DataFrame(rows).set_index("year")
print("Christmas week against the surrounding weeks (log scale)\n", x.round(3))
print("STL range:", x.STL.max().round(2), "to", x.STL.min().round(2), "| MSTL largest absolute value:", x.MSTL.abs().max().round(3))

hol = holidays.Australia(subdiv="NSW", years=range(2020, 2027))
rows = []
for d, name in sorted(hol.items()):
    d = pd.Timestamp(d)
    if name != "Good Friday" or d not in a.index or d < pd.Timestamp("2020-03-01"):
        continue
    ref = [r for k in range(2, 11) for r in (d - pd.Timedelta(days=k), d + pd.Timedelta(days=k)) if r in a.index]
    rows.append({"date": d.date(), **{m: a.loc[d, m] - a.loc[ref, m].median() for m in ("STL", "MSTL")}})
g = pd.DataFrame(rows).set_index("date")
print("\nGood Friday against the neighbouring days (log scale)\n", g.round(2), "\nmean:", g.mean().round(2).to_dict())

oc = pd.read_csv(f"{REPO}/results/opal/offline_changes.csv")
print("\noffline change dates:\n", oc.groupby(["adjustment", "method"]).size().unstack())
