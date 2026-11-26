"""Opal data and the events list.

Opal: daily NSW tap-ons by mode from Transport for NSW Opal Tap Data (counts rounded to the nearest
100; files refreshed for four days after each day). Main series: train tap-ons, 1 Jan 2020 -
28 Sep 2026. Days with no file (3-4 Jun 2021) and days with fewer than 20 reported hours
(31 Jul, 1 Aug 2025) are left missing.

Data faults follow the publisher's own definition (Opal Patronage documentation v2.0, s.5):
"data variations may occur due to incomplete or unavailable data at the time of extraction ...
examples where data discrepancies are more likely: 28 January 2025 - 5 February 2025,
28 January 2026 - 12 February 2026", plus the forum notice of under-reported bus taps
(18 Jun - 2 Jul 2026, bus only) and the incomplete/missing days above.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "opal"
START, END = "2020-01-01", "2026-09-28"
MISSING = ["2021-06-03", "2021-06-04"]
INCOMPLETE = ["2025-07-31", "2025-08-01"]


def opal_daily(mode: str = "train") -> pd.Series:
    d = pd.read_csv(SRC / "opal_daily_nsw.csv", parse_dates=["date"])
    s = d[d["mode"] == mode].set_index("date")["tap_ons"].astype(float)
    s = s.reindex(pd.date_range(START, END, freq="D"))
    s[s.index.isin(pd.to_datetime(INCOMPLETE))] = np.nan
    s.name = f"{mode}_tap_ons"
    return s


def events() -> pd.DataFrame:
    e = pd.read_csv(SRC / "events_verified.csv", parse_dates=["start", "end"])
    return e
