"""Scoring of alarms on real data, with chance baselines.

Real-data measures (one design for every method):
  n_alarms, alarms_per_year
  near_real      alarms within +-7 days of a scored boundary (frozen list)
  share_real     near_real / n_alarms
  episodes caught (of 15), and for each episode the first alarm relative to the episode start
  overlay class of every other alarm, in order: school holiday (+-2 d), public holiday / Christmas
                 (20 Dec - 15 Jan, or 0-2 days after a public holiday), after-event (8-28 d after a
                 boundary), explained event (policy / structural / fault rows, +-7 d), unexplained
  chance baseline: circular shift of the whole alarm sequence by a random offset (2,000 shifts);
                 report the null median of share_real and episodes caught and a one-sided p-value
  lift per class = (share of alarms in class) / (share of days in class)
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data" / "opal"
# NSW public school holidays (Eastern division), NSW Department of Education "School holidays" calendar
# (education.nsw.gov.au), first and last day of the holiday; Dec-Jan blocks run to the day before Term 1.
SCHOOL = [("2020-04-10", "2020-04-26"), ("2020-07-04", "2020-07-19"), ("2020-09-26", "2020-10-11"), ("2020-12-19", "2021-01-26"),
          ("2021-04-02", "2021-04-18"), ("2021-06-26", "2021-07-11"), ("2021-09-18", "2021-10-04"), ("2021-12-18", "2022-01-30"),
          ("2022-04-09", "2022-04-25"), ("2022-07-02", "2022-07-17"), ("2022-09-24", "2022-10-09"), ("2022-12-21", "2023-01-29"),
          ("2023-04-07", "2023-04-23"), ("2023-07-01", "2023-07-16"), ("2023-09-23", "2023-10-08"), ("2023-12-20", "2024-01-29"),
          ("2024-04-13", "2024-04-28"), ("2024-07-06", "2024-07-21"), ("2024-09-28", "2024-10-13"), ("2024-12-19", "2025-02-05"),
          ("2025-04-12", "2025-04-29"), ("2025-07-05", "2025-07-21"), ("2025-09-27", "2025-10-13"), ("2025-12-20", "2026-02-01"),
          ("2026-04-03", "2026-04-21"), ("2026-07-04", "2026-07-20"), ("2026-09-26", "2026-10-12")]
WIN = 7


def scored():
    s = pd.read_csv(EV / "events_scored_FROZEN.csv", parse_dates=["date"])
    return s


def day_classes(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean class membership for every day."""
    d = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=d)
    sc = scored()
    near = np.zeros(len(d), bool)
    after = np.zeros(len(d), bool)
    for b in sc["date"]:
        delta = (d - b).days
        near |= np.abs(delta) <= WIN
        after |= (delta >= 8) & (delta <= 28)
    out["real"] = near
    sch = np.zeros(len(d), bool)
    for a, b in SCHOOL:
        sch |= (d >= pd.Timestamp(a) - pd.Timedelta(days=2)) & (d <= pd.Timestamp(b) + pd.Timedelta(days=2))
    out["school"] = sch
    ph = holidays.Australia(subdiv="NSW", years=range(2019, 2027))
    phd = pd.DatetimeIndex(sorted(ph.keys()))
    pub = np.zeros(len(d), bool)
    for h in phd:
        delta = (d - h).days
        pub |= (delta >= 0) & (delta <= 2)
    xmas = ((d.month == 12) & (d.day >= 20)) | ((d.month == 1) & (d.day <= 15))
    out["holiday"] = pub | xmas
    out["after_event"] = after
    ver = pd.read_csv(EV / "events_verified.csv", parse_dates=["start", "end"])
    ex = np.zeros(len(d), bool)
    for _, r in ver[ver["kind"].isin(["policy", "structural", "fault"]) & ver["confidence"].isin(["high", "medium"])].iterrows():
        ex |= (d >= r["start"] - pd.Timedelta(days=WIN)) & (d <= r["end"] + pd.Timedelta(days=WIN))
    out["explained"] = ex
    return out


ORDER = ["real", "school", "holiday", "after_event", "explained"]


def classify(alarm_idx, cls: pd.DataFrame) -> list[str]:
    out = []
    for a in alarm_idx:
        row = cls.iloc[a]
        out.append(next((c for c in ORDER if row[c]), "unexplained"))
    return out


def episodes_caught(alarm_dates: pd.DatetimeIndex, dirs=None, after: int = WIN, before: int = 0,
                    sc: pd.DataFrame | None = None, credit: str = "nearest") -> tuple[int, dict]:
    """An episode is caught if an alarm falls from `before` days before to `after` days after one of its
    boundaries and (when `dirs` is given) moves in the boundary's direction: down for an onset, up for an end
    or easing. Each alarm is credited to one boundary only (the nearest), so one alarm cannot catch two
    episodes. The direction check stops the upward 'echo' that a same-weekday difference shows seven days after
    a one-day drop from counting as a catch. `before=WIN` gives the symmetric window used for 'near'.
    `sc` replaces the frozen list by a subset of it (sensitivity analyses only; default: the frozen list).
    credit="all" drops the one-alarm-one-episode crediting: an episode is caught if any alarm lies in the window of
    any of its boundaries, so one alarm can catch two episodes. With before=after=WIN and dirs=None this is the rule
    as frozen on 22 Oct 2026 (events_scored_RULES.md and score.py at commit 962c5b2); nearest-boundary crediting was
    added on 23 Oct at 02:45 (commit 4a17fd6) and the direction check at 03:29 (commit 4caf1ae)."""
    sc = (scored() if sc is None else sc).sort_values("date")
    want = np.where(sc["boundary"].astype(str).str.lower().isin(["onset", "start"]), -1, 1)
    alarm_dates = pd.DatetimeIndex(alarm_dates)
    best = {}
    if credit == "all":
        dv = None if dirs is None else np.asarray(dirs)
        for b, ep, w in zip(sc["date"], sc["episode"], want):
            d = np.asarray((alarm_dates - b).days)
            ok = (d >= -before) & (d <= after)
            if dv is not None:
                ok &= dv == w
            if ok.any():
                x = alarm_dates[ok].min()
                best[ep] = min(best.get(ep, x), x)
        first = {ep: int((best[ep] - sc[sc["episode"] == ep]["date"].min()).days) for ep in best}
        return len(best), first
    for k, x in enumerate(alarm_dates):
        d = np.array([(x - b).days for b in sc["date"]])
        ok = (d >= -before) & (d <= after)
        if dirs is not None:
            ok &= want == dirs[k]
        if not ok.any():
            continue
        j = np.flatnonzero(ok)[np.argmin(np.abs(d[ok]))]
        ep = sc["episode"].iloc[j]
        best[ep] = min(best.get(ep, x), x)
    first = {ep: int((best[ep] - sc[sc["episode"] == ep]["date"].min()).days) for ep in best}
    return len(best), first


def score(alarm_idx, dates, t_start: int, t_end: int | None = None, n_shift=2000, seed=0, dirs=None) -> dict:
    d = pd.DatetimeIndex(dates)
    t_end = len(d) if t_end is None else t_end
    mon = d[t_start:t_end]
    years = len(mon) / 365.25
    cls = day_classes(d)
    keep = [k for k, x in enumerate(alarm_idx) if t_start <= x < t_end]
    a = [alarm_idx[k] for k in keep]
    dv = None if dirs is None else np.array([dirs[k] for k in keep])
    labels = classify(a, cls)
    n = len(a)
    ad = d[a] if n else pd.DatetimeIndex([])
    caught, first = episodes_caught(ad, dv)
    res = {"n_alarms": n, "alarms_per_year": n / years, "near_real": labels.count("real"),
           "share_real": labels.count("real") / n if n else np.nan, "episodes_caught": caught}
    for c in ORDER[1:] + ["unexplained"]:
        res[f"n_{c}"] = labels.count(c)
    res["unexplained_per_year"] = labels.count("unexplained") / years
    res["non_event_per_year"] = (n - labels.count("real")) / years
    # lift per class
    mcls = cls.iloc[t_start:t_end]
    for c in ORDER:
        frac_days = mcls[c].mean()
        res[f"lift_{c}"] = (sum(1 for x in a if cls.iloc[x][c]) / n) / frac_days if n and frac_days > 0 else np.nan
    # circular-shift null
    if n:
        rng = np.random.default_rng(seed)
        L = t_end - t_start
        rel = np.array(a) - t_start
        realmask = mcls["real"].to_numpy()
        null_share, null_caught = [], []
        for _ in range(n_shift):
            u = rng.integers(30, L - 30)
            sh = (rel + u) % L
            null_share.append(realmask[sh].mean())
            null_caught.append(episodes_caught(mon[sh], dv)[0])
        null_share, null_caught = np.array(null_share), np.array(null_caught)
        res["null_share_real"] = float(np.median(null_share))
        res["p_share_real"] = float((null_share >= res["share_real"]).mean())
        res["null_episodes"] = float(np.median(null_caught))
        res["p_episodes"] = float((null_caught >= caught).mean())
    res["first_alarm_rel"] = first
    return res
