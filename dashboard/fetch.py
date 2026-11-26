"""Data refresh for the dashboard. Works only on the dashboard's own cache folder.

Same sources and time rules as live/data_fetch.py (the thesis fetcher), cut down to what the
dashboard needs:
  * AEMO aggregated price and demand, region NSW1, monthly CSVs. SETTLEMENTDATE is NEM time (UTC+10,
    no daylight saving) and marks the END of the interval; 30-minute intervals up to the one ending
    2021-10-01 00:00, 5-minute intervals after. A NEM day D covers intervals that START in D.
  * Open-Meteo archive (model ecmwf_ifs), hourly temperature at Sydney and Penrith, timezone GMT+10.

The cache starts as a copy of the thesis data files (the "seed"). Each run downloads again every month
from the month after the last complete day in the cache up to the current month, so a month file that
was saved while still filling (the October 2026 one, for example) is always replaced.
The seed folder is only read, never written.
"""
from __future__ import annotations

import io
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

AEMO_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_NSW1.csv"
REGION = "NSW1"
NEM_TZ = timezone(timedelta(hours=10))
FIVE_MIN_START = pd.Timestamp("2021-10-01 00:05:00")
UA = {"User-Agent": "Mozilla/5.0 (student research dashboard; daily fetch)"}
SITES = {"sydney": (-33.87, 151.21), "penrith": (-33.75, 150.69)}
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_MODEL = "ecmwf_ifs"
CDD_BASE, HDD_BASE = 24.0, 18.0
SEED_FILES = ("nsw_demand_daily.csv", "sydney_temp_daily_sydney.csv", "sydney_temp_daily_penrith.csv",
              "events_electricity.csv")


def nem_now() -> datetime:
    return datetime.now(NEM_TZ).replace(tzinfo=None)


def _get(url, params=None, retries=4, timeout=60):
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
            if r.status_code in (403, 404):
                break
        except requests.RequestException as e:
            last = e
        time.sleep(2 ** k)
    raise RuntimeError(f"failed {url}: {last}")


def seed_cache(cache: Path, seed: Path) -> list[str]:
    """Copy the thesis data files into the cache if they are not there yet."""
    cache.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SEED_FILES:
        if not (cache / name).exists():
            shutil.copy2(seed / name, cache / name)
            copied.append(name)
    return copied


# ----------------------------------------------------------------------------- AEMO demand
def _months(first: pd.Timestamp, last: pd.Timestamp) -> list[str]:
    y, m = first.year, first.month
    out = []
    while (y, m) <= (last.year, last.month):
        out.append(f"{y:04d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _daily_from_raw(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Daily mean demand from raw interval records (same steps as data_fetch.build_30min and build_daily)."""
    df = pd.concat(frames, ignore_index=True)
    df = df[df["REGION"] == REGION].copy()
    df["end"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    df = df.sort_values("end").drop_duplicates("end", keep="last").reset_index(drop=True)
    df["interval_min"] = np.where(df["end"] >= FIVE_MIN_START, 5, 30)
    df["start"] = df["end"] - pd.to_timedelta(df["interval_min"], unit="min")
    hh_end = df["start"].dt.floor("30min") + pd.Timedelta(minutes=30)
    g = df.groupby(hh_end)
    hh = pd.DataFrame({"demand_mw": g["TOTALDEMAND"].mean(), "n_obs": g["TOTALDEMAND"].count()})
    hh = hh.reindex(pd.date_range(hh.index.min(), hh.index.max(), freq="30min"))
    hh["n_obs"] = hh["n_obs"].fillna(0).astype(int)
    n_exp = np.where(hh.index > FIVE_MIN_START, 6, 1)
    hh["partial"] = (hh["n_obs"] > 0) & (hh["n_obs"] < n_exp)
    day = (hh.index - pd.Timedelta(minutes=30)).normalize()
    gd = hh.groupby(day)
    out = pd.DataFrame({
        "mean_mw": gd["demand_mw"].mean(),
        "max_mw_30min": gd["demand_mw"].max(),
        "min_mw_30min": gd["demand_mw"].min(),
        "n_30min": gd["demand_mw"].count(),
        "n_30min_partial": gd["partial"].sum(),
    })
    start_day = df["start"].dt.normalize()
    e = (df["TOTALDEMAND"] * df["interval_min"] / 60.0).groupby(start_day).sum()
    out["energy_mwh"] = e.reindex(out.index)
    out["n_raw"] = df.groupby(start_day)["TOTALDEMAND"].count().reindex(out.index).fillna(0).astype(int)
    out["n_raw_expected"] = np.where(out.index >= pd.Timestamp("2021-10-01"), 288, 48)
    out["complete"] = (out["n_raw"] == out["n_raw_expected"]) & (out["n_30min"] == 48)
    out.loc[~out["complete"], "energy_mwh"] = np.nan
    out["energy_mwh_scaled"] = out["mean_mw"] * 24.0
    out.index.name = "date_nem"
    return out


def refresh_demand(cache: Path) -> dict:
    f = cache / "nsw_demand_daily.csv"
    old = pd.read_csv(f, parse_dates=["date_nem"]).set_index("date_nem")
    last_ok = old.index[old["complete"].astype(bool)].max()
    now = pd.Timestamp(nem_now())
    yms = _months(last_ok + pd.Timedelta(days=1), now)
    raw_dir = cache / "raw"
    raw_dir.mkdir(exist_ok=True)
    frames, got = [], []
    for ym in yms:
        try:
            r = _get(AEMO_URL.format(ym=ym))
        except RuntimeError:
            if ym == yms[-1] and now.day == 1:      # the new month's file may not exist in its first hour
                continue
            raise
        if not r.content.startswith(b"REGION"):
            raise RuntimeError(f"{ym}: unexpected content {r.content[:80]!r}")
        (raw_dir / f"PRICE_AND_DEMAND_{ym}_{REGION}.csv").write_bytes(r.content)
        frames.append(pd.read_csv(io.BytesIO(r.content)))
        got.append(ym)
        time.sleep(0.5)
    if not frames:
        return {"months_fetched": [], "last_complete_day": str(last_ok.date())}
    new = _daily_from_raw(frames).round(3)
    first_new = new.index.min()
    both = old.index.intersection(new.index)
    both = both[old.loc[both, "complete"].astype(bool)]
    drift = float((old.loc[both, "mean_mw"] - new.loc[both, "mean_mw"]).abs().max()) if len(both) else 0.0
    merged = pd.concat([old[old.index < first_new], new])
    tmp = f.with_suffix(".part")
    merged.to_csv(tmp, date_format="%Y-%m-%d")
    tmp.replace(f)
    last_new = merged.index[merged["complete"].astype(bool)].max()
    return {"months_fetched": got, "last_complete_day": str(last_new.date()),
            "last_interval_end": str(pd.to_datetime(frames[-1]["SETTLEMENTDATE"].iloc[-1])),
            "max_change_in_days_already_saved_mw": drift}


# ----------------------------------------------------------------------------- temperature
def _fetch_archive(site: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = SITES[site]
    r = _get(ARCHIVE_URL, params=dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                                      hourly="temperature_2m", timezone="GMT+10", models=ARCHIVE_MODEL))
    h = r.json()["hourly"]
    return pd.DataFrame({"time_nem": pd.to_datetime(h["time"]), "temperature_2m": h["temperature_2m"]})


def _daily_temp(h: pd.DataFrame) -> pd.DataFrame:
    x = h.dropna(subset=["temperature_2m"])
    g = x.groupby(x["time_nem"].dt.normalize())["temperature_2m"]
    d = pd.DataFrame({"t_mean": g.mean(), "t_max": g.max(), "t_min": g.min(), "n_hours": g.count()})
    d["cdd24"] = (d["t_mean"] - CDD_BASE).clip(lower=0)
    d["hdd18"] = (HDD_BASE - d["t_mean"]).clip(lower=0)
    d.index.name = "date_nem"
    return d


def refresh_temp(cache: Path) -> dict:
    """Append the complete NEM days after the last saved one (earlier days are left exactly as they were)."""
    end = pd.Timestamp((nem_now().date() - timedelta(days=1)).isoformat())
    added = {}
    for site in SITES:
        f = cache / f"sydney_temp_daily_{site}.csv"
        old = pd.read_csv(f, index_col=0, parse_dates=True)
        start = old.index.max() + pd.Timedelta(days=1)
        if start > end:
            added[site] = 0
            continue
        d = _daily_temp(_fetch_archive(site, start.date().isoformat(), end.date().isoformat()))
        d = d[d["n_hours"] == 24]
        # only keep an unbroken run of new days, so a late day is fetched again tomorrow
        keep = []
        for k, day in enumerate(pd.date_range(start, end, freq="D")):
            if day not in d.index:
                break
            keep.append(day)
        d = d.loc[keep]
        tmp = f.with_suffix(".part")
        pd.concat([old, d.round(3)]).to_csv(tmp, date_format="%Y-%m-%d")
        tmp.replace(f)
        added[site] = len(d)
        time.sleep(0.5)
    return added


def refresh(cache: Path, seed: Path) -> dict:
    info = {"seeded": seed_cache(cache, seed)}
    try:
        info["demand"] = refresh_demand(cache)
    except Exception as e:      # keep the saved copy if the feed fails
        info["demand_error"] = repr(e)
    try:
        info["temp_days_added"] = refresh_temp(cache)
    except Exception as e:
        info["temp_error"] = repr(e)
    return info
