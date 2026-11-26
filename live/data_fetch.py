"""Data acquisition for the live NSW electricity demand changepoint chapter.

Downloads and processes:
  * AEMO aggregated price and demand, region NSW1, monthly CSVs (Jan 2019 -> current month)
  * Open-Meteo ERA5-based archive hourly temperature_2m for Sydney and Penrith
  * Open-Meteo forecast API (past_days) for the last couple of days (live feed)

Time conventions
  * AEMO SETTLEMENTDATE is NEM time = AEST = UTC+10, no daylight saving, and it
    marks the END of the interval. Intervals are 30 min up to and including the
    interval ending 2021-10-01 00:00, and 5 min from the interval ending
    2021-10-01 00:05 (five-minute settlement start).
  * All timestamps written by this module are naive NEM-time (UTC+10) stamps.
  * Half-hour series is labelled by interval END (AEMO convention): the value at
    10:30 is the mean of the 5-min intervals ending 10:05 ... 10:30.
  * A NEM day D covers intervals ending D 00:05 (or 00:30) ... D+1 00:00, i.e.
    the interval START lies in [D 00:00, D+1 00:00).
  * Open-Meteo is requested with timezone=GMT+10 (fixed offset); hourly stamps
    are instantaneous values at that time; daily stats use hours 00:00..23:00.

Usage (from the repository root):
  python live/data_fetch.py all
  python live/data_fetch.py aemo | build | temp | recent | latency
"""
from __future__ import annotations

import io
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1] / "data" / "electricity"
RAW = BASE / "raw"
AEMO_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_NSW1.csv"
REGION = "NSW1"
START_MONTH = (2019, 1)
NEM_TZ = timezone(timedelta(hours=10))
FIVE_MIN_START = pd.Timestamp("2021-10-01 00:05:00")  # first 5-min interval end
UA = {"User-Agent": "Mozilla/5.0 (research data fetch; DATA5925 thesis)"}

SITES = {
    "sydney": (-33.87, 151.21),
    "penrith": (-33.75, 150.69),
}
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# "best_match" in the archive resolves to ecmwf_ifs (IFS HRES 9 km, available to
# about the current hour, incl. forecast hours for today); pinned for reproducibility.
# ERA5 ("era5") is ~5 days behind real time and on a 0.25 deg grid.
ARCHIVE_MODEL = "ecmwf_ifs"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CDD_BASE, HDD_BASE = 24.0, 18.0


# ----------------------------------------------------------------------------- helpers
def nem_now() -> datetime:
    return datetime.now(NEM_TZ).replace(tzinfo=None)


def months(start=START_MONTH, end: tuple[int, int] | None = None) -> list[str]:
    if end is None:
        n = nem_now()
        end = (n.year, n.month)
    y, m = start
    out = []
    while (y, m) <= end:
        out.append(f"{y:04d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _get(url: str, params=None, retries: int = 4, timeout: int = 60) -> requests.Response:
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
            if r.status_code in (403, 404):
                break
        except requests.RequestException as e:  # network error -> retry
            last = e
        time.sleep(2 ** k)
    raise RuntimeError(f"failed {url}: {last}")


# ----------------------------------------------------------------------------- AEMO
def raw_path(ym: str) -> Path:
    return RAW / f"PRICE_AND_DEMAND_{ym}_{REGION}.csv"


def download_month(ym: str, force: bool = False, sleep: float = 0.5) -> Path:
    """Download one monthly file. Existing files are skipped unless force."""
    p = raw_path(ym)
    if p.exists() and p.stat().st_size > 0 and not force:
        return p
    RAW.mkdir(parents=True, exist_ok=True)
    r = _get(AEMO_URL.format(ym=ym))
    if not r.content.startswith(b"REGION"):
        raise RuntimeError(f"{ym}: unexpected content {r.content[:80]!r}")
    tmp = p.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(p)
    time.sleep(sleep)
    return p


def download_all(force: bool = False) -> list[Path]:
    """All months Jan 2019 -> current NEM month. The current month (and the
    previous month during the first two days of a month, since its tail may
    still be filling) is always re-downloaded."""
    ms = months()
    n = nem_now()
    always = {ms[-1]}
    if n.day <= 2 and len(ms) > 1:
        always.add(ms[-2])
    paths = []
    for ym in ms:
        paths.append(download_month(ym, force=force or ym in always))
    return paths


def load_raw(paths=None) -> pd.DataFrame:
    if paths is None:
        paths = sorted(RAW.glob(f"PRICE_AND_DEMAND_*_{REGION}.csv"))
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["source_file"] = p.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["REGION"] == REGION].copy()
    df["end"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    df = (df.sort_values(["end", "source_file"])  # duplicates across files: keep last file
            .drop_duplicates("end", keep="last")
            .reset_index(drop=True))
    df["interval_min"] = np.where(df["end"] >= FIVE_MIN_START, 5, 30)
    df["start"] = df["end"] - pd.to_timedelta(df["interval_min"], unit="min")
    return df


def build_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Half-hourly mean TOTALDEMAND labelled by interval end, on a complete grid."""
    hh_end = df["start"].dt.floor("30min") + pd.Timedelta(minutes=30)
    g = df.groupby(hh_end)
    out = pd.DataFrame({
        "demand_mw": g["TOTALDEMAND"].mean(),
        "rrp_mean": g["RRP"].mean(),
        "n_obs": g["TOTALDEMAND"].count(),
    })
    grid = pd.date_range(out.index.min(), out.index.max(), freq="30min")
    out = out.reindex(grid)
    out.index.name = "interval_end_nem"
    out["n_obs"] = out["n_obs"].fillna(0).astype(int)
    out["n_expected"] = np.where(out.index > FIVE_MIN_START, 6, 1)
    # the half-hour ending 2021-10-01 00:30 is the first 5-min-based one
    out["missing"] = out["n_obs"] == 0
    out["partial"] = (out["n_obs"] > 0) & (out["n_obs"] < out["n_expected"])
    return out


def build_daily(hh: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    day = (hh.index - pd.Timedelta(minutes=30)).normalize()  # date of interval start
    g = hh.groupby(day)
    raw_day = df.groupby(df["start"].dt.normalize())
    out = pd.DataFrame({
        "mean_mw": g["demand_mw"].mean(),
        "max_mw_30min": g["demand_mw"].max(),
        "min_mw_30min": g["demand_mw"].min(),
        "n_30min": g["demand_mw"].count(),
        "n_30min_partial": g["partial"].sum(),
    })
    # energy from raw records: MW * interval hours (exact, no interpolation)
    e = (df["TOTALDEMAND"] * df["interval_min"] / 60.0).groupby(df["start"].dt.normalize()).sum()
    out["energy_mwh"] = e.reindex(out.index)
    out["n_raw"] = raw_day["TOTALDEMAND"].count().reindex(out.index).fillna(0).astype(int)
    exp = pd.Series(np.where(out.index >= pd.Timestamp("2021-10-01"), 288, 48), index=out.index)
    out["n_raw_expected"] = exp
    out["complete"] = (out["n_raw"] == out["n_raw_expected"]) & (out["n_30min"] == 48)
    out.loc[~out["complete"], "energy_mwh"] = np.nan  # only report energy for full days
    out["energy_mwh_scaled"] = out["mean_mw"] * 24.0  # mean-based estimate, any completeness
    out.index.name = "date_nem"
    return out


def build_aemo(write: bool = True):
    df = load_raw()
    hh = build_30min(df)
    daily = build_daily(hh, df)
    if write:
        hh_out = hh.copy()
        hh_out["demand_mw"] = hh_out["demand_mw"].round(3)
        hh_out["rrp_mean"] = hh_out["rrp_mean"].round(3)
        hh_out.to_csv(BASE / "nsw_demand_30min.csv.gz", date_format="%Y-%m-%d %H:%M:%S",
                      compression="gzip")
        daily.round(3).to_csv(BASE / "nsw_demand_daily.csv", date_format="%Y-%m-%d")
    return df, hh, daily


# ----------------------------------------------------------------------------- temperature
def _om_frame(js: dict) -> pd.DataFrame:
    h = js["hourly"]
    return pd.DataFrame({"time_nem": pd.to_datetime(h["time"]), "temperature_2m": h["temperature_2m"]})


def fetch_archive(site: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = SITES[site]
    frames = []
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:  # yearly chunks
        ce = min(e, pd.Timestamp(year=s.year, month=12, day=31))
        r = _get(ARCHIVE_URL, params=dict(latitude=lat, longitude=lon,
                                          start_date=s.date().isoformat(), end_date=ce.date().isoformat(),
                                          hourly="temperature_2m", timezone="GMT+10",
                                          models=ARCHIVE_MODEL))
        frames.append(_om_frame(r.json()))
        s = ce + pd.Timedelta(days=1)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True)


def fetch_recent(site: str, past_days: int = 3, forecast_days: int = 2) -> pd.DataFrame:
    """Forecast endpoint: past_days of best-estimate (analysis/short forecasts) plus forecast."""
    lat, lon = SITES[site]
    r = _get(FORECAST_URL, params=dict(latitude=lat, longitude=lon, hourly="temperature_2m",
                                       timezone="GMT+10", past_days=past_days,
                                       forecast_days=forecast_days))
    return _om_frame(r.json())


def daily_temp(h: pd.DataFrame) -> pd.DataFrame:
    x = h.dropna(subset=["temperature_2m"])
    g = x.groupby(x["time_nem"].dt.normalize())["temperature_2m"]
    d = pd.DataFrame({"t_mean": g.mean(), "t_max": g.max(), "t_min": g.min(), "n_hours": g.count()})
    d["cdd24"] = (d["t_mean"] - CDD_BASE).clip(lower=0)
    d["hdd18"] = (HDD_BASE - d["t_mean"]).clip(lower=0)
    d.index.name = "date_nem"
    return d


def build_temp(end: str | None = None):
    # default: last COMPLETE NEM day (yesterday); today's hours would be forecasts
    end = end or (nem_now().date() - timedelta(days=1)).isoformat()
    out = {}
    hourly_all = []
    for site in SITES:
        h = fetch_archive(site, "2019-01-01", end)
        h["site"] = site
        hourly_all.append(h)
        d = daily_temp(h)
        d.round(3).to_csv(BASE / f"sydney_temp_daily_{site}.csv", date_format="%Y-%m-%d")
        out[site] = (h, d)
    H = pd.concat(hourly_all).pivot(index="time_nem", columns="site", values="temperature_2m")
    H.columns = [f"temp_{c}" for c in H.columns]
    H.to_csv(BASE / "sydney_temp_hourly.csv.gz", date_format="%Y-%m-%d %H:%M", compression="gzip")
    return out


# ----------------------------------------------------------------------------- live
def aemo_latency() -> dict:
    """How far the current-month file lags real time (NEM time)."""
    ym = months()[-1]
    url = AEMO_URL.format(ym=ym)
    h = requests.head(url, headers=UA, timeout=30, allow_redirects=True)
    r = _get(url)
    last = pd.read_csv(io.BytesIO(r.content))["SETTLEMENTDATE"].iloc[-1]
    last = pd.Timestamp(datetime.strptime(last, "%Y/%m/%d %H:%M:%S"))
    now = pd.Timestamp(nem_now())
    return {"file": url, "last_modified_http": h.headers.get("Last-Modified"),
            "last_interval_end_nem": last, "nem_now": now,
            "lag_hours": round((now - last).total_seconds() / 3600, 2)}


def extend_temp(end: str | None = None) -> dict:
    """Append the complete NEM days after the last saved one to the saved temperature files (same archive
    model as build_temp, so earlier days are left exactly as they were)."""
    end = pd.Timestamp(end or (nem_now().date() - timedelta(days=1)).isoformat())
    added = {}
    hourly_new = []
    for site in SITES:
        f = BASE / f"sydney_temp_daily_{site}.csv"
        old = pd.read_csv(f, index_col=0, parse_dates=True)
        start = old.index.max() + pd.Timedelta(days=1)
        if start > end:
            added[site] = 0
            continue
        h = fetch_archive(site, start.date().isoformat(), end.date().isoformat())
        d = daily_temp(h)
        d = d[d["n_hours"] == 24]
        pd.concat([old, d.round(3)]).to_csv(f, date_format="%Y-%m-%d")
        h["site"] = site
        hourly_new.append(h[h["time_nem"].dt.normalize().isin(d.index)])
        added[site] = len(d)
    if hourly_new and sum(len(h) for h in hourly_new):
        H = pd.concat(hourly_new).pivot(index="time_nem", columns="site", values="temperature_2m")
        H.columns = [f"temp_{c}" for c in H.columns]
        fh = BASE / "sydney_temp_hourly.csv.gz"
        old = pd.read_csv(fh, index_col=0, parse_dates=True)
        H = pd.concat([old, H[~H.index.isin(old.index)]])
        H.to_csv(fh, date_format="%Y-%m-%d %H:%M", compression="gzip")
    return added


def live_update() -> dict:
    """One live step: refresh the current month's AEMO file and rebuild the daily series; append the latest
    complete days of temperature."""
    download_all()
    df, hh, daily = build_aemo()
    added = extend_temp()
    return {"last_interval_end": df["end"].max(), "nem_now": nem_now(), "daily_tail": daily.tail(3),
            "temp_days_added": added}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("aemo", "all"):
        ps = download_all()
        print(f"AEMO files: {len(ps)}")
    if cmd in ("build", "aemo", "all"):
        df, hh, daily = build_aemo()
        print("last interval end (NEM):", df["end"].max(), " now (NEM):", nem_now())
        print("30-min missing:", int(hh["missing"].sum()), "partial:", int(hh["partial"].sum()))
        print("incomplete days:", int((~daily["complete"]).sum()))
    if cmd in ("temp", "all"):
        build_temp()
        print("temperature written")
    if cmd == "latency":
        print(aemo_latency())
    if cmd == "recent":
        for s in SITES:
            print(s, daily_temp(fetch_recent(s)).tail(5))
