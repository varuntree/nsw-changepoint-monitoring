"""Build data.json for the dashboard.

    python build_data.py               refresh the electricity data, run both monitors, write data.json
    python build_data.py --no-refresh  use the cache as it is (no downloads)

Electricity (live): AEMO NSW1 daily mean demand, monitored with the configuration in config.json
(past-only regression forecast with weekday, time of year and temperature; FOCuS; AR(1) threshold).
Train tap-ons (static file): Transport for NSW Opal data, monitored with Holt-Winters + EWMA (AR(1)
threshold), and compared with the 15 scored real changes.

The thesis code (cpd/, scripts/electricity.py) is imported, not copied. Its data folders are only read:
all downloads go to ./cache. Layouts supported:
    <repo>/dashboard/build_data.py with <repo>/cpd            (public repository)
or set CPD_ROOT to the folder that holds cpd/.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True     # the thesis folders are only read, not even for .pyc files
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _code_root() -> Path:
    for c in (os.environ.get("CPD_ROOT"), HERE.parent):
        if c and (Path(c) / "cpd" / "monitor.py").exists():
            return Path(c)
    raise SystemExit("cannot find the cpd package; set CPD_ROOT")


CODE = _code_root()
sys.path[:0] = [str(HERE), str(CODE), str(CODE / "scripts")]

import electricity as E  # noqa: E402  (thesis script: load, temps, adj_regt_past)
import fetch  # noqa: E402
from cpd import adjust as A  # noqa: E402
from cpd import calibrate, monitor, score  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

SEED = Path(E.SRC)                 # thesis electricity files: read only
CACHE = HERE / "cache"
E.SRC = CACHE                      # every read below goes to the dashboard's own copy
TABLE = CODE / "results" / "calib" / "h_table.json"
CONFIG = HERE / "config.json"
OUT = HERE / "data.json"
TRY_MAX_DAYS = 60                  # the "Try it" slider reaches this far back
STATUS_WINDOW = 7                  # status says "Alarm" for this many days after an alarm

OPAL_MONITOR = {"adj": "HW", "det": "EWMA", "rule": "ar1", "start": "2020-03-01"}
EPISODE_LABELS = {
    "covid_onset": "COVID-19 onset", "covid_easing": "First COVID-19 easing",
    "flood_2021_03": "Floods, March 2021", "delta_onset": "Delta lockdown starts",
    "delta_end": "Delta lockdown ends", "omicron_onset": "Omicron wave",
    "omicron_end": "Omicron work-from-home ends", "shutdown_2022_02": "Rail network shutdown",
    "flood_2022_03": "Floods, March 2022", "strike_2022_06": "Rail dispute, June 2022",
    "flood_2022_07": "Floods, July 2022", "strike_2022_08": "Rail dispute, August 2022",
    "strike_2025_01": "Rail work bans, January 2025", "strike_2025_02": "Rail dispute, February 2025",
    "outage_2025_05": "Homebush wiring failure",
}
CLASS_LABELS = {"covid": "COVID-19", "flood": "Flood", "shutdown": "Rail dispute", "industrial": "Rail dispute",
                "outage": "Outage"}


# ----------------------------------------------------------------------------- helpers
def num(a, nd):
    """List for JSON: rounded, NaN as null."""
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in np.asarray(a, float)]


def trace(r, det, rule, tabs, t0, method):
    """Run the thesis monitor (cpd.monitor.run) and rebuild what it saw each day: scale, standardised
    surprise, detector statistic and threshold. The alarms are the monitor's own; the rebuild is checked
    against them."""
    r = np.asarray(r, float)
    n = len(r)
    res = monitor.run(r, "surprise", det, rule, tabs, t0, method=method)
    s, z, stat, h = (np.full(n, np.nan) for _ in range(4))
    blind = np.zeros(n, dtype=np.bool_)
    for k, (st, ph, hh) in enumerate(zip(res.seg_starts, res.phis, res.hs)):
        sc = monitor.running_scale(r.copy(), blind, st, n)
        zz = r[st:n] / sc
        path = D.stat_path(det, zz, ph)
        cross = np.flatnonzero(path > hh)
        if k < len(res.alarms):
            assert len(cross) and st + int(cross[0]) == res.alarms[k], "trace does not match monitor.run"
            end = res.alarms[k] + 1
        else:
            assert len(cross) == 0, "trace does not match monitor.run"
            end = n
        s[st:end], z[st:end], stat[st:end], h[st:end] = sc[:end - st], zz[:end - st], path[:end - st], hh
        if k < len(res.alarms):
            blind[end:end + monitor.RELEARN] = True
    assert (blind == res.blind).all()
    return res, s, z, stat, h


def status_of(dates, alarms):
    last = dates[-1]
    recent = [a for a in alarms if (last - dates[a]).days < STATUS_WINDOW]
    return {"state": "alarm" if recent else "normal", "window_days": STATUS_WINDOW,
            "alarm_day": str(dates[recent[-1]].date()) if recent else None}


def series_block(dates, actual, r, res, s, z, stat, h, t0):
    lx = np.log(np.asarray(actual, float))
    return {
        "start": str(dates[0].date()), "n": len(dates), "t_start": int(t0),
        "last_day": str(dates[-1].date()),
        "actual": num(actual, 1), "expected": num(np.exp(lx - r), 1),
        "r": num(r, 6), "s": num(s, 6), "z": num(z, 4), "stat": num(stat, 4), "h": num(h, 4),
        "blind": [int(i) for i in np.flatnonzero(res.blind)],
        "segments": [{"start": int(a), "phi": round(float(p), 6), "h": round(float(hh), 6)}
                     for a, p, hh in zip(res.seg_starts, res.phis, res.hs)],
        "status": status_of(dates, res.alarms),
    }


def h_table(det):
    t = json.loads(TABLE.read_text())
    m = t["surprise"][det]
    keys = sorted(m, key=float)
    return {"phi": [float(k) for k in keys], "h": [m[k] for k in keys], "false_alarm_chance": t["meta"]["fa"],
            "year_days": t["meta"]["year"]}


def monitor_params():
    return {"relearn": monitor.RELEARN, "scale_lag": monitor.SCALE_LAG, "scale_win": monitor.SCALE_WIN,
            "scale_every": monitor.SCALE_EVERY, "phi_win": monitor.PHI_WIN, "phi_min": monitor.PHI_MIN}


# ----------------------------------------------------------------------------- electricity
def regt_trace(x, dates, Z, keep_from, window=365, refit=7, min_obs=56):
    """The thesis forecast (electricity.adj_regt_past), step for step, also keeping for each recent refit the
    28 residuals whose median sets the level. The browser uses them to work out how the forecast would
    re-learn its level after an inserted drop. Output is checked against the thesis function."""
    import statsmodels.api as sm
    lx = np.log(np.asarray(x, float))
    X0 = A._reg_design(dates)
    X = np.column_stack([X0, Z])
    n = len(lx)
    r = np.full(n, np.nan)
    params = None
    k0 = X0.shape[1]
    refits = []
    for t in range(n):
        if (t % refit == 0) or params is None:
            lo = max(0, t - window)
            ok = ~np.isnan(lx[lo:t]) & ~np.isnan(X[lo:t]).any(axis=1)
            if ok.sum() >= min_obs:
                cols = list(range(7)) + (list(range(7, k0)) if (t - lo) >= 365 else []) + list(range(k0, X.shape[1]))
                fit = sm.RLM(lx[lo:t][ok], X[lo:t][ok][:, cols], M=sm.robust.norms.HuberT()).fit()
                params = np.zeros(X.shape[1])
                params[cols] = fit.params
                rec = slice(max(lo, t - 28), t)
                okr = ~np.isnan(lx[rec]) & ~np.isnan(X[rec]).any(axis=1)
                if okr.sum() >= 7:
                    e = lx[rec][okr] - X[rec][okr] @ params
                    params[0] += np.median(e)
                    if t >= keep_from:
                        refits.append({"t": int(t), "idx": [int(i) for i in np.arange(rec.start, rec.stop)[okr]],
                                       "e": num(e, 6)})
        if params is not None and not np.isnan(lx[t]) and not np.isnan(X[t]).any():
            r[t] = lx[t] - X[t] @ params
    return r, refits


def explain_elec(day, ev):
    """Likely reason, added after the alarm (same rule as live/update.py): the Christmas to New Year
    period, or a dated event within 7 days (at most 31 days long, high or medium confidence), or nothing."""
    e = ev.copy()
    e["end"] = e["end"].fillna(e["start"])
    e = e[((e["end"] - e["start"]).dt.days <= 31) & e["confidence"].isin(["high", "medium"])
          & ~e["event"].str.contains("Recurring|recurring", na=False)]
    if (day.month == 12 and day.day >= 20) or (day.month == 1 and day.day <= 15):
        return "holiday", "Christmas and New Year holiday period"
    hits = e[(e["start"] - pd.Timedelta(days=7) <= day) & (e["end"] + pd.Timedelta(days=7) >= day)]
    out = [short(t) for t in hits["event"].tolist()[:2]]
    if out:
        return "event", "; ".join(out)
    return "unknown", "No known event in our list"


def short(text, n=90):
    text = str(text).replace("—", ",").replace("–", "-").strip()
    for stop in (": ", " ("):
        if stop in text[12:n + 1]:
            text = text[:text.index(stop, 12)]
            break
    if len(text) > n:
        text = text[:n].rsplit(" ", 1)[0].rstrip(",;") + " ..."
    return text


def electricity(refresh: bool) -> dict:
    import statsmodels.api  # noqa: F401  (imported first: it resets warning filters on import)
    warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")
    cfg = json.loads(CONFIG.read_text())
    info = fetch.refresh(CACHE, SEED) if refresh else {"seeded": fetch.seed_cache(CACHE, SEED), "skipped": True}
    x = E.load()
    Z = E.temps(x.index)
    # stop at the last day that has both demand and temperature
    ok = ~np.isnan(x.to_numpy()) & ~np.isnan(Z).any(axis=1)
    last = int(np.flatnonzero(ok)[-1])
    x, Z = x.iloc[:last + 1], Z[:last + 1]
    dates = x.index
    n = len(x)
    r, refits = regt_trace(x.to_numpy(), dates, Z, keep_from=n - TRY_MAX_DAYS - 14)
    r_thesis = E.adj_regt_past(x.to_numpy(), dates, Z)
    assert np.allclose(r, r_thesis, equal_nan=True, atol=1e-12), "forecast copy differs from the thesis function"
    tabs = calibrate.load(TABLE, "surprise")
    t0 = int(np.searchsorted(dates, pd.Timestamp(cfg.get("start", "2020-01-01"))))
    res, s, z, stat, h = trace(r, cfg["det"], cfg["rule"], tabs, t0, cfg["adj"])
    ev = pd.read_csv(CACHE / "events_electricity.csv", parse_dates=["start", "end"])
    alarms = []
    for a, st, d in zip(res.alarms, res.starts, res.dirs):
        kind, why = explain_elec(dates[a], ev)
        alarms.append({"i": int(a), "date": str(dates[a].date()), "direction": int(d), "kind": kind, "reason": why})
    out = series_block(dates, x.to_numpy(), r, res, s, z, stat, h, t0)
    out.update({
        "key": "elec", "live": True, "tab": "Electricity", "title": "NSW electricity demand",
        "measure": "Daily average demand", "unit": "MW", "source": "AEMO",
        "monitor": {"adj": cfg["adj"], "det": cfg["det"], "rule": cfg["rule"], "start": cfg.get("start", "2020-01-01"),
                    "forecast": "REGT", "label": "Regression forecast (weekday, time of year, temperature) + FOCuS",
                    **monitor_params()},
        "h_table": h_table(cfg["det"]),
        "alarms": alarms,
        "tryit": {"max_days": TRY_MAX_DAYS, "refit_every": 7, "refits": refits},
        "refresh": json.loads(json.dumps(info, default=str)),
    })
    return out


# ----------------------------------------------------------------------------- Opal train tap-ons
def opal() -> dict:
    m = OPAL_MONITOR
    x = opal_daily("train")
    dates = x.index
    r = A.adj_HW(x.to_numpy(), dates)
    tabs = calibrate.load(TABLE, "surprise")
    t0 = int(np.searchsorted(dates, pd.Timestamp(m["start"])))
    res, s, z, stat, h = trace(r, m["det"], m["rule"], tabs, t0, m["adj"])
    alarm_dates = dates[res.alarms]

    # check against the thesis results on disk (when they are there)
    checks = {}
    f = CODE / "results" / "opal" / "alarms.csv"
    if f.exists():
        al = pd.read_csv(f, parse_dates=["alarm"])
        al = al[(al["method"] == m["adj"]) & (al["detector"] == m["det"]) & (al["rule"] == m["rule"])]
        checks["alarms_match_thesis_results"] = bool(list(al["alarm"]) == list(alarm_dates))
        checks["thesis_alarm_count"] = int(len(al))

    sc = score.scored().sort_values("date")
    cls = score.day_classes(dates)
    labels = score.classify(res.alarms, cls)
    ver = pd.read_csv(score.EV / "events_verified.csv", parse_dates=["start", "end"])
    ver = ver[ver["kind"].isin(["policy", "structural", "fault"]) & ver["confidence"].isin(["high", "medium"])]
    alarms = []
    for a, d, lab in zip(res.alarms, res.dirs, labels):
        day = dates[a]
        delta = np.array([(day - b).days for b in sc["date"]])
        if lab == "real":
            ep = sc["episode"].iloc[int(np.argmin(np.abs(delta)))]
            why = f"Real change: {EPISODE_LABELS[ep]}"
        elif lab == "after_event":
            j = np.flatnonzero((delta >= 8) & (delta <= 28))
            why = f"Aftermath of a real change ({EPISODE_LABELS[sc['episode'].iloc[int(j[-1])]]})"
        elif lab == "school":
            why = "School holidays"
        elif lab == "holiday":
            why = "Public holiday or Christmas period"
        elif lab == "explained":
            hit = ver[(ver["start"] - pd.Timedelta(days=score.WIN) <= day) & (ver["end"] + pd.Timedelta(days=score.WIN) >= day)]
            why = short(hit["event"].iloc[0])
        else:
            why = "No known cause"
        alarms.append({"i": int(a), "date": str(day.date()), "direction": int(d), "kind": lab, "reason": why})

    caught, first = score.episodes_caught(alarm_dates, np.array(res.dirs))
    episodes = []
    for ep, g in sc.groupby("episode", sort=False):
        d0 = g["date"].min()
        episodes.append({"id": ep, "label": EPISODE_LABELS[ep], "date": str(d0.date()),
                         "i": int(np.searchsorted(dates, d0)), "type": CLASS_LABELS[g["class"].iloc[0]],
                         "caught": ep in first, "days_to_alarm": int(first[ep]) if ep in first else None})
    episodes.sort(key=lambda e: e["date"])

    out = series_block(dates, x.to_numpy(), r, res, s, z, stat, h, t0)
    out.update({
        "key": "opal", "live": False, "tab": "Train tap-ons", "title": "NSW train tap-ons",
        "measure": "Daily Opal tap-ons on trains", "unit": "tap-ons", "source": "Transport for NSW Opal",
        "monitor": {"adj": m["adj"], "det": m["det"], "rule": m["rule"], "start": m["start"], "forecast": "HW",
                    "label": "Holt-Winters forecast + EWMA",
                    "ewma_lambda": D.PARAMS["EWMA"]["lam"],
                    "hw": {"warm": 28, "alpha": A.HW_ALPHA, "gamma": A.HW_GAMMA, "cap": A.HW_CAP},
                    **monitor_params()},
        "h_table": h_table(m["det"]),
        "alarms": alarms,
        "episodes": episodes, "episodes_caught": int(caught), "episode_window_days": score.WIN,
        "tryit": {"max_days": TRY_MAX_DAYS},
        "checks": checks,
    })
    return out


def main():
    refresh = "--no-refresh" not in sys.argv
    now = dt.datetime.now(dt.timezone.utc)
    nem = now.astimezone(dt.timezone(dt.timedelta(hours=10)))
    data = {
        "built_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "built_nem": nem.strftime("%Y-%m-%d %H:%M"),
        "series": {"elec": electricity(refresh), "opal": opal()},
    }
    tmp = OUT.with_suffix(".part")
    tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    tmp.replace(OUT)
    e, o = data["series"]["elec"], data["series"]["opal"]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")
    print("electricity: last complete day", e["last_day"], "| status", e["status"]["state"], "| alarms",
          [a["date"] for a in e["alarms"]], "| refresh", e["refresh"])
    print("opal: data to", o["last_day"], "| alarms", len(o["alarms"]), "| real changes caught", o["episodes_caught"],
          "of", len(o["episodes"]), "| checks", o["checks"])


if __name__ == "__main__":
    main()
