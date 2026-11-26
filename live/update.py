"""Live monitor for NSW electricity demand: refresh the data, run the monitor, write dashboard.html.

Run once a day shortly after 00:10 NEM time (AEMO rebuilds the current-month file just after midnight):
    python live/update.py
The GitHub Actions workflow .github/workflows/dashboard.yml runs it daily and publishes the page.
The monitor configuration (seasonal adjustment, detector, threshold rule) is read from live/config.json,
which is chosen from the back-test in Chapter 6.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from cpd import calibrate, monitor  # noqa: E402

SRC = ROOT / "data" / "electricity"
CONFIG = HERE / "config.json"
OUT = HERE / "dashboard.html"


def refresh_data():
    sys.path.insert(0, str(HERE))
    import data_fetch as F
    try:
        return F.live_update()
    except Exception as e:  # keep the last saved copy if the feed fails
        return {"error": repr(e)}


def run_monitor(cfg):
    import electricity as E
    x = E.load()
    dates = x.index
    Z = E.temps(dates)
    adj = E.adjust_all(x, Z)
    r = adj[cfg["adj"]]
    tabs = calibrate.load(ROOT / "results" / "calib" / "h_table.json", "surprise")
    t0 = int(np.searchsorted(dates, pd.Timestamp(cfg.get("start", "2020-01-01"))))
    res = monitor.run(r, "surprise", cfg["det"], cfg["rule"], tabs, t0, method=cfg["adj"])
    normal = np.exp(np.log(x.to_numpy()) - r)
    return x, pd.Series(normal, index=dates), res


def explain(day, ev):
    """Likely explanation added after the alarm: dated events (at most 31 days long, high or medium
    confidence) within 7 days, the Christmas-New Year period, or nothing."""
    e = ev.copy()
    e["end"] = e["end"].fillna(e["start"])
    e = e[((e["end"] - e["start"]).dt.days <= 31) & e["confidence"].isin(["high", "medium"])
          & ~e["event"].str.contains("Recurring|recurring", na=False)]
    if (day.month == 12 and day.day >= 20) or (day.month == 1 and day.day <= 15):
        return "Christmas\u2013New Year holiday period"   # the holiday explains it; demand records in it are effects
    hits = e[(e["start"] - pd.Timedelta(days=7) <= day) & (e["end"] + pd.Timedelta(days=7) >= day)]
    out = hits["event"].str.slice(0, 110).tolist()[:2]
    return "; ".join(out) or "no known event in the list (worth a look)"


def refresh_line(info):
    """One human-readable line about where the data came from (no raw dictionaries on the page)."""
    saved = dt.datetime.fromtimestamp((SRC / "nsw_demand_daily.csv").stat().st_mtime)
    if "last_interval_end" in info:
        return (f"Data refreshed {dt.datetime.now():%d %b %Y %H:%M}; latest AEMO interval ends "
                f"{pd.Timestamp(info['last_interval_end']):%d %b %Y %H:%M} (NEM time).")
    if info.get("skipped"):
        return f"Using saved data (last refreshed {saved:%d %b %Y %H:%M}); no download this run."
    return f"Data refresh failed, so the saved data from {saved:%d %b %Y %H:%M} are used."


def figure(x, normal, res, days=120):
    """The last `days` days in the thesis style (scripts/figstyle.py): demand in ink, the expected normal
    dashed grey, alarms as red triangles. Returns (base64 PNG, number of alarms shown)."""
    try:
        import figstyle as S
        S.setup()
        ink, muted, red = S.INK, S.MUTED, S.RED
    except ImportError:
        ink, muted, red = "#1b2330", "#6b7486", "#c8372d"
    plt.rcParams.update({"font.size": 13, "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 13})
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    end = x.index[-1]
    w = slice(end - pd.Timedelta(days=days), end)
    ax.plot(x.loc[w].index, x.loc[w] / 1000, color=ink, lw=1.4, label="daily mean demand")
    ax.plot(normal.loc[w].index, normal.loc[w] / 1000, color=muted, lw=1.2, ls="--", label="expected normal (from past data)")
    shown = [x.index[a] for a in res.alarms if x.index[a] >= end - pd.Timedelta(days=days)]
    top = float((x.loc[w] / 1000).max())
    if shown:
        ax.plot(shown, [top * 1.04] * len(shown), "v", color=red, ms=9, ls="", label="alarm")
    ax.set_ylabel("GW")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, borderaxespad=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.grid(True, color="#e6e9ef", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), len(shown)


def main():
    cfg = json.loads(CONFIG.read_text())
    info = refresh_data() if "--no-refresh" not in sys.argv else {"skipped": True}
    x, normal, res = run_monitor(cfg)
    ev = pd.read_csv(SRC / "events_electricity.csv", parse_dates=["start", "end"])
    last = x.index[-1]
    recent = [a for a in res.alarms if x.index[a] >= last - pd.Timedelta(days=30)]
    status = "ALARM: demand has moved away from its normal pattern" if recent else "Normal"
    rows = "".join(f"<tr><td style='white-space:nowrap'>{x.index[a]:%a %d %b %Y}</td><td>{explain(x.index[a], ev)}</td></tr>" for a in res.alarms[::-1][:12])
    img, n_shown = figure(x, normal, res)
    marks = "Red triangles are alarms." if n_shown else "No alarms in the last 120 days."
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>NSW electricity demand monitor</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:16px;max-width:1000px;margin:24px auto;padding:0 16px;color:#1d2433}}
.st{{font-size:22px;font-weight:700;padding:12px 16px;border-radius:8px;background:{'#fde4e4' if recent else '#e3f4e8'}}}
table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border-bottom:1px solid #e6e8ee;padding:6px 8px;text-align:left}}
.small{{color:#5b6475;font-size:14px}}h1{{font-size:26px;margin:8px 0}}h2{{font-size:19px;margin:18px 0 8px}}</style></head><body>
<h1>NSW electricity demand: changepoint monitor</h1>
<p class="small">Last complete day: <b>{last:%A %d %B %Y}</b> (NEM time) · page built {dt.datetime.now():%d %b %Y %H:%M} · monitor: {cfg['label']}<br>{refresh_line(info)}</p>
<div class="st">{status}</div>
<h2>Last 120 days</h2><img src="data:image/png;base64,{img}" style="width:100%">
<p class="small">{marks} The dashed line is what the monitor expected from past data only (weekday, time of year and temperature). An alarm means the gap between the two has shifted in a way that is unlikely under normal day-to-day variation. The threshold is set for a 5% chance of a false alarm in a simulated normal year; on 2020 to 2026 data the monitor raised about 0.7 alarms a year, four of five on Christmas days.</p>
<h2>Alarms since 2020</h2><table id="alarms"><tr><th>Day</th><th>Possible explanation (added after the alarm, from the events list)</th></tr>{rows}</table>
<h2>How it works</h2><p>1 · forecast normal demand for today from past data only · 2 · compare with what happened (log ratio) · 3 · standardise by recent typical variation · 4 · {cfg['det']} accumulates evidence of a lasting shift · 5 · alarm when the evidence crosses a threshold set by simulation.</p>
<p class="small">Data: Australian Energy Market Operator (AEMO), Aggregated price and demand data, NSW1 (used with attribution under AEMO's copyright permissions). Weather data by Open-Meteo.com (CC BY 4.0), ECMWF IFS model.</p>
</body></html>"""
    OUT.write_text(html)
    print("wrote", OUT, "status:", status)


if __name__ == "__main__":
    main()
