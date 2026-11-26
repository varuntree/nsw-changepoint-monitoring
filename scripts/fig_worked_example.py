"""Figure 4.2 of the thesis: one real alarm traced through the four daily steps.

Monitor: Holt-Winters forecast, capped CUSUM, AR(1) threshold. Event: the June 2021 lockdown (Delta wave onset).
Reads   results/opal/adjusted.csv (Holt-Winters forecast errors), data via cpd.data.opal_daily (tap-ons),
        results/calib/h_table.json (thresholds), results/opal/alarms.csv (to confirm the alarm dates),
        data/opal/events_scored_FROZEN.csv (boundaries).
Writes  figures/F25_worked_example.pdf (.png) and prints the values quoted in the text.
Usage   python scripts/fig_worked_example.py      (part of make figures)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import figstyle_extra as S

REPO, FIG = S.paths()
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
from cpd import calibrate, monitor  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

S.setup()
ADJ, DET, RULE = "HW", "CUSUM-r", "ar1"
ALARM, SHOW_FROM, SHOW_TO = "2021-06-30", "2021-06-05", "2021-07-04"

adj = pd.read_csv(REPO / "results/opal/adjusted.csv", parse_dates=["date"]).set_index("date")
dates = adj.index
r = adj[ADJ].to_numpy()
x = opal_daily("train").reindex(dates)
tabs = calibrate.load(REPO / "results/calib/h_table.json", "surprise")
t_start = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
res = monitor.run(r, "surprise", DET, RULE, tabs, t_start, method=ADJ)

# the run must reproduce the published alarm list exactly
ref = pd.read_csv(REPO / "results/opal/alarms.csv")
ref = ref[(ref.method == ADJ) & (ref.detector == DET) & (ref.rule == RULE)]
assert [str(dates[a].date()) for a in res.alarms] == list(ref.alarm), "alarm dates differ from results/opal/alarms.csv"

i = [str(dates[a].date()) for a in res.alarms].index(ALARM)
ta, s0, ph, h = res.alarms[i], res.seg_starts[i], res.phis[i], res.hs[i]
blind = np.zeros(len(r), dtype=np.bool_)
for a in res.alarms[:i]:
    blind[a + 1:a + 1 + monitor.RELEARN] = True
scale = monitor.running_scale(monitor.delevel(r, False), blind, s0, len(r))
z = r[s0:] / scale
path = D.stat_path(DET, z, ph)
assert int(np.flatnonzero(path > h)[0]) == ta - s0
d = dates[s0:]
df = pd.DataFrame({"taps": x.to_numpy()[s0:], "error": r[s0:], "scale": scale, "z": z, "stat": path}, index=d)
df["forecast"] = df.taps / np.exp(df.error)
df["zc"] = df.z.clip(-3, 3)

ev = pd.read_csv(REPO / "data/opal/events_scored_FROZEN.csv", parse_dates=["date"])
bounds = list(ev[ev.episode == "delta_onset"].date)
facts = {"segment_start": str(dates[s0].date()), "phi_hat": round(ph, 2), "threshold": round(h, 1),
         "threshold_independent": round(float(tabs[DET](0.0)), 1), "alarm": str(dates[ta].date()),
         "estimated_change_date": str(ref.start_est.iloc[i]),
         "first_day_above_independent_threshold": str(df.index[np.flatnonzero(df.stat.to_numpy() > tabs[DET](0.0))[0]].date())}
print(facts)
for day in ("2021-06-14", "2021-06-18", "2021-06-19", "2021-06-23", "2021-06-26", "2021-06-29", "2021-06-30"):
    q = df.loc[day]
    print(day, f"taps {q.taps:,.0f} forecast {q.forecast:,.0f} error {q.error:.3f} "
               f"({100 * (np.exp(q.error) - 1):.0f}%) scale {q.scale:.3f} z {q.z:.1f} capped {q.zc:.1f} stat {q.stat:.1f}")
# uncapped CUSUM on the same forecast errors, under both threshold rules (quoted in the text; each run has its own
# segments, so its own scale, autocorrelation estimate and threshold)
for rule, day in (("ar1", "2021-06-26"), ("indep", "2021-06-14")):
    ru = monitor.run(r, "surprise", "CUSUM", rule, tabs, t_start, method=ADJ)
    al = [str(dates[a].date()) for a in ru.alarms]
    j = al.index(day)                      # fails if the uncapped alarm is not on this day
    bl = np.zeros(len(r), dtype=np.bool_)
    for a in ru.alarms[:j]:
        bl[a + 1:a + 1 + monitor.RELEARN] = True
    sc = monitor.running_scale(monitor.delevel(r, False), bl, ru.seg_starts[j], len(r))
    pu = pd.Series(D.stat_path("CUSUM", r[ru.seg_starts[j]:] / sc, ru.phis[j]), index=dates[ru.seg_starts[j]:])
    print("uncapped CUSUM", rule, "alarms in June 2021:", [a for a in al if a.startswith("2021-06")],
          "threshold", round(ru.hs[j], 1), "statistic on 14 June", round(pu.loc["2021-06-14"], 1))

w = df.loc[SHOW_FROM:SHOW_TO]
fig, ax = plt.subplots(3, 1, figsize=(S.TW, 4.1), sharex=True, layout="constrained",
                       gridspec_kw={"height_ratios": [1.1, 1.0, 1.0]})
for a in ax:
    a.yaxis.grid(True)
    for b in bounds:
        a.axvline(b, color=S.GREEN, lw=0.9, ls=(0, (4, 2)), zorder=1)
ax[0].plot(w.index, w.forecast / 1e3, color=S.GREY, lw=1.3)
ax[0].plot(w.index, w.taps / 1e3, color=S.INK, lw=1.3)
ax[0].set_ylabel("tap-ons a day\n(thousands)")
ax[0].set_ylim(0, 1350)
ax[0].text(pd.Timestamp("2021-06-30"), 800, "forecast", color=S.MUTED, va="bottom", ha="center")
ax[0].text(pd.Timestamp("2021-06-30"), 30, "observed", color=S.INK, va="bottom", ha="center")
ax[0].text(pd.Timestamp("2021-06-14"), 35, "public holiday, 14 June", color=S.INK, ha="center", va="bottom")
ax[0].text(bounds[0] - pd.Timedelta(days=0.5), 1330, "23 June\nrestrictions", color=S.GREEN, ha="right", va="top")
ax[0].text(bounds[1] + pd.Timedelta(days=0.5), 1330, "26 June\nstay-at-home order", color=S.GREEN, ha="left", va="top")

ax[1].axhline(0, color=S.MUTED, lw=0.6, ls=":")
for v in (-3, 3):
    ax[1].axhline(v, color=S.MUTED, lw=0.7, ls=(0, (2, 2)))
ax[1].plot(w.index, w.z, color=S.GREY, lw=1.1)
ax[1].plot(w.index, w.zc, color=S.INK, lw=1.3)
ax[1].set_ylim(-20, 5)
ax[1].set_ylabel("standardised\nforecast error")
ax[1].text(pd.Timestamp("2021-06-05") + pd.Timedelta(days=0.3), -3.6, "cap at -3".replace("-", "−"), color=S.MUTED, va="top")
ax[1].text(pd.Timestamp("2021-06-14") + pd.Timedelta(days=0.6), -10.5, "uncapped", color=S.MUTED, ha="left", va="center")
ax[1].text(pd.Timestamp("2021-06-30"), -2.4, "capped", color=S.INK, ha="center", va="bottom")

stop = ta - s0
ws = df.loc[SHOW_FROM:dates[ta]]
ax[2].plot(ws.index, ws.stat, color=S.INK, lw=1.3)
ax[2].axhline(h, color=S.RED, lw=0.9, ls=(0, (4, 2)))
ax[2].text(pd.Timestamp("2021-06-05") + pd.Timedelta(days=0.3), h * 1.03, f"threshold {h:.1f}", color=S.RED, va="bottom")
ax[2].plot([dates[ta]], [df.stat.iloc[stop]], marker="v", color=S.RED, ms=6, ls="none", clip_on=False, zorder=5)
ax[2].text(dates[ta] + pd.Timedelta(days=0.7), df.stat.iloc[stop] * 1.02, "alarm, 30 June", color=S.INK, va="bottom", ha="left")
ax[2].set_ylim(0, h * 1.32)
ax[2].set_ylabel("capped CUSUM\nstatistic")
ax[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
ax[2].set_xlim(pd.Timestamp(SHOW_FROM), pd.Timestamp(SHOW_TO) + pd.Timedelta(days=1))
ax[2].set_xlabel("2021")
S.save(fig, FIG, "F25_worked_example")
