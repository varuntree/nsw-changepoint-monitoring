"""Figure 6.1 of the thesis: the same five detectors, calibrated for 5% a year, in three cases.

Case 1  simulated Gaussian AR(1) year with no change           results/calib/validation.csv (kind surprise, phi 0.4, rule ar1)
Case 2  simulated normal year with Opal's holidays, yearly     results/sim/summary/fa.csv (phi 0.4, noise normal, calendar 1,
        memory forecast                                         change none, adj YM, rule ar1)
Case 3  real Opal series 2020-2026, yearly memory forecast     results/opal/scores.csv (method YM, rule ar1, non_event_per_year)
Writes  figures/F26_three_cases.pdf (.png) and prints the plotted values.
Usage   python scripts/fig_three_cases.py      (part of make figures)
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd

import figstyle_extra as S

REPO, FIG = S.paths()
S.setup()
DETS = ["CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r"]
NAME = {"CUSUM": "CUSUM", "EWMA": "EWMA", "FOCuS": "FOCuS", "CPM": "CPM", "CUSUM-r": "capped CUSUM"}
TARGET = 5.0
PER_YEAR = -math.log(1 - TARGET / 100)      # alarms a year implied by 5% under a constant hazard

v = pd.read_csv(REPO / "results/calib/validation.csv")
v = v[(v.phi == 0.4) & (v.kind == "surprise") & (v.rule == "ar1")].set_index("det").fa * 100
f = pd.read_csv(REPO / "results/sim/summary/fa.csv")
f = f[(f.phi == 0.4) & (f.noise == "normal") & (f.calendar == 1) & (f.change == "none") & (f.adj == "YM")
      & (f.rule == "ar1")].set_index("det").fa * 100
o = pd.read_csv(REPO / "results/opal/scores.csv")
o = o[(o.method == "YM") & (o.rule == "ar1")].set_index("detector").non_event_per_year
tab = pd.DataFrame({"case1_pct": v[DETS], "case2_pct": f[DETS], "case3_per_year": o[DETS]})
print(tab.round(2).to_string())
print("target as alarms a year:", round(PER_YEAR, 3))

fig, ax = plt.subplots(1, 3, figsize=(S.TW, 2.25), sharey=True, layout="constrained")
y = list(range(len(DETS)))[::-1]
panels = [("1. Simulated AR(1) year", tab.case1_pct, "years with a false alarm (%)", 100, TARGET, "{:.1f}"),
          ("2. With holidays", tab.case2_pct, "years with a false alarm (%)", 100, TARGET, "{:.1f}"),
          ("3. Real Opal", tab.case3_per_year, "non-event alarms a year", 7, PER_YEAR, "{:.1f}")]
for a, (title, vals, xlabel, xmax, target, fmt) in zip(ax, panels):
    over = vals > (1.5 * target)
    a.barh(y, vals, height=0.62, color=[S.RED if k else S.GREY for k in over], zorder=2)
    a.axvline(target, color=S.INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    for yi, val in zip(y, vals):
        txt = "100" if val >= 99.95 else fmt.format(val)
        inside = val > 0.72 * xmax
        a.text(val - 0.02 * xmax if inside else max(val, target) + 0.03 * xmax, yi, txt, va="center",
               ha="right" if inside else "left", color="white" if inside else S.INK)
    a.set_title(title)
    a.set_xlabel(xlabel)
    a.set_xlim(0, xmax)
    a.xaxis.grid(True)
    a.tick_params(axis="y", length=0)
ax[0].set_yticks(y, [NAME[d] for d in DETS])
ax[0].text(TARGET + 3, 4.62, "5% target", color=S.INK, va="bottom", ha="left")
ax[2].text(PER_YEAR + 0.25, 4.62, "5% target: 0.05 a year", color=S.INK, va="bottom", ha="left")
for a in ax:
    a.set_ylim(-0.55, 5.25)
S.save(fig, FIG, "F26_three_cases")
