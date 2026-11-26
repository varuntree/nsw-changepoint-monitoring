"""Figure 8.1 of the thesis: non-event alarms a year against episodes caught under both rules, every Opal monitor.

Reads   results/opal/scores.csv (non_event_per_year), results/opal/rule_variants.csv (frozen: main rule;
        r0_7_dir: direction-aware check). All 150 monitors; the smaller of the two counts is plotted.
Writes  figures/F28_quiet_or_sensitive.pdf (.png) and prints the labelled points.
Usage   python scripts/fig_quiet_or_sensitive.py      (part of make figures)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle_extra as S

REPO, FIG = S.paths()
S.setup()
key = ["method", "detector", "rule"]
sc = pd.read_csv(REPO / "results/opal/scores.csv")[key + ["non_event_per_year", "n_alarms"]]
rv = pd.read_csv(REPO / "results/opal/rule_variants.csv")[key + ["frozen", "r0_7_dir"]]
d = sc.merge(rv, on=key, validate="one_to_one")
d["both"] = d[["frozen", "r0_7_dir"]].min(axis=1)
LAB = [("YM", "CUSUM-r", "ar1", "yearly memory + capped CUSUM", (2.0, 0.1), "left"),
       ("HW", "EWMA", "ar1", "Holt-Winters + EWMA", (0.15, 8.9), "left"),
       ("A", "CUSUM", "ar1", "same-weekday difference + CUSUM", (0.15, 7.4), "left"),
       ("YM", "CUSUM", "indep", "yearly memory + CUSUM, independent threshold", (0.15, 10.4), "left")]
rng = np.random.default_rng(1)
jit = rng.uniform(-0.13, 0.13, len(d))
fig, ax = plt.subplots(figsize=(S.TW * 0.86, 2.9), layout="constrained")
ax.scatter(d.non_event_per_year, d.both + jit, s=13, color=S.MUTED, alpha=0.6, zorder=2, linewidths=0)
for m, det, rule, text, pos, ha in LAB:
    q = d[(d.method == m) & (d.detector == det) & (d.rule == rule)].iloc[0]
    print(m, det, rule, round(q.non_event_per_year, 1), int(q.frozen), int(q.r0_7_dir), int(q.both))
    ax.scatter([q.non_event_per_year], [q.both], s=30, color=S.INK, zorder=4)
    ax.annotate(text, xy=(q.non_event_per_year, q.both), xytext=pos, ha=ha, va="center", color=S.INK,
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.6, shrinkA=1, shrinkB=3))
quiet = d[d.non_event_per_year < 1.5]
print("monitors below 1.5 non-event alarms a year:", len(quiet), "most episodes under both rules:", int(quiet.both.max()))
print("monitors with at least 7 under both rules:", int((d.both >= 7).sum()), "fewest non-event alarms a year among them:",
      round(d[d.both >= 7].non_event_per_year.min(), 1))
ax.set_xlim(0, 7.6)
ax.set_ylim(-0.6, 11.2)
ax.set_yticks([0, 3, 6, 9])
ax.set_xlabel("non-event alarms a year")
ax.set_ylabel("episodes caught under\nboth rules (of 15)")
ax.yaxis.grid(True)
S.save(fig, FIG, "F28_quiet_or_sensitive")
