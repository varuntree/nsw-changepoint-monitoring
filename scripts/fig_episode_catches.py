"""Figure 6.3 of the thesis: how many monitors catch each of the 15 scored episodes under each scoring rule.

Reads   results/opal/rule_variants.csv (episodes_frozen: main rule; episodes_headline: direction-aware check;
        the 137 monitors with at least ten alarms), data/opal/events_scored_FROZEN.csv (episodes, announced or not).
Writes  figures/F27_episode_catches.pdf (.png) and prints the plotted counts.
Usage   python scripts/fig_episode_catches.py      (part of make figures)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import figstyle_extra as S

REPO, FIG = S.paths()
S.setup()
LABEL = {"covid_onset": "COVID-19 onset, Mar 2020", "covid_easing": "COVID-19 easing, May 2020",
         "flood_2021_03": "Floods, Mar 2021", "delta_onset": "Delta wave onset, Jun 2021",
         "delta_end": "Delta wave end, Oct 2021", "omicron_onset": "Omicron wave onset, Dec 2021",
         "omicron_end": "Omicron wave end, Feb 2022", "shutdown_2022_02": "Shutdown, Feb 2022",
         "flood_2022_03": "Floods, Mar 2022", "strike_2022_06": "Rail dispute, Jun 2022",
         "flood_2022_07": "Floods, Jul 2022", "strike_2022_08": "Rail dispute, Aug 2022",
         "strike_2025_01": "Rail dispute, Jan 2025", "strike_2025_02": "Rail dispute, Feb 2025",
         "outage_2025_05": "Outage, May 2025"}

rv = pd.read_csv(REPO / "results/opal/rule_variants.csv").fillna("")
rv = rv[rv.n_alarms >= 10]
n = len(rv)
ev = pd.read_csv(REPO / "data/opal/events_scored_FROZEN.csv")
eps = list(dict.fromkeys(ev.episode))
unannounced = {e for e in eps if set(ev[ev.episode == e].announced_before) == {"no"}}
main = {e: sum(e in s.split(";") for s in rv.episodes_frozen) for e in eps}
both = {e: sum(e in a.split(";") and e in b.split(";") for a, b in zip(rv.episodes_frozen, rv.episodes_headline)) for e in eps}
check = {e: sum(e in s.split(";") for s in rv.episodes_headline) for e in eps}
assert both == check, "a direction-aware catch that the main rule does not count"
tab = pd.DataFrame({"episode": [LABEL[e] for e in eps], "unannounced": [e in unannounced for e in eps],
                    "main_rule": [main[e] for e in eps], "both_rules": [both[e] for e in eps]})
print(n, "monitors")
print(tab.to_string())

fig, ax = plt.subplots(figsize=(S.TW, 3.55), layout="constrained")
y = list(range(len(eps)))[::-1]
ax.barh(y, tab.main_rule, height=0.66, color=S.LIGHT, zorder=2)
ax.barh(y, tab.both_rules, height=0.66, color=S.INK, zorder=3)
for yi, m, b in zip(y, tab.main_rule, tab.both_rules):
    ax.text(m + 1.5, yi, f"{b} and {m}", va="center", ha="left", color=S.INK)
ax.set_yticks(y, [f"{lab}{' *' if u else ''}" for lab, u in zip(tab.episode, tab.unannounced)])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, n)
ax.set_xticks([0, 25, 50, 75, 100, 125, n])
ax.set_xlabel(f"monitors that catch the episode (of {n})")
ax.xaxis.grid(True)
ax.text(n - 1, y[7], "dark: caught under both rules\nlight: under the main rule only\n* not announced in advance", color=S.INK,
        va="center", ha="right", linespacing=1.35)
S.save(fig, FIG, "F27_episode_catches")
