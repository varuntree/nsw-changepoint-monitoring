"""Past-only versus retrospective adjustment on Opal, paired by detector and threshold rule (Section 6.5 of the
thesis), for every detector and both scoring rules.

Pairs: MSTL refitted on the past (MSTL_p) vs MSTL fitted to the whole series, and the regression refitted on the past
(REG_p) vs the whole-series regression (REG); every detector run with both, under each threshold rule. For each pair:
episodes caught of 15 under the main rule and the direction-aware check (opal/rule_variants.csv), alarms a year and
non-event alarms a year (opal/scores.csv). The pairs differ in fitting and level treatment as well as in what the fit
can see, so a difference is not purely the value of hindsight.
Outputs: results/opal/past_retro_pairs.csv (one row per pair), results/opal/past_retro_summary.csv (counts of pairs
in which the past-only version catches more / the same / fewer, and has fewer alarms)."""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "opal"
PAIRS = (("MSTL", "MSTL_p", "MSTL"), ("regression", "REG_p", "REG"))


def main():
    rv = pd.read_csv(R / "rule_variants.csv")
    sc = pd.read_csv(R / "scores.csv")
    m = rv.merge(sc[["method", "detector", "rule", "alarms_per_year", "non_event_per_year"]], on=["method", "detector", "rule"])
    rows = []
    for fam, past, retro in PAIRS:
        for (det, rule), _ in m[m.method == past].groupby(["detector", "rule"]):
            p = m[(m.method == past) & (m.detector == det) & (m.rule == rule)].iloc[0]
            q = m[(m.method == retro) & (m.detector == det) & (m.rule == rule)]
            if not len(q):
                continue
            q = q.iloc[0]
            rows.append(dict(family=fam, detector=det, rule=rule,
                             past_frozen=int(p.frozen), retro_frozen=int(q.frozen),
                             past_dir=int(p.r0_7_dir), retro_dir=int(q.r0_7_dir),
                             past_alarms=int(p.n_alarms), retro_alarms=int(q.n_alarms),
                             past_alarms_per_year=p.alarms_per_year, retro_alarms_per_year=q.alarms_per_year,
                             past_non_event_per_year=p.non_event_per_year, retro_non_event_per_year=q.non_event_per_year))
    t = pd.DataFrame(rows)
    t["frozen_diff"] = t.past_frozen - t.retro_frozen
    t["dir_diff"] = t.past_dir - t.retro_dir
    t.to_csv(R / "past_retro_pairs.csv", index=False)
    summ = []
    for rule in ("ar1", "indep", "both"):
        x = t if rule == "both" else t[t.rule == rule]
        for score, col in (("frozen", "frozen_diff"), ("direction-aware", "dir_diff")):
            summ.append(dict(threshold_rule=rule, scoring_rule=score, pairs=len(x), past_more=int((x[col] > 0).sum()),
                             same=int((x[col] == 0).sum()), past_fewer=int((x[col] < 0).sum()),
                             past_fewer_alarms=int((x.past_alarms < x.retro_alarms).sum()),
                             past_fewer_non_event=int((x.past_non_event_per_year < x.retro_non_event_per_year).sum()),
                             median_diff=float(np.median(x[col]))))
    s = pd.DataFrame(summ)
    s.to_csv(R / "past_retro_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
