"""Block-length sensitivity of the test-bed intervals. Compares summary.csv (3-month blocks) with
summary_6m.csv (6-month blocks, testbed_analyse.py --block-months 6 --out summary_6m.csv): for the cells of
Table 6.2 (A, B, Holt-Winters, yearly memory x 20% and 40% lasting drops and the 20% 14-day drop x the seven
detectors of the table, AR(1) rule, all start days) and of repository table testbed_full (every adjustment, setting and detector, AR(1)),
how many intervals exclude zero ("starred") with each block length, how many 3-month stars survive, and the
median half-width. Same for contrasts.csv / contrasts_6m.csv and matched_contrasts(_6m).csv.
Output: results/testbed/block_sensitivity.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

TB = Path(__file__).resolve().parents[1] / "results" / "testbed"
DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "FOCuS-r"]
key = ["subset", "setting", "adj", "det", "rule"]
a = pd.read_csv(TB / "summary.csv")
b = pd.read_csv(TB / "summary_6m.csv")
m = a.merge(b, on=key, suffixes=("_3", "_6"))
m = m[(m.subset == "all") & (m.rule == "ar1")]
rows = []


def summ(name, d, lo3, hi3, lo6, hi6):
    s3 = (d[lo3] > 0) | (d[hi3] < 0)
    s6 = (d[lo6] > 0) | (d[hi6] < 0)
    rows.append(dict(table=name, cells=len(d), starred_3m=int(s3.sum()), starred_6m=int(s6.sum()),
                     surviving=int((s3 & s6).sum()), new_6m=int((~s3 & s6).sum()),
                     median_halfwidth_3m=float(np.median((d[hi3] - d[lo3]) / 2)),
                     median_halfwidth_6m=float(np.median((d[hi6] - d[lo6]) / 2))))


t62 = m[m.adj.isin(["A", "B", "HW", "YM"]) & m.setting.isin(["drop20", "drop40", "temp20"]) & m.det.isin(DETS)]
summ("Table 6.2 (gain cells)", t62, "ci_lo_3", "ci_hi_3", "ci_lo_6", "ci_hi_6")
summ("testbed_full (all AR(1) cells)", m, "ci_lo_3", "ci_hi_3", "ci_lo_6", "ci_hi_6")
for f in ("contrasts", "matched_contrasts"):
    c = pd.read_csv(TB / f"{f}.csv").merge(pd.read_csv(TB / f"{f}_6m.csv"), on=[x for x in pd.read_csv(TB / f"{f}.csv").columns
                                                                                if x in ("setting", "a", "b", "adj", "capped", "uncapped")],
                                           suffixes=("_3", "_6"))
    summ(f, c, "lo_3", "hi_3", "lo_6", "hi_6")
out = pd.DataFrame(rows)
out.to_csv(TB / "block_sensitivity.csv", index=False)
print(out.round(3).to_string(index=False))
