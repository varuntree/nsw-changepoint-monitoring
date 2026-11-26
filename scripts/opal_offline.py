"""Offline (hindsight) benchmarks on the real Opal series: PELT, Binary Segmentation and WBS on the
retrospective STL, MSTL and regression (REG) seasonally adjusted log series, scaled by a robust long-run standard deviation
(MAD of the series minus its 29-day running median, inflated by sqrt((1+phi)/(1-phi))). Standard settings:
PELT penalty 3 log n, Binary Segmentation threshold sqrt(3 log n), WBS threshold 1.3 sqrt(2 log n) with 5,000
random intervals; minimum segment 7 days. Estimated change dates are scored like alarms (within 7 days of a
scored boundary, episodes caught, circular-shift chance baseline). Output: results/opal/offline.csv and
offline_changes.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import kernels as K  # noqa: E402
from cpd import score  # noqa: E402

OP = HERE / "results" / "opal"


def scaled(s: pd.Series):
    x = s.interpolate(limit_direction="both")
    r = x - x.rolling(29, center=True, min_periods=7).median()
    r = r.dropna()
    sd = 1.4826 * np.median(np.abs(r - r.median()))
    phi = float(np.clip(r.autocorr(1), 0, 0.9))
    sd_lr = sd * np.sqrt((1 + phi) / (1 - phi))
    return ((x - x.median()) / sd_lr).to_numpy(), phi


def main():
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    rows, ch = [], []
    rng = np.random.default_rng(0)
    for m in ("STL", "MSTL", "REG"):      # REG added last so the STL and MSTL rows are unchanged
        y, phi = scaled(adj[m])
        n = len(y)
        ints = np.sort(rng.integers(0, n, size=(5000, 2)), axis=1)
        res = {"PELT": K.pelt_l2(y, 3 * np.log(n), 7), "BinSeg": K.binseg_l2(y, np.sqrt(3 * np.log(n)), 7, 200),
               "WBS": K.wbs_l2(y, 1.3 * np.sqrt(2 * np.log(n)), 7, ints)}
        for meth, cps in res.items():
            allc = sorted(int(c) for c in cps)
            bnd = [0] + allc + [n]
            # direction of each change: later segment mean minus earlier segment mean
            dirs_all = {c: (1 if y[c:bnd[k + 2]].mean() > y[bnd[k]:c].mean() else -1) for k, c in enumerate(allc)}
            idx = [c for c in allc if c >= t0]
            sc = score.score(idx, dates, t0, n_shift=1000, dirs=[dirs_all[c] for c in idx])
            sc.pop("first_alarm_rel", None)
            sc.update(adjustment=m, method=meth, phi=phi)
            rows.append(sc)
            ch += [dict(adjustment=m, method=meth, change_date=dates[i]) for i in idx]
    pd.DataFrame(rows).to_csv(OP / "offline.csv", index=False)
    pd.DataFrame(ch).to_csv(OP / "offline_changes.csv", index=False)
    print(pd.DataFrame(rows)[["adjustment", "method", "n_alarms", "near_real", "share_real", "null_share_real", "episodes_caught", "null_episodes"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
