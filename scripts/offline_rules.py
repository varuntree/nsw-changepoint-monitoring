"""Offline segmentation on Opal under both scoring rules .

opal/offline_changes.csv stores the change dates only. This recomputes the segmentation exactly as
scripts/opal_offline.py (same scaling, penalties, minimum segment, interval seed and order), checks the dates against
the stored file, takes each change's direction as the sign of the later minus the earlier segment mean, and scores the
dates with our scorer under the main rule (+-7 days, any direction, an alarm may credit several episodes) and
the direction-aware rule (0..+7 days, direction checked; must equal offline.csv episodes_caught).
Output: results/opal/offline_rules.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import kernels as K  # noqa: E402
from cpd import score as SC  # noqa: E402
from opal_offline import scaled  # noqa: E402

R = HERE / "results" / "opal"
SERIES = ("STL", "MSTL", "REG")


def changes():
    adj = pd.read_csv(R / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    rng = np.random.default_rng(0)
    out = []
    for m in SERIES:
        y, _ = scaled(adj[m])
        n = len(y)
        ints = np.sort(rng.integers(0, n, size=(5000, 2)), axis=1)
        res = {"PELT": K.pelt_l2(y, 3 * np.log(n), 7), "BinSeg": K.binseg_l2(y, np.sqrt(3 * np.log(n)), 7, 200),
               "WBS": K.wbs_l2(y, 1.3 * np.sqrt(2 * np.log(n)), 7, ints)}
        for meth, cps in res.items():
            allc = sorted(int(c) for c in cps)
            bnd = [0] + allc + [n]
            for k, c in enumerate(allc):
                if c >= t0:
                    out.append(dict(adjustment=m, method=meth, change_date=dates[c],
                                    direction=1 if y[c:bnd[k + 2]].mean() > y[bnd[k]:c].mean() else -1))
    out = pd.DataFrame(out)
    stored = pd.read_csv(R / "offline_changes.csv", parse_dates=["change_date"])
    assert stored.reset_index(drop=True).equals(out[["adjustment", "method", "change_date"]].reset_index(drop=True))
    return out


def main():
    ch = changes()
    off = pd.read_csv(R / "offline.csv")
    rows = []
    for (m, meth), g in ch.groupby(["adjustment", "method"], sort=False):
        g = g.sort_values("change_date")
        cds = pd.DatetimeIndex(g.change_date)
        nf, _ = SC.episodes_caught(cds, None, after=7, before=7, credit="all")
        nd, _ = SC.episodes_caught(cds, list(g.direction))
        q = off[(off.adjustment == m) & (off.method == meth)]
        assert int(q.episodes_caught.iloc[0]) == nd
        rows.append(dict(adjustment=m, method=meth, n_changes=len(g), frozen=int(nf), direction_aware=int(nd)))
    t = pd.DataFrame(rows)
    t.to_csv(R / "offline_rules.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
