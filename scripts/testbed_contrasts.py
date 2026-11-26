"""Paired block-bootstrap CIs for differences in detection gain between two monitors on the test bed.
Start days are grouped into blocks of --block-months consecutive calendar months (default 3, longer than the
dependence between nearby start days; 1 reproduces the original month blocks) and whole blocks are resampled.
Intervals are unadjusted for multiplicity and indicative.
Usage: python scripts/testbed_contrasts.py [--block-months 3] [--out contrasts.csv]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "testbed"
CON = [("drop40", ("B", "FOCuS-r"), ("B", "FOCuS")), ("drop40", ("HW", "FOCuS-r"), ("HW", "FOCuS")),
       ("drop40", ("B", "FOCuS-r"), ("YM", "FOCuS-r")), ("drop40", ("B", "FOCuS-r"), ("HW", "CUSUM")),
       ("drop20", ("B", "FOCuS-r"), ("YM", "CUSUM")), ("drop20", ("B", "FOCuS-r"), ("YM", "FOCuS-r")),
       ("temp20", ("B", "FOCuS-r"), ("A", "CUSUM")), ("temp40", ("B", "FOCuS-r"), ("A", "CUSUM")),
       ("temp40", ("A", "EWMA"), ("A", "CUSUM")),
       # the Holt-Winters + EWMA recommendation against its alternatives (appended, so the rows above
       # keep their random draws)
       ("temp20", ("HW", "EWMA"), ("HW", "FOCuS")), ("temp20", ("HW", "EWMA"), ("B", "FOCuS-r")),
       ("temp20", ("HW", "EWMA"), ("HW", "FOCuS-r")), ("drop40", ("HW", "EWMA"), ("HW", "FOCuS")),
       ("drop40", ("HW", "EWMA"), ("B", "FOCuS-r")), ("drop40", ("HW", "EWMA"), ("HW", "FOCuS-r"))]


def blocks_of(dates, block_months):
    p = pd.PeriodIndex(pd.to_datetime(dates), freq="M")
    k = p.year * 12 + p.month
    return np.asarray((k - k.min()) // block_months)


def main(block_months=3, out_name="contrasts.csv"):
    p = pd.read_csv(R / "pairs.csv", parse_dates=["date"])
    p = p[p.rule == "ar1"]
    p["hit"] = p.first_post.between(0, 59).astype(float)

    def gain(setting, adj, det):
        a = p[(p.adj == adj) & (p.det == det)]
        ins = a[a.setting == setting].set_index("tau").hit
        ctl = a[a.setting == "none0"].set_index("tau").hit
        return (ins - ctl.reindex(ins.index)).rename("g")

    rng = np.random.default_rng(3)
    rows = []
    for st, m1, m2 in CON:
        d = (gain(st, *m1) - gain(st, *m2)).dropna()
        blk = blocks_of(p.drop_duplicates("tau").set_index("tau").date.reindex(d.index), block_months)
        blocks = [d.values[blk == b] for b in np.unique(blk)]
        bs = [np.mean(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])) for _ in range(2000)]
        rows.append(dict(setting=st, a=" + ".join(m1), b=" + ".join(m2), diff=d.mean(), lo=np.quantile(bs, .025),
                         hi=np.quantile(bs, .975), n=len(d), block_months=block_months, n_blocks=len(blocks)))
    out = pd.DataFrame(rows)
    out.to_csv(R / out_name, index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-months", type=int, default=3)
    ap.add_argument("--out", default="contrasts.csv")
    a = ap.parse_args()
    main(a.block_months, a.out)
