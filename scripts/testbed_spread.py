"""Test bed for the variance detectors: the spread insertions of testbed.py (variance of the log
series around a hindsight baseline multiplied by 1.5 or 2 from tau) monitored with variance CUSUM, variance EWMA
and variance FOCuS after A, B, Holt-Winters and the yearly memory, both threshold rules, on the same eligible
start days, windows and pairing as testbed.py (it calls testbed.job with these detectors and settings).
Summarised with testbed_analyse.summarise (3-month blocks), "all" start days.
Outputs: results/testbed/spread_pairs.csv, spread_summary.csv (the variance-detector rows, then the existing
mean-detector rows of testbed/summary.csv for spread1.5 and spread2 for comparison; column `source`)."""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import testbed  # noqa: E402
from cpd import adjust as A  # noqa: E402
from cpd.data import opal_daily  # noqa: E402
from testbed_analyse import summarise  # noqa: E402

OUT = HERE / "results" / "testbed"
ADJS = ("A", "B", "HW", "YM")
VDETS = ("CUSUM-var", "EWMA-var", "FOCuS-var")
SETTINGS = [("spread", 1.5), ("spread", 2.0)]


def job(args):
    testbed.ADJS, testbed.DETS, testbed.SETTINGS = ADJS, VDETS, SETTINGS
    return testbed.job(args)


def main():
    x = opal_daily("train")
    dates = x.index
    taus, screened, _ = testbed.eligible(x)
    xv = x.to_numpy().copy()
    xi = pd.Series(xv).interpolate(limit_direction="both").to_numpy()
    from statsmodels.tsa.seasonal import STL
    fit = STL(np.log(xi), period=7, robust=True).fit()
    stl_base = np.asarray(fit.trend) + np.asarray(fit.seasonal)       # as testbed.main
    jobs = [(int(t), xv, dates, stl_base, screened, None) for t in taus]
    rows = []
    with ProcessPoolExecutor(10) as ex:
        for rr in ex.map(job, jobs, chunksize=4):
            rows += rr
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "spread_pairs.csv", index=False)
    s = pd.DataFrame(summarise(df, "all", np.random.default_rng(2026), 3)).assign(source="variance detectors")
    old = pd.read_csv(OUT / "summary.csv")
    old = old[(old.subset == "all") & old.setting.isin(["spread1.5", "spread2"]) & old.adj.isin(ADJS)].assign(source="mean detectors (summary.csv)")
    out = pd.concat([s, old], ignore_index=True)
    out.to_csv(OUT / "spread_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.rule == "ar1"].pivot_table(index=["setting", "adj"], columns="det", values="delta").round(2).to_string())
    print(out[(out.rule == "ar1") & (out.setting == "spread2")].pivot_table(index="adj", columns="det", values="detect_ctl").round(2).to_string())


if __name__ == "__main__":
    main()
