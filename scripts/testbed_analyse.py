"""Summarise the paired real-data test bed (results/testbed/pairs.csv).

For each setting x adjustment x detector x rule (and for all start days / screened only / school-holiday
windows / held-out 2025-26):
  detect_ins   P(first alarm in [tau, tau+60) | inserted)
  detect_ctl   P(alarm in [tau, tau+60) | not inserted)   (the background rate on the same days)
  delta        detect_ins - detect_ctl (paired)
  delay        median and restricted mean of the first post-insertion alarm (days, +1) over pairs whose
               control has no alarm in [tau-14, tau+60)
  ci           95% interval of delta by a block bootstrap over blocks of --block-months consecutive calendar
               months (default 3; 1 gives the original month blocks), 1,000 resamples
Writes results/testbed/summary.csv and results/testbed/rq3.csv (past-only vs retrospective pairs).
Usage: python scripts/testbed_analyse.py [--block-months 3] [--out summary.csv]  (--out != summary.csv skips rq3.csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
TB = HERE / "results" / "testbed"


def school_mask(dates):
    from cpd.score import SCHOOL
    d = pd.to_datetime(dates)
    m = np.zeros(len(d), bool)
    for a, b in SCHOOL:
        m |= (d >= pd.Timestamp(a) - pd.Timedelta(days=60)) & (d <= pd.Timestamp(b))
    return m


def summarise(df, label, rng, block_months=3):
    rows = []
    key = ["adj", "det", "rule"]
    ctl = df[df["setting"] == "none0"].set_index(key + ["tau"])
    for (setting, adj, det, rule), g in df[df["setting"] != "none0"].groupby(["setting"] + key):
        c = ctl.loc[(adj, det, rule)].reindex(g["tau"]) if (adj, det, rule) in ctl.index.droplevel(-1) else None
        if c is None:
            continue
        ins = (g["first_post"].to_numpy() >= 0)
        cc = (c["first_post"].to_numpy() >= 0)
        clean = (c["first_post"].to_numpy() < 0) & (c["early14"].to_numpy() == 0)
        dl = g["first_post"].to_numpy()[clean]
        dl = np.where(dl >= 0, dl + 1, 61)
        per = pd.to_datetime(g["date"]).dt.to_period("M")
        k = per.dt.year * 12 + per.dt.month
        months = ((k - k.min()) // block_months).to_numpy() if block_months > 1 else per.to_numpy()
        um = np.unique(months)
        diff = ins.astype(float) - cc.astype(float)
        boots = []
        idx_by_m = {m: np.flatnonzero(months == m) for m in um}
        for _ in range(1000):
            pick = rng.choice(um, size=len(um), replace=True)
            ii = np.concatenate([idx_by_m[m] for m in pick])
            boots.append(diff[ii].mean())
        rows.append(dict(subset=label, setting=setting, adj=adj, det=det, rule=rule, n=len(g),
                         detect_ins=ins.mean(), detect_ctl=cc.mean(), delta=diff.mean(),
                         ci_lo=float(np.quantile(boots, 0.025)), ci_hi=float(np.quantile(boots, 0.975)),
                         n_clean=int(clean.sum()), median_delay=float(np.median(dl)) if len(dl) and (dl <= 60).mean() >= 0.5 else np.inf,
                         rmd=float(dl.mean()) if len(dl) else np.nan, detect_clean=float((dl <= 60).mean()) if len(dl) else np.nan,
                         block_months=block_months, n_blocks=len(um)))
    return rows


def main(bm=3, out_name="summary.csv"):
    import sys
    sys.path.insert(0, str(HERE))
    df = pd.read_csv(TB / "pairs.csv")
    rng = np.random.default_rng(2026)
    rows = []
    rows += summarise(df, "all", rng, bm)
    rows += summarise(df[df["screened"]], "screened", rng, bm)
    sm = school_mask(df["date"])
    rows += summarise(df[sm], "near_school_holiday", rng, bm)
    rows += summarise(df[pd.to_datetime(df["date"]) >= "2025-01-01"], "heldout_2025_26", rng, bm)
    out = pd.DataFrame(rows)
    out.to_csv(TB / out_name, index=False)
    if out_name != "summary.csv":
        return
    # RQ3: past-only vs retrospective twins (same model fitted two ways)
    a = out[(out["subset"] == "all") & out["adj"].isin(["REG", "REG_p"])]
    a.pivot_table(index=["setting", "det", "rule"], columns="adj", values=["delta", "rmd"]).to_csv(TB / "rq3.csv")
    print(out[(out["subset"] == "all") & (out["rule"] == "ar1")].pivot_table(index=["setting"], columns=["adj", "det"], values="delta").round(2).to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-months", type=int, default=3)
    ap.add_argument("--out", default="summary.csv")
    a = ap.parse_args()
    main(a.block_months, a.out)
