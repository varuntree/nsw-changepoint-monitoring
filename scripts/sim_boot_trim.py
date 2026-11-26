"""Robustness of the resampled-noise result.

The block-bootstrap noise pool (results/sim/generator_resid.csv, the generator fit's 773 normal days) already
excludes the eight reader-off days and every disruption or fault day, so removing the reader-off days (and the week
after) reproduces the existing "boot" runs exactly (checked and recorded in boot_trim_pool.csv). The stricter check run
here removes the pool's heaviest tail: the 23 days more than 5 robust SD from the median (16 at New Year, Easter, the
June long weekend and Christmas; 7 other single days, mostly weekends; listed in boot_trim_removed_days.csv). Same design as the "boot" conditions (phi = 0.4 slot, holidays in the data, no change and
2 and 4 SD drops), same seeds (seed_noise = "boot"), 10,000 runs per condition.

Outputs: results/sim_trim/<condition>/b*.npz, results/sim_trim/summary/fa.csv, detect.csv (same columns as
results/sim/summary), results/sim_trim/summary/boot_trim_pool.csv and boot_vs_trim.csv (side by side with "boot").
Usage: python scripts/sim_boot_trim.py [runs]"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

HERE = Path(__file__).resolve().parents[1]
os.environ["SIM_OUT"] = str(HERE / "results" / "sim_trim")   # before sim_run is imported (also in workers)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import sim_analyse as SA  # noqa: E402
import sim_run as SR  # noqa: E402
from cpd import sim as S  # noqa: E402
from cpd.adjust import READER_OFF  # noqa: E402

OUT = HERE / "results" / "sim_trim"
SUM = OUT / "summary"


def pool_table(g):
    r = pd.read_csv(S.RESID, parse_dates=["date"]).set_index("date")["resid"]
    ro = pd.DatetimeIndex(list(READER_OFF))
    wk = pd.DatetimeIndex(sorted({d + pd.Timedelta(days=k) for d in ro for k in range(8)}))
    keep = (r - r.median()).abs() <= S.TRIM_SD * g["sigma"]
    rows = [dict(pool="boot (as run)", days=len(r), reader_off_days_in_pool=int(r.index.isin(ro).sum()),
                 reader_off_or_week_after_in_pool=int(r.index.isin(wk).sum()), excess_kurtosis=kurtosis(r),
                 sd_over_sigma=r.std() / g["sigma"]),
            dict(pool="boot without reader-off days and the week after", days=int((~r.index.isin(wk)).sum()),
                 reader_off_days_in_pool=0, reader_off_or_week_after_in_pool=0, excess_kurtosis=kurtosis(r[~r.index.isin(wk)]),
                 sd_over_sigma=r[~r.index.isin(wk)].std() / g["sigma"]),
            dict(pool=f"boottrim (|r - median| <= {S.TRIM_SD:g} sigma)", days=int(keep.sum()), reader_off_days_in_pool=0,
                 reader_off_or_week_after_in_pool=0, excess_kurtosis=kurtosis(r[keep]), sd_over_sigma=r[keep].std() / g["sigma"])]
    pd.DataFrame(rows).to_csv(SUM / "boot_trim_pool.csv", index=False)
    r[~keep].rename("resid").to_csv(SUM / "boot_trim_removed_days.csv", index_label="date")


def summarise():
    rows_fa, rows_det = [], []
    sa_sim = SA.SIM
    SA.SIM = OUT
    try:
        for d in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name.startswith("phi")):
            c = SA.parse(d.name)
            L = SA.load(d)
            first, adj, dets = L["first"], L["adj"], L["dets"]
            n = first.shape[0]
            if c["change"] == "none":
                even, odd = L["batch"] % 2 == 0, L["batch"] % 2 == 1
                for a, an in enumerate(adj):
                    for k, dn in enumerate(dets):
                        for r, rn in enumerate(SA.RULES[:2]):
                            p = float((first[:, a, k, r] >= 0).mean())
                            rows_fa.append({**c, "adj": an, "det": dn, "rule": rn, "fa": p, "se": np.sqrt(p * (1 - p) / n), "n": n})
                        h = np.quantile(L["mx"][even, a, k], 0.95)
                        p = float((L["mx"][odd, a, k] > h).mean())
                        rows_fa.append({**c, "adj": an, "det": dn, "rule": "oracle", "fa": p,
                                        "se": np.sqrt(p * (1 - p) / odd.sum()), "n": int(odd.sum())})
                continue
            tau = L["taus"].astype(int)
            for a, an in enumerate(adj):
                for k, dn in enumerate(dets):
                    for r, rn in enumerate(SA.RULES):
                        T = first[:, a, k, r].astype(int)
                        T = np.where(T < 0, 10_000, T)
                        pre = T < tau
                        dl = (T - tau + 1)[~pre]
                        det60 = dl <= 60
                        rows_det.append({**c, "adj": an, "det": dn, "rule": rn, "n": n, "pre_fa": float(pre.mean()),
                                         "detect60": float(det60.mean()) if len(dl) else np.nan,
                                         "F7": float((dl <= 7).mean()) if len(dl) else np.nan,
                                         "rmd": float(np.minimum(dl, 61).mean()) if len(dl) else np.nan})
    finally:
        SA.SIM = sa_sim
    fa, det = pd.DataFrame(rows_fa), pd.DataFrame(rows_det)
    fa.to_csv(SUM / "fa.csv", index=False)
    det.to_csv(SUM / "detect.csv", index=False)
    # side by side with the original resampled runs
    fa0 = pd.read_csv(HERE / "results" / "sim" / "summary" / "fa.csv")
    de0 = pd.read_csv(HERE / "results" / "sim" / "summary" / "detect.csv")
    k = ["adj", "det", "rule"]
    a = fa0[(fa0.noise == "boot") & (fa0.change == "none")][k + ["fa"]].merge(fa[k + ["fa"]], on=k, suffixes=("_boot", "_trim"))
    a["measure"] = "false-alarm years"
    b = de0[(de0.noise == "boot") & (de0["size"] == -2.0)][k + ["detect60"]].merge(
        det[det["size"] == -2.0][k + ["detect60"]], on=k, suffixes=("_boot", "_trim"))
    b["measure"] = "2 SD drop found within 60 days"
    c4 = de0[(de0.noise == "boot") & (de0["size"] == -4.0)][k + ["detect60"]].merge(
        det[det["size"] == -4.0][k + ["detect60"]], on=k, suffixes=("_boot", "_trim"))
    c4["measure"] = "4 SD drop found within 60 days"
    out = pd.concat([a.rename(columns={"fa_boot": "boot", "fa_trim": "trim"}),
                     b.rename(columns={"detect60_boot": "boot", "detect60_trim": "trim"}),
                     c4.rename(columns={"detect60_boot": "boot", "detect60_trim": "trim"})])
    out.to_csv(SUM / "boot_vs_trim.csv", index=False)
    # ledger-ready summary: (a) uncapped detectors at the AR(1) threshold, (b) the ranking at the matched threshold
    unc = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR"]
    fa_ = out[(out.measure == "false-alarm years") & (out.rule == "ar1") & out.det.isin(unc) & out.adj.isin(["A", "B", "HW", "YM", "oracle"])]
    rows = [dict(key=f"fa_ar1_uncapped_{w}_{s}", value=100 * getattr(fa_[w], s)()) for w in ("boot", "trim") for s in ("min", "max")]
    for meas, tag in (("2 SD drop found within 60 days", "2sd"), ("4 SD drop found within 60 days", "4sd")):
        x = out[(out.measure == meas) & (out.rule == "oracle") & (out.adj == "oracle")]
        for _, r in x.iterrows():
            rows += [dict(key=f"detect_{tag}_{r.det}_{w}", value=100 * r[w]) for w in ("boot", "trim")]
    pd.DataFrame(rows).to_csv(SUM / "verdict_values.csv", index=False)
    return out


def main(runs=10000):
    SUM.mkdir(parents=True, exist_ok=True)
    g = json.loads((HERE / "results" / "sim" / "generator.json").read_text())
    pool_table(g)
    phi = round(g["phi"], 2)
    conds = [dict(phi=phi, noise="boottrim", calendar=True, change=ch, size=sz, seed_noise="boot")
             for ch, sz in (("none", 0.0), ("mean", -2.0), ("mean", -4.0))]
    gj = json.dumps(g)
    with ProcessPoolExecutor(SR.WORKERS) as ex:
        list(ex.map(SR.batch_job, [(conds[0], b, SR.BATCH, None, gj) for b in range(runs // SR.BATCH)]))
        orc = SR.oracle_thresholds(conds[0])
        (OUT / SR.cid(conds[0]) / "oracle.json").write_text(json.dumps(orc, indent=1))
        list(ex.map(SR.batch_job, [(c, b, SR.BATCH, orc, gj) for c in conds[1:] for b in range(runs // SR.BATCH)]))
    out = summarise()
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
