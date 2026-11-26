"""Offline benchmarks on a fixed scale.

sim_run.py runs PELT, Binary Segmentation and WBS on the monitor's z = r / s_t, whose running scale s_t (MAD over
days t-119 .. t-29) is inflated for a few weeks after a large step, which adds spurious changes. Here the same
true-pattern ("oracle") series are regenerated from the same seeds as sim_run.batch_job (SeedSequence([crc32(cid),
batch]); first 4 batches = 2,000 runs per condition) and standardised by ONE robust scale, 1.4826 x MAD of the
last 365 training days, with no running scale. Penalties are calibrated on the no-change runs exactly as
sim_run.oracle_thresholds (95th percentile of the single-split and WBS statistics; PELT penalty = threshold^2).
Conditions: phi = 0.4, normal noise, calendar 0 and 1, no change and drops of 1, 2 and 4 SD.
Output: results/sim/summary/offline_fixed.csv (detect7, median_loc_error, spurious_per_run; scale = "fixed"),
with the matching running-scale rows of offline.csv appended for comparison (scale = "running", 10,000 runs)."""
from __future__ import annotations

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import kernels as K  # noqa: E402
from cpd import sim as S  # noqa: E402
from sim_run import BATCH, OUT, WBS_M, cid, offline_stats  # noqa: E402

NB = 4


def job(args):
    c, b, thr, gj = args
    gen = S.Generator(json.loads(gj))
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(cid(c).encode()), b]))
    ints = np.sort(rng.integers(0, S.MON, size=(WBS_M, 2)), axis=1)
    rows = []
    for _ in range(BATCH):
        tau = int(rng.integers(60, 301))
        lx = gen.draw(rng, c["phi"], c["noise"], c["calendar"], c["change"], c["size"], tau)
        r = lx - gen.normal_part(c["calendar"])
        tr = r[S.TRAIN - 365:S.TRAIN]
        s = 1.4826 * np.median(np.abs(tr - np.median(tr)))
        z = np.ascontiguousarray(r[S.TRAIN:] / s)
        v, w = offline_stats(z, ints)
        row = dict(tau=tau, single=v, wbs=w)
        if thr is not None:
            for name, cp in (("PELT", K.pelt_l2(z, thr[0] ** 2, 7)), ("BinSeg", K.binseg_l2(z, thr[0], 7, 50)),
                             ("WBS", K.wbs_l2(z, thr[1], 7, ints))):
                row[f"{name}_near"] = int(cp[np.argmin(np.abs(cp - tau))]) if len(cp) else -1
                row[f"{name}_spur"] = int((np.abs(cp - tau) > 7).sum()) if c["change"] != "none" else len(cp)
        rows.append(row)
    return rows


def main():
    gj = (OUT / "generator.json").read_text()
    rows = []
    with ProcessPoolExecutor(10) as ex:
        for cal in (0, 1):
            base = dict(phi=0.4, noise="normal", calendar=bool(cal))
            none = dict(base, change="none", size=0.0)
            nr = pd.DataFrame([r for rr in ex.map(job, [(none, b, None, gj) for b in range(NB)]) for r in rr])
            thr = (float(np.quantile(nr.single, 0.95)), float(np.quantile(nr.wbs, 0.95)))
            for size in (0.0, -1.0, -2.0, -4.0):
                c = dict(base, change="none" if size == 0 else "mean", size=size)
                d = pd.DataFrame([r for rr in ex.map(job, [(c, b, thr, gj) for b in range(NB)]) for r in rr])
                for m in ("PELT", "BinSeg", "WBS"):
                    near = d[f"{m}_near"].to_numpy()
                    err = np.abs(near - d.tau.to_numpy())[near >= 0]
                    rows.append(dict(scale="fixed", phi=0.4, noise="normal", calendar=cal, change=c["change"], size=size,
                                     adj="oracle", method=m, n=len(d), thr_single=thr[0], thr_wbs=thr[1],
                                     detect7=float(((near >= 0) & (np.abs(near - d.tau) <= 7)).mean()) if size else np.nan,
                                     median_loc_error=float(np.median(err)) if size and len(err) else np.nan,
                                     spurious_per_run=float(d[f"{m}_spur"].mean())))
    out = pd.DataFrame(rows)
    old = pd.read_csv(OUT / "summary" / "offline.csv")
    old = old[(old.phi == 0.4) & (old.noise == "normal") & (old.adj == "oracle") & (old.change == "mean")].assign(scale="running", n=10000)
    out = pd.concat([out, old], ignore_index=True)
    out.to_csv(OUT / "summary" / "offline_fixed.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.pivot_table(index=["calendar", "size", "method"], columns="scale",
                          values=["detect7", "median_loc_error", "spurious_per_run"]).round(3).to_string())


if __name__ == "__main__":
    main()
