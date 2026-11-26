"""Online localisation in simulation: how close is the estimated change start at the first
alarm to the true change? The simulation batches store only first crossings, so the series are regenerated from
the sim_run seeds (SeedSequence([crc32(cid), batch]), batches 0-3 = 2,000 runs) for phi = 0.4, normal noise, no
holidays, drops of 1, 2 and 4 SD, on the true-pattern (oracle) and Holt-Winters surprises. CUSUM, FOCuS and CPM
alarm at their matched thresholds (oracle.json of the no-change condition); for every run whose first alarm comes
at or after the change, detectors.change_start gives tau_hat. Reports the median and 90th percentile of
|tau_hat - tau| and the share within 7 days, over detected runs (first alarm within 60 days).
Output: results/sim/summary/localise.csv"""
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
from cpd import detectors as D  # noqa: E402
from cpd import sim as S  # noqa: E402
from sim_run import BATCH, OUT, WBS_M, cid  # noqa: E402

DETS = ("CUSUM", "FOCuS", "CPM")
ADJS = ("oracle", "HW")


def job(args):
    c, b, orc, gj = args
    gen = S.Generator(json.loads(gj))
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(cid(c).encode()), b]))
    rng.integers(0, S.MON, size=(WBS_M, 2))
    rows = []
    for _ in range(BATCH):
        tau = int(rng.integers(60, 301))
        lx = gen.draw(rng, c["phi"], c["noise"], c["calendar"], c["change"], c["size"], tau)
        paths, ph, zs = S.run_series(lx, gen, c["calendar"], None, dets=DETS)
        for adj in ADJS:
            p = ph[adj][0]
            for det in DETS:
                cross = np.flatnonzero(paths[(adj, det)] > orc[f"{adj}|{det}"])
                if not len(cross) or cross[0] < tau or cross[0] - tau >= 60:
                    continue
                t = int(cross[0])
                st = D.change_start(det, zs[adj], t, p)
                rows.append(dict(size=c["size"], adj=adj, det=det, tau=tau, alarm=t, start_est=st, err=abs(st - tau)))
    return rows


def main():
    gj = (OUT / "generator.json").read_text()
    base = dict(phi=0.4, noise="normal", calendar=False)
    orc = json.loads((OUT / cid(dict(base, change="none", size=0.0)) / "oracle.json").read_text())
    jobs = [(dict(base, change="mean", size=s), b, orc, gj) for s in (-1.0, -2.0, -4.0) for b in range(4)]
    with ProcessPoolExecutor(10) as ex:
        rows = [r for rr in ex.map(job, jobs) for r in rr]
    d = pd.DataFrame(rows)
    out = d.groupby(["size", "adj", "det"]).agg(n_detected=("err", "size"), median_abs_error=("err", "median"),
                                               p90_abs_error=("err", lambda e: float(np.quantile(e, 0.9))),
                                               share_within_7=("err", lambda e: float((e <= 7).mean())),
                                               median_delay=("alarm", lambda a: float(np.median(a - d.loc[a.index, "tau"])))).reset_index()
    out["runs"] = 4 * BATCH
    out.to_csv(OUT / "summary" / "localise.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
