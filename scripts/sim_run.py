"""Run the Opal-like simulation. Two passes per (phi, noise, calendar):
  pass 1  no-change runs: maximum statistic over the monitoring year for every adjustment x detector, and
          the offline single-split statistic (max |CUSUM contrast|) and the WBS statistic. The 95%
          quantiles give the pipeline-oracle thresholds (5% chance of a false alarm per year).
  pass 2  change runs: first crossing of every threshold rule (independent, AR(1) table at phi_hat,
          pipeline-oracle) for every adjustment x detector; offline change estimates.
Batches of BATCH runs are written as .npz files and skipped if present (resumable). Seeds come from
SeedSequence([condition id, batch]) so results do not depend on the number of workers.
Usage: python scripts/sim_run.py [core|robust|wide|boot|all] [runs]
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import calibrate  # noqa: E402
from cpd import kernels as K  # noqa: E402
from cpd import sim as S  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

OUT = Path(os.environ.get("SIM_OUT", HERE / "results" / "sim"))   # SIM_OUT: robustness runs elsewhere
OUT.mkdir(parents=True, exist_ok=True)
BATCH = 500
TABLE = HERE / "results" / "calib" / "h_table.json"
WBS_M = 2000
WORKERS = 10


def conditions(phi_opal):
    core = []
    for phi in (0.0, phi_opal, 0.6):
        for cal in (False, True):
            core.append(dict(phi=phi, noise="normal", calendar=cal, change="none", size=0.0))
            for d in (0.5, 1.0, 2.0, 4.0):
                core.append(dict(phi=phi, noise="normal", calendar=cal, change="mean", size=-d))
            core.append(dict(phi=phi, noise="normal", calendar=cal, change="var", size=2.0))
    rob = [dict(phi=phi_opal, noise="t5", calendar=True, change="none", size=0.0),
           dict(phi=phi_opal, noise="t5", calendar=True, change="mean", size=-1.0),
           dict(phi=phi_opal, noise="t5", calendar=True, change="var", size=2.0),
           dict(phi=phi_opal, noise="normal", calendar=True, change="mean", size=1.0),
           dict(phi=phi_opal, noise="normal", calendar=True, change="var", size=1.5),
           dict(phi=phi_opal, noise="normal", calendar=True, change="var", size=3.0)]
    # real holiday scale: Gaussian noise with the raw residual SD
    wide = [dict(phi=phi_opal, noise="wide", calendar=True, change="none", size=0.0),
            dict(phi=phi_opal, noise="wide", calendar=True, change="mean", size=-2.0),
            dict(phi=phi_opal, noise="wide", calendar=True, change="mean", size=-4.0)]
    # block-bootstrap noise from the fitted residuals
    boot = [dict(phi=phi_opal, noise="boot", calendar=True, change="none", size=0.0),
            dict(phi=phi_opal, noise="boot", calendar=True, change="mean", size=-2.0),
            dict(phi=phi_opal, noise="boot", calendar=True, change="mean", size=-4.0)]
    return core, rob, wide, boot


def cid(c):
    return f"phi{c['phi']:.2f}_{c['noise']}_cal{int(c['calendar'])}_{c['change']}{c['size']:+.1f}"


def offline_stats(z, ints):
    """Single-split statistic (max |CUSUM contrast|) and WBS statistic on a monitoring-year z."""
    z = np.where(np.isnan(z), 0.0, z)
    Sx = np.concatenate([[0.0], np.cumsum(z)])
    v, _ = K._best_split(Sx, 0, len(z), 7)
    best = v
    for a, b in ints:
        if b - a >= 14:
            w, _ = K._best_split(Sx, a, b, 7)
            best = max(best, w)
    return v, best


def batch_job(args):
    c, b, runs, oracle, gen_json = args
    f = OUT / cid(c) / f"b{b:03d}.npz"
    if f.exists():
        return str(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    gen = S.Generator(json.loads(gen_json))
    tabs = {k: calibrate.load(TABLE, k) for k in ("surprise", "level")}
    key = zlib.crc32(cid({**c, "noise": c.get("seed_noise", c["noise"])}).encode())   # seed_noise: reuse seeds
    rng = np.random.default_rng(np.random.SeedSequence([key, b]))
    ints = np.sort(rng.integers(0, S.MON, size=(WBS_M, 2)), axis=1)
    spread = c["change"] in ("var", "none")
    dets = S.DETS + (S.SPREAD if spread else ())
    nA, nD = len(S.ADJ), len(dets)
    first = np.full((runs, nA, nD, 3), -1, np.int16)
    mx = np.zeros((runs, nA, nD), np.float32)
    phis = np.zeros((runs, nA), np.float32)
    taus = np.zeros(runs, np.int16)
    off = np.zeros((runs, 2, 2), np.float32)              # (oracle, HW) x (single split, WBS) statistic
    offcp = np.full((runs, 2, 3), -1, np.int16)           # PELT, BinSeg, WBS: nearest change to tau (oracle thr)
    offspur = np.zeros((runs, 2, 3), np.int16)
    for i in range(runs):
        tau = int(rng.integers(60, 301))
        taus[i] = tau
        lx = gen.draw(rng, c["phi"], c["noise"], c["calendar"], c["change"], c["size"], tau)
        paths, ph, zs = S.run_series(lx, gen, c["calendar"], None, spread=spread)
        for a, adj in enumerate(S.ADJ):
            p, kind = ph[adj]
            phis[i, a] = p
            for d, det in enumerate(dets):
                path = paths[(adj, det)]
                mx[i, a, d] = np.nanmax(path)
                t = tabs[kind]
                h_ind = t[det](0.0) if det != "FOCuS-AR" else t[det](p)
                h_ar = t[det](p)
                h_or = oracle.get(f"{adj}|{det}", np.inf) if oracle else np.inf
                first[i, a, d] = S.first_cross_multi(path, (h_ind, h_ar, h_or))
        # offline benchmark on oracle- and HW-adjusted z (same scaling as the monitor)
        for j, adj in enumerate(("oracle", "HW")):
            zz = zs[adj]
            v, w = offline_stats(zz, ints)
            off[i, j] = (v, w)
            if oracle and f"{adj}|single" in oracle:
                thr, thw = oracle[f"{adj}|single"], oracle[f"{adj}|wbs"]
                zc = np.where(np.isnan(zz), 0.0, zz)
                cps = [K.pelt_l2(zc, thr ** 2, 7), K.binseg_l2(zc, thr, 7, 50), K.wbs_l2(zc, thw, 7, ints)]
                for m, cp in enumerate(cps):
                    if len(cp):
                        k = int(np.argmin(np.abs(cp - tau)))
                        offcp[i, j, m] = cp[k]
                        offspur[i, j, m] = int((np.abs(cp - tau) > 7).sum())
    np.savez_compressed(f, first=first, mx=mx, phis=phis, taus=taus, off=off, offcp=offcp, offspur=offspur,
                        dets=np.array(dets), adj=np.array(S.ADJ))
    return str(f)


def oracle_thresholds(c_none):
    d = OUT / cid(c_none)
    fs = sorted(d.glob("b*.npz"))
    mx = np.concatenate([np.load(f)["mx"] for f in fs])
    off = np.concatenate([np.load(f)["off"] for f in fs])
    dets = list(np.load(fs[0])["dets"])
    o = {}
    for a, adj in enumerate(S.ADJ):
        for k, det in enumerate(dets):
            o[f"{adj}|{det}"] = float(np.quantile(mx[:, a, k], 0.95))
    for j, adj in enumerate(("oracle", "HW")):
        o[f"{adj}|single"] = float(np.quantile(off[:, j, 0], 0.95))
        o[f"{adj}|wbs"] = float(np.quantile(off[:, j, 1], 0.95))
    return o


def main(which="core", runs=10000):
    x = opal_daily("train")
    ev = pd.read_csv(HERE / "data" / "opal" / "events_verified.csv", parse_dates=["start", "end"])
    gpath = OUT / "generator.json"
    g = json.loads(gpath.read_text()) if gpath.exists() else S.fit_generator(x, ev, gpath)
    phi_opal = round(g["phi"], 2)
    core, rob, wide, boot = conditions(phi_opal)
    if which == "boot" and not S.RESID.exists():
        S.fit_residuals(x, ev)
    conds = {"core": core, "robust": rob, "wide": wide, "boot": boot}.get(which, core + rob + wide + boot)
    gj = json.dumps(g)
    groups = {}
    for c in conds:
        groups.setdefault((c["phi"], c["noise"], c["calendar"]), []).append(c)
    with ProcessPoolExecutor(WORKERS) as ex:
        for key, cs in groups.items():
            none = [c for c in cs if c["change"] == "none"]
            if not none:
                none = [dict(phi=key[0], noise=key[1], calendar=key[2], change="none", size=0.0)]
            nr = runs
            jobs = [(none[0], b, BATCH, None, gj) for b in range(nr // BATCH)]
            list(ex.map(batch_job, jobs))
            orc = oracle_thresholds(none[0])
            (OUT / cid(none[0]) / "oracle.json").write_text(json.dumps(orc, indent=1))
            # re-run none with oracle thresholds available? first crossings for oracle rule on none runs are
            # derived later from mx (P(mx > h_oracle) = 5% by construction).
            jobs = [(c, b, BATCH, orc, gj) for c in cs if c["change"] != "none" for b in range(nr // BATCH)]
            list(ex.map(batch_job, jobs))
            print("done", key, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "core", int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
