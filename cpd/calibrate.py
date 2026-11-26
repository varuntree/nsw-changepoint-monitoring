"""Threshold tables h(phi): for each detector, the threshold that gives a 5% chance of at least one false
alarm in 365 in-control days, for the WHOLE monitoring procedure (running scale from past data, centre
estimated from 28 days for level series), under a stationary Gaussian AR(1) with lag-1 correlation phi.
Calibrated on one set of seeds and validated on another.

A detector added later is calibrated with `add`: the same jobs, seeds and code path as `build`, run for the new
detectors only and merged into the existing table (detectors do not draw random numbers, so every series, and
hence every existing threshold, is the same as in a full rebuild). Its validation rows are appended the same way.
Usage: python -m cpd.calibrate TABLE [reps]                  build every detector
       python -m cpd.calibrate add TABLE DET [DET ...]       add detectors to an existing table
       python -m cpd.calibrate validate TABLE OUT [DET ...]  validate (all detectors, or append the listed ones)"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import detectors as D
from .kernels import ar1
from .monitor import delevel, phi_hat, running_scale

WARM, YEAR, FA = 150, 365, 0.05
GRID = np.round(np.arange(0.0, 0.901, 0.05), 2)
DETS = ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "FOCuS-var", "CUSUM-var", "CUSUM-r", "EWMA-r", "FOCuS-r",
        "EWMA-var", "CUSUM-k0.25", "CUSUM-k1", "EWMA-l0.05", "EWMA-l0.2")


def max_stats(phi: float, kind: str, reps: int, seed: int, dets=DETS) -> dict:
    rng = np.random.default_rng(seed)
    out = {d: np.empty(reps) for d in dets}
    blind = np.zeros(WARM + YEAR, dtype=np.bool_)
    for i in range(reps):
        y = ar1(WARM + YEAR, phi, rng.standard_normal(WARM + YEAR))
        if kind == "level":
            dl = delevel(y, True)
            c = float(np.median(y[WARM - 28:WARM]))
        else:
            dl, c = y, 0.0
        s = running_scale(dl, blind, WARM, WARM + YEAR)
        z = (y[WARM:] - c) / s
        for d in dets:
            out[d][i] = D.stat_path(d, z, phi).max()
    return out


def _job(args):
    phi, kind, reps, seed, dets = args
    return phi, kind, max_stats(phi, kind, reps, seed, dets)


def build(out_path: Path, reps=10000, chunk=1000, workers=6, seed0=20261022, dets=DETS, table=None):
    """Thresholds for `dets`; when `table` (an existing table) is given, they are merged into it."""
    jobs = []
    for kind in ("surprise", "level"):
        for gi, phi in enumerate(GRID):
            for c in range(reps // chunk):
                jobs.append((float(phi), kind, chunk, seed0 + 100000 * (kind == "level") + 1000 * gi + c, tuple(dets)))
    acc = {}
    with ProcessPoolExecutor(workers) as ex:
        for phi, kind, ms in ex.map(_job, jobs, chunksize=1):
            for d, v in ms.items():
                acc.setdefault((kind, d, phi), []).append(v)
    table = {} if table is None else table
    for (kind, d, phi), vs in acc.items():
        v = np.concatenate(vs)
        table.setdefault(kind, {}).setdefault(d, {})[str(phi)] = float(np.quantile(v, 1 - FA))
        for fa in (0.01, 0.10):
            table.setdefault(f"{kind}@{fa}", {}).setdefault(d, {})[str(phi)] = float(np.quantile(v, 1 - fa))
    table["meta"] = {"reps_per_grid_point": reps, "warm": WARM, "year": YEAR, "fa": FA, "seed0": seed0}
    out_path.write_text(json.dumps(table, indent=1))
    return table


def add(table_path: Path, dets, workers=6):
    """Calibrate extra detectors with the seeds and replicate count recorded in the table and merge them in."""
    table = json.loads(Path(table_path).read_text())
    meta = table["meta"]
    return build(Path(table_path), reps=meta["reps_per_grid_point"], workers=workers, seed0=meta["seed0"],
                 dets=tuple(dets), table=table)


def validate(table_path: Path, out_path: Path, reps=5000, workers=6, seed0=99, dets=DETS, append=False):
    """Realised false-alarm rate at the table threshold h(phi_hat), with phi_hat estimated from 365 past
    days exactly as the monitor does, on fresh seeds. With append=True the rows for `dets` replace any
    existing rows for those detectors in out_path and the other rows are kept."""
    jobs = [(float(phi), kind, reps, seed0 + 7919 * i + (kind == "level") * 13)
            for i, phi in enumerate((0.0, 0.2, 0.4, 0.6, 0.8)) for kind in ("surprise", "level")]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_val_job, [(j, str(table_path), tuple(dets)) for j in jobs]):
            rows += r
    import pandas as pd
    new = pd.DataFrame(rows)
    if append and Path(out_path).exists():
        old = pd.read_csv(out_path)
        new = pd.concat([old[~old["det"].isin(dets)], new], ignore_index=True)
    new.to_csv(out_path, index=False)


def _val_job(arg):
    (phi, kind, reps, seed), tp, dets = arg
    tabs = load(Path(tp), kind)
    rng = np.random.default_rng(seed)
    W = 400
    blind = np.zeros(W + YEAR, dtype=np.bool_)
    hits = {(d, r): 0 for d in dets for r in ("indep", "ar1")}
    for _ in range(reps):
        y = ar1(W + YEAR, phi, rng.standard_normal(W + YEAR))
        dl = delevel(y, kind == "level")
        ph = phi_hat(dl, blind, W)
        c = float(np.median(y[W - 28:W])) if kind == "level" else 0.0
        s = running_scale(dl, blind, W, W + YEAR)
        z = (y[W:] - c) / s
        for d in dets:
            m = D.stat_path(d, z, ph).max()
            hits[(d, "ar1")] += m > tabs[d](ph)
            hits[(d, "indep")] += m > (tabs[d](ph) if d == "FOCuS-AR" else tabs[d](0.0))
    return [dict(phi=phi, kind=kind, det=d, rule=r, fa=v / reps, se=np.sqrt((v / reps) * (1 - v / reps) / reps), reps=reps)
            for (d, r), v in hits.items()]


def load(path: Path, kind: str):
    t = json.loads(Path(path).read_text())[kind]
    fns = {}
    for d, m in t.items():
        keys = sorted(m, key=float)
        xs = np.array([float(k) for k in keys])
        ys = np.array([m[k] for k in keys])
        fns[d] = (lambda xs, ys: (lambda phi: float(np.interp(min(max(phi, 0.0), 0.9), xs, ys))))(xs, ys)
    return fns


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        extra = tuple(sys.argv[4:])
        validate(Path(sys.argv[2]), Path(sys.argv[3]), dets=extra or DETS, append=bool(extra))
    elif sys.argv[1] == "add":
        add(Path(sys.argv[2]), sys.argv[3:])
    else:
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
        build(Path(sys.argv[1]), reps=reps)
