"""Average run length in control (ARL0) equivalent to the false-alarm target.

Two numbers per target:
  geometric   if false alarms arrived at a constant daily rate p, a 5% chance of at least one in 365 days means
              (1 - p)^365 = 0.95, so ARL0 = 1 / p (also for the 1% and 10% targets);
  simulated   the in-control run length of the procedure itself at the calibrated threshold: stationary Gaussian
              AR(1) surprises (phi = 0 and 0.4), 150 warm-up days, running scale from past data exactly as in
              cpd/calibrate.py, AR(1)-rule threshold h(phi) from results/calib/h_table.json (5% target), run
              until the first alarm or T_MAX days. ARL0 is estimated as total days monitored / number of alarms
              (the exponential maximum-likelihood estimate, which allows for the few runs cut at T_MAX); the
              share of runs alarming within 365 days is a fresh-seed check of the 5% target.
CPM is left out: its cost grows with the square of the run length (tens of thousands of days here).
Output: results/calib/arl0.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import calibrate  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd.kernels import ar1  # noqa: E402
from cpd.monitor import running_scale  # noqa: E402

TABLE = HERE / "results" / "calib" / "h_table.json"
OUT = HERE / "results" / "calib" / "arl0.csv"
WARM, T_MAX, YEAR = 150, 60000, 365
REPS, CHUNK = 2000, 50
PHIS = (0.0, 0.4)
DETS = ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CUSUM-r", "EWMA-r", "FOCuS-r", "CUSUM-var", "EWMA-var", "FOCuS-var")


def job(args):
    phi, seed, reps = args
    tab = calibrate.load(TABLE, "surprise")
    rng = np.random.default_rng(seed)
    n = WARM + T_MAX
    blind = np.zeros(n, dtype=np.bool_)
    out = {d: np.empty(reps, np.int64) for d in DETS}
    for i in range(reps):
        y = ar1(n, phi, rng.standard_normal(n))
        z = y[WARM:] / running_scale(y, blind, WARM, n)
        for d in DETS:
            out[d][i] = first_alarm(d, z, phi, tab[d](phi))
    return phi, out


def first_alarm(det, z, phi, h):
    """Run length (1-based) of the first crossing, or -1 within T_MAX. Every statistic is causal (its value at t
    uses z[:t+1] only), so it is computed on prefixes of doubling length (the exact FOCuS search is O(n^2))."""
    m = 4000
    while True:
        m = min(m, len(z))
        cross = np.flatnonzero(D.stat_path(det, z[:m], phi) > h)
        if len(cross):
            return int(cross[0]) + 1
        if m == len(z):
            return -1
        m *= 2


def main():
    rows = []
    for fa in (0.01, 0.05, 0.10):
        p = 1 - (1 - fa) ** (1 / YEAR)
        rows.append(dict(kind="geometric", target=fa, phi=np.nan, det="any", arl0_days=1 / p, arl0_years=1 / p / 365.25))
    jobs = [(phi, 20261023 + 1000 * k + c, CHUNK) for k, phi in enumerate(PHIS) for c in range(REPS // CHUNK)]
    acc = {}
    with ProcessPoolExecutor(10) as ex:
        for phi, out in ex.map(job, jobs):
            for d, v in out.items():
                acc.setdefault((phi, d), []).append(v)
    for (phi, d), vs in acc.items():
        rl = np.concatenate(vs)
        hit = rl > 0
        t = np.where(hit, rl, T_MAX)
        arl = t.sum() / max(hit.sum(), 1)
        p365 = float((hit & (rl <= YEAR)).mean())
        rows.append(dict(kind="simulated", target=0.05, phi=phi, det=d, reps=len(rl), arl0_days=arl,
                         arl0_years=arl / 365.25, arl0_se_days=arl / np.sqrt(max(hit.sum(), 1)),
                         median_rl_days=float(np.median(t)), censored=int((~hit).sum()), p_alarm_365=p365,
                         p_alarm_365_se=np.sqrt(p365 * (1 - p365) / len(rl)), h=calibrate.load(TABLE, "surprise")[d](phi)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
