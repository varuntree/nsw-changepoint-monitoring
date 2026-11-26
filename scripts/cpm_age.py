"""CPM and the age of a monitoring segment. The thresholds are calibrated for a 5% chance of a false
alarm in the first 365 days of a segment; on Opal a segment can last years. Here: stationary Gaussian AR(1)
surprises (phi = 0.4), 150 warm-up days and 4 x 365 monitored days, running scale as in cpd/calibrate.py, CPM
(and, for comparison, CUSUM and FOCuS) at its AR(1)-rule threshold h(0.4), no restart; 1,000 runs (seed 20261024).
Reports P(first alarm in year k | no alarm before), k = 1..4, and the cumulative chance by the end of year k.
Output: results/calib/cpm_age.csv"""
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

WARM, YEARS, PHI, REPS = 150, 4, 0.4, 1000
DETS = ("CPM", "CUSUM", "FOCuS")


def job(seed):
    tab = calibrate.load(HERE / "results" / "calib" / "h_table.json", "surprise")
    rng = np.random.default_rng(seed)
    n = WARM + 365 * YEARS
    blind = np.zeros(n, dtype=np.bool_)
    out = {d: [] for d in DETS}
    for _ in range(REPS // 20):
        y = ar1(n, PHI, rng.standard_normal(n))
        z = y[WARM:] / running_scale(y, blind, WARM, n)
        for d in DETS:
            c = np.flatnonzero(D.stat_path(d, z, PHI) > tab[d](PHI))
            out[d].append(int(c[0]) if len(c) else -1)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(job, [20261024 + i for i in range(20)]))
    rows = []
    for d in DETS:
        t = np.concatenate([r[d] for r in res])
        for k in range(1, YEARS + 1):
            lo, hi = 365 * (k - 1), 365 * k
            at_risk = ((t < 0) | (t >= lo)).sum()
            ev = ((t >= lo) & (t < hi)).sum()
            rows.append(dict(det=d, year=k, runs=len(t), at_risk=int(at_risk), alarms=int(ev),
                             p_first_alarm_given_none_before=ev / at_risk, cumulative=float(((t >= 0) & (t < hi)).mean())))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "calib" / "cpm_age.csv", index=False)
    print(out.round(3).to_string(index=False))
