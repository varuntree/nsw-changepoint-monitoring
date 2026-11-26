"""Published in-control run lengths for the CUSUM and EWMA kernels (appendix implementation-check table).

20,000 independent standard normal series of 4,000 days each, fixed seed; a series without an alarm counts
4,000 (a handful at h = 5 and for EWMA). Writes results/calib/kernel_arl.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import kernels as K  # noqa: E402

REPS, N, SEED = 20_000, 4_000, 20261024
CHECKS = [("CUSUM", "k=0.5, h=4", 168, K.cusum_path, 4.0, (0.5,)),
          ("CUSUM", "k=0.5, h=5", 465, K.cusum_path, 5.0, (0.5,)),
          ("EWMA", "lambda=0.1, L=2.814", 500, K.ewma_path, 2.814, (0.1,))]


def main():
    rows = []
    for det, setting, ref, fn, h, a in CHECKS:
        rng = np.random.default_rng(SEED)
        rl = np.empty(REPS)
        for i in range(REPS):
            t = K.first_cross(fn(rng.standard_normal(N), *a), h)
            rl[i] = t + 1 if t >= 0 else N
        rows.append(dict(det=det, setting=setting, published=ref, arl=rl.mean(), se=rl.std(ddof=1) / np.sqrt(REPS),
                         censored=int((rl >= N).sum()), reps=REPS, days=N, seed=SEED))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "calib" / "kernel_arl.csv", index=False)
    print(out.to_string(index=False))
    ewma_var()


def ewma_var(reps=2000, n=2000, h=3.0, seed=3):
    """Variance EWMA (lambda 0.1) run lengths at h = 3 for sd 1, sqrt(2) (doubled variance) and sqrt(0.5) (halved), the
    same design as tests/test_ewma_var.py::test_detects_increase_not_decrease. Writes results/calib/ewma_var_arl.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for lab, sd in (("in control", 1.0), ("variance doubled", np.sqrt(2.0)), ("variance halved", np.sqrt(0.5))):
        rl = np.empty(reps)
        for i in range(reps):
            t = K.first_cross(K.ewma_var_path(sd * rng.standard_normal(n), 0.1), h)
            rl[i] = t + 1 if t >= 0 else n
        rows.append(dict(setting=lab, sd=sd, h=h, arl=rl.mean(), censored=int((rl >= n).sum()), reps=reps, days=n))
    out = pd.DataFrame(rows)
    out["ratio_to_in_control"] = out.arl[0] / out.arl
    out.to_csv(HERE / "results" / "calib" / "ewma_var_arl.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
