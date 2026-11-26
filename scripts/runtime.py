"""Computing time per detector: single thread, after JIT compilation and a warm-up, for a stream of
n observations of N(0,1) (n = 365, 1,095 and 2,463, the Opal length). Each repeat times a batch of calls long
enough (>= 2 ms) for the timer to be exact and records the time per call; the result is the median of REPS
repeats. Reports milliseconds per run and nanoseconds per observation; the 10-observation call is the fixed
per-call overhead. Output: results/runtime.csv"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import detectors as D  # noqa: E402
from cpd import kernels as K  # noqa: E402

rng = np.random.default_rng(1)
rows = []
DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "FOCuS-var", "CUSUM-var"]


REPS = 51


def timeit(f, reps=REPS, batch_s=2e-3):
    for _ in range(3):  # compile and warm caches
        f()
    t = time.perf_counter()
    f()
    one = max(time.perf_counter() - t, 1e-7)
    inner = max(1, int(batch_s / one))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(inner):
            f()
        ts.append((time.perf_counter() - t) / inner)
    return float(np.median(ts)) * 1000


for n in (10, 365, 1095, 2463):
    z = rng.standard_normal(n)
    for d in DETS:
        ms = timeit(lambda: D.stat_path(d, z, 0.4))
        rows.append(dict(method=d, n=n, ms=ms))
    zs = np.ascontiguousarray(z)
    ints = np.sort(rng.integers(0, n, size=(5000, 2)), axis=1) if n > 20 else np.zeros((0, 2), np.int64)
    rows.append(dict(method="PELT", n=n, ms=timeit(lambda: K.pelt_l2(zs, 3 * np.log(max(n, 2)), 7))))
    rows.append(dict(method="BinSeg", n=n, ms=timeit(lambda: K.binseg_l2(zs, 4.0, 7, 50))))
    rows.append(dict(method="WBS", n=n, ms=timeit(lambda: K.wbs_l2(zs, 4.5, 7, ints))))
import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
df["ns_per_obs"] = 1e6 * df["ms"] / df["n"]
df["reps"] = REPS
df.to_csv(HERE / "results" / "runtime.csv", index=False)
print(df.pivot_table(index="method", columns="n", values="ms").round(3).to_string())
