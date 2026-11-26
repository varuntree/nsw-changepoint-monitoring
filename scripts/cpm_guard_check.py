"""CPM kernel guard (found by edge-case tests of the independent CPM code, 27 Oct 2026): check that the all-ties guard added to
cpd.kernels.cpm_mw_path changes no result.

The guard only acts when every observed value so far is equal (tie-corrected variance exactly 0), where the
unguarded kernel raised ZeroDivisionError. Any stored result was therefore computed without reaching it. This script
also checks that directly: the unguarded and guarded kernels give identical paths on 400 random series (Gaussian,
rounded, capped at +-3, 20% missing) and on every Opal surprise and level series in opal/adjusted.csv (raw and
capped), and that the unguarded kernel raises on an all-equal start.
Output: results/calib/cpm_guard_check.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import kernels as K  # noqa: E402


@njit(cache=False)
def cpm_unguarded(z, startup):     # cpd/kernels.py cpm_mw_path before the guard (commit 1f65506)
    n_all = z.shape[0]
    out = np.zeros(n_all)
    xs = np.empty(n_all)
    U = np.zeros(n_all)
    n = 0
    tie_sum = 0.0
    for t in range(n_all):
        x = z[t]
        if np.isnan(x):
            out[t] = out[t - 1] if t > 0 else 0.0
            continue
        run = 0.0
        same = 0
        for k in range(n):
            d = x - xs[k]
            if d > 0:
                run += 1.0
            elif d < 0:
                run -= 1.0
            else:
                same += 1
            U[k] += run
        if same > 0:
            a = same + 1.0
            b = float(same)
            tie_sum += (a * a * a - a) - (b * b * b - b)
        xs[n] = x
        n += 1
        if n < startup:
            out[t] = 0.0
            continue
        corr = (n + 1.0) - tie_sum / (n * (n - 1.0))
        best = 0.0
        for k in range(n - 1):
            n1 = k + 1
            n2 = n - n1
            if n1 < 2 or n2 < 2:
                continue
            v = abs(U[k]) / np.sqrt(n1 * n2 * corr / 3.0)
            if v > best:
                best = v
        out[t] = best
    return out


def main():
    rows = []
    rng = np.random.default_rng(20261027)
    for i in range(400):
        z = rng.standard_normal(int(rng.integers(30, 600)))
        kind = ("gaussian", "rounded", "capped", "missing")[i % 4]
        if kind == "rounded":
            z = np.round(z, 1)
        elif kind == "capped":
            z = np.clip(2 * z, -3, 3)
        elif kind == "missing":
            z[rng.random(len(z)) < 0.2] = np.nan
        a, b = cpm_unguarded(z, 20), K.cpm_mw_path(z, 20)
        rows.append(dict(input=f"random {kind}", series=i, n=len(z), max_abs_diff=float(np.nanmax(np.abs(a - b)))))
    adj = pd.read_csv(HERE / "results" / "opal" / "adjusted.csv")
    for col in [c for c in adj.columns if c != "date"]:
        x = adj[col].to_numpy(float)
        med = np.nanmedian(x)
        z = (x - med) / (1.4826 * np.nanmedian(np.abs(x - med)))
        for cap in (False, True):
            zz = np.clip(z, -3, 3) if cap else z
            a, b = cpm_unguarded(zz, 20), K.cpm_mw_path(zz, 20)
            rows.append(dict(input=f"Opal {col}" + (" capped" if cap else ""), series=0, n=int(np.isfinite(zz).sum()),
                             max_abs_diff=float(np.nanmax(np.abs(a - b)))))
    try:
        cpm_unguarded(np.zeros(30), 20)
        raised = "no"
    except ZeroDivisionError:
        raised = "ZeroDivisionError"
    rows.append(dict(input="all-equal start (30 zeros): unguarded kernel", series=0, n=30, max_abs_diff=np.nan, note=raised))
    rows.append(dict(input="all-equal start (30 zeros): guarded kernel", series=0, n=30,
                     max_abs_diff=float(np.abs(K.cpm_mw_path(np.zeros(30), 20)).max()), note="returns 0"))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "calib" / "cpm_guard_check.csv", index=False)
    ok = out[out.input.str.startswith(("random", "Opal"))]
    print(f"{len(ok)} series, max |difference| {ok.max_abs_diff.max():.3g}; unguarded on all-equal start: {raised}")


if __name__ == "__main__":
    main()
