"""CPM (Mann-Whitney) kernel against published limits.

1. Hawkins and Deng (2010), J. Qual. Technol. 42(2), Table 1 (p. 168), in-control ARL 500 (alpha = 0.002), warm-up
   14 (testing from the 15th observation), limits carried forward between tabulated n (3.214 beyond n = 1,000):
   - per-step hazard at the tabulated n: 20,000 N(0,1) series of length 100; target 0.002 at every n
   - in-control ARL: 4,000 series up to n = 4,000 (censored runs counted at 4,000); target 500
2. The R package cpm 2.3 limits for ARL 500 with startup 20, our setting (data/cpm/cpm23_MWARL500.csv; the
   package uses the mean of the last 101 values beyond its table): the same ARL check.
Seed 2026. Output: results/calib/cpm_hd_check.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import kernels as K  # noqa: E402

HD = {15: 3.069, 16: 3.047, 17: 3.043, 18: 3.034, 19: 3.054, 20: 3.059, 22: 3.082, 24: 3.096, 26: 3.108, 28: 3.121,
      30: 3.130, 35: 3.149, 40: 3.162, 45: 3.171, 50: 3.178, 60: 3.188, 70: 3.194, 80: 3.199, 90: 3.200, 100: 3.203,
      125: 3.206, 150: 3.209, 200: 3.210, 250: 3.212, 300: 3.215, 500: 3.213, 1000: 3.214}
LMAX = 4000


def limits(kind):
    h = np.full(LMAX + 1, np.nan)
    if kind == "hd":
        last = None
        for n in range(15, LMAX + 1):
            last = HD.get(n, last)
            h[n] = last
        return h, 15
    t = pd.read_csv(HERE / "data" / "cpm" / "cpm23_MWARL500.csv")
    h[t.n.to_numpy()] = t.h.to_numpy()
    h[t.n.max() + 1:] = t.h.to_numpy()[-101:].mean()
    return h, 20


def hazard_job(seed, reps=1000, L=100):
    h, start = limits("hd")
    rng = np.random.default_rng(seed)
    surv, alarms = np.zeros(L + 1), np.zeros(L + 1)
    for _ in range(reps):
        D = K.cpm_mw_path(rng.standard_normal(L), start)
        for n in range(start, L + 1):
            surv[n] += 1
            if D[n - 1] > h[n]:
                alarms[n] += 1
                break
    return surv, alarms


def arl_job(args):
    kind, seed, reps = args
    h, start = limits(kind)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        D = K.cpm_mw_path(rng.standard_normal(LMAX), start)
        idx = np.flatnonzero(D[start - 1:] > h[start:])
        out.append(idx[0] + 1 if len(idx) else LMAX - start + 1)
    return out


def main():
    rows = []
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(hazard_job, [2026 * 100 + i for i in range(20)]))
        surv = sum(r[0] for r in res)
        alarms = sum(r[1] for r in res)
        tabn = [n for n in HD if n <= 100]
        a = alarms[tabn].sum() / surv[tabn].sum()
        rows.append(dict(check="hazard at tabulated n (H&D Table 1, ARL 500, warm-up 14)", runs=int(surv[15]), value=a,
                         se=np.sqrt(a * (1 - a) / surv[tabn].sum()), target=0.002))
        a_all = alarms[15:].sum() / surv[15:].sum()
        rows.append(dict(check="hazard at all n 15-100 (carry-forward)", runs=int(surv[15]), value=a_all,
                         se=np.sqrt(a_all * (1 - a_all) / surv[15:].sum()), target=0.002))
        rows.append(dict(check="P(alarm by n = 100)", runs=int(surv[15]), value=alarms[15:].sum() / surv[15],
                         se=np.nan, target=1 - 0.998 ** 86))
        for kind, label in (("hd", "in-control ARL, H&D Table 1 (warm-up 14)"),
                            ("cpm", "in-control ARL, cpm 2.3 MWARL500 (startup 20, our setting)")):
            rl = np.concatenate(list(ex.map(arl_job, [(kind, 2026 + 7 * i + (kind == "cpm"), 200) for i in range(20)])))
            rows.append(dict(check=label, runs=len(rl), value=rl.mean(), se=rl.std() / np.sqrt(len(rl)), target=500,
                             censored=int((rl == LMAX - (15 if kind == "hd" else 20) + 1).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "calib" / "cpm_hd_check.csv", index=False)
    print(out.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
