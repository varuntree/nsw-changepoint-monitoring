"""Holt-Winters smoothing weight.

The simulation's Holt-Winters forecast uses alpha = 0.05 (level) and gamma = 0.10 (weekday effects). Here the same
series (regenerated from the sim_run seeds: SeedSequence([crc32(cid), batch]), all 20 batches = 10,000 runs per
condition) are adjusted with alpha in {0.01, 0.02, 0.05, 0.10} (gamma 0.10, cap 3, 28-day warm-up), then standardised
exactly as sim.run_series (running scale, phi_hat), for phi = 0.4, Gaussian noise, holidays off and on, no change and
drops of 1, 2 and 4 SD; and a "frozen" variant: alpha 0.05 in training, then the level frozen
from the monitoring start (weekday effects still update), i.e. the level is reset only when a segment starts. Detectors CUSUM, EWMA, FOCuS, capped CUSUM. Pass 1: no-change runs give the AR(1) and
independent false-alarm rates and the matched thresholds (95th percentile of the yearly maximum, as
sim_run.oracle_thresholds). Pass 2: detection within 60 days, F(7) and restricted mean delay at the matched threshold
(definitions of sim_analyse). The alpha = 0.05 rows must equal fa.csv / detect.csv (HW) to 4 dp (asserted).
Output: results/sim/summary/hw_alpha.csv"""
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
from cpd import adjust as A  # noqa: E402
from cpd import calibrate  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd import sim as S  # noqa: E402
from cpd.monitor import delevel, phi_hat, running_scale  # noqa: E402
from sim_run import BATCH, OUT, TABLE, WBS_M, cid  # noqa: E402

ALPHAS = (0.01, 0.02, 0.05, 0.10, "frozen")   # "frozen": alpha 0.05 in training, level frozen from the monitoring start
DETS = ("CUSUM", "EWMA", "FOCuS", "CUSUM-r")
SIZES = (-1.0, -2.0, -4.0)


from numba import njit  # noqa: E402


@njit(cache=True)
def _hw_frozen(lx, dow, warm, alpha, gamma, cap, t_freeze):
    """adjust._hw with the level frozen from t_freeze on (weekday effects and scale keep updating): the forecast level
    is reset only when a segment starts, as a monitor that relearns the level after each alarm would do."""
    n = lx.shape[0]
    r = np.full(n, np.nan)
    s_sum = np.zeros(7)
    s_cnt = np.zeros(7)
    tot = 0.0
    cnt = 0
    for t in range(warm):
        if not np.isnan(lx[t]):
            s_sum[dow[t]] += lx[t]
            s_cnt[dow[t]] += 1
            tot += lx[t]
            cnt += 1
    level = tot / max(cnt, 1)
    season = np.zeros(7)
    for k in range(7):
        season[k] = s_sum[k] / s_cnt[k] - level if s_cnt[k] > 0 else 0.0
    res = np.empty(warm)
    m = 0
    for t in range(warm):
        if not np.isnan(lx[t]):
            res[m] = lx[t] - level - season[dow[t]]
            m += 1
    scale = 1.4826 * np.median(np.abs(res[:m] - np.median(res[:m]))) if m > 3 else 0.05
    if scale <= 0:
        scale = 0.05
    for t in range(warm, n):
        k = dow[t]
        if np.isnan(lx[t]):
            continue
        e = lx[t] - (level + season[k])
        r[t] = e
        ec = min(max(e, -cap * scale), cap * scale)
        scale = 0.9 * scale + 0.1 * 1.2533 * abs(ec)
        a = alpha if t < t_freeze else 0.0
        level += a * ec
        season[k] += gamma * (1.0 - a) * ec
        mu = season.mean()
        for j in range(7):
            season[j] -= mu
    return r


def hw(x, alpha):
    lx = A._log(x)
    dow = pd.DatetimeIndex(S.DATES).dayofweek.to_numpy().astype(np.int64)
    if alpha == "frozen":
        return _hw_frozen(np.ascontiguousarray(lx), dow, 28, A.HW_ALPHA, A.HW_GAMMA, A.HW_CAP, S.TRAIN)
    return A._hw(np.ascontiguousarray(lx), dow, 28, alpha, A.HW_GAMMA, A.HW_CAP)


def job(args):
    c, b, gj, h_or = args
    gen = S.Generator(json.loads(gj))
    tabs = calibrate.load(TABLE, "surprise")
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(cid(c).encode()), b]))
    rng.integers(0, S.MON, size=(WBS_M, 2))
    blind = np.zeros(S.N, dtype=np.bool_)
    mx = np.zeros((BATCH, len(ALPHAS), len(DETS)), np.float32)
    first = np.full((BATCH, len(ALPHAS), len(DETS), 3), -1)
    taus = np.zeros(BATCH, int)
    for i in range(BATCH):
        tau = int(rng.integers(60, 301))
        taus[i] = tau
        lx = gen.draw(rng, c["phi"], c["noise"], c["calendar"], c["change"], c["size"], tau)
        x = np.exp(lx)
        for a, al in enumerate(ALPHAS):
            r = hw(x, al)
            dl = delevel(np.ascontiguousarray(r), False)
            ph = phi_hat(dl, blind, S.TRAIN)
            z = r[S.TRAIN:] / running_scale(dl, blind, S.TRAIN, S.N)
            for k, det in enumerate(DETS):
                path = D.stat_path(det, z, ph)
                mx[i, a, k] = np.nanmax(path)
                ho = h_or[(c["calendar"], a, k)] if h_or else np.inf
                first[i, a, k] = S.first_cross_multi(path, (tabs[det](0.0), tabs[det](ph), ho))
    return mx, first, taus


def main():
    gj = (OUT / "generator.json").read_text()
    rows = []
    with ProcessPoolExecutor(10) as ex:
        h_or = {}
        for cal in (False, True):
            c = dict(phi=0.4, noise="normal", calendar=cal, change="none", size=0.0)
            res = list(ex.map(job, [(c, b, gj, None) for b in range(20)]))
            mx = np.concatenate([r[0] for r in res])
            first = np.concatenate([r[1] for r in res])
            for a, al in enumerate(ALPHAS):
                for k, det in enumerate(DETS):
                    h_or[(cal, a, k)] = float(np.quantile(mx[:, a, k], 0.95))
                    for j, rule in ((0, "indep"), (1, "ar1")):
                        rows.append(dict(calendar=int(cal), alpha=al, det=det, change="none", size=0.0, rule=rule,
                                         false_alarm=float((first[:, a, k, j] >= 0).mean())))
        for cal in (False, True):
            for size in SIZES:
                c = dict(phi=0.4, noise="normal", calendar=cal, change="mean", size=size)
                res = list(ex.map(job, [(c, b, gj, h_or) for b in range(20)]))
                first = np.concatenate([r[1] for r in res])
                tau = np.concatenate([r[2] for r in res])
                for a, al in enumerate(ALPHAS):
                    for k, det in enumerate(DETS):
                        T = first[:, a, k, 2].astype(int)
                        T = np.where(T < 0, 10_000, T)
                        post = T >= tau
                        dl = (T - tau + 1)[post]
                        rows.append(dict(calendar=int(cal), alpha=al, det=det, change="mean", size=size, rule="oracle",
                                         pre_fa=float((~post).mean()), detect60=float((dl <= 60).mean()),
                                         F7=float((dl <= 7).mean()), rmd=float(np.minimum(dl, 61).mean())))
    out = pd.DataFrame(rows)
    # the base alpha reproduces the main simulation
    fa = pd.read_csv(OUT / "summary" / "fa.csv")
    det = pd.read_csv(OUT / "summary" / "detect.csv")
    out.to_csv(OUT / "summary" / "hw_alpha.csv", index=False)
    base = (fa.phi == 0.4) & (fa.noise == "normal") & (fa.adj == "HW")
    based = (det.phi == 0.4) & (det.noise == "normal") & (det.adj == "HW") & (det.rule == "oracle") & (det.change == "mean")
    for _, r in out[out.alpha.astype(str) == "0.05"].iterrows():
        if r["change"] == "none":
            v = fa[base & (fa.calendar == r["calendar"]) & (fa.det == r["det"]) & (fa.rule == r["rule"])].fa.iloc[0]
            assert abs(v - r["false_alarm"]) < 5e-5, (r.to_dict(), v)
        else:
            v = det[based & (det.calendar == r["calendar"]) & (det["size"] == r["size"]) & (det.det == r["det"])].detect60.iloc[0]
            assert abs(v - r["detect60"]) < 5e-5, (r.to_dict(), v)
    pd.set_option("display.width", 250)
    print(out[out.change == "none"].pivot_table(index=["calendar", "rule", "det"], columns="alpha", values="false_alarm").round(3).to_string())
    print(out[out.change == "mean"].pivot_table(index=["calendar", "size", "det"], columns="alpha", values="detect60").round(3).to_string())


if __name__ == "__main__":
    main()
