"""Thresholds calibrated on real in-control Opal surprises (block bootstrap) instead of simulated AR(1) noise.
For each past-only adjustment and detector: resample the surprises of a calm training period (1 Jan 2023 to
31 Jul 2024, holidays included, no scored change) in circular blocks of 28 days, run the same procedure
(150 warm-up days, running scale, 365 monitored days) and take the 95th percentile of the yearly maximum
statistic as a constant threshold. Then monitor the whole series with that threshold and score as usual.
Writes results/opal/bootstrap.csv, with the simulated thresholds for comparison: h_indep = h(0) (independent rule),
h_sim = median over segments of the AR(1)-rule threshold h(phi_hat) actually used on this series, and the ratios
h_boot / h_indep and h_boot / h_sim. Also writes results/opal/bootstrap_heldout.csv: the same bootstrap-threshold
alarms scored on the held-out period only (1 Jan 2025 to 28 Sep 2026), which the calm calibration window
(Jan 2023 to Jul 2024) does not overlap."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import calibrate, detectors as D, monitor, score  # noqa: E402
from cpd.monitor import running_scale  # noqa: E402

OUT = HERE / "results" / "opal"
WARM, YEAR, BLOCK, REPS = 150, 365, 28, 2000
# the first three adjustments and six detectors are the original set; the rest were added on 24 Oct 2026 for the
# real-data ranking at a matched calibration (seeds depend on list positions, so the original rows are unchanged)
ADJS = ["A", "B", "HW", "YM", "MSTL_p", "REG_p"]
DETS = ["CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r", "FOCuS-r", "EWMA-r"]


def boot_h(train, det, seed):
    rng = np.random.default_rng(seed)
    n, L = len(train), WARM + YEAR
    blind = np.zeros(L, dtype=np.bool_)
    mx = np.empty(REPS)
    for i in range(REPS):
        starts = rng.integers(0, n, L // BLOCK + 2)
        y = np.concatenate([np.take(train, np.arange(s, s + BLOCK), mode="wrap") for s in starts])[:L]
        s = running_scale(y, blind, WARM, L)
        mx[i] = D.stat_path(det, y[WARM:] / s, 0.0).max()
    return float(np.quantile(mx, 0.95))


def job(args):
    adj, det, r, dates, t0 = args
    m = (dates >= "2023-01-01") & (dates <= "2024-07-31")
    rr = r
    if adj == "YM":       # the yearly memory is applied inside the monitor; resample its surprises
        from cpd.adjust import yearly_memory
        rr = r - yearly_memory(r, np.isnan(r))
    train = rr[m]
    train = train[~np.isnan(train)]
    h = boot_h(train, det, seed=hash((adj, det)) % 2**31 if False else 1000 + ADJS.index(adj) * 10 + DETS.index(det))
    res = monitor.run(r, "surprise", det, "ar1", {det: (lambda phi, h=h: h)}, t0, method=adj)
    sc = score.score(res.alarms, dates, t0, n_shift=1000, dirs=res.dirs)
    sc.update(method=adj, detector=det, h_boot=h, blind_days_per_year=res.blind.sum() / ((len(r) - t0) / 365.25))
    tabs = calibrate.load(HERE / "results" / "calib" / "h_table.json", "surprise")
    sim = monitor.run(r, "surprise", det, "ar1", tabs, t0, method=adj)
    sc.update(h_indep=tabs[det](0.0), h_sim=float(np.median(sim.hs)), phi_hat_median=float(np.median(sim.phis)))
    sc.update(ratio_indep=h / sc["h_indep"], ratio_sim=h / sc["h_sim"])
    t_ho = int(np.searchsorted(dates, pd.Timestamp("2025-01-01")))
    ho = score.score(res.alarms, dates, t_ho, n_shift=1000, dirs=res.dirs)
    ho.pop("first_alarm_rel", None)
    ho.update(method=adj, detector=det, h_boot=h)
    return sc, ho


if __name__ == "__main__":
    df = pd.read_csv(OUT / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = df.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    jobs = [(a, d, df[a].to_numpy(), dates, t0) for a in ADJS for d in DETS]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(job, jobs))
    out = pd.DataFrame([r[0] for r in res])
    out.to_csv(OUT / "bootstrap.csv", index=False)
    pd.DataFrame([r[1] for r in res]).to_csv(OUT / "bootstrap_heldout.csv", index=False)
    print(out[["method", "detector", "h_boot", "n_alarms", "near_real", "share_real", "null_share_real", "episodes_caught",
               "non_event_per_year", "unexplained_per_year", "blind_days_per_year"]].round(2).to_string(index=False))
