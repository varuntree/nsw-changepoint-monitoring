"""Test bed at a matched background.

The test bed of testbed.py compares monitors at their own background rates, so a quieter monitor (capped) and a
noisier one (uncapped) are not compared on equal terms. Here every monitor gets the threshold that gives the SAME
background: 20% of the unmodified windows have an alarm within the 60 days after tau.

For each start day tau of testbed.py (same eligible days) and each setting (none, permanent drops of 10, 20 and
40%, 14-day drops of 20 and 40%), the series is modified and re-adjusted exactly as in testbed.py, the monitor's
first segment is run from tau - 90 (fresh start: running scale, phi_hat and centre from past data, as in
monitor.run) and the detector statistic path is stored; we keep its maximum over [tau - 90, tau) and over
[tau, tau + 60). A window counts as an alarm at threshold h if the post-tau maximum exceeds h (one segment, no
relearn; the control windows define the background in the same way). Per monitor, h20 is the 80th percentile of
the post-tau maximum over the control windows, so the background is 20% for every monitor; the gain is
P(max > h20 | inserted) - 0.20 (paired), with a 3-month block bootstrap interval, and the capped-minus-uncapped
difference in gain for each adjustment (CUSUM-r vs CUSUM, EWMA-r vs EWMA, FOCuS-r vs FOCuS).
Outputs: results/testbed/matched_pairs.csv, matched.csv, matched_contrasts.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import adjust as A  # noqa: E402
from cpd import calibrate, monitor  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd.data import opal_daily  # noqa: E402
from testbed import POST, PRE, adjust_one, eligible, insert  # noqa: E402
from testbed_contrasts import blocks_of  # noqa: E402

OUT = HERE / "results" / "testbed"
ADJS = ("A", "B", "HW", "YM")
DETS = ("CUSUM", "EWMA", "FOCuS", "CUSUM-r", "EWMA-r", "FOCuS-r", "CPM")   # CPM added last
SETTINGS = [("none", 0.0), ("drop", 10), ("drop", 20), ("drop", 40), ("temp", 20), ("temp", 40)]
BG = 0.20


def segment_paths(r, t0, t1, dets):
    """Statistic paths of monitor.run's first segment on surprises r over [t0, t1) (no alarm yet)."""
    r = np.ascontiguousarray(np.asarray(r, float)[:t1])
    blind = np.zeros(len(r), dtype=np.bool_)
    dl = monitor.delevel(r, False)
    ph = monitor.phi_hat(dl, blind, t0)
    s = monitor.running_scale(dl, blind, t0, t1)
    z = r[t0:t1] / s
    return {d: D.stat_path(d, z, ph) for d in dets}, ph


def surprises(adj, x, dates, tau):
    if adj == "YM":
        hw, _ = adjust_one("HW", x, dates, tau)
        return hw - A.yearly_memory(hw, np.isnan(hw))
    return adjust_one(adj, x, dates, tau)[0]


def job(args):
    tau, x, dates = args
    t0, t1 = tau - PRE, min(len(x), tau + POST)
    rows = []
    for kind, size in SETTINGS:
        xx = x if kind == "none" else insert(x, tau, kind, size, None)
        for adj in ADJS:
            paths, _ = segment_paths(surprises(adj, xx, dates, tau), t0, t1, DETS)
            for det, p in paths.items():
                rows.append(dict(tau=int(tau), date=str(dates[tau].date()), setting=f"{kind}{size:g}", adj=adj,
                                 det=det, max_pre=float(np.nanmax(p[:PRE])), max_post=float(np.nanmax(p[PRE:]))))
    return rows


def check_against_monitor(x, dates, taus, n=6):
    """The stored path is the monitor's: its first crossing of the AR(1) threshold is monitor.run's first alarm."""
    tabs = calibrate.load(HERE / "results" / "calib" / "h_table.json", "surprise")
    for tau in taus[:: max(1, len(taus) // n)]:
        t0, t1 = tau - PRE, tau + POST
        for adj in ADJS:
            r = surprises(adj, x, dates, tau)
            paths, ph = segment_paths(r, t0, t1, DETS)
            rr = adjust_one("HW", x, dates, tau)[0] if adj == "YM" else r
            for det in DETS:
                res = monitor.run(rr, "surprise", det, "ar1", tabs, t0, t1, method=adj)
                cross = np.flatnonzero(paths[det] > tabs[det](ph))
                assert (res.alarms[0] if res.alarms else -1) == (t0 + cross[0] if len(cross) else -1), (tau, adj, det)


def analyse(df, block_months=3, suffix=""):
    rng = np.random.default_rng(5)
    ctl = df[df.setting == "none0"].set_index(["adj", "det", "tau"])["max_post"]
    rows, crows = [], []
    gains = {}
    for (adj, det), g in ctl.groupby(level=[0, 1]):
        h = float(np.quantile(g.values, 1 - BG))
        c_hit = (g.values > h).astype(float)
        for st in [f"{k}{s:g}" for k, s in SETTINGS[1:]]:
            ins = df[(df.setting == st) & (df.adj == adj) & (df.det == det)].set_index("tau")["max_post"]
            ins = ins.reindex(g.index.get_level_values(2))
            hit = (ins.values > h).astype(float)
            gains[(st, adj, det)] = pd.Series(hit - c_hit, index=ins.index)
            rows.append(dict(setting=st, adj=adj, det=det, h20=h, background=c_hit.mean(), detect=hit.mean(),
                             gain=(hit - c_hit).mean(), n=len(hit)))
    dates = df.drop_duplicates("tau").set_index("tau")["date"]
    for (st, adj, det), g in gains.items():
        if not det.endswith("-r"):
            continue
        d = (g - gains[(st, adj, det[:-2])]).dropna()
        blk = blocks_of(dates.reindex(d.index), block_months)
        bl = [d.values[blk == b] for b in np.unique(blk)]
        bs = [np.mean(np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])) for _ in range(2000)]
        crows.append(dict(setting=st, adj=adj, capped=det, uncapped=det[:-2], diff=d.mean(),
                          lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n=len(d), block_months=block_months,
                          n_blocks=len(bl)))
    m = pd.DataFrame(rows)
    # detectors added later go last so the earlier rows keep their random draws
    m["_o"] = (m.det == "CPM").astype(int)
    m = m.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    # interval for each gain as well
    los, his = [], []
    for _, r in m.iterrows():
        g = gains[(r.setting, r.adj, r.det)]
        blk = blocks_of(dates.reindex(g.index), block_months)
        bl = [g.values[blk == b] for b in np.unique(blk)]
        bs = [np.mean(np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])) for _ in range(1000)]
        los.append(np.quantile(bs, .025))
        his.append(np.quantile(bs, .975))
    m["ci_lo"], m["ci_hi"] = los, his
    m = m.sort_values(["setting", "adj", "det"], kind="stable").reset_index(drop=True)
    m.to_csv(OUT / f"matched{suffix}.csv", index=False)
    c = pd.DataFrame(crows)
    c.to_csv(OUT / f"matched_contrasts{suffix}.csv", index=False)
    print(m.pivot_table(index=["setting", "adj"], columns="det", values="gain").round(2).to_string())
    print(c.round(2).to_string(index=False))


def main():
    x = opal_daily("train")
    dates = x.index
    taus, _, _ = eligible(x)
    xv = x.to_numpy().copy()
    check_against_monitor(xv, dates, taus)
    print("first-segment paths match monitor.run", flush=True)
    rows = []
    with ProcessPoolExecutor(10) as ex:
        for rr in ex.map(job, [(int(t), xv, dates) for t in taus], chunksize=4):
            rows += rr
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "matched_pairs.csv", index=False)
    analyse(df)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyse":       # analyse [block_months] -> matched{_6m}.csv
        bm = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        analyse(pd.read_csv(OUT / "matched_pairs.csv"), bm, "" if bm == 3 else f"_{bm}m")
    else:
        main()
