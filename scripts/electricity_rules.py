"""Electricity under every scoring rule, and the temperature contrast with an interval.

1. Every monitor of electricity.py (same adjusted series, thresholds and monitor) is rerun to recover alarm directions
   (sign of the surprises from the estimated start to the alarm, as monitor.run), and scored against the frozen
   electricity list under the frozen rule (+-7 days, any direction, an alarm may credit several boundaries) and the
   direction-aware rule (0..+7 days, direction checked, nearest boundary):
     results/electricity/rule_variants.csv   adj, det, rule, alarms, frozen, direction_aware
     results/electricity/rule_credits.csv    every alarm within +-7 days of a scored boundary: date, direction, boundary
2. The 10% test-bed gain of the forecast with temperature (REGT_p) + FOCuS minus the yearly memory + FOCuS, with a
   block bootstrap over 3-month blocks of start days (2,000 resamples, seed 3; testbed_contrasts method):
     results/extra/elec_contrast_ci.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import calibrate, monitor, score  # noqa: E402
from electricity import ADJS, DETS, RULES, SRC, T_START, TABLE  # noqa: E402
from testbed_contrasts import blocks_of  # noqa: E402

OUT = HERE / "results" / "electricity"


def scored():
    return pd.read_csv(SRC / "events_electricity_scored_FROZEN.csv", parse_dates=["date"])


def job(args):
    a, det, rule, r, dates, t0 = args
    res = monitor.run(r, "surprise", det, rule, calibrate.load(TABLE, "surprise"), t0, method=a)
    ad = dates[res.alarms] if res.alarms else pd.DatetimeIndex([])
    dv = np.array(res.dirs)
    sc = scored()
    fz = score.episodes_caught(ad, None, after=7, before=7, sc=sc, credit="all")[0]
    da = score.episodes_caught(ad, dv, sc=sc)[0]
    credits = []
    for x, d in zip(ad, dv):
        for _, b in sc.iterrows():
            if abs((x - b.date).days) <= 7:
                credits.append(dict(adj=a, det=det, rule=rule, alarm=x.date(), direction=int(d), episode=b.episode,
                                    boundary=b.boundary, boundary_date=b.date.date(), days_from_boundary=(x - b.date).days))
    return dict(adj=a, det=det, rule=rule, alarms=len(ad), frozen=fz, direction_aware=da), credits


def contrast():
    tb = pd.read_csv(OUT / "testbed.csv")
    dates = pd.DatetimeIndex(pd.read_csv(OUT / "adjusted.csv", parse_dates=[0]).iloc[:, 0])
    t = tb[tb["size"] == 10.0].copy()
    t["hit"] = t.first_post.between(0, 59).astype(float)

    def gain(adj, det):
        g = t[(t.adj == adj) & (t.det == det)]
        ins = g[g.which == "drop"].set_index("tau").hit
        ctl = g[g.which == "none"].set_index("tau").hit
        return ins - ctl.reindex(ins.index)

    rows = []
    rng = np.random.default_rng(3)
    for a1, d1, a2, d2 in [("REGT_p", "FOCuS", "YM", "FOCuS"), ("REGT_p", "CUSUM", "YM", "CUSUM")]:
        g1, g2 = gain(a1, d1), gain(a2, d2)
        d = (g1 - g2).dropna()
        for bm in (3, 6):
            blk = blocks_of(dates[d.index.to_numpy()], bm)
            bl = [d.values[blk == b] for b in np.unique(blk)]
            bs = [np.mean(np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])) for _ in range(2000)]
            rows.append(dict(setting="drop10", a=f"{a1} + {d1}", b=f"{a2} + {d2}", gain_a=g1.mean(), gain_b=g2.mean(),
                             diff=d.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n=len(d),
                             block_months=bm, n_blocks=len(bl)))
    out = pd.DataFrame(rows)
    (HERE / "results" / "extra").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / "extra" / "elec_contrast_ci.csv", index=False)
    print(out.round(3).to_string(index=False))


def main():
    adj = pd.read_csv(OUT / "adjusted.csv", parse_dates=[0]).set_index(pd.read_csv(OUT / "adjusted.csv").columns[0])
    adj.index = pd.DatetimeIndex(adj.index)
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp(T_START)))
    jobs = [(a, d, ru, adj[a].to_numpy(), dates, t0) for a in ADJS for d in DETS for ru in RULES
            if not (d == "FOCuS-AR" and ru == "indep")]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(job, jobs))
    rv = pd.DataFrame([r[0] for r in res])
    cr = pd.DataFrame([c for r in res for c in r[1]])
    old = pd.read_csv(OUT / "scores.csv")
    chk = rv.merge(old[["adj", "det", "rule", "n_alarms", "episodes_caught"]], on=["adj", "det", "rule"])
    assert (chk.alarms == chk.n_alarms).all() and (chk.direction_aware == chk.episodes_caught).all()
    rv.to_csv(OUT / "rule_variants.csv", index=False)
    cr.to_csv(OUT / "rule_credits.csv", index=False)
    print(rv[(rv.frozen > 0) | (rv.direction_aware > 0)].to_string(index=False))
    print(cr[cr.rule == "ar1"].to_string(index=False))
    contrast()


if __name__ == "__main__":
    main()
