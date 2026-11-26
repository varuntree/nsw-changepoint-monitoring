"""Real-data ranking at a matched alarm budget.

Monitors raise very different numbers of alarms on Opal, and episodes caught grow with the alarm count. Here every
monitor (adjustment x detector; at a fixed constant threshold the threshold rule plays no part) gets the constant
threshold h that makes it raise the same number of alarms B over 1 Mar 2020 - 28 Sep 2026, with the usual
procedure (running scale, centre, phi_hat for FOCuS-AR, 28-day relearn). h is found by bisection on log h; the
achieved count is the one closest to B (exact in most cells). Budgets:
  median     the median alarm count of the 150 monitors of scores.csv
  quartile   their lower quartile
  calm       instead of a count, each past-only monitor's threshold set on a calm stretch of the real series
             (opal_bootstrap.py, h_boot), so the alarm counts differ
Episodes caught are scored under the direction-aware rule (headline), the +-7-day rule with nearest-boundary
crediting, and the rule as frozen on 22 Oct (+-7 days, any direction, an alarm may catch several episodes), each
with the circular-shift chance median and p (seed 0, 1,000 shifts). Because every monitor of a budget raises the
same number of alarms, the chance baseline is the same for all of them up to timing, and the counts are directly
comparable. Best-of-family correction per budget (common offsets, seed 7, 2,000 shifts) over all its monitors.
Outputs: results/opal/budget.csv, budget_familywise.csv, budget_alarms.csv"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import monitor, score  # noqa: E402
from opal_extra import _shift_test  # noqa: E402

OP = HERE / "results" / "opal"
METHODS = ["raw", "A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG"]
DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
KIND = {m: ("level" if m in ("raw", "STL", "MSTL", "REG") else "surprise") for m in METHODS}
RULESETS = [("dir", (7, 0, True)), ("pm7_nodir", (7, 7, False)), ("frozen", (7, 7, False, "all"))]


def run_h(r, m, det, h, t0):
    return monitor.run(r, KIND[m], det, "ar1", {det: (lambda phi, h=h: h)}, t0, method=m)


def find_h(r, m, det, B, t0, iters=40):
    lo, hi = np.log(1e-3), np.log(1e5)
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n = len(run_h(r, m, det, float(np.exp(mid)), t0).alarms)
        if best is None or abs(n - B) < abs(best[1] - B) or (abs(n - B) == abs(best[1] - B) and n <= B):
            best = (float(np.exp(mid)), n)
        if n == B:
            break
        if n > B:
            lo = mid
        else:
            hi = mid
    return best[0]


def job(args):
    m, det, budget, B, h, r, dates, t0 = args
    if h is None:
        h = find_h(r, m, det, B, t0)
    res = run_h(r, m, det, h, t0)
    mon = dates[t0:]
    rel = np.array(res.alarms, int) - t0
    dv = np.array(res.dirs)
    sc = score.scored()
    out = _shift_test((rel, dv, mon, [(*rs, sc) if len(rs) == 3 else (*rs[:3], sc, rs[3]) for _, rs in RULESETS], 1000))
    s = score.score(res.alarms, dates, t0, n_shift=0, dirs=res.dirs)
    row = dict(budget=budget, target=B, method=m, detector=det, h=h, n_alarms=len(rel),
               non_event_per_year=s["non_event_per_year"], unexplained_per_year=s["unexplained_per_year"],
               blind_days_per_year=res.blind.sum() / ((len(r) - t0) / 365.25))
    for (name, _), (obs, med, p, eps) in zip(RULESETS, out):
        row.update({name: obs, f"{name}_null": med, f"{name}_p": p, f"{name}_episodes": ";".join(eps)})
    al = pd.DataFrame(dict(budget=budget, method=m, detector=det, alarm=dates[res.alarms] if len(rel) else [],
                           direction=res.dirs if len(rel) else []))
    return row, al


def fw_job(args):
    cells, mon, us = args
    L = len(mon)
    sc = score.scored()
    out = []
    for u in us:
        c = {name: max(score.episodes_caught(mon[(rel + u) % L], dv if rs[2] else None, after=rs[0], before=rs[1], sc=sc,
                                             credit=rs[3] if len(rs) > 3 else "nearest")[0] for rel, dv in cells)
             for name, rs in RULESETS}
        out.append(c)
    return out


def main():
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    n150 = pd.read_csv(OP / "scores.csv").n_alarms
    budgets = {"median": int(np.median(n150)), "quartile": int(np.quantile(n150, 0.25))}
    jobs = [(m, d, b, B, None, adj[m].to_numpy(), dates, t0) for b, B in budgets.items() for m in METHODS for d in DETS]
    boot = pd.read_csv(OP / "bootstrap.csv")
    jobs += [(r.method, r.detector, "calm", np.nan, float(r.h_boot), adj[r.method].to_numpy(), dates, t0) for _, r in boot.iterrows()]
    with ProcessPoolExecutor(10) as ex:
        out = list(ex.map(job, jobs, chunksize=2))
    rows = pd.DataFrame([o[0] for o in out])
    rows.to_csv(OP / "budget.csv", index=False)
    alarms = pd.concat([o[1] for o in out], ignore_index=True)
    alarms.to_csv(OP / "budget_alarms.csv", index=False)
    # best-of-family correction per budget
    mon = dates[t0:]
    L = len(mon)
    rng = np.random.default_rng(7)
    us = [int(rng.integers(30, L - 30)) for _ in range(2000)]
    fws = []
    for b in list(budgets) + ["calm"]:
        cells = []
        for (m, d), g in alarms[alarms.budget == b].groupby(["method", "detector"]):
            idx = np.array([dates.get_loc(a) for a in g.alarm]) - t0
            if len(idx):
                cells.append((idx, g.direction.to_numpy()))
        with ProcessPoolExecutor(10) as ex:
            got = [x for part in ex.map(fw_job, [(cells, mon, us[i::10]) for i in range(10)]) for x in part]
        sub = rows[rows.budget == b]
        for name, _ in RULESETS:
            null = np.array([g[name] for g in got])
            obs = int(sub[name].max())
            fws.append(dict(budget=b, rule=name, monitors=len(cells), obs_max=obs, p=float((null >= obs).mean()),
                            null_median=float(np.median(null)), null_95=float(np.quantile(null, 0.95)),
                            **{f"p_ge{k}": float((null >= k).mean()) for k in range(4, 13)}))
    pd.DataFrame(fws).to_csv(OP / "budget_familywise.csv", index=False)
    pd.set_option("display.width", 250)
    for b in list(budgets) + ["calm"]:
        print(b, budgets.get(b, "boot"))
        print(rows[rows.budget == b].sort_values(["dir", "frozen"], ascending=False).head(10)[
            ["method", "detector", "n_alarms", "dir", "dir_p", "pm7_nodir", "frozen", "frozen_p", "non_event_per_year"]].round(3).to_string(index=False))
    print(pd.DataFrame(fws).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
