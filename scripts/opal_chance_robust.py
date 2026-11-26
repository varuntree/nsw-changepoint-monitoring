"""Robustness of the Opal chance baseline.

The circular shift of score.score moves a monitor's whole alarm sequence by one random offset around the
2020-2026 series, so alarms from the COVID years can land in the calm years and back. Two stricter baselines:
  era    the monitored days are split at 2023-01-01; each era's alarms are shifted circularly WITHIN the era by
         its own offset drawn from 30 .. (era length - 30);
  local  one offset u = +-U{30, ..., 180} days for all of a monitor's alarms, reflected at the ends of the series
         (a shifted day d < 0 becomes -d, and d > L-1 becomes 2(L-1) - d).
Rules: direction-aware (0..+7 days after a boundary, direction checked; the scores.csv rule), "orig" (+-7 days, no
direction, each alarm credited to its nearest boundary) and "frozen" (the rule as frozen on 22 Oct: +-7 days, no
direction, an alarm may catch several episodes). Scoring with score.episodes_caught on the frozen list.

Outputs (results/opal/):
  chance_local.csv             every monitor (150), both rules: observed episodes, and for the circular (seed 0,
                               1,000 shifts, = rule_variants.csv, asserted), era and local baselines (seed 11,
                               2,000 shifts each) the chance median and one-sided p
  chance_local_familywise.csv  best-of-136 (cells with >= 10 alarms) under circular shifts (= familywise.csv /
                               rule_variants_familywise.csv) and era shifts (one pair of era offsets per draw,
                               shared by all cells, seed 7, 2,000 draws): P(best >= k), k = 4..12, both rules
  familywise_bothrules.csv     the "holds under both rules" statistic: max over the 136 cells of min(direction-
                               aware count, original count); observed value and P(max-min >= observed) under the
                               circular and era shifts (seed 7, 2,000 draws)
  bothrules_survivors.csv      every family monitor's min(direction-aware, frozen) count and its adjusted p
  no_dq_2025_01.csv, familywise_no_dq.csv, bothrules_survivors_no_dq.csv   the same without the 15 Jan 2025
                               boundary (strike_2025_01), as no_covid_onset.csv (`python ... r4` runs only these)
  no_covid_onset.csv           without the March-2020 onset episode (14 scored episodes): every monitor's count
                               and circular-shift p (seed 0, 2,000 shifts) under both rules, and the best-of-136
                               rows (seed 7, 2,000) for both rules
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
from cpd import score  # noqa: E402
from opal_extra import _cells, _setup  # noqa: E402

OP = HERE / "results" / "opal"
ERA_SPLIT = "2023-01-01"
RULES = {"dir": dict(after=7, before=0, use_dirs=True, credit="nearest"),
         "orig": dict(after=7, before=7, use_dirs=False, credit="nearest"),     # +-7, nearest-boundary crediting
         "frozen": dict(after=7, before=7, use_dirs=False, credit="all")}       # the rule as frozen on 22 Oct
WORKERS = 10


def _era_bounds(mon):
    e1 = int(np.searchsorted(mon, pd.Timestamp(ERA_SPLIT)))
    return [(0, e1), (e1, len(mon))]


def draws(kind, n, seed, L, eras):
    """Shift parameters: circular -> offset u; era -> (u1, u2); local -> signed offset u."""
    rng = np.random.default_rng(seed)
    if kind == "circular":
        return [int(rng.integers(30, L - 30)) for _ in range(n)]
    if kind == "era":
        return [tuple(int(rng.integers(30, (b - a) - 30)) for a, b in eras) for _ in range(n)]
    if kind == "local":
        return [int(rng.integers(30, 181)) * int(rng.choice([-1, 1])) for _ in range(n)]
    raise ValueError(kind)


def apply(kind, rel, p, L, eras):
    if kind == "circular":
        return (rel + p) % L
    if kind == "era":
        out = rel.copy()
        for (a, b), u in zip(eras, p):
            m = (rel >= a) & (rel < b)
            out[m] = a + (rel[m] - a + u) % (b - a)
        return out
    sh = rel + p
    sh = np.where(sh < 0, -sh, sh)
    return np.where(sh >= L, 2 * (L - 1) - sh, sh)


def count(mon, idx, dv, rule, sc):
    r = RULES[rule]
    return score.episodes_caught(mon[idx], dv if r["use_dirs"] else None, after=r["after"], before=r["before"], sc=sc,
                                 credit=r["credit"])[0]


def monitor_job(args):
    rel, dv, mon, eras, specs, sc = args
    L = len(mon)
    out = {}
    for kind, n, seed in specs:
        ps = draws(kind, n, seed, L, eras)
        for rule in RULES:
            obs = count(mon, rel, dv, rule, sc)
            null = np.array([count(mon, apply(kind, rel, p, L, eras), dv, rule, sc) for p in ps]) if len(rel) else np.zeros(1)
            out[(kind, rule)] = (obs, float(np.median(null)), float((null >= obs).mean()) if len(rel) else np.nan)
    return out


def fw_job(args):
    cells, mon, eras, kind, ps, sc = args
    L = len(mon)
    res = []
    for p in ps:
        c = {rule: [count(mon, apply(kind, rel, p, L, eras), dv, rule, sc) for rel, dv in cells] for rule in RULES}
        res.append((max(c["dir"]), max(c["orig"]), max(min(a, b) for a, b in zip(c["dir"], c["orig"])),
                    max(c["frozen"]), max(min(a, b) for a, b in zip(c["dir"], c["frozen"]))))
    return res


def familywise(cells, mon, eras, kind, sc, n=2000, seed=7):
    ps = draws(kind, n, seed, len(mon), eras)
    parts = [ps[i::WORKERS] for i in range(WORKERS)]
    with ProcessPoolExecutor(WORKERS) as ex:
        got = list(ex.map(fw_job, [(cells, mon, eras, kind, p, sc) for p in parts]))
    arr = np.empty((n, 5))
    for i in range(WORKERS):
        arr[i::WORKERS] = got[i]
    return arr          # columns: best dir, best orig (+-7 nearest), best min(dir, orig), best frozen, best min(dir, frozen)


def main():
    al, dates, t0 = _setup()
    mon = dates[t0:]
    eras = _era_bounds(mon)
    sc = score.scored()
    cells = _cells(al, dates, t0)

    # 1. per-monitor baselines
    specs = [("circular", 1000, 0), ("era", 2000, 11), ("local", 2000, 11)]
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(monitor_job, [(rel, dv, mon, eras, specs, sc) for _, rel, dv in cells], chunksize=2))
    rows = []
    for (k, rel, _), r in zip(cells, res):
        row = dict(method=k[0], detector=k[1], rule=k[2], n_alarms=len(rel))
        for (kind, rule), (obs, med, p) in r.items():
            row[f"{rule}_obs"] = obs
            row[f"{rule}_{kind}_null"] = med
            row[f"{rule}_{kind}_p"] = p
        rows.append(row)
    loc = pd.DataFrame(rows)
    rv = pd.read_csv(OP / "rule_variants.csv")
    chk = loc.merge(rv, on=["method", "detector", "rule"])
    assert (chk.dir_obs == chk.r0_7_dir).all() and (chk.orig_obs == chk.pm7_nodir).all()
    has = chk.n_alarms_x > 0
    assert np.allclose(chk.loc[has, "dir_circular_p"], chk.loc[has, "r0_7_dir_p"])
    assert np.allclose(chk.loc[has, "orig_circular_p"], chk.loc[has, "pm7_nodir_p"])
    assert (chk.frozen_obs == chk.frozen).all() and np.allclose(chk.loc[has, "frozen_circular_p"], chk.loc[has, "frozen_p"])
    loc.to_csv(OP / "chance_local.csv", index=False)
    print("per-monitor baselines written; circular rows reproduce rule_variants.csv", flush=True)

    # 2-3. best-of-136 and the both-rules statistic, circular and era shifts
    big = [(rel, dv) for _, rel, dv in cells if len(rel) >= 10]
    obs_dir = max(count(mon, rel, dv, "dir", sc) for rel, dv in big)
    obs_orig = max(count(mon, rel, dv, "orig", sc) for rel, dv in big)
    obs_mm = max(min(count(mon, rel, dv, "dir", sc), count(mon, rel, dv, "orig", sc)) for rel, dv in big)
    obs_fz = max(count(mon, rel, dv, "frozen", sc) for rel, dv in big)
    obs_mmf = max(min(count(mon, rel, dv, "dir", sc), count(mon, rel, dv, "frozen", sc)) for rel, dv in big)
    fwrows, bothrows = [], []
    arrs = {}
    for kind in ("circular", "era"):
        arr = familywise(big, mon, eras, kind, sc)
        arrs[kind] = arr
        for j, (rule, obs) in ((0, ("dir", obs_dir)), (1, ("orig", obs_orig)), (3, ("frozen", obs_fz))):
            fwrows.append(dict(shift=kind, rule=rule, cells=len(big), obs_max=obs, p=float((arr[:, j] >= obs).mean()),
                               null_median=float(np.median(arr[:, j])), null_95=float(np.quantile(arr[:, j], 0.95)),
                               **{f"p_ge{k}": float((arr[:, j] >= k).mean()) for k in range(4, 13)}))
        for j, pair, obs in ((2, "dir+orig", obs_mm), (4, "dir+frozen", obs_mmf)):
            bothrows.append(dict(shift=kind, rules=pair, cells=len(big), obs_max_min=obs, p=float((arr[:, j] >= obs).mean()),
                                 null_median=float(np.median(arr[:, j])), null_95=float(np.quantile(arr[:, j], 0.95)),
                                 **{f"p_ge{k}": float((arr[:, j] >= k).mean()) for k in range(4, 11)}))
    fw = pd.DataFrame(fwrows)
    old = pd.read_csv(OP / "rule_variants_familywise.csv").set_index("rule_variant")
    for rule, name in (("dir", "r0_7_dir"), ("orig", "pm7_nodir"), ("frozen", "frozen")):
        r = fw[(fw["shift"] == "circular") & (fw.rule == rule)].iloc[0]
        for k in range(6, 12):
            assert abs(r[f"p_ge{k}"] - old.loc[name, f"p_best_episodes_ge{k}"]) < 1e-12, (rule, k)
    fw.to_csv(OP / "chance_local_familywise.csv", index=False)
    pd.DataFrame(bothrows).to_csv(OP / "familywise_bothrules.csv", index=False)
    print(fw.round(4).to_string(index=False))
    print(pd.DataFrame(bothrows).round(4).to_string(index=False), flush=True)

    # survivors of the both-rules statistic
    survivors(cells, mon, eras, sc, arrs)

    # 4. without the March-2020 onset
    without_episode(cells, mon, eras, sc, "covid_onset", "no_covid_onset.csv")


def survivors(cells, mon, eras, sc, arrs, out="bothrules_survivors.csv"):
    """Every family monitor (>= 10 alarms) with min(direction-aware, frozen) counts and its adjusted p: the chance that
    the best min-count over the family, under the common shifts, reaches it (circular and era shifts)."""
    rows = []
    for k, rel, dv in cells:
        if len(rel) < 10:
            continue
        d, f = count(mon, rel, dv, "dir", sc), count(mon, rel, dv, "frozen", sc)
        m = min(d, f)
        row = dict(method=k[0], detector=k[1], rule=k[2], n_alarms=len(rel), dir=d, frozen=f, min_count=m)
        for kind, arr in arrs.items():
            row[f"p_adj_{kind}"] = float((arr[:, 4] >= m).mean())
            row[f"crit95_{kind}"] = float(np.quantile(arr[:, 4], 0.95))
        row["survives"] = row["p_adj_circular"] < 0.05
        rows.append(row)
    o = pd.DataFrame(rows).sort_values(["min_count", "p_adj_circular"], ascending=[False, True])
    o.to_csv(OP / out, index=False)
    print(o[o.survives].to_string(index=False), flush=True)
    return o


def without_episode(cells, mon, eras, sc, episode, out_name, fw_name=None, surv_name=None):
    """Drop one episode from the scored set and redo every monitor's counts and own p (circular, 2,000 shifts, both
    rules), the best-of-family rows and the both-rules statistic (and its survivors)."""
    big = [(rel, dv) for _, rel, dv in cells if len(rel) >= 10]
    sc2 = sc[sc.episode != episode]
    n_ep = sc2.episode.nunique()
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(monitor_job, [(rel, dv, mon, eras, [("circular", 2000, 0)], sc2) for _, rel, dv in cells],
                          chunksize=2))
    rows = []
    for (k, rel, _), r in zip(cells, res):
        row = dict(row="monitor", method=k[0], detector=k[1], rule=k[2], n_alarms=len(rel), episodes_scored=n_ep)
        for (kind, rule), (obs, med, p) in r.items():
            row[f"{rule}_obs"] = obs
            row[f"{rule}_null"] = med
            row[f"{rule}_p"] = p
        rows.append(row)
    arr = familywise(big, mon, eras, "circular", sc2)
    o_d = max(count(mon, rel, dv, "dir", sc2) for rel, dv in big)
    o_o = max(count(mon, rel, dv, "orig", sc2) for rel, dv in big)
    o_f = max(count(mon, rel, dv, "frozen", sc2) for rel, dv in big)
    o_mm = max(min(count(mon, rel, dv, "dir", sc2), count(mon, rel, dv, "frozen", sc2)) for rel, dv in big)
    fws = []
    for j, (rule, obs) in ((0, ("dir", o_d)), (1, ("orig", o_o)), (3, ("frozen", o_f)), (4, ("dir+frozen min", o_mm))):
        r = dict(row="best_of_family", rule=rule, cells=len(big), episodes_scored=n_ep, obs_max=obs,
                 p=float((arr[:, j] >= obs).mean()), null_95=float(np.quantile(arr[:, j], 0.95)),
                 **{f"p_ge{k}": float((arr[:, j] >= k).mean()) for k in range(4, 13)})
        rows.append(r)
        fws.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OP / out_name, index=False)
    if fw_name:
        pd.DataFrame(fws).to_csv(OP / fw_name, index=False)
    if surv_name:
        survivors(cells, mon, eras, sc2, {"circular": arr}, surv_name)
    print(out[out.row != "monitor"].dropna(axis=1).round(4).to_string(index=False), flush=True)
    return out


def r4():
    """Survivors of the both-rules statistic, and everything without the 15 Jan 2025 boundary
    (strike_2025_01, a one-boundary episode)."""
    al, dates, t0 = _setup()
    mon = dates[t0:]
    eras = _era_bounds(mon)
    sc = score.scored()
    cells = _cells(al, dates, t0)
    big = [(rel, dv) for _, rel, dv in cells if len(rel) >= 10]
    arrs = {kind: familywise(big, mon, eras, kind, sc) for kind in ("circular", "era")}
    survivors(cells, mon, eras, sc, arrs)
    without_episode(cells, mon, eras, sc, "strike_2025_01", "no_dq_2025_01.csv", "familywise_no_dq.csv",
                    "bothrules_survivors_no_dq.csv")

def r5():
    """(1) The both-rules statistic (best over the 136 family cells of min(direction-aware, frozen))
    against the best direction-aware count, shift by shift, for the same circular and era shifts (seed 7, 2,000 each)
    -> bothrules_equivalence.csv. (2) Holt-Winters + EWMA under each of the five rule versions of tab:rule-variants:
    count, chance median, p (own) and p (adjusted, best of 137) -> hw_ewma_rule_versions.csv."""
    al, dates, t0 = _setup()
    mon = dates[t0:]
    eras = _era_bounds(mon)
    sc = score.scored()
    cells = _cells(al, dates, t0)
    big = [(rel, dv) for _, rel, dv in cells if len(rel) >= 10]
    obs_dir = max(count(mon, rel, dv, "dir", sc) for rel, dv in big)
    obs_mmf = max(min(count(mon, rel, dv, "dir", sc), count(mon, rel, dv, "frozen", sc)) for rel, dv in big)
    rows = []
    for kind in ("circular", "era"):
        arr = familywise(big, mon, eras, kind, sc)
        d, m = arr[:, 0], arr[:, 4]
        rows.append(dict(shift=kind, shifts=len(arr), cells=len(big), obs_best_dir=obs_dir, obs_best_min_dir_frozen=obs_mmf,
                         share_shifts_equal=float((d == m).mean()), max_abs_diff=float(np.abs(d - m).max()),
                         **{f"p_dir_ge{k}": float((d >= k).mean()) for k in range(6, 10)},
                         **{f"p_min_ge{k}": float((m >= k).mean()) for k in range(6, 10)}))
    eq = pd.DataFrame(rows)
    eq.to_csv(OP / "bothrules_equivalence.csv", index=False)
    print(eq.round(4).to_string(index=False))
    rv = pd.read_csv(OP / "rule_variants.csv")
    fw = pd.read_csv(OP / "rule_variants_familywise.csv").set_index("rule_variant")
    r = rv[(rv.method == "HW") & (rv.detector == "EWMA") & (rv.rule == "ar1")].iloc[0]
    out = []
    for v, label in (("frozen", "+-7 d, any boundary (frozen rule)"), ("pm7_nodir", "+-7 d, nearest boundary"),
                     ("pm7_dir", "+-7 d, nearest boundary, direction"), ("r0_7_nodir", "0-7 d, nearest earlier boundary"),
                     ("r0_7_dir", "0-7 d, nearest earlier boundary, direction (direction-aware rule)")):
        k = int(r[v])
        out.append(dict(rule_version=v, label=label, episodes=k, chance_median=float(r[f"{v}_null"]), p_own=float(r[f"{v}_p"]),
                        p_adjusted=float(fw.loc[v, f"p_best_episodes_ge{k}"]), best_of_137=int(fw.loc[v, "obs_max_episodes"])))
    hw = pd.DataFrame(out)
    hw.to_csv(OP / "hw_ewma_rule_versions.csv", index=False)
    print(hw.to_string(index=False))


if __name__ == "__main__":
    {"r4": r4, "r5": r5}.get((sys.argv[1:] or [""])[0], main)()
