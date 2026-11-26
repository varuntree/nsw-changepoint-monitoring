"""Extra Opal analyses for Chapter 5:
  yearly.csv       alarms per calendar year for every cell (false-alarm behaviour in quiet years 2023-2026)
  heldout.csv      scores on the held-out period 2025-01-01 .. 2026-09-28 (pre-registered for the yearly memory)
  covid.csv        first alarm around COVID onset: days from 16 and 23 March 2020, and the chance of a
                   background alarm in the 14 days before (1 - exp(-14 r / 365), r = method's alarms a year
                   outside scored windows)
  holds.csv        alarm stability: does the level 28-88 days after an alarm differ from the 28 days before
                   the estimated change start, judged on the retrospective STL series with a long-run scale
                   (|D| > 1.96)? With a circular-shift chance value.
  sensitivity.csv  primary cells re-run with CUSUM k (0.25, 1), EWMA lambda (0.05, 0.2), false-alarm target
                   (1%, 10%) and relearn window (14 days)
Further analyses (sensitivities only; the frozen event list and the headline rule are unchanged):
  rule_variants.csv          every monitor's episodes caught under the headline rule (0..+7 days, direction
                             checked), the rule frozen on 22 Oct (`frozen`: +-7 days, any direction, an alarm may
                             catch several episodes), the frozen rule + nearest-boundary crediting (`pm7_nodir`), +
                             direction (`pm7_dir`), the 0..+7 rule without direction, and 3- and 14-day windows;
                             each with its circular-shift chance median and p-value
  rule_variants_familywise.csv  best-of-136 correction (as familywise.csv) under each rule variant
  familywise_no_omicron.csv  best-of-136 correction and per-monitor tests with the Omicron episode removed
  blind_misses.csv           missed episodes whose 0..+7-day windows fell wholly or partly in a relearn period
  announced_split.csv        episodes caught among the announced and unannounced episodes, with chance p-values
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
from cpd import calibrate, monitor, score  # noqa: E402
from cpd import detectors as D  # noqa: E402
from cpd import kernels as K  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

OP = HERE / "results" / "opal"
TABLE = HERE / "results" / "calib" / "h_table.json"
KINDS = {m: ("level" if m in ("STL", "MSTL", "REG", "raw") else "surprise")
         for m in ["raw", "A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG"]}
PRIMARY = [("A", "CUSUM"), ("A", "FOCuS"), ("A", "EWMA"), ("A", "CPM"), ("B", "CUSUM"), ("HW", "CUSUM"),
           ("HW", "FOCuS"), ("YM", "CUSUM"), ("YM", "FOCuS"), ("STL", "CUSUM"), ("HW", "EWMA")]


def yearly():
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm"])
    al["year"] = al["alarm"].dt.year
    t = al.groupby(["method", "detector", "rule", "year"]).size().unstack(fill_value=0)
    t.to_csv(OP / "yearly.csv")
    return t


def heldout():
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm"])
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2025-01-01")))
    rows = []
    for (m, d, r), g in al.groupby(["method", "detector", "rule"]):
        idx = [dates.get_loc(a) for a in g["alarm"]]
        sc = score.score(idx, dates, t0, n_shift=500, dirs=list(g["direction"]))
        sc.update(method=m, detector=d, rule=r)
        sc.pop("first_alarm_rel", None)
        rows.append(sc)
    pd.DataFrame(rows).to_csv(OP / "heldout.csv", index=False)


def covid():
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm", "start_est"])
    s = pd.read_csv(OP / "scores.csv")
    rows = []
    for (m, d, r), g in al.groupby(["method", "detector", "rule"]):
        w = g[(g["alarm"] >= "2020-03-09") & (g["alarm"] <= "2020-05-22")]
        first = w["alarm"].min() if len(w) else pd.NaT
        row = s[(s["method"] == m) & (s["detector"] == d) & (s["rule"] == r)].iloc[0]
        rate = row["non_event_per_year"]
        st = w.loc[w["alarm"] == first, "start_est"].iloc[0] if len(w) else pd.NaT
        rows.append(dict(method=m, detector=d, rule=r, first_alarm=first,
                         days_from_16mar=(first - pd.Timestamp("2020-03-16")).days if pd.notna(first) else np.nan,
                         days_from_23mar=(first - pd.Timestamp("2020-03-23")).days if pd.notna(first) else np.nan,
                         chance_alarm_14d_before=1 - np.exp(-14 * rate / 365),
                         start_est=st, days_alarm_after_start_est=(first - st).days if pd.notna(first) else np.nan,
                         start_est_from_16mar=(st - pd.Timestamp("2020-03-16")).days if pd.notna(first) else np.nan))
    pd.DataFrame(rows).to_csv(OP / "covid.csv", index=False)


def holds():
    from cpd.adjust import adj_STL_retro
    x = opal_daily("train")
    lx = adj_STL_retro(x.to_numpy(), x.index)
    s = pd.Series(lx, index=x.index)
    res = s - s.rolling(29, center=True, min_periods=7).median()
    sd = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res)))
    r = res.dropna()
    phi = float(np.clip(r.autocorr(1), 0, 0.9))
    sd_lr = sd * np.sqrt((1 + phi) / (1 - phi))
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm", "start_est"])
    n = len(s)

    def stat(a_i, st_i, h=60):
        before = s.iloc[max(0, st_i - 28):st_i].dropna()
        after = s.iloc[a_i + 28:a_i + 28 + h].dropna()
        if len(before) < 14 or len(after) < 30:
            return np.nan
        return abs(after.mean() - before.mean()) / (sd_lr * np.sqrt(1 / len(after) + 1 / len(before)))

    rows = []
    rng = np.random.default_rng(4)
    for (m, d, r_), g in al.groupby(["method", "detector", "rule"]):
        ai = np.array([s.index.get_loc(a) for a in g["alarm"]])
        si = np.array([s.index.get_loc(b) for b in g["start_est"]])
        st = np.array([stat(a, b) for a, b in zip(ai, si)])
        ok = ~np.isnan(st)
        share = float((st[ok] > 1.96).mean()) if ok.any() else np.nan
        null = []
        lo = int(np.searchsorted(s.index, pd.Timestamp("2020-03-01")))
        L = n - lo
        for _ in range(300):
            u = rng.integers(30, L - 30)
            a2 = lo + (ai - lo + u) % L
            s2 = np.clip(a2 - (ai - si), 0, n - 1)
            v = np.array([stat(a, b) for a, b in zip(a2, s2)])
            v = v[~np.isnan(v)]
            null.append((v > 1.96).mean() if len(v) else np.nan)
        rows.append(dict(method=m, detector=d, rule=r_, n=int(ok.sum()), share_hold=share,
                         null_share_hold=float(np.nanmedian(null)), phi_stl=phi))
    pd.DataFrame(rows).to_csv(OP / "holds.csv", index=False)


def _quick_table(det, param, fa=0.05, reps=6000, grid=np.round(np.arange(0, 0.91, 0.1), 1)):
    from cpd.calibrate import WARM, YEAR
    from cpd.monitor import running_scale
    rng = np.random.default_rng(11)
    blind = np.zeros(WARM + YEAR, dtype=np.bool_)
    hs = []
    for phi in grid:
        m = np.empty(reps)
        for i in range(reps):
            y = K.ar1(WARM + YEAR, phi, rng.standard_normal(WARM + YEAR))
            sc = running_scale(y, blind, WARM, WARM + YEAR)
            z = y[WARM:] / sc
            m[i] = (K.cusum_path(z, param) if det == "CUSUM" else K.ewma_path(z, param)).max()
        hs.append(np.quantile(m, 1 - fa))
    hs = np.array(hs)
    return lambda p: float(np.interp(min(max(p, 0), 0.9), grid, hs))


def _sens_job(args):
    m, d, variant, r, dates, t0 = args
    kind = KINDS[m]
    tabs = calibrate.load(TABLE, kind)
    relearn = monitor.RELEARN
    old = dict(D.PARAMS["CUSUM"]), dict(D.PARAMS["EWMA"])
    try:
        if variant.startswith("k="):
            D.PARAMS["CUSUM"]["k"] = float(variant[2:])
            tabs = dict(tabs)
            tabs["CUSUM"] = _quick_table("CUSUM", float(variant[2:]))
        elif variant.startswith("lam="):
            D.PARAMS["EWMA"]["lam"] = float(variant[4:])
            tabs = dict(tabs)
            tabs["EWMA"] = _quick_table("EWMA", float(variant[4:]))
        elif variant.startswith("fa="):
            tabs = calibrate.load(TABLE, f"{kind}@{variant[3:]}")
        elif variant == "relearn=14":
            relearn = 14
        res = monitor.run(r, kind, d, "ar1", tabs, t0, method=m, relearn=relearn)
    finally:
        D.PARAMS["CUSUM"].update(old[0])
        D.PARAMS["EWMA"].update(old[1])
    sc = score.score(res.alarms, dates, t0, n_shift=500, dirs=res.dirs)
    sc.pop("first_alarm_rel", None)
    sc.update(method=m, detector=d, variant=variant)
    return sc


def sensitivity():
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    jobs = []
    for m, d in PRIMARY:
        vs = ["fa=0.01", "fa=0.1", "relearn=14"]
        if d == "CUSUM":
            vs += ["k=0.25", "k=1.0"]
        if d == "EWMA":
            vs += ["lam=0.05", "lam=0.2"]
        jobs += [(m, d, v, adj[m].to_numpy(), dates, t0) for v in vs]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(_sens_job, jobs))
    pd.DataFrame(rows).to_csv(OP / "sensitivity.csv", index=False)


def familywise(n_shift=2000, min_alarms=10):
    """Correction for picking the best of many cells: shift every cell's alarms by the same random offset,
    take the maximum share near a scored change (and maximum episodes caught) over all cells with at least
    `min_alarms` alarms, and compare with the observed maxima."""
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm"])
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    cls = score.day_classes(dates)
    real = cls["real"].to_numpy()[t0:]
    mon = dates[t0:]
    L = len(mon)
    cells = []
    for k, g in al.groupby(["method", "detector", "rule"]):
        idx = np.array([dates.get_loc(a) for a in g["alarm"]]) - t0
        dv = g["direction"].to_numpy()[idx >= 0]
        idx = idx[idx >= 0]
        if len(idx) >= min_alarms:
            cells.append(((k, dv), idx))
    obs_share = max(real[i].mean() for _, i in cells)
    obs_ep = max(score.episodes_caught(mon[i], kd[1])[0] for kd, i in cells)
    rng = np.random.default_rng(7)
    ms, me = [], []
    for _ in range(n_shift):
        u = rng.integers(30, L - 30)
        sh = [((i + u) % L, kd[1]) for kd, i in cells]
        ms.append(max(real[s].mean() for s, _ in sh))
        me.append(max(score.episodes_caught(mon[s], dv)[0] for s, dv in sh))
    out = dict(cells=len(cells), obs_max_share=obs_share, p_share=float((np.array(ms) >= obs_share).mean()),
               null_max_share_median=float(np.median(ms)), null_max_share_95=float(np.quantile(ms, 0.95)),
               obs_max_episodes=obs_ep, p_episodes=float((np.array(me) >= obs_ep).mean()),
               null_max_episodes_median=float(np.median(me)), null_max_episodes_95=float(np.quantile(me, 0.95)),
               **{f"p_best_episodes_ge{k}": float((np.array(me) >= k).mean()) for k in (4, 5, 6, 7, 8)})
    pd.Series(out).to_csv(OP / "familywise.csv", header=["value"])
    print(out)




def unexplained_null(n_shift=1000):
    """Chance baseline for 'unexplained alarms a year': shift each cell's alarms by random offsets and count
    how many land on days that no class explains."""
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm"])
    adj = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    cls = score.day_classes(dates).iloc[t0:]
    unexpl = ~cls[["real", "school", "holiday", "after_event", "explained"]].any(axis=1).to_numpy()
    L = len(unexpl)
    years = L / 365.25
    rng = np.random.default_rng(11)
    rows = []
    for (m, d, r), g in al.groupby(["method", "detector", "rule"]):
        idx = np.array([dates.get_loc(a) for a in g["alarm"]]) - t0
        idx = idx[idx >= 0]
        if not len(idx):
            continue
        obs = unexpl[idx].sum() / years
        null = [unexpl[(idx + rng.integers(30, L - 30)) % L].sum() / years for _ in range(n_shift)]
        rows.append(dict(method=m, detector=d, rule=r, unexplained_per_year=obs, null_unexplained_per_year=float(np.median(null)),
                         p_lower=float((np.array(null) <= obs).mean()), share_days_unexplained=float(unexpl.mean())))
    pd.DataFrame(rows).to_csv(OP / "unexplained_null.csv", index=False)


# ------------------------------------------------------------------------------------------ further analyses
T_START = "2020-03-01"
# rule variants: (name, after, before, direction checked, crediting). The rule history (git, events_scored_RULES.md):
#   frozen       22 Oct 2026 (commit 962c5b2): +-7 days of any boundary, any direction, an alarm may catch several
#                episodes (credit "all")
#   pm7_nodir    + one alarm credited to its nearest boundary only (added 23 Oct 02:45, commit 4a17fd6, together
#                with the 0..+7 window, i.e. r0_7_nodir)
#   pm7_dir      + direction check (23 Oct 03:29, commit 4caf1ae); r0_7_dir = the direction-aware headline rule
RULE_VARIANTS = [("r0_7_dir", 7, 0, True, "nearest"), ("r0_7_nodir", 7, 0, False, "nearest"),
                 ("pm7_dir", 7, 7, True, "nearest"), ("pm7_nodir", 7, 7, False, "nearest"),
                 ("r0_3_dir", 3, 0, True, "nearest"), ("r0_14_dir", 14, 0, True, "nearest"),
                 ("frozen", 7, 7, False, "all"), ("frozen_dir", 7, 7, True, "all"), ("r0_7_dir_all", 7, 0, True, "all")]


def _setup():
    al = pd.read_csv(OP / "alarms.csv", parse_dates=["alarm", "start_est"])
    dates = pd.read_csv(OP / "adjusted.csv", parse_dates=["date"])["date"]
    dates = pd.DatetimeIndex(dates)
    t0 = int(np.searchsorted(dates, pd.Timestamp(T_START)))
    return al, dates, t0


def _cells(al, dates, t0):
    out = []
    for k, g in al.groupby(["method", "detector", "rule"]):
        idx = np.array([dates.get_loc(a) for a in g["alarm"]]) - t0
        keep = idx >= 0
        out.append((k, idx[keep], g["direction"].to_numpy()[keep]))
    return out


def _shift_test(args):
    """Episodes caught and the circular-shift chance distribution (same shifts as score.score: seed 0,
    offsets 30..L-30) for one monitor under several (after, before, use_dirs, scored subset[, credit]) rules."""
    rel, dv, mon, rules, n_shift = args
    L = len(mon)
    rng = np.random.default_rng(0)
    us = [rng.integers(30, L - 30) for _ in range(n_shift)]
    res = []
    for rule in rules:
        after, before, use_dirs, sc = rule[:4]
        cr = rule[4] if len(rule) > 4 else "nearest"
        d = dv if use_dirs else None
        obs, first = score.episodes_caught(mon[rel], d, after=after, before=before, sc=sc, credit=cr)
        null = np.array([score.episodes_caught(mon[(rel + u) % L], d, after=after, before=before, sc=sc, credit=cr)[0]
                         for u in us]) if len(rel) else np.zeros(1)
        res.append((obs, float(np.median(null)), float((null >= obs).mean()), sorted(first)))
    return res


def rule_variants(n_shift=1000):
    al, dates, t0 = _setup()
    mon = dates[t0:]
    sc = score.scored()
    cells = _cells(al, dates, t0)
    rules = [(a, b, u, sc, c) for _, a, b, u, c in RULE_VARIANTS]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(_shift_test, [(rel, dv, mon, rules, n_shift) for _, rel, dv in cells], chunksize=4))
    rows = []
    for (k, rel, _), rr in zip(cells, res):
        row = dict(method=k[0], detector=k[1], rule=k[2], n_alarms=len(rel))
        for (name, *_), (obs, nm, p, eps) in zip(RULE_VARIANTS, rr):
            row[name] = obs
            row[f"{name}_null"] = nm
            row[f"{name}_p"] = p
        row["episodes_headline"] = ";".join(rr[0][3])
        row["episodes_pm7_nodir"] = ";".join(rr[3][3])
        row["episodes_frozen"] = ";".join(rr[6][3])
        rows.append(row)
    out = pd.DataFrame(rows)
    s = pd.read_csv(OP / "scores.csv")
    chk = out.merge(s, on=["method", "detector", "rule"])
    has = chk["p_episodes"].notna()
    assert (chk["r0_7_dir"] == chk["episodes_caught"]).all()
    assert (chk.loc[has, "r0_7_dir_p"] == chk.loc[has, "p_episodes"]).all()
    out.to_csv(OP / "rule_variants.csv", index=False)
    print(out.sort_values("frozen", ascending=False).head(8).to_string(index=False))


def _fw_job(args):
    cells, mon, after, before, use_dirs, sc, us, credit = args
    L = len(mon)
    return [max(score.episodes_caught(mon[(rel + u) % L], dv if use_dirs else None, after=after, before=before,
                                      sc=sc, credit=credit)[0] for rel, dv in cells) for u in us]


def _familywise_eps(after=7, before=0, use_dirs=True, sc=None, n_shift=2000, min_alarms=10, credit="nearest"):
    """familywise() for the episode count only, under any rule or scored subset: same 136 cells, same 2,000
    offsets (seed 7)."""
    al, dates, t0 = _setup()
    mon = dates[t0:]
    sc = score.scored() if sc is None else sc
    cells = [(rel, dv) for _, rel, dv in _cells(al, dates, t0) if len(rel) >= min_alarms]
    obs = max(score.episodes_caught(mon[rel], dv if use_dirs else None, after=after, before=before, sc=sc,
                                    credit=credit)[0] for rel, dv in cells)
    rng = np.random.default_rng(7)
    L = len(mon)
    us = [int(rng.integers(30, L - 30)) for _ in range(n_shift)]
    parts = [us[i::10] for i in range(10)]
    with ProcessPoolExecutor(10) as ex:
        got = list(ex.map(_fw_job, [(cells, mon, after, before, use_dirs, sc, p, credit) for p in parts]))
    me = np.empty(n_shift)
    for i in range(10):
        me[i::10] = got[i]
    out = dict(cells=len(cells), obs_max_episodes=obs, p_episodes=float((me >= obs).mean()),
               null_max_episodes_median=float(np.median(me)), null_max_episodes_95=float(np.quantile(me, 0.95)))
    out.update({f"p_best_episodes_ge{k}": float((me >= k).mean()) for k in range(4, 13)})
    return out


def rule_variants_familywise():
    rows = []
    for name, a, b, u, c in RULE_VARIANTS:
        rows.append(dict(rule_variant=name, after=a, before=b, direction=u, credit=c, **_familywise_eps(a, b, u, credit=c)))
    out = pd.DataFrame(rows)
    # the headline rule must reproduce familywise.csv (same cells and offsets)
    fw = pd.read_csv(OP / "familywise.csv", index_col=0)["value"]
    h = out.iloc[0]
    for k in ("obs_max_episodes", "p_episodes", "p_best_episodes_ge6", "p_best_episodes_ge7"):
        assert abs(h[k] - fw[k]) < 1e-12, (k, h[k], fw[k])
    out.to_csv(OP / "rule_variants_familywise.csv", index=False)
    print(out.round(4).to_string(index=False))


def familywise_no_omicron():
    """Sensitivity for the Christmas/Omicron confound: drop the Omicron onset episode from the scored set
    (14 episodes) and redo the best-of-136 correction and the per-monitor chance tests."""
    sc = score.scored()
    sc14 = sc[sc["episode"] != "omicron_onset"]
    fw = _familywise_eps(sc=sc14)
    al, dates, t0 = _setup()
    mon = dates[t0:]
    rows = [dict(row="best_of_136", episodes_scored=sc14["episode"].nunique(), **fw)]
    cells = {k: (rel, dv) for k, rel, dv in _cells(al, dates, t0)}
    keys = [k for k in cells if k[2] == "ar1"]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(_shift_test, [(*cells[k], mon, [(7, 0, True, sc14), (7, 0, True, sc)], 2000) for k in keys]))
    for k, ((o14, n14, p14, e14), (o15, n15, p15, _)) in zip(keys, res):
        rows.append(dict(row="monitor", method=k[0], detector=k[1], rule=k[2], n_alarms=len(cells[k][0]),
                         episodes_scored=14, caught=o14, null_median=n14, p=p14, caught_15=o15, null_median_15=n15,
                         p_15=p15, caught_omicron=o15 > o14))
    out = pd.DataFrame(rows)
    out.to_csv(OP / "familywise_no_omicron.csv", index=False)
    print(out.iloc[:1].T.to_string())
    print(out[out["row"] == "monitor"].sort_values("caught", ascending=False).head(10).to_string(index=False))


def blind_misses():
    """For every monitor: the episodes it missed whose 0..+7-day windows (all boundaries) lie wholly inside a
    relearn period (the 28 days after an alarm, as in monitor.run), or partly (some window day blind)."""
    al, dates, t0 = _setup()
    sc = score.scored()
    rows = []
    for (m, d, r), g in al.groupby(["method", "detector", "rule"]):
        g = g[g["alarm"] >= T_START]
        caught, first = score.episodes_caught(pd.DatetimeIndex(g["alarm"]), g["direction"].to_numpy())
        blind = set()
        for a in g["alarm"]:
            blind.update(pd.date_range(a + pd.Timedelta(days=1), a + pd.Timedelta(days=28)))
        wholly, partly = [], []
        for ep, e in sc.groupby("episode"):
            if ep in first:
                continue
            days = [b + pd.Timedelta(days=k) for b in e["date"] for k in range(8)]
            nb = sum(x in blind for x in days)
            if nb == len(days):
                wholly.append(ep)
            elif nb > 0:
                partly.append(ep)
        rows.append(dict(method=m, detector=d, rule=r, n_alarms=len(g), caught=caught, missed=15 - caught,
                         missed_wholly_blind=len(wholly), missed_partly_blind=len(partly),
                         wholly=";".join(wholly), partly=";".join(partly)))
    out = pd.DataFrame(rows)
    out.to_csv(OP / "blind_misses.csv", index=False)
    print(out[out["rule"] == "ar1"].sort_values("caught", ascending=False).head(8).to_string(index=False))


def _split_job(args):
    """Caught episodes under the headline rule (frozen list, one boundary per alarm) split into announced and
    unannounced, observed and under the circular shifts of score.score (seed 0)."""
    rel, dv, mon, un, n_shift = args
    L = len(mon)
    rng = np.random.default_rng(0)
    us = [rng.integers(30, L - 30) for _ in range(n_shift)]

    def split(idx):
        caught = score.episodes_caught(mon[idx], dv)[1]
        return sum(e not in un for e in caught), sum(e in un for e in caught), sorted(caught)

    obs = split(rel)
    null = np.array([split((rel + u) % L)[:2] for u in us]) if len(rel) else np.zeros((1, 2))
    return obs, null


def announced_split(n_shift=2000):
    """Episodes caught (headline rule, scored against the whole frozen list as in scores.csv) split into the 9
    announced (announced_before yes/partly) and 6 unannounced (no) episodes, each count with its own
    circular-shift chance median and p-value (the same shifts split the same way)."""
    al, dates, t0 = _setup()
    mon = dates[t0:]
    sc = score.scored()
    unq = sc.groupby("episode")["announced_before"].apply(lambda v: (v == "no").all())
    un = set(unq[unq].index)
    cells = _cells(al, dates, t0)
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(_split_job, [(rel, dv, mon, un, n_shift) for _, rel, dv in cells], chunksize=4))
    rows = []
    for (k, rel, _), ((oa, ou, eps), null) in zip(cells, res):
        rows.append(dict(method=k[0], detector=k[1], rule=k[2], n_alarms=len(rel), caught=oa + ou,
                         announced_caught=oa, announced_of=len(unq) - len(un), announced_null=float(np.median(null[:, 0])),
                         announced_p=float((null[:, 0] >= oa).mean()),
                         unannounced_caught=ou, unannounced_of=len(un), unannounced_null=float(np.median(null[:, 1])),
                         unannounced_p=float((null[:, 1] >= ou).mean()),
                         announced_episodes=";".join(e for e in eps if e not in un),
                         unannounced_episodes=";".join(e for e in eps if e in un)))
    out = pd.DataFrame(rows)
    chk = out.merge(pd.read_csv(OP / "scores.csv"), on=["method", "detector", "rule"])
    assert (chk["caught"] == chk["episodes_caught"]).all()
    out.to_csv(OP / "announced_split.csv", index=False)
    print(out[out["rule"] == "ar1"].sort_values("caught", ascending=False).head(8).to_string(index=False))

def ewma_direction():
    """EWMA has no change-start estimator, so monitor.run takes an EWMA alarm's direction from the sign of z on the
    alarm day. Check that this always agrees with the sign of the EWMA statistic itself, for every
    past-only forecast x {EWMA, capped EWMA} x both rules -> results/extra/ewma_direction.csv."""
    import opal_case as oc
    orig = D.change_start
    rec = []

    def patched(det, z, a, ph=0.0):
        if det in ("EWMA", "EWMA-r"):
            zz = np.asarray(z, float)[:a + 1]
            if det == "EWMA-r":
                zz = np.clip(zz, -D.CLIP, D.CLIP)
            lam, e = D.PARAMS["EWMA"]["lam"], 0.0
            for v in zz:
                if not np.isnan(v):
                    e = lam * v + (1 - lam) * e
            rec.append((np.sign(e), np.sign(zz[a])))
        return orig(det, z, a, ph)

    df, kinds = oc.adjusted()
    dates = df.index
    t0 = int(np.searchsorted(dates, pd.Timestamp(oc.T_START)))
    rows = []
    D.change_start = patched
    try:
        for m in ["A", "B", "HW", "YM", "MSTL_p", "REG_p"]:
            for det in ["EWMA", "EWMA-r"]:
                for ru in ["ar1", "indep"]:
                    rec.clear()
                    res = monitor.run(df[m].to_numpy(), kinds[m], det, ru, calibrate.load(TABLE, kinds[m]), t0, method=m)
                    r = np.array(rec).reshape(-1, 2)
                    rows.append(dict(method=m, detector=det, rule=ru, alarms=len(res.alarms),
                                     disagreements=int((r[:, 0] != r[:, 1]).sum()) if len(r) else 0))
    finally:
        D.change_start = orig
    out = pd.DataFrame(rows)
    (HERE / "results" / "extra").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / "extra" / "ewma_direction.csv", index=False)
    print(out.to_string(index=False))


def phi_fallback():
    """How often phi_hat = 0 (fewer than 56 days or 28 consecutive pairs, or a negative estimate) in the
    main monitors (A, B, HW, YM x CUSUM, EWMA, FOCuS, capped CUSUM, capped FOCuS; AR(1) rule), so the threshold is
    h(0) -> results/extra/phi_fallback.csv: segments, segments with phi_hat = 0, share of monitored days at h(0),
    first and last such segment start."""
    import opal_case as oc
    df, kinds = oc.adjusted()
    dates = df.index
    t0 = int(np.searchsorted(dates, pd.Timestamp(oc.T_START)))
    rows = []
    for m in ["A", "B", "HW", "YM"]:
        for det in ["CUSUM", "EWMA", "FOCuS", "CUSUM-r", "FOCuS-r"]:
            res = monitor.run(df[m].to_numpy(), kinds[m], det, "ar1", calibrate.load(TABLE, kinds[m]), t0, method=m)
            ends = res.seg_starts[1:] + [len(df)]
            days = np.array([e - s_ for s_, e in zip(res.seg_starts, ends)])
            z = np.array(res.phis) == 0
            starts0 = [str(dates[s_].date()) for s_, zz in zip(res.seg_starts, z) if zz]
            rows.append(dict(method=m, detector=det, segments=len(res.phis), segments_phi0=int(z.sum()),
                             share_days_phi0=float(days[z].sum() / days.sum()), first_phi0=starts0[0] if starts0 else "",
                             last_phi0=starts0[-1] if starts0 else "", phi0_starts=";".join(starts0)))
    out = pd.DataFrame(rows)
    (HERE / "results" / "extra").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / "extra" / "phi_fallback.csv", index=False)
    print(out.drop(columns="phi0_starts").to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1:] or ["yearly", "heldout", "covid", "holds", "sensitivity", "familywise", "unexplained_null",
                            "rule_variants", "rule_variants_familywise", "familywise_no_omicron", "blind_misses",
                            "announced_split", "ewma_direction", "phi_fallback"]
    for w in what:
        globals()[w]()
        print("done", w, flush=True)


