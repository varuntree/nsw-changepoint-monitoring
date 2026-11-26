"""Summarise the simulation (results/sim/*/b*.npz) into tables for Chapter 4.

Outputs (results/sim/summary/):
  fa.csv          realised P(false alarm within the monitoring year) by condition x adjustment x detector x rule,
                  with Monte Carlo SE; pipeline-oracle validated split-half (threshold from even batches,
                  rate on odd batches)
  detect.csv      per change condition x adjustment x detector x rule: pre-change false alarm P(T < tau),
                  conditional detection P(T - tau < 60 | T >= tau), median delay, restricted mean delay
                  E[min(T - tau + 1, 61) | T >= tau], delay curve F(d) at d = 7, 14, 30
  offline.csv     PELT / BinSeg / WBS on oracle- and HW-adjusted series: detection within +-7 days,
                  median localisation error, spurious changes per run
  shares.csv      within-condition decomposition (logit scale) of detection at matched false-alarm rate
                  (oracle rule) over adjustment x detector, and of calibration error (realised FA) over
                  adjustment x detector x rule: share of sum of squares for each main effect and interaction
  tornado.csv     mean absolute change in the outcome when one factor is switched, others held fixed
  delay.csv       detection-speed panel: phi = 0.4, normal noise, mean drops of 0.5 to 4 SD; true pattern
                  and Holt-Winters without holidays, yearly memory with holidays; every mean detector; matched
                  ("oracle") and AR(1) rules; detect60, F7, F14, F30, restricted mean delay, median delay
  variance.csv    variance-change panel: phi = 0.4, variance factor v = 1.5, 2, 3 (normal noise, holidays),
                  v = 2 without holidays and with t5 noise; true pattern, Holt-Winters, yearly memory; the three
                  spread detectors (CUSUM-var, EWMA-var, FOCuS-var) and the mean detectors
  shares_spread.csv  as shares.csv for the variance changes, with the detector factor running over the three
                  spread detectors plus CUSUM, EWMA, FOCuS and CPM
  ranges.csv      effect sizes behind the shares: for each condition and rule, the range (max - min, in
                  percentage points) of the false-alarm rate across adjustments for each detector and across
                  detectors for each adjustment (A, B, HW, YM x CUSUM, EWMA, FOCuS, CPM); the same for detection
                  within 60 days at the matched rate
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
SIM = HERE / "results" / "sim"
OUT = SIM / "summary"
OUT.mkdir(parents=True, exist_ok=True)
RULES = ("indep", "ar1", "oracle")
REAL_ADJ = ("A", "B", "HW", "YM")


def parse(name):
    phi, noise, cal, ch = name.split("_")
    change = "none" if ch.startswith("none") else ("mean" if ch.startswith("mean") else "var")
    size = float(ch[len(change):])
    return dict(phi=float(phi[3:]), noise=noise, calendar=int(cal[3:]), change=change, size=size)


def load(d):
    fs = sorted(d.glob("b*.npz"))
    if not fs:
        return None
    z = [np.load(f) for f in fs]
    out = {k: np.concatenate([q[k] for q in z]) for k in ("first", "mx", "phis", "taus", "off", "offcp", "offspur")}
    out["dets"] = list(z[0]["dets"])
    out["adj"] = list(z[0]["adj"])
    # detectors added later by scripts/sim_addon.py (x*.npz beside each batch): append along the detector axis
    xs = sorted(d.glob("x*.npz"))
    if xs:
        assert len(xs) == len(fs), d
        x = [np.load(f) for f in xs]
        out["first"] = np.concatenate([out["first"], np.concatenate([q["first"] for q in x])], axis=2)
        out["mx"] = np.concatenate([out["mx"], np.concatenate([q["mx"] for q in x])], axis=2)
        out["dets"] += list(x[0]["dets"])
    out["batch"] = np.concatenate([np.full(len(q["taus"]), i) for i, q in enumerate(z)])
    return out


def main():
    conds = sorted([p for p in SIM.iterdir() if p.is_dir() and p.name.startswith("phi")])
    fa_rows, det_rows, off_rows = [], [], []
    for d in conds:
        c = parse(d.name)
        L = load(d)
        if L is None:
            continue
        first, adj, dets = L["first"], L["adj"], L["dets"]
        n = first.shape[0]
        if c["change"] == "none":
            even, odd = L["batch"] % 2 == 0, L["batch"] % 2 == 1
            for a, an in enumerate(adj):
                for k, dn in enumerate(dets):
                    for r, rn in enumerate(RULES[:2]):
                        p = float((first[:, a, k, r] >= 0).mean())
                        fa_rows.append({**c, "adj": an, "det": dn, "rule": rn, "fa": p, "se": np.sqrt(p * (1 - p) / n), "n": n})
                    h = np.quantile(L["mx"][even, a, k], 0.95)
                    p = float((L["mx"][odd, a, k] > h).mean())
                    fa_rows.append({**c, "adj": an, "det": dn, "rule": "oracle", "fa": p,
                                    "se": np.sqrt(p * (1 - p) / odd.sum()), "n": int(odd.sum())})
            continue
        tau = L["taus"].astype(int)
        for a, an in enumerate(adj):
            for k, dn in enumerate(dets):
                for r, rn in enumerate(RULES):
                    T = first[:, a, k, r].astype(int)
                    T = np.where(T < 0, 10_000, T)
                    pre = T < tau
                    post = ~pre
                    dl = (T - tau + 1)[post]
                    det60 = (dl <= 60)
                    rmd = np.minimum(dl, 61)
                    row = {**c, "adj": an, "det": dn, "rule": rn, "n": n,
                           "pre_fa": float(pre.mean()), "detect60": float(det60.mean()) if post.any() else np.nan,
                           "se_detect": float(np.sqrt(det60.mean() * (1 - det60.mean()) / max(post.sum(), 1))),
                           "median_delay": float(np.median(dl)) if (dl <= 60).mean() >= 0.5 else np.inf,
                           "rmd": float(rmd.mean()) if post.any() else np.nan}
                    for dd in (7, 14, 30):
                        row[f"F{dd}"] = float((dl <= dd).mean()) if post.any() else np.nan
                    det_rows.append(row)
        for j, an in enumerate(("oracle", "HW")):
            for m, mn in enumerate(("PELT", "BinSeg", "WBS")):
                cp = L["offcp"][:, j, m].astype(int)
                hit = (cp >= 0) & (np.abs(cp - tau) <= 7)
                err = np.abs(cp - tau)[cp >= 0]
                off_rows.append({**c, "adj": an, "method": mn, "detect7": float(hit.mean()),
                                 "median_loc_error": float(np.median(err)) if len(err) else np.nan,
                                 "spurious_per_run": float(L["offspur"][:, j, m].mean())})
    fa = pd.DataFrame(fa_rows)
    det = pd.DataFrame(det_rows)
    off = pd.DataFrame(off_rows)
    fa.to_csv(OUT / "fa.csv", index=False)
    det.to_csv(OUT / "detect.csv", index=False)
    off.to_csv(OUT / "offline.csv", index=False)
    shares(fa, det)
    panels(det)
    ranges(fa, det)
    tuning(fa, det)


TUNE = ["CUSUM-k0.25", "CUSUM", "CUSUM-k1", "EWMA-l0.05", "EWMA", "EWMA-l0.2"]


def tuning(fa, det):
    """Tuning sensitivity: CUSUM with k = 0.25, 0.5, 1 and EWMA with lambda = 0.05, 0.1, 0.2;
    phi = 0.4, normal noise, no holidays (10,000 runs per condition, from sim_addon.py --tuning); matched and AR(1)
    rules; true pattern, Holt-Winters and yearly memory. Output tuning.csv (with the parameter in its own column)."""
    d = det[(det["phi"] == 0.4) & (det["noise"] == "normal") & (det["calendar"] == 0) & det["det"].isin(TUNE)
            & det["adj"].isin(["oracle", "HW", "YM"]) & det["rule"].isin(["oracle", "ar1"])]
    f = fa[(fa["phi"] == 0.4) & (fa["noise"] == "normal") & (fa["calendar"] == 0) & fa["det"].isin(TUNE)
           & fa["adj"].isin(["oracle", "HW", "YM"]) & fa["rule"].isin(["oracle", "ar1"])]
    par = {"CUSUM-k0.25": ("CUSUM", "k", 0.25), "CUSUM": ("CUSUM", "k", 0.5), "CUSUM-k1": ("CUSUM", "k", 1.0),
           "EWMA-l0.05": ("EWMA", "lambda", 0.05), "EWMA": ("EWMA", "lambda", 0.1), "EWMA-l0.2": ("EWMA", "lambda", 0.2)}
    out = d[COLS].copy()
    out = out.merge(f[["adj", "det", "rule", "fa"]].rename(columns={"fa": "false_alarm_rate"}), on=["adj", "det", "rule"], how="left")
    out["family"] = out["det"].map(lambda x: par[x][0])
    out["param"] = out["det"].map(lambda x: par[x][1])
    out["value"] = out["det"].map(lambda x: par[x][2])
    out.to_csv(OUT / "tuning.csv", index=False)


SHARE_DETS = ["CUSUM", "EWMA", "FOCuS", "CPM"]
MEAN_ALL = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
SPREAD_ALL = ["CUSUM-var", "EWMA-var", "FOCuS-var"]
COLS = ["phi", "noise", "calendar", "change", "size", "adj", "det", "rule", "n", "pre_fa", "detect60", "se_detect",
        "F7", "F14", "F30", "rmd", "median_delay"]


def panels(det):
    d = det[(det["phi"] == 0.4) & det["rule"].isin(["oracle", "ar1"])]
    sel = [("true pattern, no holidays", "oracle", 0), ("Holt-Winters, no holidays", "HW", 0),
           ("yearly memory, holidays", "YM", 1)]
    out = []
    for label, adj, cal in sel:
        g = d[(d["noise"] == "normal") & (d["change"] == "mean") & (d["size"] < 0) & (d["adj"] == adj)
              & (d["calendar"] == cal) & d["det"].isin(MEAN_ALL)]
        out.append(g[COLS].assign(setting=label, shift_sd=-g["size"]))
    # real holiday scale (noise "wide": Gaussian with the raw residual SD; shifts in those SD units)
    for adj in ("oracle", "A", "B", "HW", "YM"):
        g = d[(d["noise"] == "wide") & (d["change"] == "mean") & (d["adj"] == adj) & d["det"].isin(MEAN_ALL)]
        out.append(g[COLS].assign(setting=f"real holiday scale, {adj}", shift_sd=-g["size"]))
    # block-bootstrap noise (noise "boot": resampled real residuals; shifts in robust-sigma units)
    for adj in ("oracle", "A", "B", "HW", "YM"):
        g = d[(d["noise"] == "boot") & (d["change"] == "mean") & (d["adj"] == adj) & d["det"].isin(MEAN_ALL)]
        out.append(g[COLS].assign(setting=f"bootstrap noise, {adj}", shift_sd=-g["size"]))
    pd.concat(out).to_csv(OUT / "delay.csv", index=False)
    g = d[(d["change"] == "var") & d["adj"].isin(["oracle", "HW", "YM"]) & d["det"].isin(SPREAD_ALL + MEAN_ALL)]
    g = g[COLS].assign(setting=np.where(g["noise"] == "t5", "t5 noise, holidays",
                                        np.where(g["calendar"] == 1, "normal noise, holidays", "normal noise, no holidays")))
    g.to_csv(OUT / "variance.csv", index=False)


def ranges(fa, det):
    rows = []
    f = fa[fa["adj"].isin(REAL_ADJ) & fa["det"].isin(SHARE_DETS) & fa["rule"].isin(["indep", "ar1"])]
    d = det[det["adj"].isin(REAL_ADJ) & det["det"].isin(SHARE_DETS) & (det["rule"] == "oracle")]
    for outcome, df, y, keys in (("false_alarm", f, "fa", ["phi", "noise", "calendar", "rule"]),
                                 ("detect60_matched", d, "detect60", ["phi", "noise", "calendar", "change", "size", "rule"])):
        for k, g in df.groupby(keys):
            for varied, held in (("adj", "det"), ("det", "adj")):
                for lev, h in g.groupby(held):
                    v = 100 * h[y]
                    rows.append({**dict(zip(keys, k)), "outcome": outcome, "varied": varied, "held": lev,
                                 "min_pct": v.min(), "max_pct": v.max(), "range_pts": v.max() - v.min(),
                                 "argmin": h.loc[v.idxmin(), varied], "argmax": h.loc[v.idxmax(), varied]})
    pd.DataFrame(rows).to_csv(OUT / "ranges.csv", index=False)


def _ss_shares(df, factors, y):
    """Balanced-design sum-of-squares shares for main effects and two-way interactions."""
    m = df[y].mean()
    tot = ((df[y] - m) ** 2).sum()
    out = {}
    for f in factors:
        g = df.groupby(f)[y].mean()
        out[f] = float(((df[f].map(g) - m) ** 2).sum() / tot) if tot > 0 else np.nan
    for f1, f2 in itertools.combinations(factors, 2):
        g12 = df.groupby([f1, f2])[y].transform("mean")
        g1 = df[f1].map(df.groupby(f1)[y].mean())
        g2 = df[f2].map(df.groupby(f2)[y].mean())
        out[f"{f1}x{f2}"] = float(((g12 - g1 - g2 + m) ** 2).sum() / tot) if tot > 0 else np.nan
    out["residual"] = 1 - sum(v for v in out.values() if not np.isnan(v))
    return out


def _tornado(df, factors, y):
    out = {}
    for f in factors:
        others = [g for g in factors if g != f]
        rng = df.groupby(others)[y].agg(lambda s: s.max() - s.min())
        out[f] = float(rng.mean())
    return out


def _logit(p):
    p = np.clip(p, 0.002, 0.998)
    return np.log(p / (1 - p))


def shares(fa, det):
    rows, trows = [], []
    keys = ["phi", "noise", "calendar"]
    # calibration error: |realised FA - 0.05| over adjustment x detector x rule (indep, ar1)
    f = fa[fa["adj"].isin(REAL_ADJ) & fa["rule"].isin(["indep", "ar1"]) & fa["det"].isin(["CUSUM", "EWMA", "FOCuS", "CPM"])].copy()
    f["y"] = _logit(f["fa"])
    for k, g in f.groupby(keys):
        rows.append({**dict(zip(keys, k)), "outcome": "false_alarm_logit", **_ss_shares(g, ["adj", "det", "rule"], "y")})
        trows.append({**dict(zip(keys, k)), "outcome": "false_alarm_rate", **_tornado(g, ["adj", "det", "rule"], "fa")})
    # detection at matched false-alarm rate (oracle rule)
    d = det[det["adj"].isin(REAL_ADJ) & (det["rule"] == "oracle") & det["det"].isin(["CUSUM", "EWMA", "FOCuS", "CPM"])].copy()
    d["y"] = _logit(d["detect60"])
    for k, g in d.groupby(keys + ["change", "size"]):
        rows.append({**dict(zip(keys + ["change", "size"], k)), "outcome": "detect60_logit_matched", **_ss_shares(g, ["adj", "det"], "y")})
        trows.append({**dict(zip(keys + ["change", "size"], k)), "outcome": "detect60_matched", **_tornado(g, ["adj", "det"], "detect60")})
    pd.DataFrame(rows).to_csv(OUT / "shares.csv", index=False)
    pd.DataFrame(trows).to_csv(OUT / "tornado.csv", index=False)
    # variance changes with the spread detectors included in the detector factor
    rows = []
    d = det[det["adj"].isin(REAL_ADJ) & (det["rule"] == "oracle") & (det["change"] == "var")
            & det["det"].isin(SHARE_DETS + SPREAD_ALL)].copy()
    d["y"] = _logit(d["detect60"])
    for k, g in d.groupby(keys + ["change", "size"]):
        rows.append({**dict(zip(keys + ["change", "size"], k)), "outcome": "detect60_logit_matched_with_spread",
                     "n_dets": g["det"].nunique(), **_ss_shares(g, ["adj", "det"], "y")})
    pd.DataFrame(rows).to_csv(OUT / "shares_spread.csv", index=False)


if __name__ == "__main__":
    main()
