"""Share of variation by detector set: how much the forecast, the threshold rule and the detector
explain, with and without the robust detectors (CPM, capped), in simulation and on Opal.

Method: sim_analyse._ss_shares (balanced sums of squares; main effects and two-way interactions, plus residual).
  Simulation (results/sim/summary): for every phi x noise x calendar and detector set
    CEF       = CUSUM, EWMA, FOCuS
    CEFP      = CEF + CPM
    CEF+cap   = CEF + capped CUSUM, EWMA, FOCuS
    CEFP+cap  = CEFP + capped CUSUM, EWMA, FOCuS
  false alarms: logit of the yearly false-alarm rate over forecast (A, B, HW, YM) x detector x rule (indep, ar1);
  detection: logit of detection within 60 days at the matched rate over forecast x detector, computed for each
  mean DROP (size < 0) and averaged over the drops.
  Opal (results/opal/scores.csv + rule_variants.csv), past-only forecasts A, B, HW, YM, MSTL_p, REG_p:
    (i) AR(1) rule x 8 detectors; (ii) AR(1) rule x CEF; (iii) both rules x CEF; (iv) both rules x the
    7 detectors that have both rules (no FOCuS-AR); also CEF+cap and CEFP for each rule setting.
  outcomes: non_event_per_year, r0_7_dir (direction-aware episodes), pm7_nodir (+-7 days, nearest-boundary
  crediting), frozen (the rule as frozen on 22 Oct); raw
  scale (counts and rates), as in the text.
Outputs: results/extra/shares_detsets.csv, results/extra/opal_shares.csv (shares in per cent)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from sim_analyse import _logit, _ss_shares  # noqa: E402

R = HERE / "results"
OUT = R / "extra"
OUT.mkdir(exist_ok=True)
ADJ = ["A", "B", "HW", "YM"]
CEF = ["CUSUM", "EWMA", "FOCuS"]
CAP = ["CUSUM-r", "EWMA-r", "FOCuS-r"]
SETS = {"CEF": CEF, "CEFP": CEF + ["CPM"], "CEF+cap": CEF + CAP, "CEFP+cap": CEF + ["CPM"] + CAP}
PAST = ["A", "B", "HW", "YM", "MSTL_p", "REG_p"]
ALL8 = CEF + ["FOCuS-AR", "CPM"] + CAP


def sim():
    fa = pd.read_csv(R / "sim/summary/fa.csv")
    det = pd.read_csv(R / "sim/summary/detect.csv")
    rows = []
    for (phi, noise, cal), _ in fa.groupby(["phi", "noise", "calendar"]):
        for name, S in SETS.items():
            f = fa[(fa.phi == phi) & (fa.noise == noise) & (fa.calendar == cal) & (fa.change == "none")
                   & fa.adj.isin(ADJ) & fa.rule.isin(["indep", "ar1"]) & fa.det.isin(S)].copy()
            if len(f):
                f["y"] = _logit(f["fa"])
                # cells strictly inside the logit clip (0.2%-99.8%): with 0 or 1 such cell the decomposition is
                # degenerate (every saturated condition gives the same shares, whatever the one free cell's value)
                free = int(((f["fa"] > 0.002) & (f["fa"] < 0.998)).sum())
                rows.append(dict(phi=phi, noise=noise, calendar=cal, detset=name, outcome="false_alarm_logit",
                                 n_cells=len(f), n_unsaturated=free, degenerate=free <= 1,
                                 **{k: 100 * v for k, v in _ss_shares(f, ["adj", "det", "rule"], "y").items()}))
            d = det[(det.phi == phi) & (det.noise == noise) & (det.calendar == cal) & (det.change == "mean")
                    & (det["size"] < 0) & det.adj.isin(ADJ) & (det.rule == "oracle") & det.det.isin(S)]
            per = []
            for _, g in d.groupby("size"):
                g = g.copy()
                g["y"] = _logit(g["detect60"])
                per.append(_ss_shares(g, ["adj", "det"], "y"))
            if per:
                # saturation flags for detection too: cells inside the logit clip, and cells on the
                # floor (< 2% found) which make the "forecast" share mostly the yearly memory against the rest
                free_d = int(((d["detect60"] > 0.002) & (d["detect60"] < 0.998)).sum())
                floor_d = int((d["detect60"] < 0.02).sum())
                rows.append(dict(phi=phi, noise=noise, calendar=cal, detset=name, outcome="detect60_logit_matched_drops",
                                 n_cells=len(d), n_sizes=len(per), n_unsaturated=free_d, n_floor=floor_d,
                                 degenerate=free_d <= len(per), floor_dominated=floor_d >= len(d) / 2,
                                 **{k: 100 * np.mean([p[k] for p in per]) for k in per[0]}))
                for (sz, _), p_ in zip(d.groupby("size"), per):      # each drop size on its own
                    rows.append(dict(phi=phi, noise=noise, calendar=cal, detset=name,
                                     outcome=f"detect60_logit_matched_size{sz:g}", n_cells=len(d[d["size"] == sz]),
                                     n_sizes=1, **{k: 100 * v for k, v in p_.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "shares_detsets.csv", index=False)
    four(out)
    return out


SETTINGS = [("normal", 0, "Gaussian, no holidays"), ("normal", 1, "Gaussian, holidays"),
            ("wide", 1, "real holiday scale"), ("t5", 1, "t5, holidays"), ("boot", 1, "resampled real residuals")]


def four(sd):
    """The forecast / detector / threshold-rule split for the three linear-sum detectors, with CPM
    (the four RQ1 detectors) and with the capped detectors, in every noise setting at phi = 0.4. Detection is pooled
    over drop sizes (matched 5% rate, so no rule term); false alarms use both threshold rules."""
    rows = []
    for noise, cal, label in SETTINGS:
        for name in SETS:
            for outc, key in (("detect60_logit_matched_drops", "detection"), ("false_alarm_logit", "false alarms")):
                r = sd[(sd.phi == 0.4) & (sd.noise == noise) & (sd.calendar == cal) & (sd.detset == name) & (sd.outcome == outc)]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append(dict(setting=label, noise=noise, calendar=cal, detectors=name, outcome=key,
                                 n_cells=int(r.n_cells), n_unsaturated=r.n_unsaturated, n_floor=r.get("n_floor"),
                                 forecast=r.adj, detector=r.det, rule=r.get("rule"),
                                 degenerate=bool(r.degenerate) if pd.notna(r.degenerate) else False,
                                 floor_dominated=bool(r.get("floor_dominated")) if pd.notna(r.get("floor_dominated")) else False))
    pd.DataFrame(rows).to_csv(OUT / "shares_four_detectors.csv", index=False)


def opal():
    s = pd.read_csv(R / "opal/scores.csv")
    rv = pd.read_csv(R / "opal/rule_variants.csv")
    s = s.merge(rv[["method", "detector", "rule", "r0_7_dir", "pm7_nodir", "frozen"]], on=["method", "detector", "rule"])
    s = s[s.method.isin(PAST)]
    rows = []
    specs = [("i", "ar1", ALL8), ("ii", "ar1", CEF), ("iii", "both", CEF), ("iv", "both", [d for d in ALL8 if d != "FOCuS-AR"]),
             ("ar1_CEFP", "ar1", SETS["CEFP"]), ("ar1_CEF+cap", "ar1", SETS["CEF+cap"]),
             ("both_CEFP", "both", SETS["CEFP"]), ("both_CEF+cap", "both", SETS["CEF+cap"])]
    for name, rules, dets in specs:
        d = s[s.detector.isin(dets) & ((s.rule == "ar1") if rules == "ar1" else s.rule.isin(["indep", "ar1"]))]
        factors = ["method", "detector"] + (["rule"] if rules == "both" else [])
        for y in ("non_event_per_year", "r0_7_dir", "pm7_nodir", "frozen"):
            rows.append(dict(spec=name, rules=rules, detectors=";".join(dets), outcome=y, n_cells=len(d),
                             **{k: 100 * v for k, v in _ss_shares(d, factors, y).items()}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "opal_shares.csv", index=False)
    return out


def _share_ci(d, col, count_col, eps, n_boot=2000, seed=1):
    M = np.array([[e in str(x).split(";") for e in eps] for x in d[col].fillna("")], dtype=float)
    if count_col is not None:
        assert (M.sum(1) == d[count_col].to_numpy()).all(), (col, count_col)
    g = d[["method", "detector"]].copy()

    def sh(w):
        g["y"] = M @ w
        s = _ss_shares(g, ["method", "detector"], "y")
        return 100 * s["method"], 100 * s["detector"]

    obs = sh(np.ones(len(eps)))
    rng = np.random.default_rng(seed)
    B = np.array([sh(np.bincount(rng.integers(0, len(eps), len(eps)), minlength=len(eps)).astype(float))
                  for _ in range(n_boot)])
    return dict(forecast_share=obs[0], detector_share=obs[1],
                forecast_lo=float(np.percentile(B[:, 0], 2.5)), forecast_hi=float(np.percentile(B[:, 0], 97.5)),
                detector_lo=float(np.percentile(B[:, 1], 2.5)), detector_hi=float(np.percentile(B[:, 1], 97.5)),
                p_forecast_gt_detector=float((B[:, 0] > B[:, 1]).mean()), n_boot=n_boot)


def opal_ci(rv_path=R / "opal/rule_variants.csv", out_path=OUT / "opal_shares_ci.csv"):
    """Episode bootstrap for the Opal share split: the 15 scored episodes are resampled with
    replacement (2,000 resamples, seed 1); each monitor's count is the sum of its catches over the resampled
    episodes (catch matrix from rule_variants.csv); shares by sim_analyse._ss_shares on forecast x detector.
    Past-only forecasts A, B, HW, YM, MSTL_p, REG_p, AR(1) threshold; detector sets CEF and all eight."""
    import sys as _s
    _s.path.insert(0, str(HERE))
    from cpd import score
    eps = sorted(score.scored().episode.unique())
    rv = pd.read_csv(rv_path)
    cols = [("direction-aware", "episodes_headline", "r0_7_dir"),
            ("+-7 nearest", "episodes_pm7_nodir" if "episodes_pm7_nodir" in rv else "episodes_original_rule", "pm7_nodir")]
    if "episodes_frozen" in rv:
        cols.append(("frozen", "episodes_frozen", "frozen"))
    rows = []
    for fname, fs, dname, dets in (("six past-only", PAST, "CEF", CEF), ("six past-only", PAST, "all 8", ALL8),
                                   ("four main (A, B, HW, YM)", ["A", "B", "HW", "YM"], "CEF", CEF)):
        d = rv[rv.method.isin(fs) & rv.detector.isin(dets) & (rv.rule == "ar1")].reset_index(drop=True)
        for rule, col, cnt in cols:
            rows.append(dict(forecasts=fname, detectors=dname, rule=rule, monitors=len(d), **_share_ci(d, col, cnt, eps)))
    out = pd.DataFrame(rows)
    if out_path is not None:
        out.to_csv(out_path, index=False)
    return out


def elec():
    """Electricity test bed: shares of the gain (detection with the drop minus without, within 60 days) over
    forecast x detector for the 5% and 10% drops (6 forecasts x 5 detectors, AR(1)); raw gains, not logits, because
    gains can be negative. Also the detector range of the 10% gain within the forecast with temperature."""
    t = pd.read_csv(R / "electricity/testbed.csv")
    t["hit"] = t.first_post.between(0, 59).astype(float)
    g = t.pivot_table(index=["size", "adj", "det", "tau"], columns="which", values="hit").reset_index()
    g["gain"] = g["drop"] - g["none"]
    gm = g.groupby(["size", "adj", "det"]).gain.mean().reset_index().rename(columns={"adj": "method", "det": "detector"})
    rows = []
    for sz, d in gm.groupby("size"):
        if sz not in (5.0, 10.0):
            continue
        rows.append(dict(size=sz, outcome="gain", n_cells=len(d), **{k: 100 * v for k, v in _ss_shares(d, ["method", "detector"], "gain").items()}))
    r10 = gm[(gm["size"] == 10.0) & (gm.method == "REGT_p")]
    r10b = r10[r10.detector != "FOCuS-AR"]
    rows.append(dict(size=10.0, outcome="REGT_p detector range", n_cells=len(r10), min_gain=r10.gain.min(), max_gain=r10.gain.max(),
                     min_gain_no_focus_ar=r10b.gain.min(), max_gain_no_focus_ar=r10b.gain.max()))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "elec_shares.csv", index=False)
    return out


if __name__ == "__main__":
    a = sim()
    b = opal()
    print(elec().round(3).to_string(index=False))
    print(opal_ci().round(3).to_string(index=False))
    pd.set_option("display.width", 250)
    print(a[(a.phi == 0.4) & (a.noise == "normal")].round(1).to_string(index=False))
    print(b.round(1).to_string(index=False))
