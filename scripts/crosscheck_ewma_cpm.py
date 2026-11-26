"""Cross-check against an independently written EWMA and Mann-Whitney CPM.

Compares the independent code (data/independent/ewma_cpm/ewma.py, cpm.py, not in this repository), its Opal
alarms and scores and its 2,000-run simulation cells (data/independent/ewma_cpm/results) with our code.
  kernels   its stat_path against cpd.kernels.ewma_path (lambda 0.1) and cpm_mw_path (startup 20) on 30 inputs:
            Gaussian, rounded (ties), capped at +-3, 15-30% missing, leading missing, step changes, an all-missing
            series, a clipped t3 series, a 2,200-day series, all-equal starts. CPM also with z + 0.0, which removes
            its signed-zero tie miscount (-0.0 and 0.0 counted as two tie groups);
  opal      alarm dates, directions and estimated change dates per monitor (10 forecasts x EWMA, capped EWMA, CPM x
            2 threshold rules) and the score rows, against opal/alarms.csv, scores.csv and rule_variants.csv;
  sim       every false-alarm and detection cell it ran (phi 0/0.4/0.6, calendar off/on, Gaussian; t5) against
            sim/summary/fa.csv and detect.csv: z = difference / combined SE. Its capped FOCuS uses focus_cpt, so its
            cells are compared with our pre-fix package values too (extra/fa_package_focus.csv).
  cpm start its change-date mapping (calendar day of the first observation after the maximising split) against
            ours (t_alarm - n2 + 1, which counts observed days back from the alarm): alarms whose dates differ and
            whether any differs in a quantity the thesis prints (opal/covid.csv first alarms, Figure 6.5).
Outputs: results/extra/crosscheck_ewma_cpm.csv (summary, ledger-ready), crosscheck_ewma_cpm_kernels.csv, crosscheck_ewma_cpm_opal_alarms.csv,
crosscheck_ewma_cpm_opal_scores.csv, crosscheck_ewma_cpm_sim_cells.csv, crosscheck_ewma_cpm_cpm_start.csv."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import kernels as K  # noqa: E402

R = HERE / "results"
OUT = R / "extra"
Y = HERE / "data" / "independent" / "ewma_cpm"
DETMAP = {"EWMA": "EWMA", "EWMA-r": "EWMA-r", "CPM": "CPM", "CUSUM": "CUSUM", "CUSUM-r": "CUSUM-r", "FOCuS": "FOCuS",
          "FOCuS-r": "FOCuS-r"}


def _load(name):
    """The independent module, compiled in memory (its numba cache=True would write a cache file next to the source and cannot
    be reloaded for a module loaded from a path)."""
    import types
    src = (Y / f"{name}.py").read_text().replace("cache=True", "cache=False")
    m = types.ModuleType(f"independent_{name}")
    sys.modules[m.__name__] = m
    exec(compile(src, str(Y / f"{name}.py"), "exec"), m.__dict__)
    return m


def inputs():
    rng = np.random.default_rng(20261027)
    out = []
    for i in range(20):
        n = int(rng.integers(30, 401))
        z = rng.standard_normal(n)
        if i % 4 == 1:
            z = np.round(z, 1)                       # heavy ties (includes -0.0)
        if i % 4 == 2:
            z = np.clip(2 * z, -3, 3)                # capped: ties at +-3
        z[rng.random(n) < rng.uniform(0.15, 0.30)] = np.nan
        if i % 5 == 0:
            z[:5] = np.nan                          # leading missing
        if i % 3 == 0:
            z[n // 2:] += rng.normal(0, 2)           # step change
        out.append((f"random {i}", z))
    out.append(("all missing", np.full(50, np.nan)))
    out.append(("t3 clipped", np.clip(rng.standard_t(3, 300), -3, 3)))
    out.append(("2,200 days", rng.standard_normal(2200)))
    out.append(("equal (25 equal values, then a shift)", np.concatenate([np.full(25, 1.5), np.full(15, 3.0)])))
    out.append(("all equal", np.zeros(40)))
    out.append(("signed zero", np.concatenate([np.arange(1.0, 19.0), [0.0, -0.0], np.arange(19.0, 25.0)])))
    return out


def kernels():
    ye, yc = _load("ewma"), _load("cpm")
    rows = []
    for name, z in inputs():
        e = np.nanmax(np.abs(ye.stat_path(z) - K.ewma_path(z, 0.1)))
        c_ours = K.cpm_mw_path(z, 20)
        c = np.nanmax(np.abs(yc.stat_path(z) - c_ours))
        c0 = np.nanmax(np.abs(yc.stat_path(z + 0.0) - c_ours))
        rows.append(dict(input=name, n=len(z), ewma_max_abs_diff=float(e), cpm_max_abs_diff=float(c),
                         cpm_max_abs_diff_signed_zero_normalised=float(c0),
                         has_signed_zero=bool(np.any((z == 0) & np.signbit(z)) and np.any((z == 0) & ~np.signbit(z)))))
    k = pd.DataFrame(rows)
    k.to_csv(OUT / "crosscheck_ewma_cpm_kernels.csv", index=False)
    return k


def opal():
    ours = pd.read_csv(R / "opal" / "alarms.csv")
    ours["alarm"] = pd.to_datetime(ours.alarm).dt.strftime("%Y-%m-%d")
    ours["start_est"] = pd.to_datetime(ours.start_est).dt.strftime("%Y-%m-%d")
    theirs = pd.concat([pd.read_csv(Y / "results" / "opal" / f"{f}_alarms.csv") for f in ("ewma", "ewma_capped", "cpm")])
    theirs = theirs.rename(columns={"adjustment": "method", "alarm_date": "alarm", "start_estimate_date": "start_y",
                                    "direction": "direction_y"})
    theirs["detector"] = theirs.detector.replace({"EWMA_capped": "EWMA-r", "EWMA-capped": "EWMA-r", "CPM_capped": "CPM-r"})
    o = ours[ours.detector.isin(theirs.detector.unique())]
    keys = ["method", "detector", "rule", "alarm"]
    m = o.merge(theirs[keys + ["start_y", "direction_y"]], on=keys, how="outer", indicator=True)
    m.to_csv(OUT / "crosscheck_ewma_cpm_opal_alarms.csv", index=False)
    sc = pd.concat([pd.read_csv(Y / "results" / "opal" / f"{f}_scores.csv") for f in ("ewma", "ewma_capped", "cpm")])
    sc = sc.rename(columns={"adjustment": "method"})
    sc["detector"] = sc.detector.replace({"EWMA_capped": "EWMA-r", "EWMA-capped": "EWMA-r"})
    ob = pd.read_csv(R / "opal" / "scores.csv").merge(pd.read_csv(R / "opal" / "rule_variants.csv")[
        ["method", "detector", "rule", "r0_7_dir", "frozen"]], on=["method", "detector", "rule"])
    s = sc.merge(ob, on=["method", "detector", "rule"], suffixes=("_y", ""))
    cols = ["n_alarms", "near_real", "episodes_caught", "p_episodes", "p_share_real", "blind_days_per_year", "non_event_per_year"]
    for c in cols:
        s[f"{c}_equal"] = np.isclose(s[f"{c}_y"], s[c], equal_nan=True)
    s["episodes_equal_dir_rule"] = s["episodes_caught_y"] == s["r0_7_dir"]
    s.to_csv(OUT / "crosscheck_ewma_cpm_opal_scores.csv", index=False)
    return m, s


def sim():
    fa_b = pd.read_csv(R / "sim" / "summary" / "fa.csv")
    de_b = pd.read_csv(R / "sim" / "summary" / "detect.csv")
    pk = pd.read_csv(OUT / "fa_package_focus.csv")
    k = ["phi", "noise", "calendar", "change", "size", "adj", "det", "rule"]
    rows = []
    for d in ("sim", "robust_t5"):
        for f in sorted((Y / "results" / d).glob("*_false_alarms.csv")):
            y = pd.read_csv(f)
            m = y.merge(fa_b, on=k, suffixes=("_y", "_b"))
            m["measure"], m["source"] = "false-alarm years", d
            m = m.merge(pk[k + ["fa"]].rename(columns={"fa": "fa_package"}), on=k, how="left")
            m["z"] = (m.fa_y - m.fa_b) / np.sqrt(m.se_y ** 2 + m.se_b ** 2)
            m["z_vs_package"] = (m.fa_y - m.fa_package) / np.sqrt(m.se_y ** 2 + m.fa_package * (1 - m.fa_package) / 10000)
            rows.append(m.rename(columns={"fa_y": "value_y", "fa_b": "value_b", "se_y": "se_y", "se_b": "se_b"}))
        for f in sorted((Y / "results" / d).glob("*_detection.csv")):
            y = pd.read_csv(f)
            m = y.merge(de_b, on=k, suffixes=("_y", "_b"))
            m["measure"], m["source"] = "detected within 60 days", d
            m["z"] = (m.detect60_y - m.detect60_b) / np.sqrt(m.se_detect_y ** 2 + m.se_detect_b ** 2)
            rows.append(m.rename(columns={"detect60_y": "value_y", "detect60_b": "value_b", "se_detect_y": "se_y",
                                          "se_detect_b": "se_b"}))
    c = pd.concat(rows, ignore_index=True)
    keep = k + ["measure", "source", "value_y", "se_y", "value_b", "se_b", "z", "fa_package", "z_vs_package"]
    c = c[[x for x in keep if x in c.columns]]
    c.to_csv(OUT / "crosscheck_ewma_cpm_sim_cells.csv", index=False)
    return c


def first2000():
    """The independent runs used the group harness, whose seeds are ours (SeedSequence([crc32(condition), batch])),
    so their 2,000 runs are our batches b000-b003. Recompute every cell they ran from those four batches and count
    identical cells. The matched (oracle) rule cannot be identical: their threshold is set on their own no-change runs
    (1,000 even-batch runs), ours on 10,000; FOCuS and capped FOCuS differ because theirs is focus_cpt."""
    sim = R / "sim"
    k = ["phi", "noise", "calendar", "change", "size", "adj", "det", "rule"]
    rows = []
    for d in ("sim", "robust_t5"):
        for f in sorted((Y / "results" / d).glob("*_false_alarms.csv")) + sorted((Y / "results" / d).glob("*_detection.csv")):
            y = pd.read_csv(f)
            measure = "false-alarm years" if f.name.endswith("_false_alarms.csv") else "detected within 60 days"
            for (phi, noise, cal, ch, sz), g in y.groupby(["phi", "noise", "calendar", "change", "size"]):
                ch_s = "none" if ch == "none" else ch
                cond = sim / f"phi{phi:.2f}_{noise}_cal{int(cal)}_{ch_s}{sz:+.1f}"
                fs = [cond / f"b{i:03d}.npz" for i in range(4)]
                if not all(p_.exists() for p_ in fs):
                    continue
                z = [np.load(p_) for p_ in fs]
                first = np.concatenate([q["first"] for q in z])
                taus = np.concatenate([q["taus"] for q in z]).astype(int)
                dets, adjs = list(z[0]["dets"]), list(z[0]["adj"])
                for _, r in g.iterrows():
                    if r.det not in dets or r.adj not in adjs:
                        continue
                    ri = {"indep": 0, "ar1": 1, "oracle": 2}[r.rule]
                    T = first[:, adjs.index(r.adj), dets.index(r.det), ri].astype(int)
                    if measure == "false-alarm years":
                        if r.rule == "oracle":
                            continue        # matched FA is computed from maxima on held-out runs, not first crossings
                        ours, indep = float((T >= 0).mean()), float(r.fa)
                    else:
                        T = np.where(T < 0, 10_000, T)
                        dl = (T - taus + 1)[T >= taus]
                        ours = float((dl <= 60).mean()) if len(dl) else np.nan
                        indep = float(r.detect60)
                    rows.append({**{c: r[c] for c in k}, "measure": measure, "source": d, "indep": indep, "ours_first2000": ours,
                                 "identical": bool(np.isclose(indep, ours, rtol=0, atol=1e-12) or (np.isnan(indep) and np.isnan(ours)))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "crosscheck_ewma_cpm_first2000.csv", index=False)
    return t


def cpm_start(m):
    """Her start mapping against ours on the CPM and capped CPM alarms both runs share."""
    x = m[(m["_merge"] == "both") & m.detector.isin(["CPM", "CPM-r"])].copy()
    x["start_ours"] = pd.to_datetime(x.start_est)
    x["start_indep"] = pd.to_datetime(x.start_y)
    x["days_indep_minus_ours"] = (x.start_indep - x.start_ours).dt.days
    cv = pd.read_csv(R / "opal" / "covid.csv")
    cv = cv[cv.detector == "CPM"].dropna(subset=["first_alarm"])
    firsts = set(zip(cv.method, cv.rule, cv.first_alarm))
    x["is_covid_first_alarm"] = [(a, r, d) in firsts for a, r, d in zip(x.method, x.rule, x.alarm)]
    x.to_csv(OUT / "crosscheck_ewma_cpm_cpm_start.csv", index=False)
    return x


def main():
    kk = kernels()
    m, s = opal()
    c = sim()
    cs = cpm_start(m)
    f2 = first2000()
    own = f2.det.isin(["EWMA", "EWMA-r", "CPM"]) & f2.rule.isin(["indep", "ar1"])
    cusum = f2.det.isin(["CUSUM", "CUSUM-r"]) & f2.rule.isin(["indep", "ar1"])
    focus = f2.det.isin(["FOCuS", "FOCuS-r"]) & f2.rule.isin(["indep", "ar1"])
    fa2, de2 = f2.measure == "false-alarm years", f2.measure == "detected within 60 days"
    fa = c[c.measure == "false-alarm years"]
    capf = fa[(fa.det == "FOCuS-r") & (fa.adj == "YM") & (fa.calendar == 1) & (fa.rule == "ar1") & (fa.noise == "normal")]
    other = fa[~fa.index.isin(capf.index)]
    de = c[c.measure == "detected within 60 days"]
    summ = [
        ("kernel inputs", len(kk)),
        ("EWMA max abs path difference", kk.ewma_max_abs_diff.max()),
        ("CPM max abs path difference, inputs without signed zeros", kk[~kk.has_signed_zero].cpm_max_abs_diff.max()),
        ("CPM max abs path difference, all inputs after z + 0.0", kk.cpm_max_abs_diff_signed_zero_normalised.max()),
        ("CPM max abs path difference from the independent code's signed-zero tie miscount", kk[kk.has_signed_zero].cpm_max_abs_diff.max()),
        ("Opal monitors compared", int(s.shape[0])),
        ("Opal alarms, independent code", int((m["_merge"] != "left_only").sum())),
        ("Opal alarms, ours (same monitors)", int((m["_merge"] != "right_only").sum())),
        ("Opal alarms only in one run", int((m["_merge"] != "both").sum())),
        ("Opal alarms with a different direction", int(((m["_merge"] == "both") & (m.direction != m.direction_y)).sum())),
        ("Opal score rows with every compared column equal", int(s[[f"{c_}_equal" for c_ in
                                                                    ["n_alarms", "near_real", "episodes_caught", "p_episodes", "blind_days_per_year"]]].all(axis=1).sum())),
        ("Opal score rows whose episodes equal our direction-aware count", int(s.episodes_equal_dir_rule.sum())),
        ("Opal score rows whose episodes equal our frozen-rule count", int((s.episodes_caught_y == s.frozen).sum())),
        ("simulation false-alarm cells compared", len(fa)),
        ("false-alarm cells with |z| > 3, excluding YM + capped FOCuS + holidays + AR(1)", int((other.z.abs() > 3).sum())),
        ("false-alarm max |z|, excluding those cells", float(other.z.abs().max())),
        ("YM + capped FOCuS + holidays + AR(1) cells", len(capf)),
        ("their max |z| against our exact FOCuS", float(capf.z.abs().max())),
        ("their max |z| against our pre-fix package FOCuS", float(capf.z_vs_package.abs().max())),
        ("false-alarm max |z|, independent threshold rule", float(fa[fa.rule == "indep"].z.abs().max())),
        ("false-alarm cells with |z| > 3 under the matched rule (threshold estimated on 1,000 runs in the independent code)",
         int(((other.rule == "oracle") & (other.z.abs() > 3)).sum())),
        ("t5 false-alarm cells compared", int((fa.source == "robust_t5").sum())),
        ("t5 false-alarm max |z|", float(fa[fa.source == "robust_t5"].z.abs().max())),
        ("detection cells compared", len(de)),
        ("detection cells with |z| > 4", int((de.z.abs() > 4).sum())),
        ("of those, capped FOCuS (package in the independent code)", int(((de.z.abs() > 4) & (de.det == "FOCuS-r")).sum())),
        ("detection cells mean z", float(de.z.replace([np.inf, -np.inf], np.nan).mean())),
        ("CPM alarms compared for the change date", len(cs)),
        ("CPM alarms whose change date differs", int((cs.days_indep_minus_ours != 0).sum())),
        ("CPM change date difference, days (min)", float(cs.days_indep_minus_ours.min())),
        ("of those, first COVID-19 alarms (opal/covid.csv, Figure 6.4)", int(((cs.days_indep_minus_ours != 0) & cs.is_covid_first_alarm).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM false-alarm cells, independent and AR(1) rules", int((own & fa2).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM false-alarm cells identical", int((own & fa2 & f2.identical).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM false-alarm cells, Gaussian", int((own & fa2 & (f2.source == "sim")).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM false-alarm cells, t5", int((own & fa2 & (f2.source == "robust_t5")).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM detection cells, independent and AR(1) rules", int((own & de2).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM detection cells identical", int((own & de2 & f2.identical).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM detection cells, Gaussian", int((own & de2 & (f2.source == "sim")).sum())),
        ("first 2,000 runs: EWMA/capped EWMA/CPM detection cells, t5", int((own & de2 & (f2.source == "robust_t5")).sum())),
        ("first 2,000 runs: CUSUM/capped CUSUM cells (group harness detectors) compared", int(cusum.sum())),
        ("first 2,000 runs: CUSUM/capped CUSUM cells identical", int((cusum & f2.identical).sum())),
        ("first 2,000 runs: FOCuS/capped FOCuS cells (focus_cpt in the independent code) compared", int(focus.sum())),
        ("first 2,000 runs: FOCuS/capped FOCuS cells identical", int((focus & f2.identical).sum())),
        ("matched-rule false-alarm cells (threshold from the independent 1,000 held-out runs vs our 5,000)", int((fa.rule == "oracle").sum())),
        ("matched-rule false-alarm max |z|", float(fa[fa.rule == "oracle"].z.abs().max())),
        ("matched-rule detection cells compared", int((de2 & (f2.rule == "oracle")).sum())),
        ("matched-rule detection cells identical", int((de2 & (f2.rule == "oracle") & f2.identical).sum())),
    ]
    out = pd.DataFrame(summ, columns=["check", "value"])
    out.to_csv(OUT / "crosscheck_ewma_cpm.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(capf[["phi", "value_y", "value_b", "fa_package", "z", "z_vs_package"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
