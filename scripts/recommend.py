"""Table 8.2 of the thesis: recommendations by a stated rule.

Rule: rank set-ups by the evidence that does not depend on the 15 Opal episodes, then use the episodes as a check.
  Simulation   % of normal years with a false alarm, AR(1) threshold, phi = 0.4, holidays in the data, at the
               simulated (noise normal, calendar 1) and the real holiday scale (noise wide)    [sim/summary/fa.csv]
               ✓ if <= 6% at both scales, × otherwise
  Test bed     gain at a matched 20% background for a 20% drop lasting 14 days and a lasting 20% drop
               [testbed/matched.csv]; ✓ if both 95% intervals (3-month blocks) exclude 0, × otherwise
  Opal check   episodes caught of 15, frozen rule and direction-aware rule [opal/rule_variants.csv]:
               ✓ if p (adjusted) < 0.05 in the monitor's family (best of the 137 AR(1)/independent monitors with >= 10
               alarms, rule_variants_familywise.csv; the 40 calm-stretch monitors, budget_familywise.csv), (✓) if only
               p (own) < 0.05, × otherwise
Rows: yearly memory + capped CUSUM; yearly memory + capped FOCuS; row 3 = the past-only set-up with the largest
matched gain for the 20% 14-day drop (printed with its runners-up); Holt-Winters + EWMA (raised by the case study);
same-weekday difference + CUSUM with the calm-stretch threshold; yearly memory + CUSUM with the independent threshold
(not recommended). Footnote values are computed here too.
Outputs: results/recommend.csv (every cell, symbol and source value), results/recommend.tex (the table body,
for tables.py / tables/recommend.tex), results/recommend_row3.csv (the row-3 ranking)."""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
NAME = {"A": "same-weekday difference", "B": "median of four", "HW": "Holt-Winters", "YM": "yearly memory"}
DNAME = {"CUSUM": "CUSUM", "EWMA": "EWMA", "FOCuS": "FOCuS", "CPM": "CPM", "CUSUM-r": "capped CUSUM",
         "EWMA-r": "capped EWMA", "FOCuS-r": "capped FOCuS"}


# The pass rules as text, so the thesis can quote them exactly
RULES = [
    ("Simulation", "passes if the simulated chance of a false alarm in a normal year is at most 6% both at the simulated "
     "and at the real holiday scale (Gaussian AR(1) noise, phi = 0.4, holidays in the data, the set-up's own threshold rule)",
     "sim/summary/fa.csv (noise normal and wide, calendar 1)"),
    ("Test bed", "passes if, at a matched 20% background, the gain for a 20% drop lasting 14 days and for a lasting 20% drop "
     "both have 95% intervals (three-month blocks) above zero", "testbed/matched.csv (temp20, drop20)"),
    ("Opal check", "supports (tick) if the episodes caught under a rule have p (adjusted, best of the monitor's family) "
     "below 0.05; beats chance alone (tick in brackets) if only p (own) is below 0.05; does not support otherwise",
     "opal/rule_variants.csv, rule_variants_familywise.csv; calm stretch: budget.csv, budget_familywise.csv"),
    ("Recommended", "passes both the simulation and the test bed; the Opal episodes are a check only", "recommend.csv passes_both"),
    ("Raised by the case study", "fails the simulation or the test bed but the Opal check supports it after adjustment "
     "under at least one rule; a hypothesis for a forward test", "recommend.csv opal_adjusted_pass"),
    ("Not recommended", "uses the independent threshold, which does not hold its target under dependence "
     "(calibration check: {indep_lo:.0f} to {indep_hi:.0f}% of normal years at phi = 0.4 for CUSUM, EWMA and FOCuS), "
     "whatever the Opal check shows",
     "calib/validation.csv; recommend.csv d28_status"),
    ("Resampled noise (reported, not a pass rule)", "simulated chance of a false alarm in a normal year when the noise is "
     "the residuals of the generator fit to Opal, resampled in 28-day blocks (holidays in the data), the set-up's own threshold rule", "sim/summary/fa.csv (noise boot)"),
    ("Calm-stretch outcome (reported, not a pass rule)", "the same forecast and detector with a constant threshold set by a "
     "block bootstrap of the January 2023 to July 2024 surprises: alarms over March 2020 to September 2026, episodes caught "
     "under each rule, and whether the COVID-19 onset is caught", "opal/budget.csv (budget calm)"),
]


def load():
    return dict(fa=pd.read_csv(R / "sim/summary/fa.csv"), m=pd.read_csv(R / "testbed/matched.csv"),
                rv=pd.read_csv(R / "opal/rule_variants.csv"), fw=pd.read_csv(R / "opal/rule_variants_familywise.csv"),
                sc=pd.read_csv(R / "opal/scores.csv"), bu=pd.read_csv(R / "opal/budget.csv"),
                bf=pd.read_csv(R / "opal/budget_familywise.csv"), bo=pd.read_csv(R / "opal/bootstrap.csv"),
                sv=pd.read_csv(R / "opal/bothrules_survivors.csv"), tb=pd.read_csv(R / "testbed/summary.csv"))


def sim_cells(X, adj, det, rule="ar1"):
    f = X["fa"]
    q = (f.phi == 0.4) & (f.adj == adj) & (f.det == det) & (f.rule == rule) & (f.change == "none")
    s = f[q & (f.noise == "normal") & (f.calendar == 1)].fa.iloc[0]
    w = f[q & (f.noise == "wide")].fa.iloc[0]
    return 100 * s, 100 * w, "✓" if max(s, w) <= 0.06 else "×"


def tb_cells(X, adj, det):
    m = X["m"]
    g = {st: m[(m.adj == adj) & (m.det == det) & (m.setting == st)].iloc[0] for st in ("temp20", "drop20")}
    ok = all(r.ci_lo > 0 or r.ci_hi < 0 for r in g.values())
    return g["temp20"].gain, g["drop20"].gain, "✓" if ok else "×", g


def opal_cells(X, adj, det, rule):
    r = X["rv"].query("method == @adj and detector == @det and rule == @rule").iloc[0]
    fw = X["fw"].set_index("rule_variant")
    out = {}
    for col, fwrow in (("frozen", "frozen"), ("r0_7_dir", "r0_7_dir")):
        k, p_own = int(r[col]), float(r[f"{col}_p"])
        in_family = r.n_alarms >= 10
        p_adj = float(fw.loc[fwrow, f"p_best_episodes_ge{k}"]) if in_family and f"p_best_episodes_ge{k}" in fw.columns else np.nan
        sym = "✓" if (in_family and p_adj < 0.05) else ("(✓)" if p_own < 0.05 else "×")
        out[col] = (k, sym, p_own, p_adj, in_family)
    return out, int(r.n_alarms)


def main():
    X = load()
    # row 3: largest matched gain, 20% for 14 days, past-only set-ups
    m = X["m"]
    r3 = m[(m.setting == "temp20") & m.adj.isin(["A", "B", "HW", "YM"])].sort_values("gain", ascending=False)
    r3[["adj", "det", "gain", "ci_lo", "ci_hi", "background"]].to_csv(R / "recommend_row3.csv", index=False)
    a3, d3 = r3.iloc[0].adj, r3.iloc[0].det
    rows_def = [("From simulation and test bed", "False alarms must be rare", "YM", "CUSUM-r", "ar1"),
                ("From simulation and test bed", "More detection, about one or two alarms a year to check", "YM", "FOCuS-r", "ar1"),
                ("From simulation and test bed", "Drops of one to two weeks, holiday alarms checked by hand", a3, d3, "ar1"),
                ("Raised by the case study", "Short disruptions, someone checks about five alarms a year", "HW", "EWMA", "ar1"),
                ("Raised by the case study", "Fewest false alarms on real data", "A", "CUSUM", "calm"),
                ("Not recommended", "Most real episodes under both rules", "YM", "CUSUM", "indep"),
                # evaluated under the same rule; not in the printed table body (the table is rebuilt by
                # tables.py from recommend.csv)
                ("Evaluated", "Rank-based detector on the simplest forecast", "A", "CPM", "indep")]
    sc, bu, bo, sv = X["sc"], X["bu"], X["bo"], X["sv"]
    out = []
    for group, situation, adj, det, rule in rows_def:
        row = dict(group=group, situation=situation, adj=adj, det=det, rule=rule,
                   setup=f"{NAME[adj]} + {DNAME[det]}" + (", threshold from a calm stretch" if rule == "calm" else
                                                         (", independent threshold" if rule == "indep" else "")))
        if rule == "calm":
            row.update(sim_sym="–", tb_sym="–")
            b = bu[(bu.budget == "calm") & (bu.method == adj) & (bu.detector == det)].iloc[0]
            bf = X["bf"].set_index(["budget", "rule"])
            for col, rname in (("frozen", "frozen"), ("r0_7_dir", "dir")):
                k = int(b[rname])
                p_own = float(b[f"{rname}_p"])
                p_adj = float(bf.loc[("calm", rname), f"p_ge{k}"])
                row[f"opal_{col}"] = k
                row[f"opal_{col}_p_own"], row[f"opal_{col}_p_adj"] = p_own, p_adj
                row[f"opal_{col}_sym"] = "✓" if p_adj < 0.05 else ("(✓)" if p_own < 0.05 else "×")
            row["non_event_per_year"] = float(b.non_event_per_year)
            row["blind_pct"] = 100 * float(bo[(bo.method == adj) & (bo.detector == det)].blind_days_per_year.iloc[0]) / 365.25
            row["n_alarms"] = int(b.n_alarms)
        else:
            if rule == "indep":
                f = X["fa"]
                v = f[(f.phi == 0.4) & (f.noise == "normal") & (f.calendar == 0) & (f.adj == adj) & (f.det == det) & (f.rule == "indep")].fa.iloc[0]
                row.update(sim_no_holidays=100 * v, sim_sym="×")
                s_, w_, _ = sim_cells(X, adj, det, "indep")
                row.update(sim_sim=s_, sim_real=w_)
            else:
                row["sim_sim"], row["sim_real"], row["sim_sym"] = sim_cells(X, adj, det)
            row["tb_temp20"], row["tb_drop20"], row["tb_sym"], g = tb_cells(X, adj, det)
            row.update(tb_temp20_lo=g["temp20"].ci_lo, tb_temp20_hi=g["temp20"].ci_hi, tb_drop20_lo=g["drop20"].ci_lo, tb_drop20_hi=g["drop20"].ci_hi)
            oc, n = opal_cells(X, adj, det, rule)
            for col, (k, sym, p_own, p_adj, fam) in oc.items():
                row.update({f"opal_{col}": k, f"opal_{col}_sym": sym, f"opal_{col}_p_own": p_own, f"opal_{col}_p_adj": p_adj,
                            "in_family": fam})
            s = sc[(sc.method == adj) & (sc.detector == det) & (sc.rule == rule)].iloc[0]
            row.update(non_event_per_year=float(s.non_event_per_year), blind_pct=100 * float(s.blind_days_per_year) / 365.25,
                       n_alarms=int(s.n_alarms))
        # resampled real residuals: % of normal years with a false alarm, same threshold rule
        if rule != "calm":
            f = X["fa"]
            row["sim_boot"] = 100 * f[(f.phi == 0.4) & (f.noise == "boot") & (f.calendar == 1) & (f.adj == adj) & (f.det == det)
                                      & (f.rule == rule) & (f.change == "none")].fa.iloc[0]
        # the same forecast and detector with the calm-stretch threshold (opal/budget.csv, budget 'calm'; 2020-26)
        cb = bu[(bu.budget == "calm") & (bu.method == adj) & (bu.detector == det)]
        if len(cb):
            cb = cb.iloc[0]
            eps = set(str(cb.dir_episodes).split(";")) | set(str(cb.frozen_episodes).split(";"))
            row.update(calm_n_alarms=int(cb.n_alarms), calm_dir=int(cb.dir), calm_frozen=int(cb.frozen),
                       calm_non_event_per_year=float(cb.non_event_per_year), calm_covid_onset="yes" if "covid_onset" in eps else "no")
        # the pass rules, applied (the rule text is in recommend_rules.csv)
        row["passes_sim"] = row.get("sim_sym") == "✓"
        row["passes_tb"] = row.get("tb_sym") == "✓"
        row["passes_both"] = row["passes_sim"] and row["passes_tb"]
        row["opal_adjusted_pass"] = any(row.get(f"opal_{c}_sym") == "✓" for c in ("frozen", "r0_7_dir"))
        row["d28_status"] = ("not recommended (independent threshold)" if rule == "indep" else
                             "recommended (passes simulation and test bed)" if row["passes_both"] else
                             "raised by the case study (Opal check passes after adjustment)" if row["opal_adjusted_pass"] else
                             "does not qualify")
        b2 = sv[(sv.method == adj) & (sv.detector == det) & (sv.rule == (rule if rule != "calm" else "-"))]
        row["holds_both_rules"] = bool(len(b2) and b2.survives.iloc[0])
        row["both_rules_p_adj"] = float(b2.p_adj_circular.iloc[0]) if len(b2) else np.nan
        out.append(row)
    t = pd.DataFrame(out)
    # footnote values
    f = X["fa"]
    fn = dict(
        a_boot_fa=100 * f[(f.noise == "boot") & (f.adj == "YM") & (f.det == "CUSUM-r") & (f.rule == "ar1")].fa.iloc[0],
        b_alarms=int(t.iloc[0].n_alarms),
        c_hw_ewma_no_holidays=100 * f[(f.phi == 0.4) & (f.noise == "normal") & (f.calendar == 0) & (f.adj == "HW") & (f.det == "EWMA") & (f.rule == "ar1")].fa.iloc[0],
        d_hw_ewma_own_gain=float(X["tb"].query("subset == 'all' and setting == 'temp20' and adj == 'HW' and det == 'EWMA' and rule == 'ar1'").delta.iloc[0]),
        d_hw_ewma_own_background=float(X["tb"].query("subset == 'all' and setting == 'temp20' and adj == 'HW' and det == 'EWMA' and rule == 'ar1'").detect_ctl.iloc[0]),
        e_hw_ewma_both_rules_p_adj=float(t.iloc[3].both_rules_p_adj), e_ym_cusum_indep_both_rules_p_adj=float(t.iloc[5].both_rules_p_adj),
        f_calm_frozen_p_adj=float(t.iloc[4].opal_frozen_p_adj), f_calm_dir_p_adj=float(t.iloc[4].opal_r0_7_dir_p_adj))
    t.to_csv(R / "recommend.csv", index=False)
    pd.Series(fn).to_csv(R / "recommend_footnotes.csv", header=["value"])
    v = pd.read_csv(R / "calib/validation.csv")
    v = v[(v.kind == "surprise") & (v.rule == "indep") & (v.phi == 0.4) & v.det.isin(["CUSUM", "EWMA", "FOCuS"])].fa
    rules = [(c, r.format(indep_lo=100 * v.min(), indep_hi=100 * v.max()), src) for c, r, src in RULES]
    pd.DataFrame(rules, columns=["criterion", "rule", "source"]).to_csv(R / "recommend_rules.csv", index=False)
    write_tex(t[~t.group.str.startswith("Evaluated")], fn)
    pd.set_option("display.width", 250)
    print("row 3 =", a3, d3, "; runners-up:", r3.iloc[1:5][["adj", "det", "gain"]].round(2).values.tolist())
    print(t[["setup", "sim_sym", "sim_sim", "sim_real", "tb_sym", "tb_temp20", "tb_drop20", "non_event_per_year", "blind_pct",
             "opal_frozen", "opal_frozen_sym", "opal_r0_7_dir", "opal_r0_7_dir_sym", "holds_both_rules"]].round(2).to_string(index=False))
    print(pd.Series(fn).round(3))


def _sym(s):
    return {"✓": r"\checkmark", "(✓)": r"(\checkmark)", "×": r"$\times$", "–": "--"}[s]


def write_tex(t, fn):
    lines = ["% written by scripts/recommend.py from results/; do not edit"]
    group = None
    for _, r in t.iterrows():
        if r.group != group:
            lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{r.group}}}}} \\")
            group = r.group
        if r.rule == "calm":
            sim, tb = "--", "--"
        elif r.rule == "indep":
            sim = rf"$\times$ {r.sim_no_holidays:.0f} (no holidays)"
            tb = rf"{_sym(r.tb_sym)} {r.tb_temp20:.2f} / {r.tb_drop20:.2f}"
        else:
            sim = rf"{_sym(r.sim_sym)} {r.sim_sim:.1f} / {r.sim_real:.1f}" if r.sim_sim < 10 else rf"{_sym(r.sim_sym)} {r.sim_sim:.0f} / {r.sim_real:.0f}"
            tb = rf"{_sym(r.tb_sym)} {r.tb_temp20:.2f} / {r.tb_drop20:.2f}"
        cells = [r.situation, r.setup, sim, tb, f"{r.non_event_per_year:.1f}", f"{r.blind_pct:.0f}",
                 f"{_sym(r.opal_frozen_sym)} {int(r.opal_frozen)}", f"{_sym(r.opal_r0_7_dir_sym)} {int(r.opal_r0_7_dir)}"]
        lines.append(" & ".join(cells) + r" \\")
    lines.append("% footnotes: " + "; ".join(f"{k} = {v:.3g}" for k, v in fn.items()))
    (R / "recommend.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
