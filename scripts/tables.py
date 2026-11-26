"""LaTeX tables for the thesis, written from result files so that no number is typed by hand.
Usage: python scripts/tables.py [names...]   -> tables/*.tex   (default: all)

Style: booktabs, no vertical rules, one name per method throughout. Captions open with
the table's takeaway; numbers quoted in a caption are computed here from the same files as the cells.
Definitions of the real-data measures live in Section 4.x (\\ref{sec:measures}); captions point there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
SIM = RES / "sim" / "summary"
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)

ANAME = {"A": "A: same-weekday difference", "B": "B: median of four", "HW": "Holt-Winters",
         "YM": "Holt-Winters, yearly memory", "MSTL_p": "MSTL refitted on the past",
         "REG_p": "regression refitted on the past", "STL": "retrospective STL", "MSTL": "retrospective MSTL",
         "REG": "retrospective regression", "raw": "no adjustment", "oracle": "true pattern removed",
         "REGT_p": "regression with temperature, refitted on the past"}
ASHORT = {"A": "A", "B": "B", "HW": "Holt-Winters", "YM": "yearly memory", "MSTL_p": "MSTL (past)",
          "REG_p": "regression (past)", "STL": "retrospective STL", "MSTL": "retrospective MSTL",
          "REG": "retrospective regression", "raw": "no adjustment", "oracle": "true pattern"}
DNAME = {"CUSUM": "CUSUM", "EWMA": "EWMA", "FOCuS": "FOCuS", "FOCuS-AR": "FOCuS-AR", "CPM": "CPM",
         "CUSUM-r": "capped CUSUM", "EWMA-r": "capped EWMA", "FOCuS-r": "capped FOCuS",
         "CUSUM-var": "variance CUSUM", "EWMA-var": "variance EWMA", "FOCuS-var": "variance FOCuS",
         "PELT": "PELT", "BinSeg": "BinSeg", "WBS": "WBS"}


def cap(s):
    return s[:1].upper() + s[1:]


def rhu(x, d=0):
    """Round half up (82.5 -> 83), not Python's round-half-even; a tiny epsilon absorbs float error."""
    f = 10 ** d
    return np.floor(x * f + 0.5 + 1e-9) / f


def pct(x):
    return "--" if pd.isna(x) else f"{rhu(100 * x):.0f}"


def sgn(n):
    n = int(n)
    return "0" if n == 0 else f"+{n}" if n > 0 else f"$-${-n}"


def num(x, d=2):
    s = f"{x:.{d}f}"
    if s.startswith("-0.") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$")


def pval(p):
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def write(name, tex):
    # first line marks the file as generated, so the number lint checks it by freshness
    (OUT / name).write_text("% written by scripts/tables.py from results/; do not edit\n" + tex.replace("\\\\\\\\", "\\\\"))


# ------------------------------------------------------------------------------------------- Opal case study
OPAL_DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
OPAL_METHODS = ["A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG", "raw"]
def H(*lines):
    """A header cell stacked over several lines."""
    return r"\begin{tabular}[b]{@{}c@{}}" + r"\\".join(lines) + r"\end{tabular}"


def opal_head(first, thr=True, budget=False):
    t = " & Thr." if thr else ""
    o = 1 if thr else 0
    nb = 3 if budget else 2
    bh = f" & {H('Budget,', 'dir.')}" if budget else ""
    return (rf" &{' &' if thr else ''} & & \multicolumn{{{nb}}}{{c}}{{Episodes (chance)}} & & & \\" "\n"
            rf"\cmidrule(lr){{{4 + o}-{3 + nb + o}}}" "\n"
            rf"{first}{t} & Alarms & {H('Near, \\%', '(chance)')} & {H('Frozen', 'rule')} & "
            rf"{H('Direction-', 'aware')}{bh} & {H('Non-', 'event')} & {H('Unex-', 'plained')} & {H('Blind', 'days')} \\")


OPAL_NOTE = (r"Near: share of alarms within 7 days of a scored boundary. Episodes: scored episodes caught (of 15) under "
             r"the frozen rule (fixed on 22 October: an alarm within 7 days either side of any boundary, in any "
             r"direction) and under the direction-aware rule (0 to 7 days after the nearest boundary, in its direction; "
             r"23 October). Brackets: chance median; $^{*}$ $p$ (own) $<0.05$. Non-event and unexplained "
             r"alarms and blind days are per year. Definitions: Table~\ref{tab:measures}.")


def _opal_data():
    s = pd.read_csv(RES / "opal" / "scores.csv")
    rv = pd.read_csv(RES / "opal" / "rule_variants.csv")
    m = s.merge(rv[["method", "detector", "rule", "frozen", "frozen_null", "frozen_p",
                    "r0_7_dir", "r0_7_dir_null", "r0_7_dir_p"]], on=["method", "detector", "rule"], how="left")
    assert (m["r0_7_dir"] == m["episodes_caught"]).all(), "rule_variants.csv headline column must match scores.csv"
    return m


def _ep(k, null, p, bold=False):
    v = f"{int(k)}{'$^{*}$' if p < 0.05 else ''}"
    return (f"\\textbf{{{v}}}" if bold else v) + f" ({null:.0f})"


def _opal_cells(q, thr=True, budget=None, lead=None):
    lead = lead or {}
    st = "$^{*}$" if q.p_share_real < 0.05 else ""
    e1 = _ep(q.frozen, q.frozen_null, q.frozen_p, lead.get("frozen") == q.frozen)
    e2 = _ep(q.r0_7_dir, q.r0_7_dir_null, q.r0_7_dir_p, lead.get("dir") == q.r0_7_dir)
    eb = ""
    if budget is not None:
        b = budget.get((q.method, q.detector))
        eb = " & " + (_ep(b.dir, b.dir_null, b.dir_p, lead.get("budget") == b.dir) if b is not None else "--")
    t = f"{'AR(1)' if q.rule == 'ar1' else 'indep.'} & " if thr else ""
    return (f"{t}{int(q.n_alarms)} & {pct(q.share_real)}{st} "
            f"({pct(q.null_share_real)}) & {e1} & {e2}{eb} & {q.non_event_per_year:.1f} & {q.unexplained_per_year:.1f} & "
            f"{q.blind_days_per_year:.0f} \\\\")


PRIMARY = [("Holt-Winters", [("HW", "EWMA", "ar1"), ("HW", "FOCuS", "ar1")]),
           ("Same-weekday difference", [("A", "CUSUM", "ar1"), ("A", "EWMA", "ar1"), ("A", "CUSUM", "indep")]),
           ("Median of four", [("B", "EWMA", "indep"), ("B", "FOCuS-r", "ar1")]),
           ("Capped detectors", [("A", "CUSUM-r", "ar1"), ("A", "FOCuS-r", "ar1")]),
           ("Yearly memory (pipeline forecast)", [("YM", "FOCuS", "ar1"), ("YM", "CUSUM", "ar1"),
                                                  ("YM", "FOCuS-r", "ar1"), ("YM", "CUSUM", "indep")]),
           ("Retrospective and reference", [("STL", "EWMA", "indep"), ("MSTL", "FOCuS", "ar1"), ("A", "CPM", "ar1"),
                                            ("raw", "CUSUM", "ar1")])]


RSHORT = {"STL": "STL (retro.)", "MSTL": "MSTL (retro.)", "REG": "regression (retro.)", "raw": "raw (no forecast)",
          "YM": "yearly mem."}


def _budget(kind="median"):
    b = pd.read_csv(RES / "opal" / "budget.csv")
    b = b[b.budget == kind]
    return {(r.method, r.detector): r for r in b.itertuples()}, b


def _both_rules_survivors():
    """Monitors (best-of-137 family, >= 10 alarms) whose min(direction-aware, frozen) count reaches the smallest k with
    adjusted p <= 0.05 under the both-rules statistic (circular shifts). Cross-checked against the analysis file
    opal/bothrules_survivors.csv when it exists."""
    rv = pd.read_csv(RES / "opal" / "rule_variants.csv")
    br = pd.read_csv(RES / "opal" / "familywise_bothrules.csv")
    br = br[(br["shift"] == "circular") & (br["rules"] == "dir+frozen")].iloc[0]
    kmin = min(int(c[4:]) for c in br.index if c.startswith("p_ge") and br[c] <= 0.05)
    sv = RES / "opal" / "bothrules_survivors.csv"
    if sv.exists():
        a = pd.read_csv(sv)
        a = {(r.method, r.detector, r.rule) for r in a[a.survives].itertuples()}
        rv1 = rv[rv.n_alarms >= 10]
        mine = {(r.method, r.detector, r.rule) for r in rv1[rv1[["r0_7_dir", "frozen"]].min(axis=1) >= kmin].itertuples()}
        assert a == mine, f"survivors differ from the analysis file: {a ^ mine}"
    rv = rv[rv.n_alarms >= 10]
    mn = rv[["r0_7_dir", "frozen"]].min(axis=1)
    surv = rv[mn >= kmin]
    return {(r.method, r.detector, r.rule) for r in surv.itertuples()}, kmin, br


def _row3_pick():
    """Table 8.2 row 3 rule: the past-only monitor with the largest matched-background gain for a 20% 14-day drop."""
    m = pd.read_csv(RES / "testbed" / "matched.csv")
    x = m[(m.setting == "temp20") & m.adj.isin(["A", "B", "HW", "YM", "MSTL_p", "REG_p"])].sort_values("gain", ascending=False)
    return x.iloc[0].adj, x.iloc[0].det


def t_opal_primary():
    """Main text: ten rows by six columns; frozen rule first; dagger = holds under both rules after adjustment.
    Every other monitor is in repository table opal_full."""
    m = _opal_data()
    surv, kmin, _ = _both_rules_survivors()
    r3 = _row3_pick()
    rows_spec = [("HW", "EWMA", "ar1"), ("HW", "FOCuS", "ar1"), ("A", "CUSUM", "ar1"), ("B", "EWMA", "indep"),
                 ("B", "FOCuS-r", "ar1"), (r3[0], r3[1], "ar1"), ("YM", "CUSUM-r", "ar1"), ("YM", "FOCuS-r", "ar1"),
                 ("YM", "FOCuS", "ar1"), ("YM", "CUSUM", "indep")]
    seen, spec = set(), []
    for x in rows_spec:  # the row-3 pick may already be listed
        if x not in seen:
            seen.add(x)
            spec.append(x)
    rows = []
    for meth, d, r in spec:
        q = m[(m.method == meth) & (m.detector == d) & (m.rule == r)].iloc[0]
        lab = f"{cap(ASHORT[meth])} + {DNAME[d]}" + (", indep." if r == "indep" else "")
        lab += (" (pipeline)" if (meth, d, r) == ("YM", "FOCuS", "ar1") else "")
        dag = "$^{\\dagger}$" if (meth, d, r) in surv else ""
        st = "$^{*}$" if q.p_share_real < 0.05 else ""
        rows.append(f"{lab}{dag} & {int(q.n_alarms)} & {pct(q.share_real)}{st} ({pct(q.null_share_real)}) & "
                    f"{_ep(q.frozen, q.frozen_null, q.frozen_p)} & {_ep(q.r0_7_dir, q.r0_7_dir_null, q.r0_7_dir_p)} & "
                    f"{q.non_event_per_year:.1f} & {pct(q.blind_days_per_year / 365)} \\\\")
    tex = (r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Several monitors catch more episodes than chance, but which leads depends on the scoring rule. Episodes of 15, chance median in brackets; $^{*}$ $p$ (own) $<0.05$; $^{\dagger}$ holds under both rules after adjustment for the best of 137 (both counts at least """ +
           f"{kmin}" + r"""). AR(1) threshold unless marked; columns: Table~\ref{tab:measures}; every monitor: Table~\ref{tab:opal-full}.}\label{tab:opal-primary}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & & & \multicolumn{2}{c}{Episodes (chance)} & & \\
\cmidrule(lr){4-5}
Monitor & Alarms & """ + H("Near, \\%", "(chance)") + r""" & """ + H("Frozen", "rule") + r""" & """ + H("Direction-", "aware") +
           r""" & """ + H("Non-event", "a year") + r""" & """ + H("Blind, \\%", "of year") + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("opal_primary.tex", tex)


def t_opal_full():
    m = _opal_data()
    assert set(m.method) == set(OPAL_METHODS) and set(m.detector) <= set(OPAL_DETS), "scores.csv has rows the table omits"
    rows, n = [], 0
    for meth in OPAL_METHODS:
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{cap(ANAME[meth])}}}}} \\\\*")
        for d in OPAL_DETS:
            for r in ["indep", "ar1"]:
                q = m[(m.method == meth) & (m.detector == d) & (m.rule == r)]
                if len(q):
                    rows.append(f"\\quad {DNAME[d]} & " + _opal_cells(q.iloc[0]))
                    n += 1
        rows.append("\\addlinespace")
    assert n == len(m), "every row of scores.csv must appear once"
    head = opal_head("Detector")
    tex = (r"""{\footnotesize\linespread{1}\selectfont\setlength{\tabcolsep}{3.2pt}
\begin{longtable}{@{}lrrrrrrrr@{}}
\caption{Opal case study, 1 March 2020 to 28 September 2026, every forecast, detector and threshold rule (Thr.). """ +
           OPAL_NOTE + r"""}\label{tab:opal-full}\\
\toprule
""" + head + r"""
\midrule
\endfirsthead
\toprule
""" + head + r"""
\midrule
\endhead
""" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{longtable}}\n")
    write("opal_full.tex", tex)


RULES = [("frozen", r"Frozen rule: $\pm$7 d, any boundary"),
         ("pm7_nodir", r"Variant 1: $\pm$7 d, nearest boundary"),
         ("pm7_dir", r"Variant 2: $\pm$7 d, nearest, direction"),
         ("r0_7_nodir", "Variant 3: 0--7 d, nearest"),
         ("r0_7_dir", "Direction-aware rule: 0--7 d, nearest, direction")]
RV_MON = [("A", "CUSUM", "ar1"), ("B", "EWMA", "indep"), ("STL", "EWMA", "indep"), ("YM", "CUSUM", "indep"),
          ("YM", "FOCuS", "indep"), ("STL", "FOCuS-AR", "ar1"),
          ("HW", "EWMA", "ar1"), ("HW", "FOCuS", "ar1"), ("A", "EWMA", "ar1"), ("B", "FOCuS-r", "ar1"), ("YM", "FOCuS", "ar1")]


def t_rule_variants():
    """Table B.1: monitors under the main rule and the later variants, in two stacked
    blocks so the table fits at footnote size; p (own) and p (adjusted, best of the family); era shifts; the
    both-rules statistic in a footer row. Reads whatever rule_variants*.csv the analysis last wrote."""
    rv = pd.read_csv(RES / "opal" / "rule_variants.csv")
    fw = pd.read_csv(RES / "opal" / "rule_variants_familywise.csv").set_index("rule_variant")
    cl = pd.read_csv(RES / "opal" / "chance_local_familywise.csv")
    br = pd.read_csv(RES / "opal" / "familywise_bothrules.csv")
    fam = int(fw.cells.iloc[0])
    lead = {c: rv[c].max() for c, _ in RULES}
    surv, kmin, _ = _both_rules_survivors()
    rmap = {"frozen": "frozen", "pm7_nodir": "orig", "r0_7_dir": "dir"}

    def block(rules, first):
        n = len(rules)
        heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\parbox[b]{{2.7cm}}{{\\centering {lab}}}}}" for _, lab in rules)
        cm = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n))
        out = [(r"\midrule" if not first else r"\toprule"), f" & {heads} \\\\", cm,
               "Monitor" + (" & " + H("Ep.", "(chance)") + " & " + H("$p$", "(adjusted)")) * n + " \\\\", r"\midrule"]
        for meth, d, r in RV_MON:
            q = rv[(rv.method == meth) & (rv.detector == d) & (rv.rule == r)].iloc[0]
            cells = []
            for col, _ in rules:
                k = int(q[col])
                pb = fw.loc[col].get(f"p_best_episodes_ge{k}", np.nan)
                cells += [_ep(k, q[f"{col}_null"], q[f"{col}_p"], k == lead[col]), "--" if pd.isna(pb) else pval(pb)]
            lab = f"{cap(ASHORT[meth])} + {DNAME[d]}" + (", indep." if r == "indep" else "")
            lab += "$^{\\dagger}$" if (meth, d, r) in surv else ""
            out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        out.append("\\addlinespace")
        out.append("Best observed, $p$ (adjusted) & " + " & ".join(
            f"{int(fw.loc[c, 'obs_max_episodes'])} & {pval(fw.loc[c, 'p_episodes'])}" for c, _ in rules) + " \\\\")
        out.append("Chance, 95\\% of the best & " + " & ".join(f"{int(fw.loc[c, 'null_max_episodes_95'])} & " for c, _ in rules) + " \\\\")
        eras = []
        for c, _ in rules:
            x = cl[(cl["shift"] == "era") & (cl["rule"] == rmap.get(c, "-"))]
            eras.append(f"{int(x.null_95.iloc[0])} & {pval(x.p.iloc[0])}" if len(x) else "-- & --")
        out.append("Era shifts: 95\\% of the best, $p$ & " + " & ".join(eras) + " \\\\")
        return out, n

    b1, n1 = block(RULES[:3], True)
    b2, n2 = block(RULES[3:], False)
    pad = " & " * (2 * (n1 - n2))  # the second block has fewer columns
    b2 = [x.replace(" \\\\", pad + " \\\\") if x.endswith(" \\\\") else x for x in b2]
    b3 = []
    nd, fnd, svn = (RES / "opal" / f for f in ("no_dq_2025_01.csv", "familywise_no_dq.csv", "bothrules_survivors_no_dq.csv"))
    nsv = 0
    if nd.exists() and fnd.exists():
        x = pd.read_csv(nd)
        x = x[x.row == "monitor"]
        fx = pd.read_csv(fnd).set_index("rule")
        surv_nd = set()
        if svn.exists():
            y = pd.read_csv(svn)
            surv_nd = {(r.method, r.detector, r.rule) for r in y[y.survives].itertuples()}
            nsv = len(surv_nd)
        heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\parbox[b]{{2.7cm}}{{\\centering {lab}}}}}" for lab in
                           ("frozen rule", "direction-aware rule"))
        b3 = [r"\midrule", f"\\multicolumn{{{1 + 2 * n1}}}{{@{{}}l}}{{\\emph{{Without the 15 January 2025 boundary (data-quality date; 14 episodes)}}}} \\\\",
              f" & {heads}" + " & " * (2 * (n1 - 2)) + " \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
              "Monitor" + (" & " + H("Ep.", "(chance)") + " & " + H("$p$", "(adjusted)")) * 2 + " & " * (2 * (n1 - 2)) + " \\\\",
              r"\midrule"]
        for meth, d, r in RV_MON:
            q = x[(x.method == meth) & (x.detector == d) & (x.rule == r)]
            if not len(q):
                continue
            q = q.iloc[0]
            cells = []
            for col, fr in (("frozen", "frozen"), ("dir", "dir")):
                k = int(q[f"{col}_obs"])
                pb = fx.loc[fr].get(f"p_ge{k}", np.nan)
                cells += [_ep(k, q[f"{col}_null"], q[f"{col}_p"]), "--" if pd.isna(pb) else pval(pb)]
            lab = f"{cap(ASHORT[meth])} + {DNAME[d]}" + (", indep." if r == "indep" else "")
            lab += "$^{\\dagger}$" if (meth, d, r) in surv_nd else ""
            b3.append(f"{lab} & " + " & ".join(cells) + " & " * (2 * (n1 - 2)) + " \\\\")
    bf = br[(br["shift"] == "circular") & (br["rules"] == "dir+frozen")].iloc[0]
    bfe = br[(br["shift"] == "era") & (br["rules"] == "dir+frozen")]
    both = (f"$^{{\\dagger}}$ Holds under both the frozen and the direction-aware rule after adjustment: at least {kmin} episodes under each; best "
            f"{int(bf.obs_max_min)}, $p$ (adjusted) {pval(bf.p)}" + (f" (era shifts {pval(bfe.p.iloc[0])})" if len(bfe) else "")
            + (f"; without 15 January 2025 the threshold is lower and {nsv} monitors hold" if nsv else ""))
    tex = (r"""{\footnotesize\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}l""" + "rr" * n1 + r"""@{}}
\caption{Episodes caught under the frozen rule, three variants and the direction-aware rule: no single monitor leads under every rule. Variant 3 was the first change after the first scored run; variants 1 and 2 change one condition at a time. Ep.: episodes of 15 (chance median; $^{*}$ $p$ (own) $<0.05$; bold: the most of all 150 monitors). $p$ (adjusted): chance that the best of the """ +
           f"{fam}" + r""" monitors with at least ten alarms catches as many when all alarm lists are shifted by one random offset; era shifts stay within 2020--22 or 2023--26. AR(1) threshold unless marked.}\label{tab:rule-variants}\\
""" + "\n".join(b1) + "\n" + "\n".join(b2) + "\n" + "\n".join(b3) + r"""
\addlinespace
\multicolumn{""" + f"{1 + 2 * n1}" + r"""}{@{}p{14cm}}{""" + both + r""".} \\
\bottomrule
\end{longtable}}
""")
    write("rule_variants.tex", tex)


def t_opal_budget():
    """App C: episodes caught when every monitor raises the same number of alarms (median and lower-quartile
    budgets) and with thresholds set on a calm stretch of real data."""
    b = pd.read_csv(RES / "opal" / "budget.csv")
    fw = pd.read_csv(RES / "opal" / "budget_familywise.csv").set_index(["budget", "rule"])
    kinds = ["median", "quartile", "calm"]
    tgt = {k: b[b.budget == k] for k in kinds}
    pick = {("HW", "EWMA"), ("A", "CUSUM")}   # the monitors named in the text, plus every column's leaders
    for k in kinds:
        x = tgt[k]
        pick |= {(r.method, r.detector) for r in x[x.dir == x.dir.max()].itertuples()}
    order = {m: i for i, m in enumerate(OPAL_METHODS)}
    pick = sorted(pick, key=lambda md: (-tgt["median"].set_index(["method", "detector"]).dir.get(md, -1),
                                        order[md[0]], OPAL_DETS.index(md[1])))
    lead = {(k, c): tgt[k][c].max() for k in kinds for c in ("dir", "frozen")}
    rows = []
    for meth, d in pick:
        cells = []
        for k in kinds:
            q = tgt[k][(tgt[k].method == meth) & (tgt[k].detector == d)]
            if not len(q):
                cells += ["--", "--"] + (["--"] if k == "calm" else [])
                continue
            q = q.iloc[0]
            if k == "calm":
                cells.append(f"{int(q.n_alarms)}")
            cells += [_ep(q.dir, q.dir_null, q.dir_p, q.dir == lead[(k, "dir")]),
                      _ep(q.frozen, q.frozen_null, q.frozen_p, q.frozen == lead[(k, "frozen")])]
        lab = f"{RSHORT.get(meth, ASHORT[meth])} + {DNAME[d]}"
        rows.append(f"{cap(lab)} & " + " & ".join(cells) + " \\\\")
    def fwrow(what):
        out = []
        for k in kinds:
            cells = [""] * (3 if k == "calm" else 2)
            for j, c in enumerate(("dir", "frozen")):
                x = fw.loc[(k, c)] if (k, c) in fw.index else None
                cells[(1 if k == "calm" else 0) + j] = ("--" if x is None else
                                                        f"{int(x.obs_max)}" if what == "max" else pval(x.p))
            out += cells
        return " & ".join(out)
    rng = {k: (int(tgt[k].n_alarms.min()), int(tgt[k].n_alarms.max())) for k in kinds}
    head1 = (f"\\multicolumn{{2}}{{c}}{{{int(tgt['median'].target.iloc[0])} alarms}} & "
             f"\\multicolumn{{2}}{{c}}{{{int(tgt['quartile'].target.iloc[0])} alarms}} & "
             f"\\multicolumn{{3}}{{c}}{{Calm-stretch threshold}}")
    tex = (r"""\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\caption{At a matched alarm budget many monitors catch six or seven of 15 episodes and no forecast or detector stands clearly apart; with calm-stretch thresholds the same-weekday difference with CUSUM leads. Each monitor gets the constant threshold that gives the budget (achieved """ +
           f"{rng['median'][0]}--{rng['median'][1]} and {rng['quartile'][0]}--{rng['quartile'][1]} alarms" +
           r"""), or the threshold set on the calm stretch (past-only forecasts only). Dir.: direction-aware rule; frozen: frozen rule; chance median in brackets; $^{*}$ $p$ (own) $<0.05$; bold: the most in the column. Rows: each column's leaders and the monitors named in the text. Best of family: the most episodes of any monitor and its $p$ (adjusted, best of 80; calm: 40 monitors).}\label{tab:opal-budget}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & """ + head1 + r""" \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}
Monitor & Dir. & Frozen & Dir. & Frozen & Alarms & Dir. & Frozen \\
\midrule
""" + "\n".join(rows) + r"""
\addlinespace
Best of family & """ + fwrow("max") + r""" \\
$p$ (adjusted) & """ + fwrow("p") + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")
    write("opal_budget.tex", tex)


SENS_MON = [("HW", "FOCuS"), ("HW", "EWMA"), ("A", "CUSUM")]
WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def t_opal_sens():
    """Section 6.7 (tab:opal-sens): episodes caught under the false-alarm target, window and relearn variants,
    and the missed episodes that fell in a relearn period. Main settings (5%, 7 days, 28-day relearn) in bold."""
    se = pd.read_csv(RES / "opal" / "sensitivity.csv")
    sc = pd.read_csv(RES / "opal" / "scores.csv")
    rv = pd.read_csv(RES / "opal" / "rule_variants.csv")
    bm = pd.read_csv(RES / "opal" / "blind_misses.csv")
    pick = lambda df, m, d, **kw: df[(df.method == m) & (df.detector == d)  # noqa: E731
                                     & np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
    rows, a28, moves = [], [], []
    for m, d in SENS_MON:
        s, r = pick(sc, m, d, rule="ar1"), pick(rv, m, d, rule="ar1")
        b = pick(bm, m, d, rule="ar1")
        f1, f10, rl = (pick(se, m, d, variant=v) for v in ("fa=0.01", "fa=0.1", "relearn=14"))
        main = int(s.episodes_caught)
        assert main == int(r.r0_7_dir) and int(b.missed) == 15 - main and int(b.n_alarms) == int(s.n_alarms)
        others = [int(f1.episodes_caught), int(f10.episodes_caught), int(r.r0_3_dir), int(r.r0_14_dir),
                  int(rl.episodes_caught)]
        moves += [abs(x - main) for x in others if x != main]
        a28.append(int(s.n_alarms))
        rows.append(f"{cap(ASHORT[m])} + {DNAME[d]} & {others[0]} & \\textbf{{{main}}} & {others[1]} & {others[2]} & "
                    f"\\textbf{{{main}}} & {others[3]} & {others[4]} ({int(rl.n_alarms)}) & "
                    f"{int(b.missed_wholly_blind)} of {int(b.missed)} & {int(b.missed_partly_blind)} of {int(b.missed)} \\\\")
    tex = (r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Episodes caught move by up to """ + f"{WORD[max(moves)]}" +
           r""" with the false-alarm target, the window after a boundary and the relearn length (of 15, direction-aware rule, AR(1) threshold; main settings in bold). Relearn 14 days: episodes, with alarms in brackets; at 28 days the alarms are """ +
           f"{a28[0]}, {a28[1]} and {a28[2]}" +
           r""". Missed in relearn: missed episodes whose whole 0--7-day window fell in a relearn period, and those where part of it did.}\label{tab:opal-sens}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Target a year} & \multicolumn{3}{c}{Window, days} & Relearn & \multicolumn{2}{c}{Missed in relearn} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){9-10}
Monitor & 1\% & \textbf{5\%} & 10\% & 3 & \textbf{7} & 14 & 14 days & wholly & partly \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("opal_sens.tex", tex)


# ------------------------------------------------------------------------------------------- test bed
TB_DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "FOCuS-r"]
TB_ADJS = ["A", "B", "HW", "YM", "REG_p", "STL", "REG"]
TB_SETS = {"drop2.5": "$-$2.5\\%", "drop5": "$-$5\\%", "drop10": "$-$10\\%", "drop20": "$-$20\\%", "drop40": "$-$40\\%",
           "temp5": "$-$5\\%, 14 d", "temp10": "$-$10\\%, 14 d", "temp20": "$-$20\\%, 14 d", "temp40": "$-$40\\%, 14 d",
           "spread1.5": "var.\\ $\\times1.5$", "spread2": "var.\\ $\\times2$"}
TB_HEAD = (r" & \multicolumn{5}{c}{Uncapped} & \multicolumn{2}{c}{Capped} \\" "\n"
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}" "\n"
           r"{} & CUSUM & EWMA & FOCuS & FOCuS-AR & CPM & CUSUM & FOCuS \\")


def _tb(s):
    s = s[(s.subset == "all") & (s.rule == "ar1")]
    return s


def _sig(q):
    return q.ci_lo > 0 or q.ci_hi < 0


def _gain_cells(qs):
    """Gain cells: bold = largest gain in the row (ties, at the printed precision, all bold); * = interval excludes 0."""
    best = max(round(q.delta, 2) for q in qs)
    out = []
    for q in qs:
        v = num(q.delta) + ("$^{*}$" if _sig(q) else "")
        out.append(f"\\textbf{{{v}}}" if round(q.delta, 2) == best else v)
    return out


def _stars_kept(s3, s6, adjs, sets):
    """How many cells starred with 3-month blocks keep their star with 6-month blocks."""
    k = lambda df: df.set_index(["adj", "setting", "det"])  # noqa: E731
    a, b = k(s3), k(s6)
    idx = [(x, y, d) for x in adjs for y in sets for d in TB_DETS if (x, y, d) in a.index]
    st3 = [i for i in idx if _sig(a.loc[i])]
    kept = [i for i in st3 if i in b.index and _sig(b.loc[i])]
    return len(kept), len(st3), len(idx)


def t_testbed():
    """Main text: A, B, Holt-Winters and the yearly memory; background and gain for three changes.
    Clean-window detection is in repository table testbed_full."""
    s = _tb(pd.read_csv(RES / "testbed" / "summary.csv"))
    s6p = RES / "testbed" / "summary_6m.csv"
    get = lambda a, k, d: s[(s.adj == a) & (s.setting == k) & (s.det == d)].iloc[0]  # noqa: E731
    adjs, sets = ["A", "B", "HW", "YM"], ["drop20", "drop40", "temp20"]
    rows, half = [], []
    for i, a in enumerate(adjs):
        rows.append(("\\addlinespace\n" if i else "") + f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{cap(ANAME[a])}}}}} \\\\*")
        rows.append("\\quad Background & " +
                    " & ".join(f"{get(a, 'drop40', d).detect_ctl:.2f}" for d in TB_DETS) + " \\\\*")
        for k, lab in zip(sets, ["Gain, 20\\% lasting", "Gain, 40\\% lasting", "Gain, 20\\% for 14 days"]):
            qs = [get(a, k, d) for d in TB_DETS]
            half += [(q.ci_hi - q.ci_lo) / 2 for q in qs]
            rows.append(f"\\quad {lab} & " + " & ".join(_gain_cells(qs)) + " \\\\")
    half = np.array(half)
    nb = int(s.n_blocks.iloc[0]) if "n_blocks" in s.columns else 12
    six = ""
    if s6p.exists():
        kept, st3, _ = _stars_kept(s, _tb(pd.read_csv(s6p)), adjs, sets)
        six = f"; with 6-month blocks {kept} of the {st3} stars remain"
    mt = pd.read_csv(RES / "testbed" / "matched.csv").set_index(["setting", "adj", "det"])
    mg = pd.Series([mt.loc[("drop40", a, "FOCuS-r")].gain - mt.loc[("drop40", a, "FOCuS")].gain for a in adjs])
    unc = s[s.det.isin(["CUSUM", "EWMA", "FOCuS"]) & s.adj.isin(["A", "B", "HW"])].detect_ctl
    cpd = s[s.det.isin(["CUSUM-r", "FOCuS-r"]) & s.adj.isin(["A", "B", "HW"])].detect_ctl
    tex = (r"""\begin{table}[t]\centering\footnotesize\linespread{1}\selectfont\setlength{\tabcolsep}{3.4pt}
\caption{Capped detectors gain the most for larger drops. They also alarm less often in background windows (""" +
           f"{cpd.min():.2f}--{cpd.max():.2f} against {unc.min():.2f}--{unc.max():.2f}" +
           r""" on A, B and Holt-Winters), but at a matched 20\% background capped FOCuS beats FOCuS by as much or more (""" +
           f"{mg.min():.2f}--{mg.max():.2f}" + r""" for a 40\% drop, Section~\ref{sec:5-testbed}). Background: alarm within 60 days with nothing inserted; gain: extra chance with an inserted change; bold: largest in the row; $^{*}$ approximate 95\% interval excludes zero. All settings: Table~\ref{tab:testbed-full}.}\label{tab:testbed}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
""" + TB_HEAD + r"""
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("testbed.tex", tex)
    print(f"  testbed main: CI half-width median {np.median(half):.3f}{six}")


def t_testbed_full():
    """Repository table testbed_full: every adjustment and setting; background and clean-window rows; the same conventions as Table 6.2."""
    s = _tb(pd.read_csv(RES / "testbed" / "summary.csv"))
    s6p = RES / "testbed" / "summary_6m.csv"
    rows, half = [], []
    for i, a in enumerate(TB_ADJS):
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{cap(ANAME[a])}}}}} \\\\*")
        bg = s[(s.adj == a) & (s.setting == "drop40")].set_index("det")
        rows.append("\\quad Background & " + " & ".join(f"{bg.loc[d].detect_ctl:.2f}" if d in bg.index else "--"
                                                         for d in TB_DETS) + " \\\\*")
        for k in TB_SETS:
            qs = [s[(s.adj == a) & (s.det == d) & (s.setting == k)] for d in TB_DETS]
            have = [q.iloc[0] for q in qs if len(q)]
            half += [(q.ci_hi - q.ci_lo) / 2 for q in have]
            cells = iter(_gain_cells(have)) if have else iter([])
            rows.append(f"\\quad {TB_SETS[k]} & " + " & ".join(next(cells) if len(q) else "--" for q in qs) + " \\\\")
        cl = [s[(s.adj == a) & (s.det == d) & (s.setting == x)] for d in TB_DETS for x in ("drop20", "drop40")]
        cc = [f"{cl[2 * j].iloc[0].detect_clean:.2f}/{cl[2 * j + 1].iloc[0].detect_clean:.2f}"
              if len(cl[2 * j]) and len(cl[2 * j + 1]) else "--" for j in range(len(TB_DETS))]
        rows.append("\\quad Clean, $-$20\\%/$-$40\\% & " + " & ".join(cc) + " \\\\")
        rows.append("\\addlinespace")
    half = np.array(half)
    six = ""
    if s6p.exists():
        kept, st3, _ = _stars_kept(s, _tb(pd.read_csv(s6p)), TB_ADJS, list(TB_SETS))
        six = f"; with 6-month blocks {kept} of the {st3} stars remain"
    head = TB_HEAD.replace("{} &", "Change &")
    tex = (r"""{\footnotesize\linespread{1}\selectfont\setlength{\tabcolsep}{3.6pt}
\begin{longtable}{@{}lrrrrrrr@{}}
\caption{Real-data test bed, all settings (AR(1) thresholds; a drop, lasting unless marked 14 d, or the variance multiplied by 1.5 or 2). Background, gain, bold and $^{*}$ as in Table~\ref{tab:testbed}. Clean: detection of a 20\% / 40\% lasting drop among windows with no background alarm (a different subset for each monitor, so indicative).}\label{tab:testbed-full}\\
\toprule
""" + head + r"""
\midrule
\endfirsthead
\toprule
""" + head + r"""
\midrule
\endhead
""" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{longtable}}\n")
    write("testbed_full.tex", tex)
    print(f"  testbed full{six}")


# ------------------------------------------------------------------------------------------- simulation
DELAY_DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
DELAY_COLS = [("true pattern, no holidays", 1.0), ("true pattern, no holidays", 2.0), ("true pattern, no holidays", 4.0),
              ("Holt-Winters, no holidays", 2.0), ("Holt-Winters, no holidays", 4.0), ("yearly memory, holidays", 4.0)]


def t_delay():
    """Restricted mean delay and share found within 7 days, matched threshold, phi = 0.4."""
    d = pd.read_csv(SIM / "delay.csv")
    d = d[(d.rule == "oracle") & np.isclose(d.phi, 0.4) & (d.noise == "normal")]
    g = lambda st, sd, det: d[(d.setting == st) & np.isclose(d.shift_sd, sd) & (d.det == det)].iloc[0]  # noqa: E731
    rows = []
    for det in DELAY_DETS:
        cells = []
        for st, sd in DELAY_COLS:
            q = g(st, sd, det)
            cells += [f"{q.rmd:.1f}", f"{q.F7:.2f}"]
        rows.append(f"{cap(DNAME[det])} & " + " & ".join(cells) + " \\\\")
        if det == "CPM":
            rows.append("\\addlinespace")
    groups = [("True pattern removed", 3), ("Holt-Winters", 2), (H("Yearly", "memory,", "holidays"), 1)]
    h1 = " & ".join(f"\\multicolumn{{{2 * n}}}{{c}}{{{t}}}" for t, n in groups)
    h2 = " & ".join(f"\\multicolumn{{2}}{{c}}{{{sd:.0f} SD}}" for _, sd in DELAY_COLS)
    c1 = "\\cmidrule(lr){2-7}\\cmidrule(lr){8-11}\\cmidrule(lr){12-13}"
    c2 = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DELAY_COLS)))
    four = ("true pattern, no holidays", 4.0)
    fa, f, e, c = (g(*four, x) for x in ("FOCuS-AR", "FOCuS", "EWMA", "CUSUM"))
    lead = sorted(DELAY_DETS, key=lambda x: g(*four, x).rmd)[:3]
    assert set(lead) == {"FOCuS-AR", "FOCuS", "EWMA"}, f"caption names the three fastest at 4 SD: {lead}"
    one = ("true pattern, no holidays", 1.0)
    b1 = sorted(DELAY_DETS, key=lambda x: g(*one, x).rmd)[:2]
    tex = (r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.9pt}
\caption{Detection speed in simulation ($\phi=0.4$, matched 5\% false-alarm rate): FOCuS-AR, FOCuS and EWMA find a 4 SD drop fastest (""" +
           f"{fa.rmd:.1f}, {f.rmd:.1f} and {e.rmd:.1f} days against {c.rmd:.1f} for CUSUM), {DNAME[b1[0]]} and {DNAME[b1[1]]} a 1 SD drop ({g(*one, b1[0]).rmd:.1f} and {g(*one, b1[1]).rmd:.1f})" +
           r"""; CPM rarely alarms within a week. RMD: restricted mean delay in days (not found within 60 days counts as 61); F(7): share found within 7 days. 10{,}000 runs per cell.}\label{tab:delay}
\begin{tabular}{@{}l""" + "rr" * len(DELAY_COLS) + r"""@{}}
\toprule
 & """ + h1 + r""" \\
""" + c1 + "\n & " + h2 + r""" \\
""" + c2 + "\nDetector" + " & RMD & F(7)" * len(DELAY_COLS) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("delay.tex", tex)


VAR_DETS = ["CUSUM-var", "EWMA-var", "FOCuS-var", "CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM"]
VAR_COLS = [("oracle", 1, 1.5, "normal", "detect60"), ("oracle", 1, 2.0, "normal", "detect60"),
            ("oracle", 1, 3.0, "normal", "detect60"), ("oracle", 1, 3.0, "normal", "F14"),
            ("oracle", 1, 2.0, "t5", "detect60"), ("oracle", 0, 2.0, "normal", "detect60"), ("YM", 0, 2.0, "normal", "detect60")]


def t_variance():
    """Detection of variance increases (within 60 days; one column within 14 days),
    matched threshold, phi = 0.4."""
    v0 = pd.read_csv(SIM / "variance.csv")
    v0 = v0[(v0.rule == "oracle") & np.isclose(v0.phi, 0.4)]
    g = lambda adj, cal, size, det, noise="normal": v0[(v0.adj == adj) & (v0.calendar == cal) & np.isclose(v0["size"], size)  # noqa: E731
                                                       & (v0.det == det) & (v0.noise == noise)]
    cell = {}
    rows = []
    for det in VAR_DETS:
        cells = []
        for adj, cal, size, noise, col in VAR_COLS:
            q = g(adj, cal, size, det, noise)
            cells.append(pct(q[col].iloc[0]) if len(q) else "--")
            cell[(det, adj, cal, size, noise, col)] = rhu(100 * q[col].iloc[0]) if len(q) else np.nan
        lab = DNAME[det] if det.endswith("-var") else f"{det} (mean)"
        rows.append(f"{cap(lab)} & " + " & ".join(cells) + " \\\\")
        if det == "FOCuS-var":
            rows.append("\\addlinespace")
    # ratios for a doubling of the variance without holidays (true pattern removed, Gaussian, within 60 days)
    d0 = lambda det: g("oracle", 0, 2.0, det).detect60.iloc[0]  # noqa: E731
    spread = ["CUSUM-var", "EWMA-var", "FOCuS-var"]
    r1 = [d0(a) / d0(m) for a in spread for m in ("CUSUM", "EWMA", "FOCuS")]
    r2 = [d0(a) / d0("FOCuS-AR") for a in spread]
    ym_hol = v0[(v0.noise == "normal") & (v0.adj == "YM") & (v0.calendar == 1) & v0.det.isin(VAR_DETS)].detect60.max()
    tex = (r"""\begin{table}[t]\centering\small\setlength{\tabcolsep}{4.2pt}
\caption{Detectors built for spread find a doubling of the variance """ +
           f"{rhu(min(r1), 1):.1f} to {rhu(max(r1), 1):.1f} times as often as CUSUM, EWMA and FOCuS, and {rhu(min(r2), 1):.1f} to {rhu(max(r2), 1):.1f} times FOCuS-AR" +
           r"""; under $t_5$ noise the advantage shrinks. Detection (\%), $\phi=0.4$, matched 5\% rate; after the yearly memory with holidays every detector finds at most """ +
           f"{100 * ym_hol:.1f}" + r"""\%.}\label{tab:variance}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & \multicolumn{5}{c}{Holidays removed exactly} & \multicolumn{2}{c}{\begin{tabular}[b]{@{}c@{}}No holidays,\\variance $\times2$\end{tabular}} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Detector & $\times1.5$ & $\times2$ & $\times3$ & """ + H("$\\times3$, within", "14 days") + r""" & $\times2$, $t_5$ & \begin{tabular}[b]{@{}c@{}}True\\pattern\end{tabular} & \begin{tabular}[b]{@{}c@{}}Yearly\\memory\end{tabular} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("variance.tex", tex)


def t_validation():
    v = pd.read_csv(RES / "calib" / "validation.csv")
    order = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r", "CUSUM-var", "EWMA-var", "FOCuS-var"]
    v = v[(v.kind == "surprise") & v.det.isin(order)]
    p = v.pivot_table(index="det", columns=["rule", "phi"], values="fa")
    phis = [0.0, 0.2, 0.4, 0.6, 0.8]
    rows = []
    for d in [x for x in order if x in p.index]:
        a = " & ".join(f"{100 * p.loc[d, ('indep', f)]:.1f}" for f in phis)
        b = " & ".join(f"{100 * p.loc[d, ('ar1', f)]:.1f}" for f in phis)
        rows.append(f"{cap(DNAME[d])} & {a} & {b} \\\\")
        if d in ("CPM", "FOCuS-r"):
            rows.append("\\addlinespace")
    reps = int(v.reps.iloc[0])
    lo = 100 * v[v.rule == "ar1"].fa.min()
    hi = 100 * v[v.rule == "ar1"].fa.max()
    tex = (r"""\begin{table}[t]\centering\small\setlength{\tabcolsep}{4.5pt}
\caption{The AR(1) threshold holds the 5\% target under dependence (""" + f"{lo:.1f} to {hi:.1f}\\%" +
           r""" for every detector and $\phi$) and the independent threshold does not. Realised chance (\%) of at least one false alarm in a year of stationary AR(1) data, on fresh seeds (""" +
           f"{reps:,}".replace(",", "{,}") + r""" years per cell; Monte Carlo SE about 0.3 points at 5\%). The independent threshold assumes $\phi=0$; the AR(1) threshold uses $\hat\phi$ estimated from the previous year, as the monitor does.}\label{tab:validation}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & \multicolumn{5}{c}{Independent, true $\phi$} & \multicolumn{5}{c}{AR(1), true $\phi$} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Detector & 0 & 0.2 & 0.4 & 0.6 & 0.8 & 0 & 0.2 & 0.4 & 0.6 & 0.8 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("validation.tex", tex)


def t_runtime():
    r = pd.read_csv(RES / "runtime.csv")
    sp = RES / "runtime_spread.csv"   # re-timed spread detectors (variance FOCuS fixed) replace the rows of runtime.csv
    ev = RES / "runtime_ewma_var.csv"
    for f in ([sp] if sp.exists() else []) + ([ev] if ev.exists() else []):
        x = pd.read_csv(f)
        x = x[~x.method.isin(set(r.method) & set(x.method)) | (f == sp)]
        r = pd.concat([r[~r.method.isin(x.method)], x], ignore_index=True)
    p = r.pivot_table(index="method", columns="n", values="ms")
    order = [m for m in ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-var", "EWMA-var", "FOCuS-var", "PELT", "BinSeg", "WBS"]
             if m in p.index]

    def us(ms):
        v = 1000 * ms  # microseconds, at least two significant digits
        return f"{v:.1f}" if v < 10 else f"{v:,.0f}".replace(",", "{,}")
    rows = []
    for m in order:
        rows.append(f"{cap(DNAME[m])} & " + " & ".join(us(p.loc[m, n]) for n in (365, 1095, 2463)) + " \\\\")
        if m == "FOCuS-var":
            rows.append("\\addlinespace")
    tex = (r"""\begin{table}[t]\centering\small
\caption{Every method is fast enough to run daily: CUSUM, EWMA and their variance versions take about a microsecond for a year of data, and the slowest method takes """ + f"{p[2463].max():.0f}" + r"""\,ms for the whole series. Computing time in microseconds ($\mu$s; 1{,}000\,$\mu$s = 1\,ms) for one run on $n$ daily observations (one thread, after compilation and a warm-up; median of 51 repeats, each timing a batch of calls lasting at least 2\,ms; Apple M4). Online detectors compute the statistic for every day (FOCuS and FOCuS-AR by direct search, quadratic in $n$); offline methods segment the whole series.}\label{tab:runtime}
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & $n=365$ & $n=1{,}095$ & $n=2{,}463$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("runtime.tex", tex)


def t_retro_extra():
    """Further retrospective adjustments, scored with the same monitor."""
    t = pd.read_csv(RES / "opal" / "retro_extra.csv")
    t = t[t.rule == "ar1"]
    rows = []
    for m, g in t.groupby("method", sort=False):
        cells = []
        for d in ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM"]:
            q = g[g.detector == d].iloc[0]
            cells.append(f"{int(q.near_real)}/{int(q.n_alarms)} ({int(q.episodes_caught)})")
        rows.append(f"{cap(m)} & " + " & ".join(cells) + " \\\\")
    best = int(t.episodes_caught.max())
    tex = (r"""\begin{table}[ht]\centering\small\setlength{\tabcolsep}{4pt}
\caption{Our further retrospective seasonal adjustments, scored with the same monitor (AR(1) threshold), catch at most """ +
           f"{best}" + r""" episodes. Cells: alarms within 7 days of a scored boundary / all alarms (episodes caught of 15, direction-aware rule).}\label{tab:retro-extra}
\begin{tabular}{@{}lccccc@{}}
\toprule
Adjustment & CUSUM & EWMA & FOCuS & FOCuS-AR & CPM \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("retro_extra.tex", tex)


VO_DETS = ["CUSUM-var", "EWMA-var", "FOCuS-var"]


def t_opal_variance():
    """App C: the variance detectors on Opal (AR(1)) and in the test bed (variance x2). Outside the
    best-of-136 family; direction-free rules only."""
    vs = pd.read_csv(RES / "opal" / "variance_scores.csv")
    vm = pd.read_csv(RES / "opal" / "variance_vs_mean.csv")
    sp = pd.read_csv(RES / "testbed" / "spread_summary.csv")
    vs, sp = vs[vs.rule == "ar1"], sp[(sp.rule == "ar1") & (sp.subset == "all") & (sp.setting == "spread2")]
    summ = vm[(vm.kind == "summary") & (vm.rule == "ar1")].set_index(["method", "detector"])
    rows = []
    for i, a in enumerate(["A", "B", "HW", "YM"]):
        rows.append(("\\addlinespace\n" if i else "") + f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{cap(ANAME[a])}}}}} \\\\*")
        for d in VO_DETS:
            q = vs[(vs.method == a) & (vs.detector == d)].iloc[0]
            st = lambda p: "$^{*}$" if p < 0.05 else ""  # noqa: E731
            w1 = pct(summ.loc[(a, d)].share_within_1d_of_mean) if (a, d) in summ.index else "--"
            sch = pct(q.n_school / q.n_alarms)
            t = sp[(sp.adj == a) & (sp.det == d)]
            tb = (f"{num(t.delta.iloc[0])}{'$^{*}$' if _sig(t.iloc[0]) else ''} & {t.detect_ctl.iloc[0]:.2f}"
                  if len(t) else "-- & --")
            rows.append(f"\\quad {cap(DNAME[d])} & {int(q.n_alarms)} & {q.non_event_per_year:.1f} & {sch} & {w1} & "
                        f"{int(q.frozen)}{st(q.frozen_p)} ({q.frozen_null:.0f}) & "
                        f"{int(q.r0_7_nodir)}{st(q.r0_7_nodir_p)} ({q.r0_7_nodir_null:.0f}) & {tb} \\\\")
    mean_best = pd.read_csv(RES / "testbed" / "summary.csv")
    mean_best = mean_best[(mean_best.subset == "all") & (mean_best.rule == "ar1") & (mean_best.setting == "spread2")
                          & mean_best.adj.isin(["A", "B", "HW", "YM"])].delta.max()
    w = vm[(vm.kind == "summary") & (vm.rule == "ar1")].share_within_1d_of_mean
    tex = (r"""\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{3.6pt}
\caption{The variance detectors on Opal (AR(1) threshold) mostly alarm within a day of a mean alarm (""" +
           f"{pct(w.min())}--{pct(w.max())}" + r"""\%), and in the test bed they find a doubled spread more often than the best mean detector (gain up to """ +
           f"{num(sp[sp.det.isin(VO_DETS)].delta.max())} against {num(mean_best)}" + r""") against high backgrounds. School: \% of alarms in school holidays; near mean: \% within a day of a CUSUM or FOCuS alarm (Holt-Winters and yearly memory only; -- elsewhere). Episodes: frozen rule and 0--7 days, any direction (chance; $^{*}$ $p$ (own) $<0.05$; outside the best-of-family adjustment).}\label{tab:opal-variance}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & & & & & \multicolumn{2}{c}{Episodes (chance)} & \multicolumn{2}{c}{Test bed, $\times2$} \\
\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Detector & Alarms & """ + H("Non-", "event") + r""" & """ + H("School,", r"\%") + r""" & """ + H("Near", r"mean, \%") + r""" & """ +
           H("Frozen", "rule") + r""" & """ + H("0 to 7 days,", "any dir.") + r""" & Gain & """ + H("Back-", "ground") + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("opal_variance.tex", tex)


def t_sim_appendix():
    """Two sub-columns per detector; blocks for phi = 0 and 0.6 and for holidays at their real size
    (noise "wide", phi = 0.4); drop sizes labelled in % of daily tap-ons."""
    import json
    g = json.loads((RES / "sim" / "generator.json").read_text())
    fa = pd.read_csv(SIM / "fa.csv")
    det = pd.read_csv(SIM / "detect.csv")
    det = det[(det.rule == "oracle") & (det.change == "mean")]
    adjs = ["oracle", "A", "B", "HW", "YM", "raw"]
    dets = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM"]
    drop = lambda sd: 100 * (1 - np.exp(-2 * sd))  # noqa: E731  (2 SD lasting drop, % of daily tap-ons)
    blocks = []
    conds = [(0.0, 0, "normal"), (0.0, 1, "normal"), (0.6, 0, "normal"), (0.6, 1, "normal"), (0.4, 1, "wide")]
    for phi, cal, noise in conds:
        sd = g["sd_raw"] if noise == "wide" else g["sigma"]
        what = ("holidays at their real size (noise SD " + f"{sd:.3f}" + ")" if noise == "wide"
                else ("holidays in the data" if cal else "no holidays"))
        rows = [f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{$\\phi={phi:g}$, {what}; 2 SD drop $\\approx$ {drop(sd):.0f}\\%}}}} \\\\*"]
        for a in adjs:
            c1 = []
            for d in dets:
                q = fa[np.isclose(fa.phi, phi) & (fa.calendar == cal) & (fa.noise == noise) & (fa.adj == a) & (fa.det == d)
                       & (fa.rule == "ar1")]
                r = det[np.isclose(det.phi, phi) & (det.calendar == cal) & (det.noise == noise) & (det.adj == a)
                        & (det.det == d) & np.isclose(det["size"], -2.0)]
                c1 += [pct(q.fa.iloc[0]) if len(q) else "--", pct(r.detect60.iloc[0]) if len(r) else "--"]
            rows.append(f"\\quad {cap(ASHORT[a] if a != 'raw' else 'raw (no forecast)')} & " + " & ".join(c1) + " \\\\")
        blocks.append("\n".join(rows))
    h1 = " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for d in dets)
    cm = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(5))
    tex = r"""\begin{table}[ht]\centering\small\setlength{\tabcolsep}{6.5pt}
\caption{Simulation at $\phi=0$ and $\phi=0.6$ gives the same picture as at $\phi=0.4$, except that at $\phi=0$ the forecasts B and Holt-Winters exceed the false-alarm target by more; with holidays at their real size the problem is smaller but the same. FA: chance (\%) of a false alarm in a year, AR(1) threshold; Det.: share (\%) of 2 SD drops detected within 60 days, matched 5\% rate. 10{,}000 runs per cell.}\label{tab:sim-appendix}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & """ + h1 + r""" \\
""" + cm + "\nForecast" + " & FA & Det." * 5 + r""" \\
\midrule
""" + "\n\\addlinespace\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("sim_appendix.tex", tex)


TUNE = [("CUSUM-k0.25", "$k=0.25$"), ("CUSUM", "$k=0.5$"), ("CUSUM-k1", "$k=1$"),
        ("EWMA-l0.05", "$\\lambda=0.05$"), ("EWMA", "$\\lambda=0.1$"), ("EWMA-l0.2", "$\\lambda=0.2$")]


def t_sim_tuning():
    """Tuning sensitivity of CUSUM (k) and EWMA (lambda), phi = 0.4, no holidays."""
    import json
    g = json.loads((RES / "sim" / "generator.json").read_text())
    t = pd.read_csv(SIM / "tuning.csv")
    t = t[(t.change == "mean") & np.isclose(t.phi, 0.4) & (t.noise == "normal") & (t.calendar == 0)]
    rows = []
    for i, a in enumerate(["oracle", "HW", "YM"]):
        rows.append(("\\addlinespace\n" if i else "") + f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{cap(ANAME[a])}}}}} \\\\*")
        fa = [t[(t.adj == a) & (t.det == d) & (t.rule == "ar1")].false_alarm_rate for d, _ in TUNE]
        rows.append("\\quad False alarms, AR(1) (\\%) & " + " & ".join(f"{100 * x.iloc[0]:.1f}" if len(x) else "--" for x in fa) + " \\\\")
        for sd in (1.0, 2.0, 4.0):
            dd = [t[(t.adj == a) & (t.det == d) & (t.rule == "oracle") & np.isclose(t["size"], -sd)].detect60 for d, _ in TUNE]
            pct_drop = 100 * (1 - np.exp(-sd * g["sigma"]))
            rows.append(f"\\quad {sd:.0f} SD drop ($\\approx${pct_drop:.0f}\\%), detected & " +
                        " & ".join(pct(x.iloc[0]) if len(x) else "--" for x in dd) + " \\\\")
    q = lambda d: t[(t.adj == "oracle") & (t.det == d) & (t.rule == "oracle") & np.isclose(t["size"], -1.0)].detect60.iloc[0]  # noqa: E731
    tex = (r"""\begin{table}[ht]\centering\small\setlength{\tabcolsep}{5pt}
\caption{A smaller smoothing constant closes most of the gap between EWMA and CUSUM for small drops (1 SD, true pattern removed: EWMA """ +
           f"{pct(q('EWMA'))}\\% with $\\lambda=0.1$ and {pct(q('EWMA-l0.05'))}\\% with $\\lambda=0.05$, against {pct(q('CUSUM'))}\\% for CUSUM" +
           r"""). Detection within 60 days at a matched 5\% rate and false alarms with the AR(1) threshold; $\phi=0.4$, no holidays; 10{,}000 runs per cell. Our settings: $k=0.5$, $\lambda=0.1$.}\label{tab:tuning}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{CUSUM} & \multicolumn{3}{c}{EWMA} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
 & """ + " & ".join(lab for _, lab in TUNE) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("tuning.tex", tex)


BOOT_DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]


def t_boot_noise():
    """Block-bootstrap noise (Opal's own residuals, 28-day blocks), phi = 0.4, holidays in the data.
    False alarms with the AR(1) threshold and detection of 2 and 4 SD drops at the matched rate."""
    import json
    g = json.loads((RES / "sim" / "generator.json").read_text())
    fa = pd.read_csv(SIM / "fa.csv")
    fa = fa[(fa.noise == "boot") & (fa.rule == "ar1")]
    det = pd.read_csv(SIM / "detect.csv")
    det = det[(det.noise == "boot") & (det.rule == "oracle") & (det.change == "mean")]
    if fa.empty:
        print("  boot_noise: no bootstrap-noise rows; skipped")
        return
    adjs = ["oracle", "A", "B", "HW", "YM", "raw"]
    lab = lambda a: cap(ASHORT[a] if a != "raw" else "raw (no forecast)")  # noqa: E731
    blocks = []
    for title, src, size in [("False alarms in a normal year, AR(1) threshold (\\%)", fa, None),
                             ("2 SD drop ($\\approx$" + f"{100 * (1 - np.exp(-2 * g['sigma'])):.0f}" + "\\%) detected within 60 days (\\%)", det, -2.0),
                             ("4 SD drop ($\\approx$" + f"{100 * (1 - np.exp(-4 * g['sigma'])):.0f}" + "\\%) detected within 60 days (\\%)", det, -4.0)]:
        rows = [f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\*"]
        for a in adjs:
            cells = []
            for d in BOOT_DETS:
                q = src[(src.adj == a) & (src.det == d)]
                if size is not None:
                    q = q[np.isclose(q["size"], size)]
                v = (q.fa if size is None else q.detect60)
                cells.append(pct(v.iloc[0]) if len(v) else "--")
            rows.append(f"\\quad {lab(a)} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(rows))
    tp = fa[(fa.adj == "oracle") & fa.det.isin(["CUSUM", "EWMA", "FOCuS", "FOCuS-AR"])].fa
    tex = (r"""\begin{table}[ht]\centering\small\setlength{\tabcolsep}{3pt}
\caption{With noise resampled from Opal's own surprises, every uncapped detector alarms in """ +
           f"{100 * tp.min():.1f} to {100 * tp.max():.0f}" + r"""\% of normal years even with the true pattern removed: heavy-tailed real noise alone defeats the AR(1) threshold. Circular block bootstrap (28-day blocks) of the generator's residuals; $\phi=0.4$, holidays in the data; detection at a matched 5\% rate. 10{,}000 runs per cell.}\label{tab:boot-noise}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & \multicolumn{5}{c}{Uncapped} & \multicolumn{3}{c}{Capped} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-9}
Forecast & CUSUM & EWMA & FOCuS & FOCuS-AR & CPM & CUSUM & EWMA & FOCuS \\
\midrule
""" + "\n\\addlinespace\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("boot_noise.tex", tex)


NOISE_DETS = ["CPM", "CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CUSUM-r", "EWMA-r", "FOCuS-r"]


def _p1(x):
    """Percent: one decimal below 10, whole numbers above (so 0.4 and 2.0 stay distinguishable)."""
    v = 100 * x
    return f"{rhu(v, 1):.1f}" if v < 9.95 else f"{rhu(v):.0f}"


def t_noise():
    """Heavy-tailed noise reverses the detector ranking. phi = 0.4, holidays in the data."""
    fa = pd.read_csv(SIM / "fa.csv")
    fa = fa[np.isclose(fa.phi, 0.4) & (fa.calendar == 1) & (fa.rule == "ar1")]
    det = pd.read_csv(SIM / "detect.csv")
    det = det[np.isclose(det.phi, 0.4) & (det.calendar == 1) & (det.rule == "oracle") & (det.change == "mean")]
    fa_rows = [("True pattern, resampled", "oracle", "boot"),
               ("Yearly memory, sim. scale", "YM", "normal"),
               ("Yearly memory, real scale", "YM", "wide"),
               ("Yearly memory, resampled", "YM", "boot")]
    det_rows = [("1 SD, Gaussian", "normal", -1.0), ("1 SD, $t_5$", "t5", -1.0),
                ("2 SD, resampled", "boot", -2.0), ("4 SD, resampled", "boot", -4.0)]
    rows = ["\\multicolumn{9}{@{}l}{\\emph{False alarms, AR(1) threshold (\\%)}} \\\\*"]
    for lab, a, n in fa_rows:
        cells = [_p1(fa[(fa.adj == a) & (fa.noise == n) & (fa.det == d)].fa.iloc[0])
                 if len(fa[(fa.adj == a) & (fa.noise == n) & (fa.det == d)]) else "--" for d in NOISE_DETS]
        rows.append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{9}{@{}l}{\\emph{Detected within 60 days, true pattern removed (\\%)}} \\\\*")
    for lab, n, sz in det_rows:
        x = det[(det.adj == "oracle") & (det.noise == n) & np.isclose(det["size"], sz)]
        cells = [_p1(x[x.det == d].detect60.iloc[0]) if len(x[x.det == d]) else "--" for d in NOISE_DETS]
        rows.append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    tex = (r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Heavy-tailed noise reverses the detector ranking: with Gaussian noise CUSUM and FOCuS find most changes; with heavy-tailed noise CUSUM, CPM and the capped detectors keep their power and FOCuS, FOCuS-AR and EWMA lose most of it. $\phi=0.4$, holidays in the data. Resampled: circular 28-day blocks of the generator's residuals; real scale: holidays against Gaussian noise at the ordinary residual SD. 10{,}000 runs per cell; full table: Table~\ref{tab:boot-noise}.}\label{tab:noise}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & & \multicolumn{4}{c}{Uncapped} & \multicolumn{3}{c}{Capped} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-9}
 & CPM & CUSUM & EWMA & FOCuS & FOCuS-AR & CUSUM & EWMA & FOCuS \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("noise.tex", tex)


SETUP_SHORT = {("YM", "CUSUM-r", "ar1"): "yearly mem. + capped CUSUM", ("YM", "FOCuS-r", "ar1"): "yearly mem. + capped FOCuS",
               ("HW", "EWMA", "ar1"): "Holt-Winters + EWMA", ("YM", "CUSUM", "indep"): "yearly mem. + CUSUM, indep.",
               ("A", "CUSUM", "calm"): "A + CUSUM (calm stretch)"}


def t_recommend():
    """Table 8.2: built from results/recommend.csv (written by recommend.py) with plain
    columns (numbers and verdicts in separate cells). Rows grouped by recommend.py's status; not-recommended rows
    grey. Pass rules: results/recommend_rules.csv (quoted in Section 8.2)."""
    src = RES / "recommend.csv"
    if not src.exists():
        print("  recommend: results/recommend.csv not found yet (pending recommend.py); skipped")
        return
    r = pd.read_csv(src)
    need = {"sim_boot", "calm_n_alarms", "calm_dir", "calm_covid_onset", "d28_status"}
    if not need <= set(r.columns):
        print(f"  recommend: recommend.csv lacks {need - set(r.columns)}; skipped")
        return
    sym = {"✓": "\\checkmark", "×": "$\\times$", "(✓)": "(\\checkmark)", "–": "--", "-": "--", "nan": "--"}
    f1 = lambda x: "--" if pd.isna(x) else (f"{x:.1f}" if x < 9.95 else f"{rhu(x):.0f}")  # noqa: E731
    f2 = lambda x: "--" if pd.isna(x) else num(x)  # noqa: E731
    r["g"] = [("Not recommended (grey)" if str(s).startswith("not recommended") else g) for g, s in zip(r.group, r.d28_status)]
    r["g"] = r["g"].replace({"From simulation and test bed": "Candidates from simulation and test bed"})
    order = list(dict.fromkeys([g for g in r.g if not g.startswith("Not")] + [g for g in r.g if g.startswith("Not")]))
    ncol = 13
    rows = []
    for g in order:
        rows.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textbf{{{g}}}}} \\\\")
        for q in r[r.g == g].itertuples():
            name = SETUP_SHORT.get((q.adj, q.det, q.rule), q.setup.replace("same-weekday difference", "A")
                                   .replace("median of four", "B").replace("independent threshold", "indep."))
            simr = f1(q.sim_sim) if q.rule != "indep" or pd.isna(getattr(q, "sim_no_holidays", np.nan)) else f1(q.sim_sim)
            rows.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{q.situation}}}}} \\\\*" if not g.startswith("Not")
                        else f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textcolor{{gray}}{{\\emph{{{q.situation}}}}}}} \\\\*")
            cells = [f"{name}",
                     f1(q.sim_sim), f1(q.sim_real), f1(q.sim_boot), sym.get(str(q.sim_sym), str(q.sim_sym)),
                     f2(q.tb_temp20), f2(q.tb_drop20), sym.get(str(q.tb_sym), str(q.tb_sym)),
                     f"{int(q.opal_frozen)}", sym.get(str(q.opal_frozen_sym), str(q.opal_frozen_sym)),
                     f"{int(q.opal_r0_7_dir)}", sym.get(str(q.opal_r0_7_dir_sym), str(q.opal_r0_7_dir_sym)),
                     "--" if pd.isna(q.calm_n_alarms) else f"{int(q.calm_n_alarms)} / {int(q.calm_dir)} / {q.calm_covid_onset}"]
            if g.startswith("Not"):
                cells = [f"\\textcolor{{gray}}{{{c}}}" for c in cells]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows = rows[:-1]
    nh = r[r.rule == "indep"]
    nh_note = ""
    if "sim_no_holidays" in r.columns and nh.sim_no_holidays.notna().any():
        nh_note = (" Independent-threshold rows fail the simulation even without holidays (" +
                   ", ".join(f"{f1(x)}\\%" for x in nh.sim_no_holidays.dropna()) + ").")
    tex = (r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{1.1pt}
\caption{Set-ups ranked by simulation and test bed, with the Opal episodes as a check; every row, chosen after the main results, is a hypothesis for a forward test (Section~\ref{sec:7-limits}; pass rules in Section~\ref{sec:7-recommend}). \checkmark{} passes or supports; (\checkmark) beats chance alone, not after adjustment for the best of the family; $\times$ does not; -- not tested.""" + nh_note + r"""}\label{tab:recommend}
\begin{tabular}{@{}lrrr@{\hspace{3pt}}crr@{\hspace{3pt}}crcr@{\hspace{2pt}}cc@{}}
\toprule
 & \multicolumn{4}{c}{""" + H("Simulated", "false alarms,", "holidays (\\%)") + r"""} & \multicolumn{3}{c}{""" + H("Test-bed", "gain at", "20\\% bkgd.") +
           r"""} & \multicolumn{4}{c}{""" + H("Opal", "episodes", "(of 15)") + r"""} & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-12}
""" + H("Situation, then", "set-up (AR(1)", "unless stated)") + r""" & """ + H("sim.", "scale") + " & " + H("real", "scale") + " & " + H("re-", "samp.") +
           r""" & pass & """ + H("20\\%,", "14 d") + " & " + H("20\\%,", "lasting") + r""" & pass & Frozen & & """ + H("Dir.-", "aware") + r""" & & """ +
           H("Calm", "stretch:", "alarms /", "episodes /", "COVID") + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("recommend.tex", tex)


GEN_GROUP = [("level", "Level"), ("weekday", "Weekday (against Monday)"), ("yearly", "Yearly pattern"),
             ("calendar", "Calendar"), ("noise", "Residuals"), ("fit", "Fit")]


def t_generator():
    """Every coefficient of the simulation generator, refitted by generator_table.py and
    checked against generator.json. Log scale; robust SE; % effect = 100 (exp(b) - 1)."""
    g = pd.read_csv(RES / "sim" / "generator_coefficients.csv")
    rows = []
    for i, (key, lab) in enumerate(GEN_GROUP):
        x = g[g.group == key]
        if not len(x):
            continue
        rows.append(("\\addlinespace\n" if i else "") + f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\*")
        for q in x.itertuples():
            term = (q.term.replace("weekday: ", "").replace(" vs Monday", "").replace("%", "\\%")
                    .replace("level mu", "level $\\mu$").replace("2023-2025", "2023--25"))
            if key in ("noise", "fit"):
                v = f"{int(q.coefficient):,}".replace(",", "{,}") if key == "fit" else (
                    f"{q.coefficient:.1f}" if "kurtosis" in q.term else f"{q.coefficient:.3f}")
                rows.append(f"\\quad {cap(term)} & {v} & & \\\\")
                continue
            se = "--" if pd.isna(q.se) else f"{q.se:.4f}"
            eff = "--" if pd.isna(q.pct_effect) else num(q.pct_effect, 1)
            rows.append(f"\\quad {cap(term)} & {num(q.coefficient, 4)} & {se} & {eff} \\\\")
    tex = (r"""\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{5pt}
\caption{The simulation generator, fitted to log daily train tap-ons on the normal days of 2023--25 by robust regression: level, weekday, yearly and calendar terms, and the residual properties the simulation uses. Coefficients on the log scale with robust standard errors; effect: $100\,(e^{b}-1)$\%.}\label{tab:generator}
\begin{tabular}{@{}lrrr@{}}
\toprule
Term & Coefficient & SE & Effect, \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("generator.tex", tex)


PR_DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]


def t_past_retro():
    """Past-only against retrospective fits of the same model (MSTL; regression), every
    detector, both threshold rules and both scoring rules; summary rows from past_retro_summary.csv."""
    p = pd.read_csv(RES / "opal" / "past_retro_pairs.csv")
    sm = pd.read_csv(RES / "opal" / "past_retro_summary.csv").set_index(["threshold_rule", "scoring_rule"])
    cols = [("ar1", "frozen"), ("ar1", "dir"), ("indep", "frozen"), ("indep", "dir")]
    rows = []
    for i, fam in enumerate(["MSTL", "regression"]):
        rows.append(("\\addlinespace\n" if i else "") + f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{cap(fam)}: refitted on the past / whole series}}}} \\\\*")
        for d in PR_DETS:
            cells = []
            for rule, sc in cols:
                q = p[(p.family == fam) & (p.detector == d) & (p.rule == rule)]
                cells += ([f"{int(q[f'past_{sc}'].iloc[0])}", f"{int(q[f'retro_{sc}'].iloc[0])}"] if len(q) else ["--", "--"])
            rows.append(f"\\quad {cap(DNAME[d])} & " + " & ".join(cells) + " \\\\")
    def srow(field, lab):
        out = []
        for rule, sc in cols:
            x = sm.loc[(rule, "frozen" if sc == "frozen" else "direction-aware")]
            out.append(f"\\multicolumn{{2}}{{c}}{{{field(x)}}}")
        return f"{lab} & " + " & ".join(out) + " \\\\"
    foot = [srow(lambda x: f"{int(x.past_more)} / {int(x.same)} / {int(x.past_fewer)}", "Past-only more / same / fewer"),
            srow(lambda x: f"{int(x.past_fewer_alarms)} of {int(x.pairs)}", "Past-only raises fewer alarms")]
    both = {sc: sm.loc[("both", sc)] for sc in ("frozen", "direction-aware")}
    tex = (r"""\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{3.6pt}
\caption{Hindsight gives no uniform advantage: fitted on the past only, the same model catches fewer episodes than its retrospective fit in """ +
           f"{int(both['frozen'].past_fewer)} of {int(both['frozen'].pairs)}" + r""" pairs under the frozen rule and """ +
           f"{int(both['direction-aware'].past_fewer)} of {int(both['direction-aware'].pairs)}" + r""" under the direction-aware rule, and more in """ +
           f"{int(both['frozen'].past_more)} and {int(both['direction-aware'].past_more)}" + r""". Episodes caught of 15 (past-only / retrospective) for every detector; the pairs differ in fitting and level treatment as well as in what the fit can see. FOCuS-AR has only the AR(1) threshold.}\label{tab:past-retro}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{AR(1) threshold} & \multicolumn{4}{c}{Independent threshold} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
 & """ + " & ".join([f"\\multicolumn{{2}}{{c}}{{{H('Frozen', 'rule')}}}", f"\\multicolumn{{2}}{{c}}{{{H('Direction-', 'aware')}}}"] * 2) + r""" \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Detector""" + " & past & retro." * 4 + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\addlinespace
""" + "\n".join(foot) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("past_retro.tex", tex)


OFF_METHODS = [("PELT", "PELT"), ("BinSeg", "BinSeg"), ("WBS", "WBS")]
OFF_SERIES = ["STL", "MSTL", "REG"]


def _offline_dirs():
    """Directions of the offline change dates. offline_changes.csv stores dates only, so the segmentation is
    recomputed exactly as scripts/opal_offline.py does (same scaling, settings, seed and order) and checked
    against the stored dates; direction = sign of the later minus the earlier segment mean."""
    sys.path.insert(0, str(HERE / "scripts"))
    sys.path.insert(0, str(HERE))
    from cpd import kernels as K
    from opal_offline import scaled
    adj = pd.read_csv(RES / "opal" / "adjusted.csv", parse_dates=["date"]).set_index("date")
    dates = adj.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    rng = np.random.default_rng(0)
    out = []
    for m in OFF_SERIES:
        y, _ = scaled(adj[m])
        n = len(y)
        ints = np.sort(rng.integers(0, n, size=(5000, 2)), axis=1)
        res = {"PELT": K.pelt_l2(y, 3 * np.log(n), 7), "BinSeg": K.binseg_l2(y, np.sqrt(3 * np.log(n)), 7, 200),
               "WBS": K.wbs_l2(y, 1.3 * np.sqrt(2 * np.log(n)), 7, ints)}
        for meth, cps in res.items():
            allc = sorted(int(c) for c in cps)
            bnd = [0] + allc + [n]
            for k, c in enumerate(allc):
                if c >= t0:
                    out.append(dict(adjustment=m, method=meth, change_date=dates[c],
                                    direction=1 if y[c:bnd[k + 2]].mean() > y[bnd[k]:c].mean() else -1))
    out = pd.DataFrame(out)
    stored = pd.read_csv(RES / "opal" / "offline_changes.csv", parse_dates=["change_date"])
    assert stored.reset_index(drop=True).equals(out[["adjustment", "method", "change_date"]].reset_index(drop=True)), \
        "recomputed offline change dates differ from offline_changes.csv"
    return out


def t_offline_matches():
    """The offline change dates at the 18 scored boundaries. Frozen rule:
    the number of the three methods with a change within 7 days either side of the boundary (any direction).
    Direction-aware rule: the number with a change 0 to 7 days after it, in its direction, credited to it as the
    nearest such boundary. Footer: episodes caught (of 15) per method, checked against our scorer and
    offline.csv."""
    ch = _offline_dirs()   # also puts the repository root on sys.path
    from cpd import score as SC
    ev = pd.read_csv(HERE / "data" / "opal" / "events_scored_FROZEN.csv", parse_dates=["date"]).sort_values("date")
    want = np.where(ev["boundary"].astype(str).str.lower().isin(["onset", "start"]), -1, 1)
    off = pd.read_csv(RES / "opal" / "offline.csv")
    hit = {}   # (series, method, rule) -> set of boundary positions matched
    totals = {}
    for (m, meth), g in ch.groupby(["adjustment", "method"]):
        g = g.sort_values("change_date")
        cds, dirs = pd.DatetimeIndex(g.change_date), g.direction.to_numpy()
        fz = {j for j, b in enumerate(ev["date"]) if (np.abs((cds - b).days) <= 7).any()}
        da = set()
        for x, dv in zip(cds, dirs):
            d = np.array([(x - b).days for b in ev["date"]])
            ok = (d >= 0) & (d <= 7) & (want == dv)
            if ok.any():
                da.add(int(np.flatnonzero(ok)[np.argmin(np.abs(d[ok]))]))
        hit[(m, meth, "frozen")], hit[(m, meth, "dir")] = fz, da
        nf, _ = SC.episodes_caught(cds, None, after=7, before=7, credit="all")
        nd, _ = SC.episodes_caught(cds, list(dirs))
        assert nf == ev.iloc[sorted(fz)].episode.nunique() and nd == ev.iloc[sorted(da)].episode.nunique()
        q = off[(off.adjustment == m) & (off.method == meth)]
        assert len(q) and int(q.episodes_caught.iloc[0]) == nd, "direction-aware count differs from offline.csv"
        totals[(m, meth, "frozen")], totals[(m, meth, "dir")] = nf, nd
    lab = {}
    for j, r in enumerate(ev.itertuples()):
        lab[j] = episode_label(r.episode)
    rows = []
    for j, r in enumerate(ev.itertuples()):
        cells = []
        for rule in ("frozen", "dir"):
            for m in OFF_SERIES:
                k = sum(j in hit[(m, meth, rule)] for meth, _ in OFF_METHODS)
                cells.append("" if k == 0 else f"{k}")
        rows.append(f"{r.date.day} {r.date:%b %Y} & {lab[j]} & " + " & ".join(cells) + " \\\\")
    foot = []
    for meth, mlab in OFF_METHODS:
        foot.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Episodes caught of 15, {mlab}}} & " +
                    " & ".join(f"{totals[(m, meth, rule)]}" for rule in ("frozen", "dir") for m in OFF_SERIES) + " \\\\")
    fz_rng = [totals[(m, meth, "frozen")] for m in OFF_SERIES for meth, _ in OFF_METHODS]
    da_rng = [totals[(m, meth, "dir")] for m in OFF_SERIES for meth, _ in OFF_METHODS]
    tex = (r"""\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{4.5pt}
\caption{With hindsight the offline methods put a change date near most scored boundaries: they catch """ +
           f"{min(fz_rng)} to {max(fz_rng)}" + r""" of the 15 episodes under the frozen rule and """ +
           f"{min(da_rng)} to {max(da_rng)}" + r""" under the direction-aware rule. Cells: how many of PELT, Binary Segmentation and WBS have a matching change date at the boundary (blank: none). Frozen rule: within 7 days either side, any direction; direction-aware rule: 0 to 7 days after, in the boundary's direction, credited to the nearest boundary. Retrospective STL, MSTL and regression series, 1 March 2020 to 28 September 2026.}\label{tab:offline-matches}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Frozen rule} & \multicolumn{3}{c}{Direction-aware rule} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Boundary & Episode""" + " & STL & MSTL & REG" * 2 + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
""" + "\n".join(foot) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("offline_matches.tex", tex)


EP_NAME = {"covid": "COVID-19", "delta": "Delta wave", "omicron": "Omicron wave", "flood": "Floods",
           "shutdown": "Shutdown", "strike": "Rail dispute", "outage": "Outage"}


def episode_label(e):
    """covid_onset -> 'COVID-19 onset'; flood_2021_03 -> 'Floods, Mar 2021'."""
    p = e.split("_")
    if len(p) == 3 and p[1].isdigit():
        return f"{EP_NAME[p[0]]}, {pd.Timestamp(int(p[1]), int(p[2]), 1):%b %Y}"
    return f"{EP_NAME[p[0]]} {p[1]}"


def t_events():
    sc = pd.read_csv(HERE / "data" / "opal" / "events_scored_FROZEN.csv", parse_dates=["date"])
    rows = []
    for _, r in sc.iterrows():
        note = str(r.note).replace("&", "and").replace("%", "\\%")
        cls = {"covid": "COVID-19"}.get(r["class"], r["class"])
        rows.append(f"{r.date.day} {r.date:%b %Y} & {episode_label(r.episode)} & {cls} & {r.announced_before} & "
                    f"\\parbox[t]{{4.9cm}}{{\\raggedright {note}\\strut}} \\\\")
    head = r"Date & Episode & Class & Announced & Note \\"
    tex = r"""{\small\setlength{\tabcolsep}{4pt}
\begin{longtable}{@{}lllll@{}}
\caption{The 18 scored boundaries (15 episodes), frozen on 22 October 2026. Each is supported by a public source in the full list of 71 events (code repository). Announced: whether the change was announced before travel changed.}\label{tab:events}\\
\toprule
""" + head + r"""
\midrule
\endfirsthead
\toprule
""" + head + r"""
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}}
"""
    write("events.tex", tex)


def t_covid():
    """Widths without hyphenation; estimated change date column (EWMA has no estimator: date = alarm)."""
    c = pd.read_csv(RES / "opal" / "covid.csv", parse_dates=["first_alarm", "start_est"])
    c = c[c.rule == "ar1"]
    has_tau = "start_est" in c.columns
    rows = []
    for i, m in enumerate(["A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL"]):
        rows.append(("\\addlinespace\n" if i else "") + f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{cap(ANAME[m])}}}}} \\\\*")
        for d in ["CUSUM", "EWMA", "FOCuS", "CPM"]:
            q = c[(c.method == m) & (c.detector == d)]
            if not len(q):
                continue
            q = q.iloc[0]
            ch = pct(q.chance_alarm_14d_before)
            if pd.isna(q.first_alarm):
                rows.append(f"\\quad {d} & none & -- & -- & -- & -- & {ch} \\\\")
                continue
            tau = f"{q.start_est.day} {q.start_est:%b}" if has_tau and pd.notna(q.start_est) else "--"
            if d == "EWMA":
                tau = "--"
            gap = f"{int(q.days_alarm_after_start_est)}" if has_tau and d != "EWMA" and pd.notna(q.start_est) else "--"
            rows.append(f"\\quad {d} & {q.first_alarm.day} {q.first_alarm:%b} & {tau} & {gap} & "
                        f"{sgn(q.days_from_16mar)} & {sgn(q.days_from_23mar)} & {ch} \\\\")
    tex = r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{5pt}
\caption{First alarm at the start of COVID-19 ($\hat\phi=0$ in this segment: the AR(1) and independent thresholds coincide). Estimated start: the change date the detector estimates at the alarm (EWMA has no estimator); days: from the estimated start to the alarm, and from the first public health order (16 March 2020) and the closure of venues (23 March) to the alarm. Chance: chance (\%) of an unrelated alarm in the 14 days before 16 March, given each monitor's own background rate.}\label{tab:covid}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & & & \multicolumn{3}{c}{Days to the alarm from} & \\
\cmidrule(lr){4-6}
Detector & First alarm & Estimated start & estimate & 16 Mar & 23 Mar & Chance (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("covid.tex", tex)


ALL = [t_validation, t_opal_primary, t_opal_full, t_rule_variants, t_opal_sens, t_opal_budget, t_opal_variance, t_testbed, t_testbed_full, t_delay, t_variance,
       t_runtime, t_retro_extra, t_sim_appendix, t_sim_tuning, t_boot_noise, t_noise, t_recommend, t_generator, t_past_retro, t_offline_matches, t_events, t_covid]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if want and f.__name__[2:] not in want:
            continue
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep going so one missing input does not stop the rest
            import traceback
            traceback.print_exc()
            print("skip", f.__name__, repr(e)[:200])
