"""The thesis tables in their final wording: tables/ (printed in the thesis) and repo-tables/ (cited from the thesis).

Every table is still computed by scripts/tables.py, into a temporary folder. This script then applies the naming
and format rules of the final text (no number is changed) and writes:
  tables/       the six tables the thesis prints (events, noise, opal_primary, recommend, rule_variants, testbed),
                byte for byte as printed;
  repo-tables/  the tables the thesis cites from the repository (Appendix A), with the same names.
The rules:
  * forecasts: one name each, as in the thesis: "same-weekday difference" (the letter A), "median of four" (B),
    "Holt-Winters" and "yearly memory"; "indep." -> "independent"; "(pipeline)" -> "the fixed monitor";
  * terms: "surprise" -> "forecast error", time-series "dependence" -> "autocorrelation";
  * scoring rules: "main rule" (fixed before the runs) and "direction-aware check";
  * pointers to tables that are not printed -> repo-tables/ in the repository (Appendix A).
repo-tables/simconditions.tex and repo-tables/checks.tex are typed by hand (they are not computed) and are left as
they are. tables/runtime_per_day.tex (Table A.2) is written here from results/runtime.csv and runtime_focus.csv.

Usage: python scripts/thesis_tables.py         (make tables)
"""
from __future__ import annotations

import importlib
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
KEEP_DIR, REPO_DIR = HERE / "tables", HERE / "repo-tables"

# tables the thesis does not print; they go to repo-tables/
CUT = ["retro_extra.tex", "past_retro.tex", "opal_full.tex", "covid.tex", "opal_variance.tex", "testbed_full.tex",
       "opal_budget.tex", "runtime.tex", "boot_noise.tex", "generator.tex", "delay.tex"]
CUT_LATER = ["sim_appendix.tex", "tuning.tex", "variance.tex", "offline_matches.tex", "opal_sens.tex",
             "validation.tex"]

REPO = r"\texttt{repo-tables/%s} in the repository (Appendix~\ref{app:code})"


def unheader(s):
    """The printed tables are kept exactly as printed, without the first-line marker that tables.py writes."""
    return s.split("\n", 1)[1] if s.startswith("% written by scripts/") else s


def _labels(s):
    """Row labels and heads shared by every printed table."""
    s = s.replace("\\small", "\\footnotesize", 1) if s.count("\\small") else s
    s = re.sub(r"(?m)^(\\textcolor\{gray\}\{)?A \+ ", r"\1Same-weekday difference + ", s)
    s = re.sub(r"(?m)^(\\textcolor\{gray\}\{)?B \+ ", r"\1Median of four + ", s)
    s = s.replace(", indep.", ", independent")
    s = s.replace("(pipeline)", "(fixed in advance)")
    s = s.replace("yearly mem. +", "Yearly memory +")
    return s


def _one(name, s):
    """Naming and layout of each table (first pass)."""
    s = _labels(s)
    if name == "opal_primary.tex":
        s = s.replace(r"\begin{tabular}[b]{@{}c@{}}Frozen\\rule\end{tabular}",
                      r"\begin{tabular}[b]{@{}c@{}}Main rule\\(frozen)\end{tabular}")
        s = s.replace(r"\begin{tabular}[b]{@{}c@{}}Direction-\\aware\end{tabular}",
                      r"\begin{tabular}[b]{@{}c@{}}Check (direc-\\tion-aware)\end{tabular}")
        s = s.replace(r"every monitor: Table~\ref{tab:opal-full}.", "every monitor: " + REPO % r"opal\_full.tex" + ".")
        s = s.replace("holds under both rules after adjustment", "passes the check: holds under both rules after adjustment")
        s = s.replace("columns: Table~\\ref{tab:measures};", "columns: Section~\\ref{sec:measures};")
        s = s.replace("AR(1) threshold unless marked;", "AR(1) threshold unless marked. Last week: the same-weekday "
                      "difference; four-week median: the median of four. Fixed: the monitor fixed in advance, "
                      "Section~\\ref{sec:pipeline};")
        s = s.replace(r"\setlength{\tabcolsep}{4pt}", r"\setlength{\tabcolsep}{2.2pt}")
        s = s.replace("(fixed in advance)", "(fixed)")
        s = s.replace(", independent$^{\\dagger}$", "$^{\\ddagger\\dagger}$")
        s = s.replace("AR(1) threshold unless marked.", "AR(1) threshold; $^{\\ddagger}$ independent threshold.")
    if name == "recommend.tex":
        s = s.replace(", independent}", "}")
        a = s.index(r"\begin{tabular}{@{}l")
        b = a + len(r"\begin{tabular}{@{}l")
        s = s[:a] + r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.8cm}" + s[b:]
        s = s.replace(r"\textbf{Not recommended (grey)}", r"\textbf{Not recommended (grey; independent threshold)}")
        s = s.replace(r"Test-bed\\gain at\\20\% bkgd.", r"Test-bed gain\\at 20\%\\background")
        s = s.replace(r"\begin{tabular}[b]{@{}c@{}}sim.\\scale\end{tabular}", r"\begin{tabular}[b]{@{}c@{}}simul.\\scale\end{tabular}")
        s = s.replace(r"\begin{tabular}[b]{@{}c@{}}re-\\samp.\end{tabular}", r"\begin{tabular}[b]{@{}c@{}}resam-\\pled\end{tabular}")
        s = s.replace("pass & Frozen & &", "pass & Main & &")
        s = s.replace(r"\begin{tabular}[b]{@{}c@{}}Dir.-\\aware\end{tabular}", "Check")
        s = s.replace(r"$\times$ does not; -- not tested.",
                      r"$\times$ does not; -- not tested. Simul.\ and real scale: holidays at the simulated and at the "
                      r"real size; resampled: Opal's residuals resampled. Main and Check: the two scoring rules.")
    if name == "rule_variants.tex":
        s = s.replace(r"\setlength{\tabcolsep}{3pt}", r"\setlength{\tabcolsep}{1.5pt}")
        a = s.index("\\midrule\n\\multicolumn{7}{@{}l}{\\emph{Without the 15 January 2025 boundary")
        b = s.index("\\addlinespace\n\\multicolumn{7}{@{}p{14cm}}{$^{\\dagger}$")
        s = s[:a] + s[b:]
        s = s.replace(r"\centering Frozen rule: $\pm$7 d, any boundary", r"\centering Main rule (frozen): $\pm$7 d, any boundary")
        s = s.replace(r"\centering Direction-aware rule: 0--7 d, nearest, direction", r"\centering Check (direction-aware): 0--7 d, nearest, direction")
    if name == "testbed.tex":
        s = s.replace("on A, B and Holt-Winters", "on the same-weekday difference, the median of four and Holt-Winters")
        s = s.replace(r"All settings: Table~\ref{tab:testbed-full}.", "All settings: " + REPO % r"testbed\_full.tex" + ".")
        s = s.replace(r"\emph{A: same-weekday difference}", r"\emph{Same-weekday difference}")
        s = s.replace(r"\emph{B: median of four}", r"\emph{Median of four}")
        # the background comparison is given in Section 6.4 of the thesis, not in the caption
        s = re.sub(r" They also alarm less often in background windows \(.*?Section~\\ref\{sec:5-testbed\}\)\.", "", s)
    if name == "offline_matches.tex":
        s = s.replace(r"\multicolumn{3}{c}{Frozen rule}", r"\multicolumn{3}{c}{Main rule (frozen)}")
        s = s.replace(r"\multicolumn{3}{c}{Direction-aware rule}", r"\multicolumn{3}{c}{Check (direction-aware)}")
    if name == "sim_appendix.tex":
        s = s.replace(r"\quad A &", r"\quad Same weekday last week &").replace(r"\quad B &", r"\quad Four-week median &")
        s = s.replace("the forecasts B and Holt-Winters", "the median of four and Holt-Winters")
    if name == "noise.tex":
        s = s.replace(r"full table: Table~\ref{tab:boot-noise}.", "full table: " + REPO % r"boot\_noise.tex" + ".")
        s = s.replace("Heavy-tailed noise reverses the detector ranking:", "Heavy-tailed noise changes which detectors work:")
    return s


def terms(s):
    """"forecast error" for the standardised one-step error, "autocorrelation" for time-series dependence."""
    for a, b in [("standardised surprises", "standardised forecast errors"),
                 ("standardised surprise", "standardised forecast error"),
                 ("Surprises", "Forecast errors"), ("surprises", "forecast errors"),
                 ("Surprise", "Forecast error"), ("surprise", "forecast error")]:
        s = s.replace(a, b)
    s = re.sub(r"\bdependence\b", "autocorrelation", s)
    s = re.sub(r"\bDependence\b", "Autocorrelation", s)
    return s


def pointers(s):
    """References to appendix material that is in the repository, not in the printed thesis."""
    for lab in ("app:sim", "app:details", "app:d-checks", "app:d-methods", "app:d-thresholds", "app:retro-extra",
                "app:opal"):
        s = s.replace(r"Appendix~\ref{" + lab + "}", r"the repository, Appendix~\ref{app:code}")
    for lab, name in (("tab:variance", "variance"), ("tab:validation", "validation"), ("tab:tuning", "tuning"),
                      ("tab:sim-appendix", "sim\\_appendix"), ("tab:offline-matches", "offline\\_matches"),
                      ("tab:opal-sens", "opal\\_sens")):
        s = s.replace(r"Table~\ref{" + lab + "}", r"\texttt{repo-tables/" + name + r".tex} in the repository")
    return s


# one name per scoring rule ("main rule", "direction-aware check"), the dagger defined as "holds under both rules",
# "the fixed monitor"; text only
FINAL = {
    "opal_primary.tex": [
        (r"$^{\dagger}$ passes the check: holds under both rules after adjustment for the best of 137 (both counts at least 7).",
         r"$^{\dagger}$ holds under both rules: at least 7 episodes under each, more than chance after adjustment for the best of 137 monitors (Appendix~\ref{app:rules})."),
        (r"Fixed: the monitor fixed in advance,", r"Fixed: the fixed monitor,"),
        (r"Last week: the same-weekday difference; four-week median: the median of four. Fixed: the fixed monitor, "
         r"Section~\ref{sec:pipeline}; columns: Section~\ref{sec:measures}; every monitor:",
         r"Fixed: the fixed monitor (Section~\ref{sec:pipeline}). Near: share of alarms within 7 days of a scored "
         r"boundary; blind: share of days in relearn periods; every monitor:"),
        (r"\begin{tabular}[b]{@{}c@{}}Main rule\\(frozen)\end{tabular}", r"\begin{tabular}[b]{@{}c@{}}Main\\rule\end{tabular}"),
        (r"\begin{tabular}[b]{@{}c@{}}Check (direc-\\tion-aware)\end{tabular}",
         r"\begin{tabular}[b]{@{}c@{}}Direction-\\aware check\end{tabular}"),
    ],
    "recommend.tex": [
        (r"with the Opal episodes as a check;", r"with the Opal episodes as supporting evidence;"),
        (r"Main and Check: the two scoring rules.", r"Main: the main rule; Direction-aware: the direction-aware check."),
        (r"about one or two alarms a year to check}", r"about one or two alarms a year to review}"),
        (r"someone checks about five alarms a year}", r"someone reviews about five alarms a year}"),
        (r"& pass & Main & & Check & &", r"& pass & \begin{tabular}[b]{@{}c@{}}Main\\rule\end{tabular} & & \begin{tabular}[b]{@{}c@{}}Direc-\\tion-\\aware\end{tabular} & &"),
    ],
    "rule_variants.tex": [
        (r"Episodes caught under the frozen rule, three variants and the direction-aware rule:",
         r"Episodes caught under the main rule, three variants and the direction-aware check:"),
        (r"\centering Main rule (frozen): $\pm$7 d, any boundary", r"\centering Main rule: $\pm$7 d, any boundary"),
        (r"\centering Check (direction-aware): 0--7 d, nearest, direction", r"\centering Direction-aware check: 0--7 d, nearest, direction"),
        (r"Holds under both the frozen and the direction-aware rule after adjustment:",
         r"Holds under both the main rule and the direction-aware check after adjustment:"),
    ],
}


def final(name, s):
    for a, b in FINAL.get(name, []):
        assert a in s, (name, a)
        s = s.replace(a, b)
    return s


def rule_names(s):
    """The two scoring rules by their names in the thesis, in the tables of repo-tables/ (the event list stays
    "frozen")."""
    for a, b in [(r"Main rule (frozen)", "Main rule"), (r"Check (direction-aware)", "Direction-aware check"),
                 (r"\begin{tabular}[b]{@{}c@{}}Frozen\\rule\end{tabular}", r"\begin{tabular}[b]{@{}c@{}}Main\\rule\end{tabular}"),
                 ("Dir.: direction-aware rule; frozen: frozen rule;", "Dir.: direction-aware check; main: main rule;"),
                 ("Frozen rule:", "Main rule:"), ("frozen rule", "main rule"),
                 ("Direction-aware rule", "Direction-aware check"), ("direction-aware rule", "direction-aware check")]:
        s = s.replace(a, b)
    s = re.sub(r"(?m)^Monitor & Dir\..*$", lambda m: m.group(0).replace("& Frozen", "& Main"), s)
    return s


def runtime_per_day():
    """Table A.2 of the thesis: computing time per daily update and for the whole Opal series.

    CUSUM, EWMA, CPM, FOCuS and FOCuS-AR from results/runtime.csv (our kernels; FOCuS and FOCuS-AR by exact
    search), pruned FOCuS from results/runtime_focus.csv (the focus_cpt software, Gaussian model)."""
    import pandas as pd
    res = HERE / "results"
    r = pd.read_csv(res / "runtime.csv").set_index(["method", "n"])
    f = pd.read_csv(res / "runtime_focus.csv").set_index(["method", "n"])
    ns = (365, 1095, 2463)
    rows = [("CUSUM", r, "CUSUM"), ("EWMA", r, "EWMA"), ("CPM", r, "CPM"),
            ("FOCuS, exact search (ours)", r, "FOCuS"), ("FOCuS-AR, exact search (ours)", r, "FOCuS-AR"),
            ("Pruned FOCuS", f, "focus_cpt gaussian")]
    reps = {int(t.loc[(m, n), "reps"]) for _, t, m in rows for n in ns}
    assert reps == {51}, reps

    def per_day(t, m, n):                    # microseconds per daily update
        return 1000 * t.loc[(m, n), "ms"] / n

    top = max(per_day(t, m, n) for _, t, m in rows for n in ns)
    assert top < 3, top                      # "under three microseconds"
    body = []
    for name, t, m in rows:
        cells = [per_day(t, m, n) for n in ns]
        whole = t.loc[(m, 2463), "ms"]
        body.append(f"{name} & " + " & ".join(f"{v:.3f}" if v < 0.1 else f"{v:.2f}" for v in cells)
                    + " & " + (f"{whole:.3f}" if whole < 0.1 else f"{whole:.1f}") + r" \\")
    return (r"""\begin{table}[!h]
\centering
\caption{Every detector updates in under three microseconds per day (Section~\ref{sec:4-time}). Computing time per daily update, in microseconds, on a series of $n$ days, and for the whole Opal series in milliseconds (one thread on an Apple M4, median of 51 repeats). Pruned FOCuS: the \texttt{focus\_cpt} software.}\label{tab:runtime}
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{3}{c}{Per day ($\mu$s)} & Whole series (ms) \\
\cmidrule(lr){2-4}
Detector & $n=365$ & $n=1{,}095$ & $n=2{,}463$ & $n=2{,}463$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def forecast_names(s):
    """One name per forecast in the tables of repo-tables/, as in the thesis text."""
    for a, b in [(r"\emph{A: same-weekday difference}", r"\emph{Same-weekday difference}"),
                 (r"\emph{B: median of four}", r"\emph{Median of four}"),
                 (r"\emph{Holt-Winters, yearly memory}", r"\emph{Yearly memory}"),
                 (r"\quad Same weekday last week &", r"\quad Same-weekday difference &"),
                 (r"\quad Four-week median &", r"\quad Median of four &"),
                 (r"\quad A &", r"\quad Same-weekday difference &"), (r"\quad B &", r"\quad Median of four &"),
                 (r"\emph{No adjustment}", r"\emph{No forecast}"),
                 ("by robust regression", "by Huber regression"),
                 ("Robust SD of the residuals (1.4826 MAD)", "MAD-based SD of the residuals (1.4826 MAD)")]:
        s = s.replace(a, b)
    s = re.sub(r"(?m)^(\\textcolor\{gray\}\{)?(A|Last week) \+ ", r"\1Same-weekday difference + ", s)
    s = re.sub(r"(?m)^(\\textcolor\{gray\}\{)?(B|Four-week median) \+ ", r"\1Median of four + ", s)
    return s


def build():
    """Return {file name: (folder, text)} for every table tables.py writes, and Table A.2."""
    tables = importlib.import_module("tables")
    out = {}
    old = tables.OUT
    with tempfile.TemporaryDirectory() as tmp:
        tables.OUT = Path(tmp)
        for f in tables.ALL:
            f()
        for p in sorted(Path(tmp).glob("*.tex")):
            raw = p.read_text()
            if p.name in CUT:
                out[p.name] = (REPO_DIR, forecast_names(rule_names(terms(raw))))
            elif p.name in CUT_LATER:
                out[p.name] = (REPO_DIR, forecast_names(rule_names(terms(_one(p.name, raw)))))
            else:
                out[p.name] = (KEEP_DIR, final(p.name, pointers(terms(_one(p.name, unheader(raw))))))
    tables.OUT = old
    out["runtime_per_day.tex"] = (KEEP_DIR, runtime_per_day())
    return out


def main():
    REPO_DIR.mkdir(exist_ok=True)
    KEEP_DIR.mkdir(exist_ok=True)
    res = build()
    for name, (folder, text) in res.items():
        (folder / name).write_text(text)
        other = (REPO_DIR if folder == KEEP_DIR else KEEP_DIR) / name
        if other.exists():
            other.unlink()
    print(f"{sum(f == KEEP_DIR for f, _ in res.values())} tables in tables/, "
          f"{sum(f == REPO_DIR for f, _ in res.values())} in repo-tables/ (plus simconditions.tex and checks.tex, typed)")


if __name__ == "__main__":
    main()
