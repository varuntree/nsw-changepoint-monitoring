# Repository appendix

Appendix A of the thesis points here for the method details and further results that the printed thesis leaves
out. There are three chapter-style appendices, each as a compiled PDF and as its LaTeX source:

| PDF | LaTeX source |
|---|---|
| [Method details](method-details.pdf) | [`method-details.tex`](method-details.tex) |
| [Additional simulation results](simulation.pdf) | [`simulation.tex`](simulation.tex) |
| [Additional Opal results](opal.pdf) | [`opal.tex`](opal.tex) |

In the PDFs the chapters are lettered M, S and O, so that their sections, tables and equations (M.1, Table S.1,
equation (M.2)) cannot be confused with those of the thesis. Every other section, figure, table and equation
number refers to the thesis. The tables these files input are in [`../tables/`](../tables/) (printed in the thesis)
and [`../repo-tables/`](../repo-tables/) (cited from it), and the works they cite are in [`refs.bib`](refs.bib).
`make appendix-pdf` (`scripts/appendix_pdf.py`) compiles the PDFs again with
[tectonic](https://tectonic-typesetting.github.io).

| File | What it holds |
|---|---|
| [`method-details.tex`](method-details.tex) | Data and events: the sources of the event list, the two data artefacts of late 2022, FOCuS on the raw counts, and the alarm classes. Forecasts and detectors: the past-only and retrospective forecasts, the exact FOCuS search, the variance detectors with their equations, estimated change dates and directions, the offline-benchmark settings and calibration, the simulation generator and its conditions (table `simconditions`), and the share-of-variation method. Thresholds: the check on fresh seeds (`repo-tables/validation`), days when $\hat\phi$ is zero, matched and real-data (bootstrap) thresholds. Test bed and checks: exclusions, intervals, the matched background, and the implementation checks (table `checks`). The tables `simconditions` and `checks` are typed in this file and also kept as `repo-tables/simconditions.tex` and `repo-tables/checks.tex`. |
| [`simulation.tex`](simulation.tex) | Further simulation results: other levels of autocorrelation ($\phi = 0$ and 0.6, `repo-tables/sim_appendix`), the variance detectors and false alarms, the Holt-Winters level weight, shares of variation in false alarms, the tuning of CUSUM and EWMA (`repo-tables/tuning`), variance changes (`repo-tables/variance`) and the offline benchmarks. |
| [`opal.tex`](opal.tex) | Further Opal results: the further retrospective adjustments, past-only against retrospective fits, offline segmentation (`repo-tables/offline_matches`), every version of the scoring rule (Table B.1 of the thesis, `tables/rule_variants`), holding under both rules, robustness of the chance verdicts, shares of variation on Opal, and the sensitivity runs (`repo-tables/opal_sens`). |

The two scoring rules are named as in the thesis: the **main rule** (an alarm within 7 days either side of any
boundary, in any direction; fixed before the runs) and the **direction-aware check** (an alarm 0 to 7 days after the
nearest boundary, in its direction). The results quoted in these files are computed from the files in `results/`.
