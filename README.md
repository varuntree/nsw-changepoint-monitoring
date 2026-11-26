# Detecting real changes in daily public data streams

Code, data and results for the master's thesis

> **Detecting Real Changes in Daily Public Data Streams: Evaluating Online Changepoint Methods on NSW Opal Train
> Travel and Electricity Demand.** Yihan Zhang, Sihan Wang and Varun Prasad Gopisetti. Supervisor: Bao Khue Tran.
> DATA5925 Data Science Project, Group 5, School of Mathematics and Statistics, UNSW Sydney, November 2026.
> Submitted for the degree of Master of Data Science and Decisions.

Appendix A of the thesis describes this repository. The state that goes with the submitted thesis is the tag
`v1.0-thesis`. The live monitor of Chapter 7 is at **https://varuntree.github.io/nsw-normal-day-monitor/**.

## What the thesis does

The thesis asks whether a live monitor that sees only past days can detect real changes in daily public data
streams that have weekly, yearly and holiday patterns and day-to-day **autocorrelation**. It compares CUSUM, EWMA,
FOCuS, FOCuS with AR(1) noise (FOCuS-AR), the rank-based change point model (CPM) and the **capped detectors**
(capped CUSUM, EWMA and FOCuS, which limit each day's standardised value to ±3), with PELT, Binary Segmentation and
WBS as offline benchmarks.

Every detector runs through the same four daily steps:

1. a past-only forecast of a normal day: the **same-weekday difference**, the **median of four**, **Holt-Winters**,
   the **yearly memory**, or MSTL or a regression refitted on the past (retrospective STL, MSTL and regression fits
   are benchmarks only);
2. the **forecast error**: observed log count minus forecast, divided by a running MAD-based scale;
3. the detector statistic;
4. a threshold calibrated by simulation for a 5% chance of a false alarm in a year, either for independent days
   (the independent threshold) or for AR(1) autocorrelation (the AR(1) threshold), with a 28-day relearn period
   after each alarm.

The comparison has four parts: a simulation fitted to NSW Opal train tap-ons (Chapter 5); a test bed of drops
inserted into the real Opal series (Chapter 6); NSW train tap-ons from 2020 to 2026, scored against 15 real episodes
fixed before the final runs (Chapter 6); and NSW electricity demand, with the live page (Chapter 7). The **fixed
monitor** (the yearly memory with FOCuS and the AR(1) threshold) was chosen in advance, before any yearly-memory
result was seen.

Real-data alarms are scored under two rules, named as in the thesis. The **main rule** was fixed before the runs: an
episode is caught if an alarm falls within 7 days either side of one of its boundaries, in either direction. The
**direction-aware check** counts an alarm only 0 to 7 days after the nearest boundary and in its expected direction.

## Contents

| Path | What |
|---|---|
| `cpd/` | The method package: forecasts (`adjust.py`), detector kernels (`kernels.py`, `detectors.py`), the monitoring procedure (`monitor.py`), threshold calibration (`calibrate.py`), the simulation generator (`sim.py`), scoring against the events with a circular-shift chance baseline (`score.py`) and the Opal loader (`data.py`). |
| `scripts/` | One script per study and check (`sim_*.py`, `testbed*.py`, `opal_*.py`, `electricity*.py`, `elec_offline.py`, `offline_rules.py`, `past_retro_pairs.py`, `retro_extra.py`, `late2022.py`, `generator_table.py`, `cpm_*.py`, `focus_check.py`, `kernel_arl.py`, `runtime*.py`, `arl0.py`, `shares_extra.py`, `recommend.py`, `mstl_holiday_check.py`, `crosscheck_*.py`); the figure scripts (`figures.py`, `thesis_figures.py`, `fig_*.py`, with their styles `figstyle.py`, `figstyle_extra.py` and `figure_names.py`, and the Inter font in `fonts/`); the table scripts (`tables.py`, `thesis_tables.py`); `numbers_extra.py` (every derived number the thesis quotes), `check_numbers.py` and `check_ledger.py` (the number check); `scorecard.py` (the quick check); and `appendix_pdf.py` (the appendix PDFs). |
| `tests/` | Validation tests: the detectors against published average run lengths, the exact FOCuS statistic against a direct search, CPM against SciPy's Mann-Whitney test and with tied values, the offline methods against `ruptures`, the variance detectors, the results files against the files they are built from, and the number check on fixtures (`tests/vectors/`, `tests/fixtures/`). |
| `ledger/`, `quoted_spec.csv`, `check_numbers_whitelist.csv` | The number ledger: one row per number quoted in the thesis, with the printed value and the result file, row and column it is recomputed from; and the rules for numbers that are not results (years, dates, settings). |
| `data/` | Opal and electricity data, event lists, fetch scripts, and the published CPM limits used to check our CPM kernel (`data/cpm/`). Licences and attribution are in [`data/LICENSES.md`](data/LICENSES.md). |
| `results/` | Result files (CSV and JSON). |
| `figures/` | Every figure (PDF, with PNG copies): the fifteen printed in the thesis and the figures we cut. |
| `tables/` | The seven computed tables printed in the thesis (LaTeX), exactly as printed. |
| `repo-tables/` | The tables the thesis cites from the repository (Appendix A), under the names it uses. |
| `appendix/` | The repository appendix: method details, further simulation results and further Opal results, as PDFs with their LaTeX sources; see [`appendix/README.md`](appendix/README.md). |
| `dashboard/` | The live page: `build_data.py` runs the electricity and Opal monitors and writes `data.json` for `index.html`; see [`dashboard/README.md`](dashboard/README.md). |
| `live/` | The electricity monitor as a script (`update.py`, `data_fetch.py`, `config.json`) and its simple page `dashboard.html`. |
| `.github/workflows/dashboard.yml` | The daily job that builds the live page and publishes it. |

## Repository appendix

The method details and further results that the printed thesis leaves out are in three short PDFs:

- [Method details](appendix/method-details.pdf) (`appendix/method-details`): the sources of the event list and the
  alarm classes; the forecast, detector and variance-detector details with their equations; the offline-benchmark
  settings and calibration; the simulation generator; and the validation of the thresholds, the test bed and the
  implementation.
- [Additional simulation results](appendix/simulation.pdf) (`appendix/simulation`): other levels of autocorrelation,
  variance detectors, the Holt-Winters level weight, shares of variation and offline benchmarks.
- [Additional Opal results](appendix/opal.pdf) (`appendix/opal`): the further retrospective adjustments, past-only
  against retrospective fits, offline segmentation, every version of the scoring rule, the sensitivity of the chance
  verdicts and shares of variation.

The LaTeX sources are next to the PDFs; `make appendix-pdf` compiles them again.

## Install

Python 3.12 is required. The code was run on macOS (Apple silicon) and Linux.

```bash
git clone https://github.com/varuntree/nsw-changepoint-monitoring.git
cd nsw-changepoint-monitoring
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Package versions are pinned in `requirements.txt`. The Makefile uses `.venv/bin/python` if it exists and `python3`
otherwise; to choose another interpreter, run `make PY=/path/to/python ...`. The FOCuS reference implementation
`focus_cpt` (GPL-3.0-or-later) is installed from PyPI as a dependency and is not distributed here. The appendix
PDFs need [tectonic](https://tectonic-typesetting.github.io); nothing else does.

## Commands

| Command | What it does | Time |
|---|---|---|
| `make tests` | Runs the validation tests. | about 1 minute |
| `make scorecard` | Re-scores every Opal monitor (10 forecasts or adjustments × 8 detectors × 2 threshold rules, 150 monitors) from `results/opal/alarms.csv` against the frozen events, with the same 1,000 circular shifts and seed as the full run. Prints the headline rows of Table 6.1, writes `results/opal/scorecard.csv`, and checks every scoring column against `results/opal/scores.csv`. Exits with code 1 on any difference. | about 1 minute on 10 cores |
| `make check-numbers` | Rebuilds the ledger (`results/extra/quoted.csv`) with `scripts/numbers_extra.py`, then checks that `tables/` and `repo-tables/` equal a fresh `scripts/thesis_tables.py` run, that every ledger row recomputes from `results/` to its printed value, and that every number in the table captions is in the ledger or the whitelist. See "The number check" below. | about 1 minute |
| `make results figures tables` | Rebuilds everything from the data: thresholds, validation, Opal, test bed, simulation, electricity back-test, kernel checks, run times and the derived numbers, then all figures and tables. | about 1.5 to 2 hours on 10 cores; several hours on a typical desktop |
| `make tables` | Writes `tables/` and `repo-tables/` from `results/` (`scripts/thesis_tables.py`, which runs `scripts/tables.py` and applies the wording of the thesis; no number changes). | seconds |
| `make figures` | Writes `figures/` from `results/`: `scripts/figures.py`, `scripts/thesis_figures.py`, the four scripts `scripts/fig_*.py`, and the simple live page. | a few minutes |
| `make mstl-check` | Prints the Christmas and Good Friday values of Section 6.2 (retrospective STL against MSTL) and the number of offline change dates on each series (`scripts/mstl_holiday_check.py`). | seconds |
| `make appendix-pdf` | Compiles `appendix/*.tex` into `appendix/*.pdf` with tectonic (`scripts/appendix_pdf.py`). | under a minute |
| `make live` | Refreshes the AEMO and Open-Meteo data and rebuilds `live/dashboard.html`. | about 2 minutes |
| `make data-opal` | Downloads the raw Opal daily files (about 2,500 files, 70 MB) and rebuilds the Opal CSVs (optional). | depends on the connection |
| `make data-electricity` | Downloads the raw AEMO files and the Open-Meteo archive (optional). | a few minutes |

**What `make results` runs.** `make results` runs `run_all.sh` and then the follow-up analyses listed in the
Makefile. `run_all.sh` writes the raw simulation runs to `results/sim/phi*/` as `.npz` batches (about 160 MB).
These are not in the repository; `scripts/sim_analyse.py` reduces them to `results/sim/summary/`.

**Reusing the Opal adjustments.** `scripts/opal_case.py` reuses `results/opal/adjusted.csv` if the file exists.
Delete it to recompute the ten seasonal adjustments as well (MSTL refitted weekly is the slow one).

**Random seeds.** All seeds are fixed in the scripts.

**Fonts.** The figures use the Inter font from `scripts/fonts/` (SIL Open Font Licence,
https://github.com/rsms/inter). With the pinned packages, `make figures` redraws the fifteen printed figures byte
for byte as they are in the thesis. Without the font files the figures fall back to DejaVu Sans.

**Dashboard screenshot.** The screenshot figure of `live/dashboard.html` (`F20_dashboard.png`, not printed in the
thesis) needs Google Chrome; without it that figure is skipped and the shipped file is kept.

## Reproducing each chapter

Each step below reads the data in `data/` or the result files of an earlier step; `make results` runs them all in
the right order. The figure and table scripts are listed in the next section.

| Chapter | What to run | Main result files |
|---|---|---|
| 3. Data | `make data-opal`, `make data-electricity` (optional: the processed data are in `data/`); `python scripts/late2022.py` for the two artefacts of late 2022 | `data/opal/`, `data/electricity/` |
| 4. Methods | `python -m cpd.calibrate results/calib/h_table.json 20000` (thresholds), `python -m cpd.calibrate validate results/calib/h_table.json results/calib/validation.csv`; `python scripts/runtime.py`, `python scripts/runtime_extra.py` (computing time, Table A.2) | `results/calib/`, `results/runtime*.csv` |
| 5. Simulation | `python scripts/sim_run.py core 10000` (and `robust`, `wide`, `boot`), `scripts/sim_addon.py`, `scripts/sim_analyse.py`, then `scripts/sim_offline_fixed.py`, `scripts/sim_localise.py`, `scripts/sim_hw_alpha.py`, `scripts/sim_boot_trim.py`, `scripts/shares_extra.py` | `results/sim/summary/`, `results/sim_trim/summary/` |
| 6. Opal | `python scripts/opal_case.py` (every monitor), `scripts/testbed.py` and `scripts/testbed_*.py` (inserted drops), then `scripts/opal_extra.py`, `opal_offline.py`, `opal_raw.py`, `opal_chance_robust.py`, `opal_variance.py`, `opal_bootstrap.py`, `opal_budget.py`, `offline_rules.py`, `past_retro_pairs.py`; `make scorecard` re-scores the alarms; `make mstl-check` gives the holiday values of Section 6.2 | `results/opal/`, `results/testbed/` |
| 7. Electricity | `python scripts/electricity.py`, `scripts/electricity_rules.py`, `scripts/elec_offline.py`; `make live` for today's data | `results/electricity/`, `live/` |
| 8. Discussion | `python scripts/recommend.py` (Table 8.2), `scripts/numbers_extra.py` (derived numbers) | `results/recommend*`, `results/extra/` |
| Appendix B | `python scripts/opal_extra.py` (every version of the scoring rule) and `scripts/opal_chance_robust.py` (their chance baselines) | `results/opal/rule_variants*.csv` |

### Files behind each thesis figure and table

Run the commands from the repository root. Figures are written to `figures/` and tables to `tables/`.

<!-- map:start -->
| Chapter | Item | File | Rebuilt by |
|---|---|---|---|
| Chapter 3: Data | Figure 3.1 | `figures/F02_opal_series.pdf` | `python scripts/thesis_figures.py F02` |
| Chapter 4: Methods | Figure 4.1 | `figures/F22_pipeline.pdf` | `python scripts/thesis_figures.py F22` |
|  | Figure 4.2 | `figures/F25_worked_example.pdf` | `python scripts/fig_worked_example.py` |
|  | Figure 4.3 | `figures/F23_scoring_rules.pdf` | `python scripts/thesis_figures.py F23` |
|  | Figure 4.4 | `figures/F24_inserted_drop.pdf` | `python scripts/thesis_figures.py F24` |
| Chapter 5: Simulation study | Figure 5.1 | `figures/F07_sim_false_alarms.pdf` | `python scripts/figures.py F07` |
|  | Figure 5.2 | `figures/F08_sim_mean_shifts.pdf` | `python scripts/figures.py F08` |
|  | Figure 5.3 | `figures/F10_sim_what_matters.pdf` | `python scripts/figures.py F10` |
|  | Table 5.1 | `tables/noise.tex` | `python scripts/thesis_tables.py` |
| Chapter 6: Opal results | Figure 6.1 | `figures/F26_three_cases.pdf` | `python scripts/fig_three_cases.py` |
|  | Table 6.1 | `tables/opal_primary.tex` | `python scripts/thesis_tables.py` |
|  | Figure 6.2 | `figures/F13_opal_grid.pdf` | `python scripts/thesis_figures.py F13` |
|  | Figure 6.3 | `figures/F27_episode_catches.pdf` | `python scripts/fig_episode_catches.py` |
|  | Figure 6.4 | `figures/F14_timeline_A_CUSUM.pdf` | `python scripts/thesis_figures.py F14` |
|  | Figure 6.5 | `figures/F16_covid.pdf` | `python scripts/thesis_figures.py F16` |
|  | Table 6.2 | `tables/testbed.tex` | `python scripts/thesis_tables.py` |
| Chapter 7: Application to NSW electricity demand | Figure 7.1 | `figures/F19_electricity_REGT_p_FOCuS.pdf` | `python scripts/thesis_figures.py F19` |
| Chapter 8: Discussion and conclusion | Figure 8.1 | `figures/F28_quiet_or_sensitive.pdf` | `python scripts/fig_quiet_or_sensitive.py` |
|  | Table 8.2 | `tables/recommend.tex` | `python scripts/thesis_tables.py` |
| Appendix A: Code, data and repository | Table A.1 | `tables/events.tex` | `python scripts/thesis_tables.py` |
|  | Table A.2 | `tables/runtime_per_day.tex` | `python scripts/thesis_tables.py` |
| Appendix B: Scoring-rule variants | Table B.1 | `tables/rule_variants.tex` | `python scripts/thesis_tables.py` |
<!-- map:end -->

Tables 2.1, 4.1 and 8.1 are typed in the thesis text. Figures 5.1, 5.2, 6.2, 6.4, 6.5 and 8.1 name the forecasts
as the text does (same-weekday difference, median of four, Holt-Winters, yearly memory). The other figures in
`figures/` (F03 to F06, F07 with the independent threshold, F09, F13 with near/all counts, F15, F19 with CUSUM,
F20, F21) are not printed.

### Tables cited from the repository

The thesis cites these tables by name ("repository table `generator`", Appendix A). They are in `repo-tables/`,
written by `python scripts/thesis_tables.py` from `results/`, except `simconditions.tex` and `checks.tex`, which are
typed (they are also in `appendix/method-details.tex`).

| Table | What | Main source |
|---|---|---|
| `opal_full` | Every one of the 150 monitors on Opal | `results/opal/scores.csv`, `rule_variants.csv` |
| `covid` | The first alarms at the start of COVID-19 | `results/opal/covid.csv` |
| `opal_variance` | The variance detectors on Opal | `results/opal/variance_*.csv`, `results/testbed/spread_summary.csv` |
| `testbed_full` | Every test-bed setting | `results/testbed/summary.csv`, `summary_6m.csv` |
| `opal_budget` | Opal at one alarm budget | `results/opal/budget*.csv` |
| `past_retro` | The past-only and retrospective pairs | `results/opal/past_retro_*.csv` |
| `offline_matches` | The offline change dates near each boundary | `results/opal/offline.csv`, `offline_changes.csv`, `adjusted.csv` |
| `opal_sens` | The sensitivity runs | `results/opal/sensitivity.csv`, `scores.csv`, `rule_variants.csv` |
| `validation` | The threshold validation on fresh seeds | `results/calib/validation.csv` |
| `simconditions` | The simulation conditions | typed |
| `sim_appendix` | The simulation at other levels of autocorrelation | `results/sim/summary/` |
| `tuning` | The tuning of CUSUM and EWMA | `results/sim/summary/` |
| `variance` | The variance detectors in simulation | `results/sim/summary/variance.csv` |
| `boot_noise` | The simulation with resampled residuals, in full | `results/sim/summary/` |
| `delay` | Delays | `results/sim/summary/delay.csv` |
| `runtime` | Computing times of every method, offline methods included (Table A.2 prints the per-day times) | `results/runtime*.csv` |
| `checks` | The implementation tests | typed; values from `results/calib/`, `results/extra/crosscheck_ewma_cpm.csv`, `crosscheck_offline.csv` |
| `generator` | The generator's coefficients | `results/sim/generator_coefficients.csv` |

`repo-tables/retro_extra.tex` (the further retrospective adjustments) is also written; `appendix/opal` describes these results.

## Events and scoring

`data/opal/events_scored_FROZEN.csv` holds the 18 scored boundaries (15 episodes). They were chosen from the 71
sourced events in `data/opal/events_verified.csv` before any monitor was scored, and the list has not been
changed since. `data/opal/events_scored_RULES.md` gives the selection rule and the dated note of 23 October 2026
that records the direction-aware check. The thesis reports both the main rule and the direction-aware check, and
Appendix B scores the leading monitors under every version of the rule.

A monitor's alarms are compared with chance by shifting the whole alarm sequence by a random offset
(`cpd/score.py`). The electricity back-test uses its own frozen list in `data/electricity/`.

## The number check

`make check-numbers` checks what needs only the repository: the tables are fresh, every ledger row is recomputed
from `results/`, and the table captions are linted. It cannot confirm that the chapter prose prints exactly the
ledger's values, because the thesis source is not in this repository. The ledger's file and anchor columns follow the
chapter files as they were when the ledger was built. The final wording of the thesis moved, cut or repeated some
sentences but changed no number, so every row still recomputes; a few numbers that the final text prints in new
places have no row of their own.

## Independently written code

An EWMA and CPM, and a PELT, Binary Segmentation and WBS, were written a second time, independently, from the same
definitions, and used to verify our implementation (repository table `checks`). Eight further retrospective seasonal
adjustments of Opal were also fitted in exploratory work. That code and those adjusted series are not in this
repository. The comparisons are `results/extra/crosscheck_ewma_cpm*.csv` (`scripts/crosscheck_ewma_cpm.py`),
`results/extra/crosscheck_offline*.csv` (`scripts/crosscheck_offline.py`, `scripts/elec_offline.py`) and
`results/opal/retro_extra.csv` (`scripts/retro_extra.py`). These scripts need the code and files in
`data/independent/`; without them `make results` keeps the shipped files, and the one test that runs that code is
skipped.

## Live page

The live page is **https://varuntree.github.io/nsw-normal-day-monitor/**. It runs the electricity monitor that the
back-test of Chapter 7 chose, as set in `dashboard/config.json` and `live/config.json`: the past-only regression
forecast with weekday, time of year and temperature; FOCuS; and the AR(1) threshold. It also shows the Opal monitor
(Holt-Winters with EWMA) on the Opal series to 28 September 2026 against the 15 scored episodes.

How it is built:

- `dashboard/build_data.py` downloads the latest AEMO demand and Open-Meteo temperature (`dashboard/fetch.py`), runs
  the monitor from 2020 with the code in `cpd/`, and writes `dashboard/data.json` for the page `dashboard/index.html`,
  which is the live page.
- `.github/workflows/dashboard.yml` runs `dashboard/build_data.py` every day at 00:40 NEM time (14:40 UTC, and again
  at 20:40 UTC if a feed was late), checks the page's detector against the Python alarms, and publishes `index.html`
  and `data.json` to the repository `varuntree/nsw-normal-day-monitor`, whose GitHub Pages site is the live page.
  Publishing needs the deploy-key secret described in [`dashboard/README.md`](dashboard/README.md).
- `live/update.py` runs the same monitor as a script and writes a simple local page, `live/dashboard.html` (status,
  the last 120 days, recent alarms with likely explanations).

GitHub pauses scheduled workflows in repositories with no activity for 60 days. To restart it, open the Actions tab
and choose "Enable workflow" or "Run workflow".

## Results files

| Directory | Contents | Thesis |
|---|---|---|
| `results/calib/` | `h_table.json` (and `h_table_flat.csv`): thresholds by detector, rule and lag-one autocorrelation φ. `validation.csv`: realised false-alarm rates on fresh seeds. `arl0.csv`, `kernel_arl.csv`, `ewma_var_arl.csv`: in-control run lengths. `focus_check*.csv`: exact FOCuS against the published algorithm. `cpm_hd_check.csv`, `cpm_age.csv`, `cpm_guard_check.csv`: CPM against published limits, and the tied-values guard. `calib_time.csv`. | Ch 4, 5; `appendix/method-details` |
| `results/sim/summary/` | `fa.csv`: false alarms. `detect.csv`, `delay.csv`: detection and delay. `variance.csv`: variance changes. `tuning.csv`, `hw_alpha.csv`: tuning and the Holt-Winters weight. `offline*.csv`, `localise.csv`: offline benchmarks and localisation. `shares*.csv`, `ranges.csv`, `tornado.csv`: what matters. `../generator.json`, `../generator_resid.csv`, `../generator_coefficients.csv`: the fitted simulation generator. | Ch 5; `appendix/simulation` |
| `results/sim_trim/summary/` | Robustness of the resampled-noise result: the same simulation with the most extreme days removed from the resampling pool (`scripts/sim_boot_trim.py`; raw runs in `results/sim_trim/phi*/` are not in the repository). | Ch 5 |
| `results/testbed/` | `pairs.csv`, `spread_pairs.csv`, `matched_pairs.csv`: paired runs with and without an inserted change (Figure 4.4). `summary*.csv`, `rq3.csv`, `spread_summary.csv`: detection gain. `contrasts*.csv`, `block_sensitivity.csv`: block-bootstrap contrasts. `matched*.csv`: test bed at a matched background. | Ch 6 |
| `results/opal/` | `adjusted.csv`: the ten adjusted series. `alarms.csv`: every alarm with direction and estimated change date. `scores.csv`: the scorecard. Further analyses: chance baselines (`familywise*.csv`, `chance_local*.csv`, `bothrules_survivors*.csv`, `unexplained_null.csv`), scoring-rule versions and data variants (`rule_variants*.csv`, `bothrules_equivalence.csv`, `hw_ewma_rule_versions.csv`, `no_covid_onset.csv`, `no_dq_2025_01.csv`), matched alarm budget (`budget*.csv`), real-data thresholds (`bootstrap*.csv`), variance detectors (`variance_*.csv`), unadjusted series (`raw_*.csv`), past-only against retrospective fits (`past_retro_*.csv`), offline segmentation (`offline*.csv`), the extreme days of 2022 (`extreme_2022_*.csv`), `covid.csv`, `yearly.csv`, `heldout.csv`, `holds.csv`, `sensitivity.csv`, `blind_misses.csv`, `announced_split.csv`, `focus_check_opal.csv`, and the further retrospective adjustments (`retro_extra.csv`). | Ch 6; Appendix B; `appendix/opal` |
| `results/electricity/` | `adjusted.csv`, `alarms.csv`, `scores.csv`: back-test. `rule_variants.csv`, `rule_credits.csv`: every scoring rule. `testbed.csv`: inserted drops. `offline_*.csv`: offline segmentation. | Ch 7 |
| `results/extra/` | Derived numbers the thesis quotes (`numbers_extra.py`, `shares_extra.py`), the ledger `quoted.csv` with its checks, and the comparisons with the independently written code (`crosscheck_ewma_cpm*.csv`, `crosscheck_offline*.csv`). | all chapters |
| `results/recommend*` | The recommendation table: every cell with its source value (`recommend.csv`), the pass rules as text (`recommend_rules.csv`), the table body (`recommend.tex`), footnotes and the row-3 ranking. | Ch 8 |
| `results/runtime*.csv` | Computing time per detector, per full monitor run, for the published FOCuS code and the spread detectors. | Ch 5; Table A.2 |

## Citation

```bibtex
@mastersthesis{zhang2026changes,
  author = {Zhang, Yihan and Wang, Sihan and Gopisetti, Varun Prasad},
  title  = {Detecting Real Changes in Daily Public Data Streams: Evaluating Online Changepoint Methods on {NSW}
            {Opal} Train Travel and Electricity Demand},
  school = {UNSW Sydney},
  type   = {Master's thesis (DATA5925 Data Science Project)},
  year   = {2026},
  month  = nov,
  note   = {Supervisor: Bao Khue Tran. Code and data: \url{https://github.com/varuntree/nsw-changepoint-monitoring},
            tag v1.0-thesis}
}
```

## Licence

The code is under the MIT licence (`LICENSE`). The data keep their sources' licences: Transport for NSW Opal Tap
Data under CC BY 4.0, AEMO data under AEMO's copyright permissions, and Open-Meteo data under CC BY 4.0. The
authors' event lists, results, figures and tables are under CC BY 4.0. The fonts in `dashboard/fonts/` and
`scripts/fonts/` are under the SIL Open Font License. Attribution and details are in [`data/LICENSES.md`](data/LICENSES.md).
