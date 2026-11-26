# Rebuild every number, figure and table of the thesis from the data in data/.
#   make tests                    validation tests (about 1 minute)
#   make scorecard                re-score the Opal alarms in results/opal/alarms.csv and check them
#                                 against results/opal/scores.csv (a few minutes)
#   make check-numbers            recompute every number the thesis quotes (ledger/) from results/
#   make results figures tables   everything (about 1.5 to 2 hours on 10 cores)
#   make live                     refresh the electricity data and rebuild live/dashboard.html
#   make mstl-check               the Christmas and Good Friday values of Section 6.2
#   make appendix-pdf             compile appendix/*.tex into appendix/*.pdf (needs tectonic)
# Python: .venv/bin/python if it exists, otherwise python3. Override with `make PY=/path/to/python ...`.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PY

.PHONY: all tests results figures tables scorecard check-numbers mstl-check appendix-pdf live data-opal data-electricity

all: tests results figures tables check-numbers

tests:
	$(PY) -m pytest -q tests

scorecard:
	$(PY) scripts/scorecard.py

results:
	./run_all.sh
	$(PY) scripts/opal_extra.py
	$(PY) scripts/opal_offline.py
	$(PY) scripts/opal_raw.py
	$(PY) scripts/opal_chance_robust.py
	$(PY) scripts/opal_chance_robust.py r4
	$(PY) scripts/opal_chance_robust.py r5
	$(PY) scripts/opal_variance.py
	@if [ -d data/independent ]; then $(PY) scripts/retro_extra.py; \
	 else echo "data/independent/ not present: results/opal/retro_extra.csv kept as shipped"; fi
	$(PY) scripts/opal_bootstrap.py
	$(PY) scripts/opal_budget.py
	$(PY) scripts/focus_check.py
	$(PY) scripts/testbed_analyse.py --block-months 3
	$(PY) scripts/testbed_contrasts.py --block-months 3
	$(PY) scripts/testbed_matched.py
	$(PY) scripts/testbed_spread.py
	$(PY) scripts/testbed_analyse.py --block-months 6 --out summary_6m.csv
	$(PY) scripts/testbed_contrasts.py --block-months 6 --out contrasts_6m.csv
	$(PY) scripts/testbed_matched.py analyse 6
	$(PY) scripts/testbed_blocks.py
	$(PY) scripts/sim_offline_fixed.py
	$(PY) scripts/sim_localise.py
	$(PY) scripts/cpm_age.py
	$(PY) scripts/cpm_hd_check.py
	$(PY) scripts/sim_hw_alpha.py
	$(PY) scripts/sim_boot_trim.py
	$(PY) scripts/electricity.py
	$(PY) scripts/electricity_rules.py
	$(PY) scripts/runtime.py
	$(PY) scripts/runtime_extra.py
	$(PY) scripts/kernel_arl.py
	$(PY) scripts/arl0.py
	$(PY) scripts/shares_extra.py
	$(PY) scripts/recommend.py
	$(PY) scripts/numbers_extra.py
	$(PY) scripts/cpm_guard_check.py
	$(PY) scripts/generator_table.py
	$(PY) scripts/past_retro_pairs.py
	@if [ -d data/independent ]; then $(PY) scripts/crosscheck_ewma_cpm.py; \
	 else echo "data/independent/ not present: results/extra/crosscheck_ewma_cpm*.csv kept as shipped"; fi
	$(PY) scripts/late2022.py
	$(PY) scripts/offline_rules.py
	@if [ -d data/independent ]; then $(PY) scripts/crosscheck_offline.py; \
	 else echo "data/independent/ not present: results/extra/crosscheck_offline*.csv kept as shipped"; fi
	$(PY) scripts/elec_offline.py

figures:
	$(PY) scripts/figures.py
	$(PY) scripts/thesis_figures.py
	$(PY) scripts/fig_worked_example.py
	$(PY) scripts/fig_three_cases.py
	$(PY) scripts/fig_episode_catches.py
	$(PY) scripts/fig_quiet_or_sensitive.py
	$(PY) live/update.py --no-refresh

# the Christmas and Good Friday values of Section 6.2 (retrospective STL against MSTL); prints only
mstl-check:
	$(PY) scripts/mstl_holiday_check.py

# the repository appendix as PDFs (appendix/*.pdf), compiled with tectonic in a temporary folder
appendix-pdf:
	$(PY) scripts/appendix_pdf.py

# tables/ (printed in the thesis) and repo-tables/ (cited from the thesis), in the wording of the final text
tables:
	$(PY) scripts/thesis_tables.py

# THESIS=/path/to/thesis (a folder with chapters/ and tables/) lints a LaTeX source token by token against the
# ledger; the ledger's anchors follow the chapter files as they were when the ledger was built (see README)
check-numbers:
	$(PY) scripts/numbers_extra.py > /dev/null
	$(PY) scripts/check_ledger.py $(if $(THESIS),--thesis $(THESIS),)

live:
	$(PY) live/update.py

# Raw data (optional: the processed files used by the analysis are already in data/)
data-opal:
	$(PY) data/opal/get_opal.py

data-electricity:
	$(PY) live/data_fetch.py all
