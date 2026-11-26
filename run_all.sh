#!/usr/bin/env bash
# Rebuild every result from the data. Order matters: thresholds -> validation -> Opal -> test bed -> simulation.
# About 1.5 to 2 hours on 10 cores. Raw simulation runs go to results/sim/phi*/ (about 160 MB, not in the repository).
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
    if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
mkdir -p results/calib results/opal results/sim
echo "[$(date +%T)] tests";          $PY -m pytest -q tests
echo "[$(date +%T)] thresholds";     $PY -m cpd.calibrate results/calib/h_table.json 20000
echo "[$(date +%T)] validation";     $PY -m cpd.calibrate validate results/calib/h_table.json results/calib/validation.csv
echo "[$(date +%T)] opal case";      $PY scripts/opal_case.py > results/opal/case.log
echo "[$(date +%T)] test bed";       $PY scripts/testbed.py > results/testbed.log
echo "[$(date +%T)] sim core";       $PY scripts/sim_run.py core 10000 > results/sim/core.log
echo "[$(date +%T)] sim robust";     $PY scripts/sim_run.py robust 10000 > results/sim/robust.log
echo "[$(date +%T)] sim wide";       $PY scripts/sim_run.py wide 10000 > results/sim/wide.log
echo "[$(date +%T)] sim boot";       $PY scripts/sim_run.py boot 10000 > results/sim/boot.log
echo "[$(date +%T)] sim add-on";     $PY scripts/sim_addon.py EWMA-var
echo "[$(date +%T)] sim tuning";     $PY scripts/sim_addon.py --tuning CUSUM-k0.25 CUSUM-k1 EWMA-l0.05 EWMA-l0.2
echo "[$(date +%T)] sim summary";    $PY scripts/sim_analyse.py
echo "[$(date +%T)] done"
