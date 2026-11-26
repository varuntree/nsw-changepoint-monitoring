"""Score the eight further retrospective seasonal adjustments of Opal fitted in exploratory work (STL, MSTL, MSTL
with the free week smoothed, Fourier regression, GAM, classical decomposition, Prophet, TBATS), exactly as
produced by their own code, with the same monitor and scoring as everything else. The adjusted files go in
data/independent/ (not in this repository).
All are retrospective, level-keeping series on the log scale. Output: results/opal/retro_extra.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import calibrate, monitor, score  # noqa: E402
from cpd.data import START, END  # noqa: E402

SRC = HERE / "data" / "independent"
FILES = {"opal_stl_adjusted": "STL", "opal_mstl_original": "MSTL", "opal_mstl_sensitivity": "MSTL, free week smoothed",
         "opal_fourier_adjusted": "Fourier regression", "opal_gam_adjusted": "GAM",
         "opal_classical_adjusted": "classical decomposition", "opal_prophet_adjusted": "Prophet",
         "opal_tbats_adjusted": "TBATS"}
TABLE = HERE / "results" / "calib" / "h_table.json"


def job(a):
    f, det, rule = a
    df = pd.read_csv(SRC / f"{f}.csv", parse_dates=["date"]).set_index("date")
    s = df["seasonally_adjusted"].astype(float).reindex(pd.date_range(START, END, freq="D"))
    dates = s.index
    t0 = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    tabs = calibrate.load(TABLE, "level")
    res = monitor.run(s.to_numpy(), "level", det, rule, tabs, t0)
    sc = score.score(res.alarms, dates, t0, n_shift=500, dirs=res.dirs)
    sc.pop("first_alarm_rel", None)
    sc.update(file=f, method=FILES[f], detector=det, rule=rule)
    return sc


if __name__ == "__main__":
    jobs = [(f, d, r) for f in FILES for d in ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM") for r in ("indep", "ar1")
            if not (d == "FOCuS-AR" and r == "indep")]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(job, jobs))
    pd.DataFrame(rows).to_csv(HERE / "results" / "opal" / "retro_extra.csv", index=False)
    print("done", len(rows))
