"""What a ready-made detector does on the raw counts (Chapter 1 and the talk's opening example).

FOCuS on log daily train tap-ons, standardised by February 2020 (the last month before COVID-19), with the
threshold for independent days (h_table surprise, phi = 0, 5% target), restarting immediately after each alarm.
Writes results/opal/raw_focus_alarms.csv and results/opal/raw_focus_summary.csv, and the same for CUSUM
(raw_cusum_*.csv).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpd import detectors as D, score  # noqa: E402
from cpd.data import opal_daily  # noqa: E402

RES = ROOT / "results"


def main(det="FOCuS", name="focus"):
    h = json.loads((RES / "calib" / "h_table.json").read_text())["surprise"][det]["0.0"]
    lx = np.log(opal_daily("train"))
    ref = lx["2020-02-01":"2020-02-29"]
    zs = ((lx - ref.mean()) / ref.std())["2020-03-01":]
    z, dates = zs.to_numpy(), zs.index
    alarms, t0 = [], 0
    while t0 < len(z) - 1:
        path = D.stat_path(det, z[t0:], 0.0)
        cross = np.flatnonzero(path > h)
        if not len(cross):
            break
        a = t0 + int(cross[0])
        alarms.append(a)
        t0 = a + 1
    ad = pd.DatetimeIndex(dates[alarms])
    bounds = pd.DatetimeIndex(score.scored().date)
    near = [bool((abs((bounds - d).days) <= 7).any()) for d in ad]
    pd.DataFrame({"alarm_date": ad.date, "weekend": ad.dayofweek >= 5, "near_scored_boundary": near}).to_csv(
        RES / "opal" / f"raw_{name}_alarms.csv", index=False)
    years = (dates[-1] - dates[0]).days / 365.25
    pd.DataFrame([{"threshold": round(h, 3), "n_alarms": len(ad), "weekend": int((ad.dayofweek >= 5).sum()),
                   "near_scored_boundary": int(sum(near)), "years": round(years, 2),
                   "alarms_per_year": round(len(ad) / years, 1), "start": dates[0].date(), "end": dates[-1].date()}]
                 ).to_csv(RES / "opal" / f"raw_{name}_summary.csv", index=False)
    print(open(RES / "opal" / f"raw_{name}_summary.csv").read())


if __name__ == "__main__":
    main()
    main("CUSUM", "cusum")       # the same with CUSUM (k = 0.5) and its independent threshold
