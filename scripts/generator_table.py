"""The fitted simulation generator as a table (repository table generator).

Refits the generator exactly as cpd.sim.fit_generator (log Opal train tap-ons 2023-2025, Huber regression, disruption,
fault and reader-off days excluded) to obtain standard errors, asserts that the coefficients equal
results/sim/generator.json, and writes one row per term: the log-scale coefficient, its robust standard error and the
effect as a % change in daily tap-ons, 100 (exp(b) - 1).
Output: results/sim/generator_coefficients.csv"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import sim as S  # noqa: E402
from cpd.data import opal_daily  # noqa: E402


def main():
    g = json.loads((HERE / "results" / "sim" / "generator.json").read_text())
    x = opal_daily("train")
    ev = pd.read_csv(HERE / "data" / "opal" / "events_verified.csv", parse_dates=["start", "end"])
    s = x.loc["2023-01-01":"2025-12-31"]
    lx = np.log(s.where(s > 0))
    bad = pd.Series(False, index=s.index)
    for _, r in ev.iterrows():
        if r["kind"] in ("disruption", "fault") or "reader" in str(r["event"]).lower() or "fare-free" in str(r["event"]).lower():
            bad |= (s.index >= r["start"]) & (s.index <= r["end"])
    W, F = S.design(s.index)
    pub, xmas, sch = S.calendar_dummies(s.index)
    metro = (s.index >= "2024-08-19").astype(float)
    tt = np.arange(len(s)) / 365.25
    X = np.column_stack([np.ones(len(s)), tt, metro, W, F, pub, xmas, sch])
    ok = lx.notna().to_numpy() & ~bad.to_numpy()
    fit = sm.RLM(lx.to_numpy()[ok], X[ok], M=sm.robust.norms.HuberT()).fit()
    b, se = fit.params, fit.bse
    assert np.allclose(b[3:9], g["weekday"][1:]) and np.allclose(b[9:15], g["fourier"])
    assert np.isclose(b[15], g["pub"]) and np.isclose(b[16], g["xmas"]) and np.isclose(b[17], g["school"])
    names = (["intercept (1 Jan 2023)", "trend (per year)", "Metro line open (from 19 Aug 2024)"]
             + [f"weekday: {d} vs Monday" for d in ("Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")]
             + [f"yearly Fourier {f}({k})" for k in (1, 2, 3) for f in ("sin", "cos")]
             + ["public holiday", "Christmas period (24 Dec to 5 Jan)", "school holidays"])
    groups = ["level"] * 3 + ["weekday"] * 6 + ["yearly"] * 6 + ["calendar"] * 3
    rows = [dict(term=n, group=gr, coefficient=float(v), se=float(e),
                 pct_effect=float(100 * (np.exp(v) - 1)) if gr in ("weekday", "calendar") or n.startswith(("trend", "Metro")) else np.nan)
            for n, gr, v, e in zip(names, groups, b, se)]
    rows.append(dict(term="level mu (mid-2024, as simulated)", group="level", coefficient=g["mu"], se=np.nan,
                     pct_effect=np.nan))
    for k, lab in (("phi", "lag-one autocorrelation of the residuals"), ("sigma", "robust SD of the residuals (1.4826 MAD)"),
                   ("sd_raw", "ordinary SD of the residuals"), ("excess_kurtosis", "excess kurtosis of the residuals"),
                   ("n_used", "days used"), ("n_excluded", "days excluded (2023-2025)")):
        rows.append(dict(term=lab, group="noise" if k not in ("n_used", "n_excluded") else "fit", coefficient=float(g[k]),
                         se=np.nan, pct_effect=np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "sim" / "generator_coefficients.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
