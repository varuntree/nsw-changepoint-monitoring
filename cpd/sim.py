"""Opal-like simulation.

Generator (fitted by fit_generator to log Opal train tap-ons 2023-01-01 .. 2025-12-31, robust regression,
fault and strike days excluded):
  log x_t = mu + weekday_{d(t)} + yearly Fourier(K=3)_t + calendar_t + e_t,   e_t AR(1), marginal sd sigma
calendar_t = public-holiday day effect (+ Christmas period effect) + school-holiday block effect, placed on
the real NSW calendar of the simulated dates (2023-01-01 onwards), switched on or off by condition.
Noise Normal or Student-t(5) scaled to unit variance, or "wide": Normal with the raw residual SD (sd_raw, about
0.097) instead of the robust sigma (about 0.049), so calendar shocks have their real size relative to the noise
(a public holiday about 5.6 SD, Christmas about 3 SD for two weeks); mean shifts are then in units of sd_raw.
Or "boot": a circular block bootstrap (28-day blocks) of the generator fit's residuals on its 773 normal days
(fit_residuals), with real tails and dependence (shifts still in robust-sigma units).
730 training days + 365 monitoring days.

Changes start at tau ~ U[60, 300] days into monitoring:
  mean shift  log level changes by -delta*sigma (delta in 0.5, 1, 2, 4; +1 also tested)
  variance    innovations multiplied by sqrt(v) from tau (v in 1.5, 2, 3), so the marginal variance
              becomes v sigma^2

For each simulated series we run every seasonal adjustment once, build z exactly as the monitor does at
the start of monitoring (running scale from past data, phi_hat from past data), compute every detector's
statistic path once, and record the first crossing for each threshold rule (independent, AR(1) table at
phi_hat, and the pipeline-oracle threshold calibrated from the no-change runs of the same condition).
"""
from __future__ import annotations

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from . import adjust as A
from . import detectors as D
from .kernels import ar1
from .monitor import delevel, phi_hat, running_scale
from .score import SCHOOL

TRAIN, MON = 730, 365
N = TRAIN + MON
DATES = pd.date_range("2023-01-01", periods=N, freq="D")
ADJ = ("oracle", "raw", "A", "B", "HW", "YM")
DETS = ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r")
SPREAD = ("FOCuS-var", "CUSUM-var", "EWMA-var")


def calendar_dummies(dates):
    d = pd.DatetimeIndex(dates)
    ph = holidays.Australia(subdiv="NSW", years=range(d.year.min(), d.year.max() + 1))
    pub = np.array([x.date() in ph for x in d], float)
    xmas = (((d.month == 12) & (d.day >= 24)) | ((d.month == 1) & (d.day <= 5))).astype(float)
    sch = np.zeros(len(d))
    for a, b in SCHOOL:
        sch[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = 1.0
    return pub, xmas, sch


def design(dates):
    d = pd.DatetimeIndex(dates)
    W = np.eye(7)[d.dayofweek][:, 1:]
    doy = d.dayofyear.to_numpy()
    F = np.column_stack([f(2 * np.pi * k * doy / 365.25) for k in (1, 2, 3) for f in (np.sin, np.cos)])
    return W, F


def fit_generator(x: pd.Series, verified_events: pd.DataFrame, out: Path) -> dict:
    """Fit the generator to log Opal train tap-ons in 2023-2025."""
    import statsmodels.api as sm
    s = x.loc["2023-01-01":"2025-12-31"]
    lx = np.log(s.where(s > 0))
    bad = pd.Series(False, index=s.index)
    for _, r in verified_events.iterrows():
        if r["kind"] in ("disruption", "fault") or "reader" in str(r["event"]).lower() or "fare-free" in str(r["event"]).lower():
            bad |= (s.index >= r["start"]) & (s.index <= r["end"])
    W, F = design(s.index)
    pub, xmas, sch = calendar_dummies(s.index)
    metro = (s.index >= "2024-08-19").astype(float)
    tt = (np.arange(len(s)) / 365.25)
    X = np.column_stack([np.ones(len(s)), tt, metro, W, F, pub, xmas, sch])
    ok = lx.notna().to_numpy() & ~bad.to_numpy()
    fit = sm.RLM(lx.to_numpy()[ok], X[ok], M=sm.robust.norms.HuberT()).fit()
    resid = lx.to_numpy() - X @ fit.params
    rr = pd.Series(resid, index=s.index)[ok]
    # AR(1) on consecutive valid days
    a = rr.to_numpy()
    cons = np.diff(rr.index.values).astype("timedelta64[D]").astype(int) == 1
    phi = float(np.corrcoef(a[:-1][cons], a[1:][cons])[0, 1])
    sd = float(1.4826 * np.median(np.abs(a - np.median(a))))
    from scipy.stats import kurtosis
    g = {"mu": float(fit.params[0] + fit.params[1] * 1.5 + fit.params[2] * 0.5),
         "weekday": [0.0] + list(map(float, fit.params[3:9])), "fourier": list(map(float, fit.params[9:15])),
         "pub": float(fit.params[15]), "xmas": float(fit.params[16]), "school": float(fit.params[17]),
         "metro": float(fit.params[2]), "trend_per_year": float(fit.params[1]),
         "phi": phi, "sigma": sd, "sd_raw": float(np.std(a)), "excess_kurtosis": float(kurtosis(a)),
         "n_used": int(ok.sum()), "n_excluded": int((~ok).sum())}
    out.write_text(json.dumps(g, indent=1))
    return g


RESID = Path(__file__).resolve().parents[1] / "results" / "sim" / "generator_resid.csv"
BOOT_BLOCK = 28
TRIM_SD = 5.0


def fit_residuals(x: pd.Series, verified_events: pd.DataFrame, out: Path = RESID) -> pd.Series:
    """The generator fit's residuals on the days it used (2023-2025 normal days), refitted exactly as
    fit_generator (same data, design and robust regression), for the block-bootstrap noise (noise "boot")."""
    import statsmodels.api as sm
    s = x.loc["2023-01-01":"2025-12-31"]
    lx = np.log(s.where(s > 0))
    bad = pd.Series(False, index=s.index)
    for _, r in verified_events.iterrows():
        if r["kind"] in ("disruption", "fault") or "reader" in str(r["event"]).lower() or "fare-free" in str(r["event"]).lower():
            bad |= (s.index >= r["start"]) & (s.index <= r["end"])
    W, F = design(s.index)
    pub, xmas, sch = calendar_dummies(s.index)
    metro = (s.index >= "2024-08-19").astype(float)
    tt = (np.arange(len(s)) / 365.25)
    X = np.column_stack([np.ones(len(s)), tt, metro, W, F, pub, xmas, sch])
    ok = lx.notna().to_numpy() & ~bad.to_numpy()
    fit = sm.RLM(lx.to_numpy()[ok], X[ok], M=sm.robust.norms.HuberT()).fit()
    rr = pd.Series(lx.to_numpy() - X @ fit.params, index=s.index)[ok]
    rr.rename("resid").to_csv(out, index_label="date")
    return rr


class Generator:
    def __init__(self, g: dict):
        self.g = g
        W, F = design(DATES)
        pub, xmas, sch = calendar_dummies(DATES)
        self.base = g["mu"] + W @ np.array(g["weekday"][1:]) + F @ np.array(g["fourier"])
        self.cal = g["pub"] * pub + g["xmas"] * xmas + g["school"] * sch

    def normal_part(self, calendar: bool):
        return self.base + (self.cal if calendar else 0.0)

    def draw(self, rng, phi, noise, calendar, change, size, tau):
        sig = self.g["sd_raw"] if noise == "wide" else self.g["sigma"]
        if noise in ("boot", "boottrim"):
            # circular block bootstrap (28-day blocks) of the fitted residuals on the generator's normal days: real
            # tails and real dependence; phi plays no part. Shift sizes stay in units of the robust sigma.
            # "boottrim" (robustness check): the same pool without the days more than TRIM_SD robust sigmas
            # from its median (23 days: 16 at New Year, Easter, the June long weekend and Christmas, which the
            # generator does not fully model, and 7 other single days, mostly weekends).
            if not hasattr(self, "_resid"):
                self._resid = pd.read_csv(RESID)["resid"].to_numpy()
            r = self._resid
            if noise == "boottrim":
                r = r[np.abs(r - np.median(r)) <= TRIM_SD * self.g["sigma"]]
            starts = rng.integers(0, len(r), N // BOOT_BLOCK + 2)
            e = np.concatenate([np.take(r, np.arange(a, a + BOOT_BLOCK), mode="wrap") for a in starts])[:N]
            e = e - np.median(r)
            lx = self.normal_part(calendar) + e
            if change == "var":
                lx[TRAIN + tau:] += (np.sqrt(size) - 1) * e[TRAIN + tau:]
            if change == "mean":
                lx[TRAIN + tau:] += size * sig
            return lx
        if noise in ("normal", "wide"):
            eps = rng.standard_normal(N)
        else:
            eps = rng.standard_t(5, N) * np.sqrt(3 / 5)
        if change == "var":
            eps[TRAIN + tau:] *= np.sqrt(size)
        e = ar1(N, phi, eps) * sig
        lx = self.normal_part(calendar) + e
        if change == "mean":
            lx[TRAIN + tau:] += size * sig
        return lx


def first_cross_multi(path, hs, t_from=0):
    out = []
    for h in hs:
        idx = np.flatnonzero(path[t_from:] > h)
        out.append(int(idx[0]) + t_from if len(idx) else -1)
    return out


def run_series(lx, gen, calendar, tables, spread=False, dets=None):
    """All adjustments x detectors for one series. Returns dict (adj, det) -> (max_stat, path) where the
    path covers the monitoring year; plus phi_hat per adjustment. `dets` restricts the detectors (used to add
    a detector to an existing run, scripts/sim_addon.py)."""
    x = np.exp(lx)
    out = {}
    phis = {}
    zs = {}
    blind = np.zeros(N, dtype=np.bool_)
    hw = A.adj_HW(x, DATES)
    for adj in ADJ:
        if adj == "oracle":
            r, kind = lx - gen.normal_part(calendar), "surprise"
        elif adj == "raw":
            r, kind = lx, "level"
        elif adj == "A":
            r, kind = A.adj_A(x), "surprise"
        elif adj == "B":
            r, kind = A.adj_B(x), "surprise"
        elif adj == "HW":
            r, kind = hw, "surprise"
        else:
            r, kind = hw - A.yearly_memory(hw, np.isnan(hw)), "surprise"
        dl = delevel(np.ascontiguousarray(r), kind == "level")
        ph = phi_hat(dl, blind, TRAIN)
        c = float(np.nanmedian(r[TRAIN - 28:TRAIN])) if kind == "level" else 0.0
        s = running_scale(dl, blind, TRAIN, N)
        z = (r[TRAIN:] - c) / s
        phis[adj] = (ph, kind)
        zs[adj] = z
        for det in (DETS + (SPREAD if spread else ())) if dets is None else dets:
            out[(adj, det)] = D.stat_path(det, z, ph)
    return out, phis, zs
