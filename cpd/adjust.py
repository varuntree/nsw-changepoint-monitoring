"""Seasonal adjustment. Every method returns a series on the log scale plus its kind:

  kind "surprise": r_t = log x_t - log m_t, where m_t is the method's estimate of normal for day t
                   made from data up to day t-1 (past-only). In control r_t is centred near 0.
  kind "level":    s_t = log x_t - seasonal part; the level (trend) is kept, so a lasting change stays a
                   step. The monitor centres it at each (re)start. Retrospective methods are all of this
                   kind (they use the whole series, so they are not available to a live monitor).

Past-only:  A   same-weekday difference                      r_t = log x_t - log x_{t-7}
            B   median of the last four same weekdays         r_t = log x_t - median(log x_{t-7k}, k=1..4)
            HW  robust Holt-Winters (level + weekday)         Gelper, Fried and Croux (2010) style
            YM  HW with a yearly memory (see yearly_memory)   computed inside the monitor
            MSTL_p  MSTL refitted weekly on the past three years (weekly + yearly once 2 years exist)
            REG_p   weekday + yearly Fourier regression refitted weekly on the past 365 days
Retrospective (kind "level"): STL, MSTL, REG (same models fitted once to the whole series).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

PAST_ONLY = ("A", "B", "HW", "YM", "MSTL_p", "REG_p")
RETRO = ("STL", "MSTL", "REG")
HW_ALPHA, HW_GAMMA, HW_CAP = 0.05, 0.10, 3.0


def _log(x):
    x = np.asarray(x, float)
    return np.log(np.where(x > 0, x, np.nan))


def adj_A(x):
    lx = _log(x)
    r = np.full(len(lx), np.nan)
    r[7:] = lx[7:] - lx[:-7]
    return r


def adj_B(x):
    lx = pd.Series(_log(x))
    lags = pd.concat([lx.shift(7 * k) for k in range(1, 5)], axis=1)
    base = lags.median(axis=1, skipna=True).where(lags.notna().sum(axis=1) >= 2)
    return (lx - base).to_numpy()


@njit(cache=True)
def _hw(lx, dow, warm, alpha, gamma, cap):
    """Robust Holt-Winters on the log scale with a level and 7 weekday effects. Forecast for day t uses
    data up to t-1. Surprises are capped at cap * running scale before updating, so one odd day cannot
    drag the baseline; the scale itself is updated with the capped surprise (absorbs real shifts)."""
    n = lx.shape[0]
    r = np.full(n, np.nan)
    # warm-up: level = mean of first `warm` valid days, weekday effects = weekday means - level
    s_sum = np.zeros(7)
    s_cnt = np.zeros(7)
    tot = 0.0
    cnt = 0
    for t in range(warm):
        if not np.isnan(lx[t]):
            s_sum[dow[t]] += lx[t]
            s_cnt[dow[t]] += 1
            tot += lx[t]
            cnt += 1
    level = tot / max(cnt, 1)
    season = np.zeros(7)
    for k in range(7):
        season[k] = s_sum[k] / s_cnt[k] - level if s_cnt[k] > 0 else 0.0
    res = np.empty(warm)
    m = 0
    for t in range(warm):
        if not np.isnan(lx[t]):
            res[m] = lx[t] - level - season[dow[t]]
            m += 1
    scale = 1.4826 * np.median(np.abs(res[:m] - np.median(res[:m]))) if m > 3 else 0.05
    if scale <= 0:
        scale = 0.05
    for t in range(warm, n):
        k = dow[t]
        if np.isnan(lx[t]):
            continue
        e = lx[t] - (level + season[k])
        r[t] = e
        ec = min(max(e, -cap * scale), cap * scale)
        scale = 0.9 * scale + 0.1 * 1.2533 * abs(ec)
        level += alpha * ec
        season[k] += gamma * (1.0 - alpha) * ec
        mu = season.mean()
        for j in range(7):
            season[j] -= mu
    return r


def adj_HW(x, dates, warm=28):
    lx = _log(x)
    dow = pd.DatetimeIndex(dates).dayofweek.to_numpy().astype(np.int64)
    return _hw(np.ascontiguousarray(lx), dow, warm, HW_ALPHA, HW_GAMMA, HW_CAP)


def _fourier(doy, K=3):
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(cols)


def _reg_design(dates):
    d = pd.DatetimeIndex(dates)
    W = np.eye(7)[d.dayofweek][:, 1:]
    F = _fourier(d.dayofyear.to_numpy())
    return np.column_stack([np.ones(len(d)), W, F])


def adj_REG_retro(x, dates):
    """Weekday + yearly Fourier (K=3) + smooth trend (cubic spline, 1 knot per 90 days) fitted once by
    robust (Huber) regression to the whole log series; returns log x minus weekday and yearly parts."""
    import statsmodels.api as sm
    from patsy import dmatrix
    lx = _log(x)
    ok = ~np.isnan(lx)
    X = _reg_design(dates)
    tt = np.arange(len(lx))
    df = max(4, len(lx) // 90)
    T = np.asarray(dmatrix(f"bs(t, df={df}, degree=3, include_intercept=True) - 1", {"t": tt}))
    # the spline basis already spans the constant, so the intercept column is dropped (avoids collinearity)
    XX = np.column_stack([X[:, 1:], T])
    fit = sm.RLM(lx[ok], XX[ok], M=sm.robust.norms.HuberT()).fit()
    seas = X[:, 1:] @ fit.params[: X.shape[1] - 1]
    return lx - seas


def adj_REG_past(x, dates, window=365, refit=7, min_obs=56, t_from=0, base=None):
    """Same weekday + Fourier model (no trend; intercept = recent level) refitted every `refit` days on
    the previous `window` days; the forecast of normal for the next days is level + weekday + yearly."""
    import statsmodels.api as sm
    lx = _log(x)
    X = _reg_design(dates)
    n = len(lx)
    r = np.full(n, np.nan) if base is None else np.array(base, float).copy()
    params = None
    for t in range(max(0, t_from - (t_from % refit)), n):
        if (t % refit == 0) or params is None:
            lo = max(0, t - window)
            ok = ~np.isnan(lx[lo:t])
            if ok.sum() >= min_obs:
                Xw = X[lo:t][ok]
                # yearly terms only once a full year of history is available
                use = slice(None) if (t - lo) >= 365 else slice(0, 7)
                fit = sm.RLM(lx[lo:t][ok], Xw[:, use], M=sm.robust.norms.HuberT()).fit()
                params = np.zeros(X.shape[1])
                params[use] = fit.params
                # intercept follows the last 28 days (recent level)
                rec = slice(max(lo, t - 28), t)
                okr = ~np.isnan(lx[rec])
                if okr.sum() >= 7:
                    params[0] += np.median(lx[rec][okr] - X[rec][okr] @ params)
        if params is not None and not np.isnan(lx[t]):
            r[t] = lx[t] - X[t] @ params
    return r


def adj_STL_retro(x, dates):
    from statsmodels.tsa.seasonal import STL
    lx = pd.Series(_log(x)).interpolate(limit_direction="both").to_numpy()
    res = STL(lx, period=7, robust=True).fit()
    out = lx - res.seasonal
    out[np.isnan(_log(x))] = np.nan
    return out


# Days on which the Opal readers were switched off (announced fare-free days: 21-25 Nov 2022, 21-22 Sep 2024,
# 26 May 2025): the published counts measure no travel. They are treated as missing when fitting MSTL, whose
# yearly component otherwise copies a 99% fall into every year (a spike of up to 6.6 on the log scale).
READER_OFF = pd.DatetimeIndex(list(pd.date_range("2022-11-21", "2022-11-25")) + list(pd.date_range("2024-09-21", "2024-09-22"))
                              + [pd.Timestamp("2025-05-26")])


def _mask_reader_off(lx, dates):
    lx = np.array(lx, float)
    lx[pd.DatetimeIndex(dates).isin(READER_OFF)] = np.nan
    return lx


def adj_MSTL_retro(x, dates):
    from statsmodels.tsa.seasonal import MSTL
    lraw = _log(x)
    lx = pd.Series(_mask_reader_off(lraw, dates)).interpolate(limit_direction="both").to_numpy()
    res = MSTL(lx, periods=(7, 365), stl_kwargs={"robust": True}).fit()
    seas = np.asarray(res.seasonal).sum(axis=1)
    out = lraw - seas
    out[np.isnan(_log(x))] = np.nan
    return out


def adj_MSTL_past(x, dates, window=1095, refit=7, min_obs=56):
    """MSTL refitted every `refit` days on the previous `window` days. The forecast of normal for day t is
    the last trend value + the weekly component for t's weekday from the last fit + (once more than
    730 days of history exist, since statsmodels drops the yearly period otherwise) the yearly
    component 364 days earlier. Before that only the weekly part is removed."""
    from statsmodels.tsa.seasonal import MSTL, STL
    lx = _log(x)
    lfit = _mask_reader_off(lx, dates)
    n = len(lx)
    d = pd.DatetimeIndex(dates)
    r = np.full(n, np.nan)
    state = None
    for t in range(n):
        if t % refit == 0 or state is None:
            lo = max(0, t - window)
            seg = pd.Series(lfit[lo:t]).interpolate(limit_direction="both").to_numpy()
            if np.isfinite(seg).sum() >= min_obs:
                if len(seg) > 730:
                    fit = MSTL(seg, periods=(7, 365), stl_kwargs={"robust": True}).fit()
                    seas = np.asarray(fit.seasonal)
                    weekly, yearly = seas[:, 0], seas[:, 1]
                else:
                    fit = STL(seg, period=7, robust=True).fit()
                    weekly, yearly = np.asarray(fit.seasonal), None
                trend = np.asarray(fit.trend)
                wk = {}
                for j in range(1, 8):
                    wk[d[t - j].dayofweek] = weekly[-j]
                state = (lo, trend[-1], wk, yearly)
        if state is None or np.isnan(lx[t]):
            continue
        lo, tr, wk, yearly = state
        y = 0.0
        if yearly is not None and t - 364 - lo >= 0:
            y = yearly[t - 364 - lo]
        r[t] = lx[t] - (tr + wk.get(d[t].dayofweek, 0.0) + y)
    return r


def adjust(method: str, x, dates):
    """Return (series, kind)."""
    if method == "raw":
        lx = _log(x)
        return lx, "level"
    if method == "A":
        return adj_A(x), "surprise"
    if method == "B":
        return adj_B(x), "surprise"
    if method in ("HW", "YM"):
        return adj_HW(x, dates), "surprise"
    if method == "MSTL_p":
        return adj_MSTL_past(x, dates), "surprise"
    if method == "REG_p":
        return adj_REG_past(x, dates), "surprise"
    if method == "STL":
        return adj_STL_retro(x, dates), "level"
    if method == "MSTL":
        return adj_MSTL_retro(x, dates), "level"
    if method == "REG":
        return adj_REG_retro(x, dates), "level"
    raise ValueError(method)


@njit(cache=True)
def yearly_memory(r, blocked, lag=364, half=7, min_valid=8, w_first=0.5):
    """Yearly factor for day t: median of the HW surprises in [t-lag-half, t-lag+half] (and the same
    window one year earlier when available), skipping days marked `blocked` (within 28 days after a past
    alarm, or missing). Shrunk by w_first while only one year of memory exists. Pre-registered 22 Oct."""
    n = r.shape[0]
    f = np.zeros(n)
    buf = np.empty(2 * (2 * half + 1))
    for t in range(n):
        m = 0
        years = 0
        for yrs in (1, 2):
            c = t - yrs * lag
            if c - half < 0:
                continue
            got = 0
            for j in range(c - half, c + half + 1):
                if not blocked[j] and not np.isnan(r[j]):
                    buf[m] = r[j]
                    m += 1
                    got += 1
            if got > 0:
                years += 1
        if m >= min_valid:
            med = np.median(buf[:m])
            f[t] = (w_first if years < 2 else 1.0) * med
    return f
