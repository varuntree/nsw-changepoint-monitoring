"""The monitoring procedure shared by the simulation, the real-data test bed and the Opal case study.

Given an adjusted series (from adjust.py), a detector and a threshold rule, the monitor works in
segments. At the start of each segment it fixes:
  centre c   for "level" series the median of the last 28 days (the relearn window after an alarm);
             0 for "surprise" series, which centre themselves (the forecast of normal re-learns);
  phi_hat    lag-1 autocorrelation of the de-levelled series over up to 365 past non-blind days
             (at least 56), bias-corrected phi + (1 + 3 phi)/n, clipped to [0, 0.9];
  threshold  h = table[detector](phi_hat) for the AR(1) rule, table[detector](0) for the independent
             rule (FOCuS-AR always uses its own table at phi_hat).
During the segment the scale s_t is re-estimated every 7 days as 1.4826 * MAD of the de-levelled
series over days [t-119, t-28) that are not blind (lag of 28 days, so a fresh change does not inflate
its own scale). The detector runs on z_t = (r_t - c) / s_t. An alarm ends the segment; the next 28 days
are blind (used to relearn) and a new segment starts after them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import detectors as D
from .adjust import yearly_memory

RELEARN = 28
SCALE_LAG, SCALE_WIN, SCALE_EVERY = 28, 91, 7
PHI_WIN, PHI_MIN = 365, 56


@dataclass
class Result:
    alarms: list = field(default_factory=list)     # alarm indices
    starts: list = field(default_factory=list)     # estimated change start per alarm
    dirs: list = field(default_factory=list)       # +1 rise / -1 fall: sign of the surprises from the estimated start to the alarm
    blind: np.ndarray | None = None                # boolean mask of relearn days
    phis: list = field(default_factory=list)       # phi_hat per segment
    hs: list = field(default_factory=list)         # threshold per segment
    seg_starts: list = field(default_factory=list)


@njit(cache=True)
def delevel(r, kind_level):
    """Level-keeping series: subtract the median of the previous 28 days. Surprise series: unchanged."""
    if not kind_level:
        return r.copy()
    n = r.shape[0]
    out = np.full(n, np.nan)
    buf = np.empty(28)
    for t in range(n):
        m = 0
        for j in range(max(0, t - 28), t):
            if not np.isnan(r[j]):
                buf[m] = r[j]
                m += 1
        if m >= 7 and not np.isnan(r[t]):
            out[t] = r[t] - np.median(buf[:m])
    return out


@njit(cache=True)
def running_scale(dl, blind, t0, t1):
    """Scale for days t0..t1-1, re-estimated every 7 days from [t-119, t-28) non-blind values."""
    out = np.empty(t1 - t0)
    buf = np.empty(SCALE_WIN)
    cur = np.nan
    for t in range(t0, t1):
        if (t - t0) % SCALE_EVERY == 0 or np.isnan(cur):
            m = 0
            lo = max(0, t - SCALE_LAG - SCALE_WIN)
            hi = max(0, t - SCALE_LAG)
            if hi - lo < 28:          # early in the series: use what exists before t
                lo, hi = max(0, t - SCALE_WIN), t
            for j in range(lo, hi):
                if not blind[j] and not np.isnan(dl[j]):
                    buf[m] = dl[j]
                    m += 1
            if m >= 14:
                med = np.median(buf[:m])
                cur = 1.4826 * np.median(np.abs(buf[:m] - med))
        out[t - t0] = cur
    return out


@njit(cache=True)
def phi_hat(dl, blind, t0):
    lo = max(0, t0 - PHI_WIN)
    xs = []
    for j in range(lo, t0):
        if not blind[j] and not np.isnan(dl[j]):
            xs.append(j)
    n = len(xs)
    if n < PHI_MIN:
        return 0.0
    # lag-1 pairs of consecutive valid days
    a = []
    b = []
    for i in range(1, n):
        if xs[i] == xs[i - 1] + 1:
            a.append(dl[xs[i - 1]])
            b.append(dl[xs[i]])
    if len(a) < PHI_MIN // 2:
        return 0.0
    A = np.array(a)
    B = np.array(b)
    A = A - A.mean()
    B = B - B.mean()
    den = np.sqrt((A * A).sum() * (B * B).sum())
    if den == 0:
        return 0.0
    ph = (A * B).sum() / den
    ph = ph + (1 + 3 * ph) / len(a)
    return min(max(ph, 0.0), 0.9)


def run(r, kind: str, det: str, rule: str, tables, t_start: int, t_end: int | None = None,
        method: str | None = None, relearn: int = RELEARN, init_centre_window: int = 28) -> Result:
    """Monitor r[t_start:t_end]. rule in {"indep", "ar1"}; tables: dict det -> callable h(phi)."""
    r = np.asarray(r, float)
    n = len(r) if t_end is None else t_end
    kind_level = kind == "level"
    blind = np.zeros(len(r), dtype=np.bool_)
    res = Result(blind=blind)
    hw = r.copy()                      # for YM: HW surprises before the yearly factor
    t0 = t_start
    first = True
    while t0 < n - 1:
        if method == "YM":
            f = yearly_memory(hw, blind | np.isnan(hw))
            r = hw - f
        dl = delevel(r, kind_level)
        ph = phi_hat(dl, blind, t0)
        if det == "FOCuS-AR":
            h = tables[det](ph)
        else:
            h = tables[det](ph if rule == "ar1" else 0.0)
        # centre
        lo = max(0, t0 - init_centre_window)
        w = r[lo:t0]
        w = w[~np.isnan(w)]
        c = (float(np.median(w)) if len(w) >= 7 else 0.0) if kind_level else 0.0
        s = running_scale(dl, blind, t0, n)
        z = (r[t0:n] - c) / s
        path = D.stat_path(det, z, ph)
        cross = np.flatnonzero(path > h)
        res.phis.append(ph)
        res.hs.append(h)
        res.seg_starts.append(t0)
        if len(cross) == 0:
            break
        a = int(cross[0])
        ta = t0 + a
        res.alarms.append(ta)
        res.starts.append(t0 + D.change_start(det, z, a, ph))
        st = max(0, min(a, D.change_start(det, z, a, ph)))
        res.dirs.append(1 if np.nansum(z[st:a + 1]) > 0 else -1)
        blind[ta + 1: ta + 1 + relearn] = True
        t0 = ta + 1 + relearn
        first = False
    return res
