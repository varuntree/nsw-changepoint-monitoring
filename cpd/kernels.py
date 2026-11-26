"""Numerical kernels (numba). Every online kernel returns the monitoring statistic at every time
point for a standardised stream z (in-control mean 0, sd 1); an alarm is the first time the
statistic exceeds the threshold h. Offline kernels return estimated change positions.

Online, mean:   CUSUM (two-sided, reference k), EWMA (two-sided, exact variance), CPM Mann-Whitney
                (mid-ranks via sign scores, tie-corrected variance), FOCuS and FOCuS-AR as exact direct searches
                over every start (focus_cpt 0.1.11 omits the first one or two starts of each series it is given;
                since 24 Oct 2026 only variance FOCuS uses the package, whose gamma family has no such gap).
Online, spread: variance CUSUM on z^2 (Hawkins and Olwell 1998) and variance EWMA on z^2 (MacGregor and
                Harris 1993).
Offline:        PELT (l2), Binary Segmentation (l2), Wild Binary Segmentation; all on a unit-variance
                series, so one penalty means the same thing for every series.
"""
from __future__ import annotations

import numpy as np
from numba import njit


# ------------------------------------------------------------------------------------- online, mean
@njit(cache=True)
def cusum_path(z, k):
    n = z.shape[0]
    out = np.empty(n)
    sp = 0.0
    sm = 0.0
    for t in range(n):
        x = z[t]
        if not np.isnan(x):
            sp = max(0.0, sp + x - k)
            sm = max(0.0, sm - x - k)
        out[t] = max(sp, sm)
    return out


@njit(cache=True)
def cusum_start(z, k, t_alarm):
    """Estimated change start for a CUSUM alarm: the last time the active side was at zero."""
    sp = 0.0
    sm = 0.0
    lp = 0
    lm = 0
    for t in range(t_alarm + 1):
        x = z[t]
        if np.isnan(x):
            continue
        sp = max(0.0, sp + x - k)
        sm = max(0.0, sm - x - k)
        if sp == 0.0:
            lp = t + 1
        if sm == 0.0:
            lm = t + 1
    return lp if sp >= sm else lm


@njit(cache=True)
def ewma_path(z, lam):
    n = z.shape[0]
    out = np.empty(n)
    w = 0.0
    m = 0
    for t in range(n):
        x = z[t]
        if not np.isnan(x):
            w = lam * x + (1.0 - lam) * w
            m += 1
        if m == 0:
            out[t] = 0.0
        else:
            var = lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * m))
            out[t] = abs(w) / np.sqrt(var)
    return out


@njit(cache=True)
def cpm_mw_path(z, startup):
    """Mann-Whitney change point model (Hawkins and Deng 2010). At each time n:
    D_n = max_k |U_{k,n}| / sd(U_{k,n}), U_{k,n} = sum_{i<=k<j<=n} sgn(x_j - x_i).
    Ties score 0 (mid-rank convention); the variance uses the tie correction
    Var = n1 n2/3 * (n+1 - sum(t^3-t)/(n(n-1))), with tie counts updated online."""
    n_all = z.shape[0]
    out = np.zeros(n_all)
    xs = np.empty(n_all)
    U = np.zeros(n_all)
    n = 0
    tie_sum = 0.0     # sum over tie groups of (t^3 - t)
    for t in range(n_all):
        x = z[t]
        if np.isnan(x):
            out[t] = out[t - 1] if t > 0 else 0.0
            continue
        run = 0.0
        same = 0
        for k in range(n):
            d = x - xs[k]
            if d > 0:
                run += 1.0
            elif d < 0:
                run -= 1.0
            else:
                same += 1
            U[k] += run
        # a value equal to `same` earlier values grows that tie group from same to same+1
        if same > 0:
            a = same + 1.0
            b = float(same)
            tie_sum += (a * a * a - a) - (b * b * b - b)
        xs[n] = x
        n += 1
        if n < startup:
            out[t] = 0.0
            continue
        corr = (n + 1.0) - tie_sum / (n * (n - 1.0))
        if corr <= 1e-9:
            # every observed value so far is equal (corr is exactly 0 only then; with one value different it is 3):
            # every U is 0, so there is no evidence of a change. Without this guard 0/0 raised ZeroDivisionError
            # (found by edge-case tests of the independent CPM code, 27 Oct 2026)
            out[t] = 0.0
            continue
        best = 0.0
        for k in range(n - 1):
            n1 = k + 1
            n2 = n - n1
            if n1 < 2 or n2 < 2:
                continue
            v = abs(U[k]) / np.sqrt(n1 * n2 * corr / 3.0)
            if v > best:
                best = v
        out[t] = best
    return out


@njit(cache=True)
def focus_exact(y):
    """Exact FOCuS / GLR-CUSUM statistic for a change in mean from a known pre-change mean 0 (unit variance):
    D_t = max over all starts 0 <= s <= t of (sum_{i=s..t} y_i)^2 / (t - s + 1), i.e. twice the Gaussian
    log-likelihood ratio, the same scale as focus_cpt. Also returns the maximising start (first post-change index).
    O(n^2) direct search over every start, including the first observations, which focus_cpt 0.1.11 leaves out.
    y must have no missing values (detectors._focus removes them first)."""
    n = y.shape[0]
    S = np.zeros(n + 1)
    for i in range(n):
        S[i + 1] = S[i] + y[i]
    out = np.zeros(n)
    arg = np.zeros(n, np.int64)
    for t in range(n):
        best = -1.0
        bs = t
        for s in range(t + 1):
            d = S[t + 1] - S[s]
            v = d * d / (t + 1 - s)
            if v > best:
                best = v
                bs = s
        out[t] = best
        arg[t] = bs
    return out, arg


@njit(cache=True)
def focus_ar_exact(x, phi):
    """Exact FOCuS-AR statistic: GLR for a change in mean of a stationary AR(1) series with known phi, using the
    Prais-Winsten transform w_0 = sqrt(1 - phi^2) x_0, w_t = x_t - phi x_{t-1}. A change whose first post-change day
    is s gives w the mean mu * a_t with a_s = 1 (s >= 1) or a_0 = sqrt(1 - phi^2) (s = 0), and a_t = 1 - phi after
    the first post-change day. D_t = max over 0 <= s <= t of (sum a w)^2 / (sum a^2), on the scale of focus_cpt's
    "arp" family, which equals this maximum restricted to s >= 2 (checked in tests/test_focus_exact.py)."""
    n = x.shape[0]
    w = np.empty(n)
    c0 = np.sqrt(1.0 - phi * phi)
    w[0] = c0 * x[0]
    for i in range(1, n):
        w[i] = x[i] - phi * x[i - 1]
    W = np.zeros(n + 1)                     # W[k] = sum of w[1..k-1]
    for i in range(1, n):
        W[i + 1] = W[i] + w[i]
    q = 1.0 - phi
    out = np.zeros(n)
    arg = np.zeros(n, np.int64)
    for t in range(n):
        # s = 0: a_0 = c0 on w_0, then q on w_1..w_t
        num = c0 * w[0] + q * (W[t + 1] - W[1])
        den = c0 * c0 + q * q * t
        best = num * num / den
        bs = 0
        for s in range(1, t + 1):
            num = w[s] + q * (W[t + 1] - W[s + 1])
            den = 1.0 + q * q * (t - s)
            v = num * num / den
            if v > best:
                best = v
                bs = s
        out[t] = best
        arg[t] = bs
    return out, arg


# ------------------------------------------------------------------------------------- online, spread
@njit(cache=True)
def var_cusum_path(z, sig2_1):
    """Upward variance CUSUM on z^2 (Hawkins and Olwell 1998): S_t = max(0, S_{t-1} + z_t^2 - k),
    k = sig2_1 ln(sig2_1) / (sig2_1 - 1), designed for an increase to variance sig2_1."""
    k = sig2_1 * np.log(sig2_1) / (sig2_1 - 1.0)
    n = z.shape[0]
    out = np.empty(n)
    s = 0.0
    for t in range(n):
        x = z[t]
        if not np.isnan(x):
            s = max(0.0, s + x * x - k)
        out[t] = s
    return out


@njit(cache=True)
def ewma_var_path(z, lam):
    """Upward variance EWMA on z^2: the exponentially weighted mean square of MacGregor and Harris (1993),
    E_t = (1 - lam) E_{t-1} + lam z_t^2 with E_0 = 1 (the in-control variance of z). Like the mean EWMA it
    is returned standardised by its exact in-control sd under iid N(0, 1), where Var(z^2) = 2:
    D_t = (E_t - 1) / sqrt(2 lam / (2 - lam) (1 - (1 - lam)^(2m))), m = number of observed days.
    One-sided (an increase in spread raises D_t); missing days are skipped."""
    n = z.shape[0]
    out = np.empty(n)
    e = 1.0
    m = 0
    for t in range(n):
        x = z[t]
        if not np.isnan(x):
            e = lam * x * x + (1.0 - lam) * e
            m += 1
        if m == 0:
            out[t] = 0.0
        else:
            var = 2.0 * lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * m))
            out[t] = (e - 1.0) / np.sqrt(var)
    return out


# ------------------------------------------------------------------------------------- offline
@njit(cache=True)
def _seg_cost(S, Q, s, t):
    d = S[t] - S[s]
    return (Q[t] - Q[s]) - d * d / (t - s)


@njit(cache=True)
def pelt_l2(x, pen, min_size):
    """PELT (Killick, Fearnhead and Eckley 2012) with the l2 (change in mean) cost. Returns the start
    index of every new segment."""
    n = x.shape[0]
    S = np.zeros(n + 1)
    Q = np.zeros(n + 1)
    for i in range(n):
        S[i + 1] = S[i] + x[i]
        Q[i + 1] = Q[i] + x[i] * x[i]
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, np.int64)
    cand = np.empty(n + 1, np.int64)
    nc = 1
    cand[0] = 0
    for t in range(min_size, n + 1):
        best = np.inf
        bs = 0
        for j in range(nc):
            s = cand[j]
            if t - s < min_size:
                continue
            c = F[s] + _seg_cost(S, Q, s, t) + pen
            if c < best:
                best = c
                bs = s
        F[t] = best
        last[t] = bs
        k = 0
        for j in range(nc):
            s = cand[j]
            if t - s < min_size or F[s] + _seg_cost(S, Q, s, t) <= F[t]:
                cand[k] = s
                k += 1
        cand[k] = t
        k += 1
        nc = k
    out = []
    t = n
    while t > 0:
        s = last[t]
        if s > 0:
            out.append(s)
        t = s
    out.reverse()
    return np.array(out, dtype=np.int64)


@njit(cache=True)
def _best_split(S, s, e, min_size):
    """Max |CUSUM contrast| over splits b in [s+min_size, e-min_size] of x[s:e]."""
    n = e - s
    best = -1.0
    bb = -1
    tot = S[e] - S[s]
    for b in range(s + min_size, e - min_size + 1):
        nl = b - s
        nr = e - b
        left = S[b] - S[s]
        c = np.sqrt(nr / (n * nl)) * left - np.sqrt(nl / (n * nr)) * (tot - left)
        c = abs(c)
        if c > best:
            best = c
            bb = b
    return best, bb


@njit(cache=True)
def binseg_l2(x, thr, min_size, max_cp):
    """Binary Segmentation (Scott and Knott 1974) with the CUSUM contrast and threshold thr."""
    n = x.shape[0]
    S = np.zeros(n + 1)
    for i in range(n):
        S[i + 1] = S[i] + x[i]
    stack_s = [0]
    stack_e = [n]
    found = []
    while len(stack_s) > 0 and len(found) < max_cp:
        s = stack_s.pop()
        e = stack_e.pop()
        if e - s < 2 * min_size:
            continue
        v, b = _best_split(S, s, e, min_size)
        if v > thr and b > 0:
            found.append(b)
            stack_s.append(s)
            stack_e.append(b)
            stack_s.append(b)
            stack_e.append(e)
    found.sort()
    return np.array(found, dtype=np.int64)


@njit(cache=True)
def wbs_l2(x, thr, min_size, ints):
    """Wild Binary Segmentation (Fryzlewicz 2014): at each step the best CUSUM contrast over the random
    intervals (rows of `ints`) inside the current segment, plus the segment itself."""
    n = x.shape[0]
    S = np.zeros(n + 1)
    for i in range(n):
        S[i + 1] = S[i] + x[i]
    stack_s = [0]
    stack_e = [n]
    found = []
    while len(stack_s) > 0:
        s = stack_s.pop()
        e = stack_e.pop()
        if e - s < 2 * min_size:
            continue
        best, bb = _best_split(S, s, e, min_size)
        for r in range(ints.shape[0]):
            a = ints[r, 0]
            b = ints[r, 1]
            if a >= s and b <= e and b - a >= 2 * min_size:
                v, p = _best_split(S, a, b, min_size)
                if v > best:
                    best = v
                    bb = p
        if best > thr and bb > 0:
            found.append(bb)
            stack_s.append(s)
            stack_e.append(bb)
            stack_s.append(bb)
            stack_e.append(e)
    found.sort()
    return np.array(found, dtype=np.int64)


# ------------------------------------------------------------------------------------- helpers
@njit(cache=True)
def ar1(n, phi, eps):
    """Stationary AR(1) with unit marginal variance from innovations eps (len n)."""
    out = np.empty(n)
    s = np.sqrt(1.0 - phi * phi)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + s * eps[t]
    return out


@njit(cache=True)
def first_cross(path, h):
    for t in range(path.shape[0]):
        if path[t] > h:
            return t
    return -1
