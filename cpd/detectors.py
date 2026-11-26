"""One interface for every online detector: stat(det, z, phi) -> (statistic path, start estimator)."""
from __future__ import annotations

import numpy as np

from . import kernels as K

MEAN_DETS = ("CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM")
ROBUST_DETS = ("CUSUM-r", "EWMA-r", "FOCuS-r")
GAMMA_SHAPE, GAMMA_SCALE0 = 0.5, 2.0   # variance FOCuS: z^2 ~ Gamma(shape 1/2, scale 2) when var(z) = 1
CLIP = 3.0
USE_PACKAGE = False   # True reproduces the runs before 24 Oct 2026 (focus_cpt for FOCuS and FOCuS-AR); checks only   # robust versions cap each standardised surprise at +-3 (bounded influence, Huber-type)
SPREAD_DETS = ("FOCuS-var", "CUSUM-var", "EWMA-var")
TUNING_DETS = ("CUSUM-k0.25", "CUSUM-k1", "EWMA-l0.05", "EWMA-l0.2")   # simulation sensitivity only
PARAMS = {"CUSUM": {"k": 0.5}, "EWMA": {"lam": 0.1}, "CPM": {"startup": 20}, "CUSUM-var": {"sig2_1": 2.0},
          "EWMA-var": {"lam": 0.1}}


def _focus(z, phi=0.0, kind="gaussian"):
    """FOCuS statistics on z with missing days removed (the reduced series is treated as consecutive, then the path
    is carried forward over missing days). Gaussian and AR(1): exact direct search over every start (kernels.py);
    gamma (variance FOCuS): the focus_cpt package, which is exact for that family."""
    z = np.asarray(z, float)
    if kind in ("gaussian", "arp") and not USE_PACKAGE:
        ok = ~np.isnan(z)
        idx = np.flatnonzero(ok)
        out = np.zeros(len(z))
        starts = np.full(len(z), -1, dtype=np.int64)
        if len(idx) == 0:
            return out, starts
        y = np.ascontiguousarray(z[ok])
        st, arg = K.focus_exact(y) if kind == "gaussian" else K.focus_ar_exact(y, float(phi))
        tmp = np.full(len(z), np.nan)
        tmp[idx] = st
        starts[idx] = idx[arg]
        out = pd_ffill0(tmp)
        return out, starts
    return _focus_pkg(z, phi, kind)


def pd_ffill0(a):
    out = np.empty(len(a))
    last = 0.0
    for i in range(len(a)):
        if not np.isnan(a[i]):
            last = a[i]
        out[i] = last
    return out


def _focus_pkg(z, phi=0.0, kind="gaussian"):
    """The focus_cpt 0.1.11 implementation (kept for the implementation check, scripts/focus_check.py, and used
    for the gamma family)."""
    import focus_cpt as fc
    z = np.asarray(z, float)
    ok = ~np.isnan(z)
    idx = np.flatnonzero(ok)
    out = np.zeros(len(z))
    starts = np.full(len(z), -1, dtype=np.int64)
    if len(idx) < 2:
        return out, starts
    y = z[ok]
    if kind == "gaussian":
        r = fc.focus_offline(y, threshold=np.inf, type="univariate", family="gaussian", theta0=0.0)
    elif kind == "arp":
        r = fc.focus_offline(y, threshold=np.inf, type="arp", family="arp", rho=[float(phi)], mu0_arp=0.0)
    elif kind == "gamma":   # z^2 ~ Gamma(shape 1/2, scale 2) in control: mean 1
        # focus_cpt's gamma theta0 is the SCALE (checked against a direct gamma GLR in tests/test_focus_var.py),
        # so the in-control mean 1 needs theta0 = 1 / shape = 2. (theta0 = 1.0, used until 23 Oct 2026, meant an
        # in-control mean of 0.5, which made the statistic drift upward in control.)
        r = fc.focus_offline(y * y + 1e-12, threshold=np.inf, type="univariate", family="gamma",
                             theta0=GAMMA_SCALE0, shape=GAMMA_SHAPE)
    else:
        raise ValueError(kind)
    st = np.asarray(r["stat"], float).ravel()
    cp = r.get("changepoint")
    tmp = np.full(len(z), np.nan)
    tmp[idx[:len(st)]] = st
    last = 0.0
    for i in range(len(z)):
        if np.isnan(tmp[i]):
            tmp[i] = last
        else:
            last = tmp[i]
    if cp is not None:
        cp = np.asarray([c if c is not None else -1 for c in cp], float).ravel()
        m = min(len(cp), len(idx))
        # changepoint is reported as a count of observations in the reduced series; map to original index
        for j in range(m):
            c = int(cp[j]) if not np.isnan(cp[j]) else -1
            starts[idx[j]] = idx[c] if 0 <= c < len(idx) else -1
    return tmp, starts


def stat_path(det: str, z, phi: float = 0.0):
    z = np.ascontiguousarray(np.asarray(z, float))
    if det.endswith("-r"):
        return stat_path(det[:-2], np.clip(z, -CLIP, CLIP), phi)
    if det == "CUSUM":
        return K.cusum_path(z, PARAMS["CUSUM"]["k"])
    if det.startswith("CUSUM-k"):          # tuning variants: CUSUM-k0.25, CUSUM-k1
        return K.cusum_path(z, float(det[7:]))
    if det.startswith("EWMA-l"):           # EWMA-l0.05, EWMA-l0.2
        return K.ewma_path(z, float(det[6:]))
    if det == "EWMA":
        return K.ewma_path(z, PARAMS["EWMA"]["lam"])
    if det == "CPM":
        return K.cpm_mw_path(z, PARAMS["CPM"]["startup"])
    if det == "FOCuS":
        return _focus(z, kind="gaussian")[0]
    if det == "FOCuS-AR":
        return _focus(z, phi=phi, kind="arp")[0]
    if det == "FOCuS-var":
        return _focus(z, kind="gamma")[0]
    if det == "CUSUM-var":
        return K.var_cusum_path(z, PARAMS["CUSUM-var"]["sig2_1"])
    if det == "EWMA-var":
        return K.ewma_var_path(z, PARAMS["EWMA-var"]["lam"])
    raise ValueError(det)


def change_start(det: str, z, t_alarm: int, phi: float = 0.0) -> int:
    """Estimated start of the change behind an alarm at t_alarm (index in z)."""
    z = np.ascontiguousarray(np.asarray(z, float))
    if det.endswith("-r"):
        return change_start(det[:-2], np.clip(z, -CLIP, CLIP), t_alarm, phi)
    if det == "CUSUM":
        return int(K.cusum_start(z, PARAMS["CUSUM"]["k"], t_alarm))
    if det.startswith("CUSUM-k"):
        return int(K.cusum_start(z, float(det[7:]), t_alarm))
    if det in ("FOCuS", "FOCuS-AR", "FOCuS-var"):
        kind = {"FOCuS": "gaussian", "FOCuS-AR": "arp", "FOCuS-var": "gamma"}[det]
        _, starts = _focus(z[: t_alarm + 1], phi=phi, kind=kind)
        s = int(starts[t_alarm])
        return s if s >= 0 else t_alarm
    if det == "CPM":
        zz = z[: t_alarm + 1]
        zz = zz[~np.isnan(zz)]
        n = len(zz)
        best, bk = -1.0, n - 1
        for k in range(2, n - 1):
            a, b = zz[:k], zz[k:]
            u = (np.sign(b[:, None] - a[None, :])).sum()
            v = abs(u) / np.sqrt(k * (n - k) * (n + 1) / 3.0)
            if v > best:
                best, bk = v, k
        return t_alarm - (n - bk) + 1
    return t_alarm
