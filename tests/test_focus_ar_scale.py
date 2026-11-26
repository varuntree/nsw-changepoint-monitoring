"""FOCuS-AR scale: the exact FOCuS-AR statistic equals (1 - phi^2) x twice the log-likelihood ratio
of a mean change in a stationary AR(1) series with unit marginal variance, computed from the explicit Gaussian
likelihood (Sigma^-1), maximised over every start."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpd import kernels as K  # noqa: E402


def two_llr(x, phi):
    n = len(x)
    i = np.arange(n)
    Si = np.linalg.inv(phi ** np.abs(i[:, None] - i[None, :]))
    out = np.empty(n)
    for t in range(n):
        y = x[:t + 1]
        S = Si if t == n - 1 else np.linalg.inv(phi ** np.abs(i[:t + 1, None] - i[None, :t + 1]))
        best = 0.0
        for s in range(t + 1):
            a = np.zeros(t + 1)
            a[s:] = 1.0
            best = max(best, (a @ S @ y) ** 2 / (a @ S @ a))
        out[t] = best
    return out


def test_focus_ar_is_scaled_twice_llr():
    rng = np.random.default_rng(5)
    for phi in (0.0, 0.2, 0.4, 0.6):
        for _ in range(50):
            n = 30
            x = K.ar1(n, phi, rng.standard_normal(n)) + (0.8 if rng.random() < 0.5 else 0.0) * (np.arange(n) >= 15)
            st, _ = K.focus_ar_exact(np.ascontiguousarray(x), phi)
            assert np.allclose(st, (1 - phi * phi) * two_llr(x, phi), atol=1e-8, rtol=1e-8)
