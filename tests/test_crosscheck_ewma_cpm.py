"""The independently written EWMA and CPM agree with our kernels, and the
new results files agree with the files they are built from."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cpd import kernels as K  # noqa: E402

Y = HERE / "data" / "independent" / "ewma_cpm"
R = HERE / "results"


def _load(name):
    """The independent module, compiled in memory (its numba cache=True would write a cache file next to the source and cannot
    be reloaded for a module loaded from a path)."""
    import types
    src = (Y / f"{name}.py").read_text().replace("cache=True", "cache=False")
    m = types.ModuleType(f"independent_{name}")
    sys.modules[m.__name__] = m
    exec(compile(src, str(Y / f"{name}.py"), "exec"), m.__dict__)
    return m


@pytest.mark.skipif(not Y.exists(), reason="the independent code is not in this repository (data/independent/)")
def test_independent_kernels_match_ours():
    ye, yc = _load("ewma"), _load("cpm")
    rng = np.random.default_rng(3)
    for _ in range(5):
        z = np.round(rng.standard_normal(200), 1) + 0.0      # ties, no signed zeros
        z[rng.random(200) < 0.2] = np.nan
        assert np.allclose(ye.stat_path(z), K.ewma_path(z, 0.1), atol=1e-12)
        assert np.allclose(yc.stat_path(z), K.cpm_mw_path(z, 20), atol=1e-12)


def test_generator_table_matches_generator_json():
    g = json.loads((R / "sim" / "generator.json").read_text())
    t = pd.read_csv(R / "sim" / "generator_coefficients.csv").set_index("term")
    assert np.isclose(t.loc["public holiday", "coefficient"], g["pub"])
    assert np.isclose(t.loc["weekday: Sunday vs Monday", "coefficient"], g["weekday"][6])
    assert np.isclose(t.loc["yearly Fourier cos(3)", "coefficient"], g["fourier"][5])


def test_past_retro_pairs_match_rule_variants():
    p = pd.read_csv(R / "opal" / "past_retro_pairs.csv")
    rv = pd.read_csv(R / "opal" / "rule_variants.csv").set_index(["method", "detector", "rule"])
    for _, r in p.iterrows():
        past, retro = ("MSTL_p", "MSTL") if r.family == "MSTL" else ("REG_p", "REG")
        assert rv.loc[(past, r.detector, r.rule), "frozen"] == r.past_frozen
        assert rv.loc[(retro, r.detector, r.rule), "r0_7_dir"] == r.retro_dir


def test_crosscheck_opal_identical():
    c = pd.read_csv(R / "extra" / "crosscheck_ewma_cpm.csv").set_index("check")["value"]
    assert c["Opal alarms only in one run"] == 0 and c["Opal alarms with a different direction"] == 0
    assert c["EWMA max abs path difference"] < 1e-12
