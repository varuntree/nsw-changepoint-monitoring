"""Add a detector to the existing simulation without re-running it (used for the variance EWMA).

For every condition that ran the spread detectors (variance changes and the no-change runs), each batch
b{b}.npz is regenerated from its own seed exactly as sim_run.batch_job does (same SeedSequence, same order of
random draws, same adjustments, running scale and phi_hat), the new detectors' statistic paths are computed on
the same z, and their maximum statistic and first crossings for the three threshold rules (independent, AR(1)
table at phi_hat, matched "oracle" threshold) are written to x{b}.npz beside the batch. One detector already in
the batch (CUSUM-var) is recomputed as well and must match the stored values exactly; this proves that the
regenerated series are the ones that were simulated. The matched thresholds of the new detectors are the 95th
percentile of their yearly maximum over the no-change runs, as in sim_run.oracle_thresholds, and are added to
that condition's oracle.json. sim_analyse.py merges x*.npz into the batch data.
If a batch already contains the detector (a fresh full run, since cpd/sim.py lists it), it is skipped.

`--replace DET` recomputes a detector that IS stored in the batches (after a fix to it) and overwrites its
columns of mx and first in b*.npz in place, with the same reproduction check and new matched thresholds in
oracle.json; every other column is left as it was. Used on 23 Oct 2026 for variance FOCuS after the gamma
parametrisation fix in cpd/detectors.py (theta0 is focus_cpt's scale: 2, not 1).
`--replace-all` does the same in every condition (mean changes too): used on 24 Oct 2026 for FOCuS, capped
FOCuS and FOCuS-AR after they moved from focus_cpt to the exact direct search (cpd/kernels.focus_exact).
Usage: python scripts/sim_addon.py [DET ...]            (default: EWMA-var)
       python scripts/sim_addon.py --replace DET [DET ...]
       python scripts/sim_addon.py --replace-all DET [DET ...]
       python scripts/sim_addon.py --tuning CUSUM-k0.25 CUSUM-k1 EWMA-l0.05 EWMA-l0.2   (phi 0.4, normal, no holidays)
"""
from __future__ import annotations

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from cpd import calibrate  # noqa: E402
from cpd import sim as S  # noqa: E402
from sim_run import BATCH, OUT, TABLE, WBS_M, cid  # noqa: E402
from sim_analyse import parse  # noqa: E402

CHECK = "CUSUM"           # stored in every batch (CUSUM-var only in variance/no-change batches); must reproduce
REPLACE_CHECK = "CUSUM"      # stored in every batch; must reproduce bit for bit
WORKERS = 10


def addon_job(args):
    c, b, new, oracle, gen_json = args
    d = OUT / cid(c)
    main = np.load(d / f"b{b:03d}.npz")
    stored = list(main["dets"])
    from cpd.detectors import SPREAD_DETS
    todo = tuple(x for x in new if x not in stored and (c["change"] in ("var", "none") or x not in SPREAD_DETS))
    if not todo:
        return None
    gen = S.Generator(json.loads(gen_json))
    tabs = {k: calibrate.load(TABLE, k) for k in ("surprise", "level")}
    key = zlib.crc32(cid(c).encode())
    rng = np.random.default_rng(np.random.SeedSequence([key, b]))
    rng.integers(0, S.MON, size=(WBS_M, 2))             # the WBS intervals drawn first in batch_job
    dets = todo + (CHECK,)
    runs = len(main["taus"])
    nA = len(S.ADJ)
    first = np.full((runs, nA, len(dets), 3), -1, np.int16)
    mx = np.zeros((runs, nA, len(dets)), np.float32)
    taus = np.zeros(runs, np.int16)
    for i in range(runs):
        tau = int(rng.integers(60, 301))
        taus[i] = tau
        lx = gen.draw(rng, c["phi"], c["noise"], c["calendar"], c["change"], c["size"], tau)
        paths, ph, _ = S.run_series(lx, gen, c["calendar"], None, dets=dets)
        for a, adj in enumerate(S.ADJ):
            p, kind = ph[adj]
            for k, det in enumerate(dets):
                path = paths[(adj, det)]
                mx[i, a, k] = np.nanmax(path)
                t = tabs[kind]
                h_or = oracle.get(f"{adj}|{det}", np.inf) if oracle else np.inf
                h_ind = t[det](p) if det == "FOCuS-AR" else t[det](0.0)      # as sim_run.batch_job
                first[i, a, k] = S.first_cross_multi(path, (h_ind, t[det](p), h_or))
    # reproduction check against the stored batch
    kc = stored.index(CHECK)
    assert np.array_equal(taus, main["taus"]), (cid(c), b, "tau")
    assert np.array_equal(mx[:, :, -1], main["mx"][:, :, kc]), (cid(c), b, "mx")
    if oracle:
        assert np.array_equal(first[:, :, -1], main["first"][:, :, kc]), (cid(c), b, "first")
    else:   # no-change runs were stored before matched thresholds existed: only the first two rules
        assert np.array_equal(first[:, :, -1, :2], main["first"][:, :, kc, :2]), (cid(c), b, "first")
    np.savez_compressed(d / f"x{b:03d}.npz", first=first[:, :, :-1], mx=mx[:, :, :-1], dets=np.array(todo))
    return str(d / f"x{b:03d}.npz")


def replace_job(args):
    c, b, dets, oracle, gen_json = args
    f = OUT / cid(c) / f"b{b:03d}.npz"
    main = dict(np.load(f))
    stored = list(main["dets"])
    gen = S.Generator(json.loads(gen_json))
    tabs = {k: calibrate.load(TABLE, k) for k in ("surprise", "level")}
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(cid(c).encode()), b]))
    rng.integers(0, S.MON, size=(WBS_M, 2))
    todo = tuple(dets) + (REPLACE_CHECK,)
    runs = len(main["taus"])
    nA = len(S.ADJ)
    first = np.full((runs, nA, len(todo), 3), -1, np.int16)
    mx = np.zeros((runs, nA, len(todo)), np.float32)
    for i in range(runs):
        tau = int(rng.integers(60, 301))
        assert tau == main["taus"][i], (cid(c), b, i)
        lx = gen.draw(rng, c["phi"], c["noise"], c["calendar"], c["change"], c["size"], tau)
        paths, ph, _ = S.run_series(lx, gen, c["calendar"], None, dets=todo)
        for a, adj in enumerate(S.ADJ):
            p, kind = ph[adj]
            for k, det in enumerate(todo):
                path = paths[(adj, det)]
                mx[i, a, k] = np.nanmax(path)
                h_or = oracle.get(f"{adj}|{det}", np.inf) if oracle else np.inf
                h_ind = tabs[kind][det](p) if det == "FOCuS-AR" else tabs[kind][det](0.0)      # as sim_run.batch_job
                first[i, a, k] = S.first_cross_multi(path, (h_ind, tabs[kind][det](p), h_or))
    kc = stored.index(REPLACE_CHECK)
    assert np.array_equal(mx[:, :, -1], main["mx"][:, :, kc]), (cid(c), b, "mx")
    rules = slice(None) if oracle else slice(0, 2)
    assert np.array_equal(first[:, :, -1, rules], main["first"][:, :, kc, rules]), (cid(c), b, "first")
    for k, det in enumerate(dets):
        j = stored.index(det)
        main["mx"][:, :, j] = mx[:, :, k]
        main["first"][:, :, j] = first[:, :, k]
    np.savez_compressed(f, **main)
    return str(f)


def replace(dets, all_conditions=False):
    gj = (OUT / "generator.json").read_text()
    conds = [parse(p.name) | {"name": p.name} for p in sorted(OUT.iterdir())
             if p.is_dir() and p.name.startswith("phi")]
    if not all_conditions:
        conds = [c for c in conds if c["change"] in ("none", "var")]
    for c in conds:
        c["calendar"] = bool(c["calendar"])
        assert cid(c) == c.pop("name")
    nb = {cid(c): len(list((OUT / cid(c)).glob("b*.npz"))) for c in conds}
    none = [c for c in conds if c["change"] == "none"]
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(replace_job, [(c, b, dets, None, gj) for c in none for b in range(nb[cid(c)])]))
        for c in none:      # new matched thresholds, as sim_run.oracle_thresholds
            d = OUT / cid(c)
            fs = sorted(d.glob("b*.npz"))
            mx = np.concatenate([np.load(f)["mx"] for f in fs])
            stored = list(np.load(fs[0])["dets"])
            orc = json.loads((d / "oracle.json").read_text())
            for a, adj in enumerate(S.ADJ):
                for det in dets:
                    orc[f"{adj}|{det}"] = float(np.quantile(mx[:, a, stored.index(det)], 0.95))
            (d / "oracle.json").write_text(json.dumps(orc, indent=1))
        jobs = []
        for c in conds:
            if c["change"] != "none":
                orc = json.loads((OUT / cid(dict(c, change="none", size=0.0)) / "oracle.json").read_text())
                jobs += [(c, b, dets, orc, gj) for b in range(nb[cid(c)])]
        done = list(ex.map(replace_job, jobs))
    print("replaced", dets, "in", len(done), "change batches and the no-change batches", flush=True)


def main(new=("EWMA-var",), only=None):
    """only: (phi, noise, calendar) to restrict to one condition group, including its mean changes (used for the
    tuning variants); default: every variance and no-change condition."""
    gj = (OUT / "generator.json").read_text()
    conds = [parse(p.name) | {"name": p.name} for p in sorted(OUT.iterdir())
             if p.is_dir() and p.name.startswith("phi")]
    if only is None:
        conds = [c for c in conds if c["change"] in ("none", "var")]
    else:
        conds = [c for c in conds if (c["phi"], c["noise"], c["calendar"]) == only]
    for c in conds:
        c["calendar"] = bool(c["calendar"])
        assert cid(c) == c.pop("name")
    nb = {cid(c): len(list((OUT / cid(c)).glob("b*.npz"))) for c in conds}
    none = [c for c in conds if c["change"] == "none"]
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(addon_job, [(c, b, new, None, gj) for c in none for b in range(nb[cid(c)])]))
        for c in none:
            d = OUT / cid(c)
            xs = sorted(d.glob("x*.npz"))
            if not xs:
                continue
            mx = np.concatenate([np.load(f)["mx"] for f in xs])
            dets = list(np.load(xs[0])["dets"])
            orc = json.loads((d / "oracle.json").read_text())
            for a, adj in enumerate(S.ADJ):
                for k, det in enumerate(dets):
                    orc[f"{adj}|{det}"] = float(np.quantile(mx[:, a, k], 0.95))
            (d / "oracle.json").write_text(json.dumps(orc, indent=1))
        jobs = []
        for c in conds:
            if c["change"] == "none":
                continue
            orc = json.loads((OUT / cid(dict(c, change="none", size=0.0)) / "oracle.json").read_text())
            jobs += [(c, b, new, orc, gj) for b in range(nb[cid(c)])]
        done = [r for r in ex.map(addon_job, jobs) if r]
    print("variance conditions written:", len(done), "batches; runs per batch", BATCH, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("--replace", "--replace-all"):
        replace(tuple(sys.argv[2:]), all_conditions=sys.argv[1] == "--replace-all")
    elif len(sys.argv) > 2 and sys.argv[1] == "--tuning":       # phi 0.4, normal, no holidays: none and mean changes
        main(("EWMA-var",) + tuple(sys.argv[2:]), only=(0.4, "normal", 0))
    else:
        main(tuple(sys.argv[1:]) or ("EWMA-var",))
