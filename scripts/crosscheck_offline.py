"""Cross-check against an independently written PELT, Binary Segmentation and WBS.

Compares the independent code (data/independent/offline/offline_methods.py, not in this repository) with our
kernels cpd.kernels.pelt_l2, binseg_l2 and wbs_l2:
  shared vector  the group's offline test vector (n = 600) and WBS intervals;
  random cases   40 series (white or AR(1) 0.6 noise, 0-5 random steps, rounded ties every fifth, a flat first third
                 every seventh) x 3 penalty multipliers (2, 3, 8 times log n) = 120 cases, random minimum segment and
                 WBS intervals (seed 2910);
  Opal           the retrospective STL, MSTL and regression series scaled as scripts/opal_offline.py, with our
                 settings (PELT 3 log n, BinSeg sqrt(3 log n), WBS 1.3 sqrt(2 log n) with 5,000 intervals, seed 0);
  its files      independent Opal change dates (STL and MSTL) against opal/offline_changes.csv, and its 6 score rows against
                 opal/offline.csv.
Outputs: results/extra/crosscheck_offline_cases.csv (every case x method), results/extra/crosscheck_offline_opal_scores.csv,
results/extra/crosscheck_offline.csv (summary, ledger-ready)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
S = HERE / "data" / "independent" / "offline"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(S))
from cpd import kernels as K  # noqa: E402
from opal_offline import scaled  # noqa: E402
from offline_methods import binseg, pelt, wbs  # noqa: E402

R = HERE / "results"
OUT = R / "extra"


def cmp(rows, label, x, pen, bthr, wthr, ints, ms=7):
    x = np.ascontiguousarray(x, float)
    ints = np.ascontiguousarray(ints, np.int64)
    h = dict(PELT=list(pelt.segment(x.copy(), pen, min_size=ms)), BinSeg=list(binseg.segment(x.copy(), bthr, min_size=ms)),
             WBS=list(wbs.segment(x.copy(), wthr, min_size=ms, intervals=ints)))
    b = dict(PELT=[int(v) for v in K.pelt_l2(x, pen, ms)], BinSeg=[int(v) for v in K.binseg_l2(x, bthr, ms, 10 ** 6)],
             WBS=[int(v) for v in K.wbs_l2(x, wthr, ms, ints)])
    for m in h:
        rows.append(dict(case=label, method=m, n=len(x), n_indep=len(h[m]), n_ours=len(b[m]),
                         identical=[int(v) for v in h[m]] == b[m], sym_diff=len(set(h[m]) ^ set(b[m]))))


def main():
    rows = []
    y = pd.read_csv(S / "vectors" / "offline_input.csv")["y"].to_numpy(float)
    n = len(y)
    ints = pd.read_csv(S / "vectors" / "wbs_intervals.csv")[["a", "b"]].to_numpy(np.int64)
    cmp(rows, "shared_vector", y, 3 * np.log(n), 4.0, 4.5, ints)
    rng = np.random.default_rng(2910)
    for i in range(40):
        n = int(rng.integers(30, 900))
        x = rng.standard_normal(n)
        if i % 4 == 1:
            for t in range(1, n):
                x[t] += 0.6 * x[t - 1]
        for _ in range(int(rng.integers(0, 6))):
            x[int(rng.integers(0, n)):] += rng.normal(0, 2)
        if i % 5 == 2:
            x = np.round(x)
        if i % 7 == 3:
            x[: n // 3] = 0.0
        iv = np.sort(rng.integers(0, n, size=(int(rng.integers(50, 2000)), 2)), axis=1)
        for pm in (2.0, 3.0, 8.0):
            cmp(rows, f"random{i}_pen{pm}", x, pm * np.log(n), np.sqrt(pm * np.log(n)), 1.3 * np.sqrt(2 * np.log(n)) * pm / 3, iv,
                ms=int(rng.choice([1, 3, 7, 14])))
    adj = pd.read_csv(R / "opal" / "adjusted.csv", parse_dates=["date"]).set_index("date")
    rng = np.random.default_rng(0)
    for m in ("STL", "MSTL", "REG"):
        yy, _ = scaled(adj[m])
        n = len(yy)
        iv = np.sort(rng.integers(0, n, size=(5000, 2)), axis=1)
        cmp(rows, f"opal_{m}", yy, 3 * np.log(n), np.sqrt(3 * np.log(n)), 1.3 * np.sqrt(2 * np.log(n)), iv)
    cases = pd.DataFrame(rows)
    cases.to_csv(OUT / "crosscheck_offline_cases.csv", index=False)
    # independent Opal change dates and score rows
    bc = pd.read_csv(R / "opal" / "offline_changes.csv", parse_dates=["change_date"])
    n_dates, dates_equal = 0, True
    for f in ("pelt", "binseg", "wbs"):
        hc = pd.read_csv(S / "results" / "opal" / f"{f}_changes.csv", parse_dates=["change_date"])
        meth = hc.detector.iloc[0]
        for a in sorted(hc.adjustment.unique()):
            hs = set(hc[hc.adjustment == a].change_date)
            bs = set(bc[(bc.adjustment == a) & (bc.method == meth)].change_date)
            n_dates += len(hs)
            dates_equal &= hs == bs
    bo = pd.read_csv(R / "opal" / "offline.csv")
    cols = ["n_alarms", "alarms_per_year", "near_real", "share_real", "null_share_real", "p_share_real", "episodes_caught",
            "null_episodes", "p_episodes", "unexplained_per_year", "non_event_per_year", "phi"]
    sc = []
    for f in ("pelt", "binseg", "wbs"):
        hs = pd.read_csv(S / "results" / "opal" / f"{f}_scores.csv")
        for _, r in hs.iterrows():
            bb = bo[(bo.adjustment == r.adjustment) & (bo.method == r.detector)].iloc[0]
            d = {c: float(abs(r[c] - bb[c])) for c in cols}
            sc.append(dict(method=r.detector, adjustment=r.adjustment, max_abs_diff=max(d.values()),
                           worst_col=max(d, key=d.get), n_alarms=int(r.n_alarms), caught=int(r.episodes_caught)))
    sc = pd.DataFrame(sc)
    sc.to_csv(OUT / "crosscheck_offline_opal_scores.csv", index=False)
    rnd = cases[cases.case.str.startswith("random")]
    summ = [("shared vector, methods identical", int(cases[cases.case == "shared_vector"].identical.sum())),
            ("random cases", rnd.case.nunique()),
            ("random cases x methods identical", int(rnd.identical.sum())),
            ("random cases x methods compared", len(rnd)),
            ("Opal series x methods identical", int(cases[cases.case.str.startswith("opal_")].identical.sum())),
            ("Opal series x methods compared", int(cases.case.str.startswith("opal_").sum())),
            ("independent Opal change dates compared", n_dates),
            ("independent Opal change dates all equal to ours", int(dates_equal)),
            ("independent Opal score rows", len(sc)),
            ("independent Opal score rows equal (max abs difference 0)", int((sc.max_abs_diff == 0).sum()))]
    out = pd.DataFrame(summ, columns=["check", "value"])
    out.to_csv(OUT / "crosscheck_offline.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
