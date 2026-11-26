"""Thesis figures (vector PDF for the thesis, PNG for checking).

One style for all figures (scripts/figstyle.py): Inter, ink/muted text, grey data, blue only for the
recommended monitor, red only for alarms, green dashed for scored episodes and boundaries, one fixed colour
per detector. No in-figure titles: the takeaway is the first sentence of the caption. Every figure is drawn
at its printed width, and save() reports the smallest printed text size (target >= 9 pt).

Usage: python scripts/figures.py [keys...]   (default: all; keys are the F-numbers below)
Figures that need results not yet produced read the planned file if it exists and otherwise skip that
element (or the whole figure) with a printed note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402
from cpd.data import opal_daily  # noqa: E402
from figstyle import (ANAME, BLUE, DET, DNAME, DNAME2, GREEN, GREEN_L, GREY, INK, LIGHT, MUTED,  # noqa: E402
                      PALE, RED, TW)

S.setup()
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
EV = HERE / "data" / "opal"
EVE = HERE / "data" / "electricity"
SIMS = RES / "sim" / "summary"
END = pd.Timestamp("2026-09-28")  # last monitored day


def save(fig, name, print_w=TW, png=True):
    S.save(fig, FIG, name, print_w, png)


def yearaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def scored_opal():
    return pd.read_csv(EV / "events_scored_FROZEN.csv", parse_dates=["date"])


# ------------------------------------------------------------------------------------------- data (Ch 3)
def f_opal_series():
    """Not printed (Figure 3.1 is drawn by scripts/thesis_figures.py). Full width; the 18 scored boundaries as green dashed lines; direct labels, no legend."""
    x = opal_daily("train") / 1e6
    sc = scored_opal()
    med = x.rolling(28, center=True, min_periods=7).median()
    fig, ax = plt.subplots(figsize=(TW, 2.45), layout="constrained")
    for d in sc["date"].unique():
        S.boundary(ax, d, lw=1.3, ls=(0, (4, 2)), alpha=1, zorder=2.5)   # above the grey data, below the median
    ax.plot(x.index, x.values, color=GREY, lw=0.35, zorder=2)
    ax.plot(med.index, med.values, color=INK, lw=1.4, zorder=3)
    ax.set_xlim(pd.Timestamp("2019-12-20"), END + pd.Timedelta(days=4))
    ax.set_ylim(0, 1.55)
    ax.set_ylabel("train tap-ons a day (millions)")
    yearaxis(ax)
    S.ygrid(ax)
    # direct labels
    ax.annotate("daily count", (pd.Timestamp("2023-09-02"), x.loc["2023-09-02"]), xytext=(pd.Timestamp("2023-02-15"), 0.22),
                color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=1))
    ax.text(pd.Timestamp("2023-06-15"), med.loc["2023-06-15"] + 0.12, "28-day median", color=INK, ha="center", va="bottom")
    ax.text(pd.Timestamp("2024-12-28"), 1.52, "scored boundaries", color=GREEN, ha="right", va="top")
    ax.text(pd.Timestamp("2020-06-10"), 1.47, "COVID-19", color=INK, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5), zorder=4)
    save(fig, "F02_opal_series")


def f_rhythm():
    """Not printed (weekly pattern, two months, autocorrelation of forecast errors)."""
    x = opal_daily("train")
    lx = np.log(x)
    q = lx.loc["2023-02-01":"2023-11-30"]
    dow = q.groupby(q.index.dayofweek).mean()
    fig, ax = plt.subplots(1, 3, figsize=(TW, 2.3), layout="constrained")
    ax[0].bar(range(7), np.exp(dow - dow.iloc[:5].mean()), color=GREY, width=0.7)
    ax[0].set_xticks(range(7), ["M", "T", "W", "T", "F", "S", "S"])
    ax[0].set_ylabel("tap-ons relative to\nan average weekday")
    ax[0].set_title("weekdays, 2023")
    w = x.loc["2023-03-06":"2023-04-30"] / 1e6
    ax[1].plot(w.index, w.values, color=INK, marker=".", ms=3, lw=0.8)
    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=3))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax[1].set_ylabel("millions a day")
    ax[1].set_title("eight weeks, 2023")
    adj = pd.read_csv(RES / "opal" / "adjusted.csv", parse_dates=["date"]).set_index("date")
    r = adj["HW"].loc["2023-01-01":"2023-12-31"].dropna()
    lags = range(1, 22)
    ax[2].bar(list(lags), [r.autocorr(k) for k in lags], color=GREY, width=0.7)
    ax[2].axhline(0, color=INK, lw=0.6)
    ax[2].set_xlabel("lag (days)")
    ax[2].set_title("forecast error autocorrelation")
    for a in ax:
        S.ygrid(a)
    save(fig, "F03_rhythm_dependence")


def f_adjust_covid():
    """Not printed. What each adjustment does to the COVID drop (23 March 2020 as the green dashed boundary)."""
    adj = pd.read_csv(RES / "opal" / "adjusted.csv", parse_dates=["date"]).set_index("date")
    w = slice("2020-02-01", "2020-05-31")
    ms = ["raw", "A", "B", "HW", "MSTL_p", "STL"]
    lab = {"raw": "log tap-ons (no adjustment)", "A": ANAME["A"], "B": ANAME["B"], "HW": "Holt-Winters",
           "MSTL_p": ANAME["MSTL_p"], "STL": "retrospective STL (level kept)"}
    ylab = {"raw": "log tap-ons", "STL": "log tap-ons,\nadjusted"}
    fig, ax = plt.subplots(3, 2, figsize=(TW, 4.5), sharex=True, layout="constrained")
    for a, m in zip(ax.flat, ms):
        s = adj[m].loc[w]
        S.boundary(a, pd.Timestamp("2020-03-23"), lw=1.0)
        if m not in ylab:
            a.axhline(0, color=GREY, lw=0.7, ls=":")
        a.plot(s.index, s.values, color=INK, lw=1.1)
        a.set_title(lab[m], fontweight="normal")
        a.set_ylabel(ylab.get(m, "forecast error (log)"))
        a.yaxis.set_major_locator(plt.MaxNLocator(4))
        S.ygrid(a)
    ax[0, 0].text(pd.Timestamp("2020-03-25"), ax[0, 0].get_ylim()[1], "23 Mar", color=GREEN, va="top", ha="left")
    for a in ax[-1]:
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        a.set_xlabel("2020")
    save(fig, "F04_adjustments_covid")


def f_detectors_toy():
    """Not placed in the thesis now: the four detector statistics on one simulated stream."""
    from cpd import detectors as D
    rng = np.random.default_rng(7)
    n, cp = 200, 120
    z = rng.standard_normal(n)
    z[cp:] += 1.0
    t = np.arange(1, n + 1)
    dets = [("CUSUM", 9.3), ("EWMA", 4.0), ("FOCuS", 22.9), ("CPM", 3.9)]
    fig, ax = plt.subplots(len(dets) + 1, 1, figsize=(TW, 5.6), sharex=True, layout="constrained")
    ax[0].plot(t, z, color=GREY, lw=0.8)
    ax[0].set_ylabel("simulated\nstream")
    for a in ax:
        S.boundary(a, cp, lw=1.0)
    ax[0].text(cp - 2, ax[0].get_ylim()[1], "mean moves from 0 to 1", color=INK, va="top", ha="right",
               bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    for a, (d, h) in zip(ax[1:], dets):
        p = D.stat_path(d, z)
        a.plot(t, p, color=DET[d], lw=1.3)
        a.axhline(h, color=MUTED, ls="--", lw=0.8)
        k = np.flatnonzero(p > h)
        if len(k):
            S.alarm_marks(a, [t[k[0]]], [p[k[0]]])
            a.annotate(f"alarm, day {t[k[0]]}", (t[k[0]], p[k[0]]), xytext=(-8, 4), textcoords="offset points",
                       color=INK, va="bottom", ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        a.set_ylabel(d)
    ax[-1].set_xlabel("day")
    save(fig, "F05_detectors_toy")


def f_thresholds():
    """Not printed. Calibrated thresholds against phi, detector colours, direct labels at the line ends."""
    t = json.loads((RES / "calib" / "h_table.json").read_text())["surprise"]
    fig, ax = plt.subplots(1, 2, figsize=(TW, 2.6), layout="constrained")
    for a, dets in zip(ax, (["CUSUM", "FOCuS", "FOCuS-AR"], ["EWMA", "CPM"])):
        for d in dets:
            k = sorted(t[d], key=float)
            xs, ys = [float(v) for v in k], [t[d][v] for v in k]
            S.det_line(a, xs, ys, d, ms=3)
            a.text(xs[-1] + 0.015, ys[-1], DNAME[d], color=INK, va="center", ha="left")
        a.set_xlabel("lag-one autocorrelation ϕ")
        a.set_xlim(-0.02, 1.14)
        a.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        S.ygrid(a)
    ax[0].set_yscale("log")
    ax[0].set_yticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
    ax[0].minorticks_off()
    ax[0].set_ylabel("threshold h (log scale)")
    ax[1].set_ylabel("threshold h (own scale)")
    save(fig, "F06_thresholds")


# ------------------------------------------------------------------------------------------- simulation (Ch 5)
FA_DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
FA_ADJS = ["oracle", "A", "B", "HW", "YM", "raw"]
FA_ROW = {"oracle": "true pattern removed", "A": "same-weekday difference", "B": "median of four", "HW": "Holt-Winters",
          "YM": "yearly memory", "raw": "no forecast"}
# diverging around the 5% target on a log scale (slate below, warm above); 2-10% is nearly white
_lo, _hi = np.log10(0.5), np.log10(100)


def _fp(v):
    return (np.log10(v) - _lo) / (_hi - _lo)


FA_CMAP = LinearSegmentedColormap.from_list("fa", [(0, "#41587a"), (_fp(1), "#8193b0"), (_fp(2), "#e6ebf2"),
                                                   (_fp(5), "#ffffff"), (_fp(10), "#fbe8d6"), (_fp(20), "#eda26c"),
                                                   (_fp(50), "#c45a1c"), (1, "#7a2a0a")])
FA_NORM = Normalize(_lo, _hi)


def _fa_panel(a, fa, cal, rule, show_cols):
    M = np.full((len(FA_ADJS), len(FA_DETS)), np.nan)
    for i, adj in enumerate(FA_ADJS):
        for j, d in enumerate(FA_DETS):
            q = fa[(fa["calendar"] == cal) & (fa["rule"] == rule) & (fa["adj"] == adj) & (fa["det"] == d)]
            if len(q):
                M[i, j] = 100 * q["fa"].iloc[0]
    a.imshow(np.log10(np.clip(M, 0.5, 100)), cmap=FA_CMAP, norm=FA_NORM, aspect="auto")
    for i in range(len(FA_ADJS)):
        for j in range(len(FA_DETS)):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt = "100" if v >= 99.5 else f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            dark = v >= 35 or v < 0.9
            col = "#8a8f98" if 2 <= v <= 10 else ("white" if dark else INK)  # on target: quiet grey
            a.text(j, i, txt, ha="center", va="center", color=col)
    a.set_xticks(range(len(FA_DETS)), [DNAME2[d] for d in FA_DETS] if show_cols else [], linespacing=1.0)
    a.xaxis.tick_top()
    a.set_yticks(range(len(FA_ADJS)), [FA_ROW[x] for x in FA_ADJS])
    a.tick_params(length=0, pad=3)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_xticks(np.arange(-0.5, len(FA_DETS)), minor=True)
    a.set_yticks(np.arange(-0.5, len(FA_ADJS)), minor=True)
    a.grid(which="minor", color="white", lw=1.5)
    a.tick_params(which="minor", length=0)
    a.axvline(4.5, color=INK, lw=0.7)  # uncapped left, capped right
    if show_cols:
        a.text(-0.62, -0.62, "past-only forecast", ha="right", va="bottom", color=MUTED)


def f_sim_fa(phi=0.4, rule="ar1", name="F07_sim_false_alarms"):
    """Fig 5.1 (AR(1) threshold) and the same figure with the independent threshold (not printed).
    Rows: adjustments; columns: detectors; one panel without and one with holidays."""
    fa = pd.read_csv(SIMS / "fa.csv")
    fa = fa[(fa["noise"] == "normal") & np.isclose(fa["phi"], phi)]
    fig = plt.figure(figsize=(TW, 4.1), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.09])
    for k, cal in enumerate((0, 1)):
        a = fig.add_subplot(gs[k, 0])
        _fa_panel(a, fa, cal, rule, show_cols=k == 0)
        a.set_ylabel("holidays in the data" if cal else "no holidays", fontweight="semibold", labelpad=6)
    cax = fig.add_subplot(gs[2, 0])
    sm = plt.cm.ScalarMappable(norm=FA_NORM, cmap=FA_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    cb.set_ticks(np.log10(ticks), labels=["≤0.5", "1", "2", "5 (target)", "10", "20", "50", "100"])
    cb.ax.tick_params(length=2)
    cb.outline.set_visible(False)
    cb.set_label("chance (%) of at least one false alarm in a normal year")
    save(fig, name)


def f_sim_fa_indep():
    f_sim_fa(rule="indep", name="F07_sim_false_alarms_indep")


MEAN_HL = ["CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r", "FOCuS-r"]   # in their detector colours
MEAN_BG = ["FOCuS-AR"]                                               # light grey


def f_sim_mean(phi=0.4):
    """Fig 5.2. Detection of lasting drops; uncapped detectors grey, CPM and the capped ones in colour."""
    det = pd.read_csv(SIMS / "detect.csv")
    g = det[(det["change"] == "mean") & (det["noise"] == "normal") & (det["rule"] == "oracle") & np.isclose(det["phi"], phi)
            & (det["size"] < 0)]
    adjs = ["oracle", "A", "B", "HW", "YM"]
    head = {"oracle": "true pattern\nremoved", "A": "same-weekday\ndifference", "B": "median\nof four",
            "HW": "Holt-Winters\n", "YM": "yearly\nmemory"}
    fig, ax = plt.subplots(2, len(adjs), figsize=(TW, 4.15), sharey=True, sharex=True, layout="constrained")
    for i, cal in enumerate((0, 1)):
        for j, adj in enumerate(adjs):
            a = ax[i, j]
            h = g[(g["adj"] == adj) & (g["calendar"] == cal)]
            for d in MEAN_BG + MEAN_HL:
                hh = h[h["det"] == d].sort_values("size", ascending=False)
                if d in MEAN_BG:
                    a.plot(-hh["size"], 100 * hh["detect60"], color=LIGHT, lw=1.1, zorder=1)
                else:
                    S.det_line(a, -hh["size"], 100 * hh["detect60"], d, ms=3.6, lw=1.3, zorder=3)
            a.set_xscale("log", base=2)
            a.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
            a.minorticks_off()
            a.set_ylim(-3, 103)
            a.set_yticks([0, 50, 100])
            S.ygrid(a)
            if i == 0:
                a.set_title(head[adj], fontweight="normal", linespacing=1.05)
        ax[i, -1].yaxis.set_label_position("right")
        ax[i, -1].set_ylabel("holidays in the data" if cal else "no holidays", rotation=270, va="bottom",
                             fontweight="semibold")
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.06, h_pad=0.06, hspace=0.06)  # keep tick labels apart
    fig.supylabel("% detected within 60 days", fontsize=S.FS)
    fig.supxlabel("size of the lasting drop (noise SDs; 1 SD ≈ 4.8% of daily tap-ons)", fontsize=S.FS)
    hs = []
    for d in MEAN_HL:
        cap = d.endswith("-r")
        hs.append(Line2D([], [], color=DET[d], ls=S.DLS[d], marker=S.DMK[d], ms=3.6, mfc="white" if cap else DET[d],
                         mec=DET[d], label=DNAME[d]))
    hs.append(Line2D([], [], color=LIGHT, lw=1.5, label="FOCuS-AR (grey)"))
    fig.legend(handles=hs, loc="outside upper center", ncol=4, handlelength=2.4, columnspacing=1.4)
    save(fig, "F08_sim_mean_shifts")


def f_sim_var(phi=0.4):
    """Variance-change panel, optional figure. Left: true pattern removed, holidays in the data, by the
    variance factor. Right: variance doubled, no holidays, by adjustment. Matched thresholds. Variance EWMA
    is drawn when detect.csv has it."""
    src = SIMS / "variance.csv" if (SIMS / "variance.csv").exists() else SIMS / "detect.csv"
    det = pd.read_csv(src)
    g = det[(det["change"] == "var") & (det["noise"] == "normal") & (det["rule"] == "oracle") & np.isclose(det["phi"], phi)]
    left = g[(g["calendar"] == 1) & (g["adj"] == "oracle")]
    if left.empty:
        print("  F09: no variance rows in detect.csv; skipped")
        return
    var_dets = [d for d in ["CUSUM-var", "EWMA-var", "FOCuS-var"] if d in set(g["det"])]
    if "EWMA-var" not in var_dets:
        print("  F09: variance EWMA not in detect.csv yet; drawn without it (pending analysis)")
    fig, ax = plt.subplots(1, 2, figsize=(TW, 2.45), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    for d in ["CUSUM", "EWMA"]:
        hh = left[left["det"] == d].sort_values("size")
        a.plot(hh["size"], 100 * hh["detect60"], color=LIGHT, lw=1.3, marker="o", ms=3)
    ends = []
    for d in var_dets:
        hh = left[left["det"] == d].sort_values("size")
        S.det_line(a, hh["size"], 100 * hh["detect60"], d)
        ends.append((100 * hh["detect60"].iloc[-1], DNAME[d], INK))
    hh = left[left["det"] == "CUSUM"].sort_values("size")
    if len(hh):
        ends.append((100 * hh["detect60"].iloc[-1], "mean CUSUM,\nEWMA", MUTED))
    ends.sort()
    merged = []  # lines ending within 4 points share one label
    for y, lab, col in ends:
        if merged and y - merged[-1][0] < 4 and col == merged[-1][2]:
            py, plab, pc = merged[-1]
            merged[-1] = ((py + y) / 2, plab + "\nand " + lab.replace("variance ", ""), pc)
        else:
            merged.append((y, lab, col))
    last = -99
    for y, lab, col in merged:  # keep direct labels at least 9 points apart
        y = max(y, last + 9 + 7 * lab.count("\n"))
        a.text(3.08, y, lab, color=col, va="center", linespacing=1.0)
        last = y
    a.set_xticks([1.5, 2, 3], ["1.5", "2", "3"])
    a.set_xlim(1.4, 4.45)
    a.set_ylim(-3, 100)
    a.set_xlabel("variance multiplied by")
    a.set_ylabel("% detected within 60 days")
    a.set_title("true pattern removed", fontweight="normal")
    S.ygrid(a)
    right = g[(g["calendar"] == 0) & np.isclose(g["size"], 2.0)]
    adjs = [x for x in ["oracle", "YM"] if x in set(right["adj"])]  # true pattern and yearly memory
    a = ax[1]
    for i, adj in enumerate(adjs):
        a.axhline(i, color=S.GRID, lw=0.6, zorder=0)
        for d in ["CUSUM", "EWMA"] + var_dets:
            q = right[(right["adj"] == adj) & (right["det"] == d)]
            if len(q):
                v = 100 * q["detect60"].iloc[0]
                if d in ("CUSUM", "EWMA"):
                    a.plot(v, i, "o", color=LIGHT, ms=5, zorder=2)
                else:  # dodge the spread detectors vertically so equal values stay visible
                    off = {"CUSUM-var": -0.14, "EWMA-var": 0.0, "FOCuS-var": 0.14}[d]
                    a.plot(v, i + off, S.DMK[d], color=DET[d], ms=5.5, zorder=3, mec="white", mew=0.5)
    a.set_yticks(range(len(adjs)), ["true pattern\nremoved" if x == "oracle" else S.ASHORT[x] for x in adjs])
    a.set_ylim(len(adjs) - 0.4, -0.6)
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.set_xlim(-2, 40)
    a.set_xlabel("% detected within 60 days")
    a.set_title("variance doubled, no holidays", fontweight="normal")
    S.xgrid(a)
    save(fig, "F09_sim_variance")


def f_sim_tornado(phi=0.4):
    """Fig 5.3. Share of variation (log-odds scale) by forecast, detector, threshold rule and
    interactions, for each noise setting: one bar among CUSUM, EWMA and FOCuS and one with CPM added
    (extra/shares_detsets.csv, detsets CEF and CEFP). False-alarm splits flagged degenerate (almost every cell at the
    bound) are not drawn."""
    ds = pd.read_csv(RES / "extra" / "shares_detsets.csv")
    ds = ds[np.isclose(ds["phi"], phi)]
    settings = [("normal", 0, "Gaussian, no holidays"), ("normal", 1, "Gaussian, holidays"),
                ("wide", 1, "holidays at real scale"), ("t5", 1, "$t_5$ noise, holidays"),
                ("boot", 1, "resampled residuals")]
    sets = [("CEF", "CUSUM, EWMA, FOCuS"), ("CEFP", "with CPM")]
    have = [(n, c, lab) for n, c, lab in settings if len(ds[(ds["noise"] == n) & (ds["calendar"] == c)])]
    parts = [("forecast", "#2b3442"), ("detector", "#6f7a8c"), ("threshold rule", "#b4bccb"), ("interactions", "white")]
    ys, ylabs, heads, y = [], [], [], 0.0
    for n, c, lab in have:
        heads.append((lab, y - 0.62))
        for k, (dset, dlab) in enumerate(sets):
            ys.append((n, c, dset, y))
            ylabs.append(dlab)
            y += 0.75
        y += 0.75
    fig, ax = plt.subplots(1, 2, figsize=(TW, 0.36 * len(ys) + 1.3), layout="constrained", sharey=True)
    nsat = 0
    for a, o, t in zip(ax, ("false_alarm_logit", "detect60_logit_matched_drops"),
                       ("false alarms", "detection, false alarms matched")):
        for n, c, dset, yy in ys:
            r = ds[(ds["noise"] == n) & (ds["calendar"] == c) & (ds["outcome"] == o) & (ds["detset"] == dset)]
            if not len(r) or bool(r["degenerate"].iloc[0]) or r[["adj", "det"]].isna().all(axis=None):
                a.text(1, yy, "saturated: not shown", va="center", ha="left", color=MUTED)
                nsat += 1
                continue
            r = r.iloc[0]
            vals = [r["adj"], r["det"], 0 if pd.isna(r["rule"]) else r["rule"]]
            vals.append(max(0.0, 100 - sum(vals)))
            left = 0.0
            for (name, col), v in zip(parts, vals):
                a.barh(yy, v, left=left, height=0.62, color=col, edgecolor="#6f7a8c" if col == "white" else "white",
                       lw=0.6, label=name)
                if v >= 9:
                    a.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center",
                           color="white" if col in ("#2b3442", "#6f7a8c") else INK)
                left += v
        a.set_xlim(0, 100)
        a.set_xticks([0, 25, 50, 75, 100])
        a.set_xlabel("share of variation (%, log-odds scale)")
        a.set_title(t, fontweight="normal")
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
    ax[0].set_yticks([yy for *_, yy in ys], ylabs)
    ax[0].set_ylim(y - 0.6, -1.05)
    for lab, yy in heads:  # setting name above each pair of bars
        ax[0].text(0, yy, lab, ha="left", va="center", color=INK, fontweight="semibold")
    h, l = ax[1].get_legend_handles_labels()
    keep = {lab: hh for hh, lab in zip(h, l)}
    fig.legend([keep[p] for p, _ in parts if p in keep], [p for p, _ in parts if p in keep], loc="outside upper center", ncol=4)
    if nsat:
        fig.text(0.01, 0.0, "Saturated: almost every cell alarms in every year, so the false-alarm split is not defined.",
                 ha="left", va="top", color=MUTED)
    save(fig, "F10_sim_what_matters")


# ------------------------------------------------------------------------------------------- Opal (Ch 6)
GRID_ORDER = ["A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG", "raw"]
GRID_ROW = {"A": "same-weekday difference", "B": "median of four", "HW": "Holt-Winters",
            "YM": "yearly memory", "MSTL_p": "MSTL refitted on the past",
            "REG_p": "regression refitted on the past", "STL": "retrospective STL", "MSTL": "retrospective MSTL",
            "REG": "retrospective regression", "raw": "no forecast"}
GRID_DETS = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
RECOMMENDED = {("HW", "EWMA")}   # Holt-Winters + FOCuS dropped from the recommendations


SHADE = LinearSegmentedColormap.from_list("ep", ["#ffffff", "#c9d0db", "#4a5568"])


def _grid_data(col):
    s = pd.read_csv(RES / "opal" / "scores.csv")
    rv = pd.read_csv(RES / "opal" / "rule_variants.csv")
    s = s.merge(rv[["method", "detector", "rule", col, f"{col}_p"]], on=["method", "detector", "rule"])
    s = s[s["rule"] == "ar1"].copy()
    s["episodes_caught"], s["p_episodes"] = s[col], s[f"{col}_p"]
    return s


def _grid_panel(ax, s, norm, recommended, compact, show_cols=True):
    nR, nC = len(GRID_ORDER), len(GRID_DETS)
    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(nR - 0.5, -0.5)
    for i, m in enumerate(GRID_ORDER):
        for j, d in enumerate(GRID_DETS):
            r = s[(s["method"] == m) & (s["detector"] == d)]
            if not len(r):
                continue
            r = r.iloc[0]
            n, ep = int(r["n_alarms"]), int(r["episodes_caught"])
            few = n < 10
            if few:  # fewer than 10 alarms: hatched, muted text
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec=LIGHT, hatch="////", lw=0, zorder=1))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none", ec="white", lw=1.5, zorder=1.5))
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=SHADE(norm(ep)), ec="white", lw=1.5, zorder=1))
            dark = (not few) and norm(ep) >= 0.6
            tc = "white" if dark else (MUTED if few else INK)
            star = "*" if r["p_episodes"] < 0.05 else ""
            bb = dict(facecolor="white", edgecolor="none", pad=0.3) if few else None
            if compact:
                ax.text(j, i, f"{ep}{star}", ha="center", va="center", fontsize=9.5, fontweight="semibold", color=tc,
                        bbox=bb, zorder=2)
            else:
                ax.text(j, i - 0.13, f"{ep}{star}", ha="center", va="center", fontsize=10.5, fontweight="semibold",
                        color=tc, bbox=bb, zorder=2)
                ax.text(j, i + 0.24, f"{int(r['near_real'])}/{n}", ha="center", va="center", fontsize=9, color=tc,
                        bbox=bb, zorder=2)
            if (m, d) in recommended:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fc="none", ec=BLUE, lw=2.0, zorder=3))
    ax.set_xticks(range(nC), [DNAME2[d] for d in GRID_DETS] if show_cols else [], linespacing=1.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(nR), [GRID_ROW[m] for m in GRID_ORDER])
    for tl, m in zip(ax.get_yticklabels(), GRID_ORDER):
        if any(m == r[0] for r in recommended):
            tl.set_color(BLUE)
            tl.set_fontweight("semibold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(5.5, color=INK, lw=0.8)   # past-only above, retrospective and none below
    ax.axhline(8.5, color=INK, lw=0.8)
    ax.axvline(4.5, color=INK, lw=0.8)   # uncapped | capped


def f_opal_grid(col="r0_7_dir", name="F13_opal_grid_single", recommended=None):
    """Single-rule grid with near/all in each cell (F13_opal_grid_original uses col="frozen")."""
    recommended = RECOMMENDED if recommended is None else recommended
    s = _grid_data(col)
    norm = Normalize(0, max(7, int(s["episodes_caught"].max())))
    fig, ax = plt.subplots(figsize=(TW, 4.35), layout="constrained")
    _grid_panel(ax, s, norm, recommended, compact=False)
    save(fig, name)


def f_opal_grid_both():
    """Not printed (Figure 6.2 is drawn by scripts/thesis_figures.py): main rule above, direction-aware check below; episodes caught (* p (own) < 0.05),
    one shade scale for both, hatched = fewer than 10 alarms, blue outline = Holt-Winters + EWMA (Table 8.2), legend."""
    from matplotlib.patches import Patch
    sf, sd = _grid_data("frozen"), _grid_data("r0_7_dir")
    vmax = int(max(sf["episodes_caught"].max(), sd["episodes_caught"].max()))
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(TW, 5.7), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.07])
    for k, (s, t) in enumerate([(sf, "main rule (fixed 22 October)"), (sd, "direction-aware check")]):
        a = fig.add_subplot(gs[k, 0])
        _grid_panel(a, s, norm, RECOMMENDED, compact=True, show_cols=k == 0)
        a.set_ylabel(t, fontweight="semibold", labelpad=6)
    cax = fig.add_subplot(gs[2, 0])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SHADE), cax=cax, orientation="horizontal")
    cb.set_ticks(range(0, vmax + 1, 2))
    cb.outline.set_visible(False)
    cb.set_label("episodes caught of 15, AR(1) threshold; * beats chance alone")
    hs = [Patch(fc="white", ec=LIGHT, hatch="////", label="fewer than 10 alarms"),
          Patch(fc="white", ec=BLUE, lw=2.0, label="Holt-Winters + EWMA (Table 8.2)")]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, handlelength=2.2)
    save(fig, "F13_opal_grid")


TL_CLASSES = [("real", "near a scored change", GREEN), ("after_event", "8–28 days after a change", GREY),
              ("school", "school holiday", GREY), ("holiday", "public holiday or Christmas", GREY),
              ("explained", "other listed explanation", GREY), ("unexplained", "unexplained", RED)]


def f_timeline(method="A", det="CUSUM", rule="ar1"):
    """Not printed (Figure 6.4 is drawn by scripts/thesis_figures.py). What each alarm of one monitor coincides with: one strip row per class."""
    from collections import Counter
    from cpd.score import classify, day_classes
    al = pd.read_csv(RES / "opal" / "alarms.csv", parse_dates=["alarm"])
    al = al[(al["method"] == method) & (al["detector"] == det) & (al["rule"] == rule)]
    x = opal_daily("train") / 1e6
    cls = day_classes(x.index)
    lab = classify([x.index.get_loc(d) for d in al["alarm"]], cls)
    n = Counter(lab)
    rows = [(k, name, c) for k, name, c in TL_CLASSES if n[k]]
    fig, ax = plt.subplots(2, 1, figsize=(TW, 3.3), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [0.9, 1.3]})
    for d in scored_opal()["date"].unique():
        S.boundary(ax[0], d, lw=1.3, ls=(0, (4, 2)), alpha=1, zorder=3)
    ax[0].plot(x.index, x.values, color=GREY, lw=0.3)
    ax[0].set_xlim(x.index[0] - pd.Timedelta(days=20), END + pd.Timedelta(days=20))
    ax[0].set_ylim(0, 1.55)
    ax[0].set_ylabel("tap-ons a day\n(millions)")
    S.ygrid(ax[0])
    for i, (k, name, c) in enumerate(rows):
        ds = [d for d, l in zip(al["alarm"], lab) if l == k]
        ax[1].axhline(i, color=GRID_C, lw=0.6, zorder=0)
        ax[1].plot(ds, [i] * len(ds), "|", color=c if c != GREY else "#6b7486", ms=11, mew=2.0)
    ax[1].set_yticks(range(len(rows)), [f"{name} ({n[k]})" for k, name, _ in rows])
    ax[1].set_ylim(len(rows) - 0.5, -0.5)
    ax[1].tick_params(axis="y", length=0)
    ax[1].spines["left"].set_visible(False)
    yearaxis(ax[1])
    save(fig, f"F14_timeline_{method}_{det}")


GRID_C = S.GRID


def f_testbed():
    """Not printed (Table 6.2 carries it). Gain from an inserted lasting drop, per adjustment,
    with the background alarm rate (no insertion) underneath when the summary has it."""
    s = pd.read_csv(RES / "testbed" / "summary.csv")
    s = s[(s["subset"] == "all") & (s["rule"] == "ar1") & s["setting"].str.startswith("drop")].copy()
    s["p"] = s["setting"].str[4:].astype(float)
    adjs = ["A", "B", "HW", "YM"]
    dets = ["CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r", "FOCuS-r"]
    has_bg = "detect_ctl" in s.columns
    fig, ax = plt.subplots(2 if has_bg else 1, len(adjs), figsize=(TW, 4.0 if has_bg else 2.5), sharex=True,
                           sharey="row", layout="constrained", squeeze=False)
    for j, adj in enumerate(adjs):
        for d in dets:
            h = s[(s["adj"] == adj) & (s["det"] == d)].sort_values("p")
            S.det_line(ax[0, j], h["p"], 100 * h["delta"], d, ms=3, lw=1.1, label=DNAME[d] if j == 0 else None)
            if has_bg:
                bg = h.groupby("p")["detect_ctl"].first()
                S.det_line(ax[1, j], bg.index, 100 * bg.values, d, ms=3, lw=1.1)
        ax[0, j].set_title(S.ASHORT[adj], fontweight="normal")
        for a in ax[:, j]:
            a.set_xscale("log")
            a.set_xticks([2.5, 5, 10, 20, 40], ["2.5", "5", "10", "20", "40"])
            a.minorticks_off()
            S.ygrid(a)
    ax[0, 0].set_ylabel("extra detection\nwithin 60 days (points)")
    if has_bg:
        ax[1, 0].set_ylabel("background: alarm\nwithout insertion (%)")
    fig.supxlabel("lasting drop inserted (%)", fontsize=S.FS)
    fig.legend(loc="outside upper center", ncol=3, handlelength=2.2, columnspacing=1.2)
    save(fig, "F15_testbed")


COVID_PICK = [("HW", "FOCuS"), ("HW", "EWMA"), ("A", "CUSUM"), ("A", "FOCuS"), ("B", "EWMA"), ("YM", "CUSUM"),
              ("A", "CPM"), ("STL", "CUSUM"), ("MSTL_p", "CUSUM")]
COVID_NAME = {"A": "same-weekday difference", "B": "median of four", "HW": "Holt-Winters", "YM": "yearly memory", "STL": "retrospective STL",
              "MSTL_p": "MSTL refitted on the past"}


def _start_est(c, alarms):
    """Estimated change date for each first alarm: covid.csv 'start_est' when the analysis adds it,
    otherwise the 'start_est' of the same alarm in alarms.csv; None if neither exists."""
    if "start_est" in c.columns:
        return {(r.method, r.detector): r.start_est for r in c.itertuples()}
    if "start_est" not in alarms.columns:
        print("  F16: no estimated change dates available; ticks skipped")
        return {}
    a = alarms[alarms["rule"] == "ar1"]
    out = {}
    for r in c.itertuples():
        q = a[(a["method"] == r.method) & (a["detector"] == r.detector) & (a["alarm"] == r.first_alarm)]
        if len(q):
            out[(r.method, r.detector)] = q["start_est"].iloc[0]
    return out


def f_covid():
    """Not printed (Figure 6.5 is drawn by scripts/thesis_figures.py). COVID onset: tap-ons against February 2020, and each monitor's first alarm (red) with its
    estimated change date (open tick)."""
    x = opal_daily("train")
    feb = x.loc["2020-02-03":"2020-02-29"]
    ref = feb.groupby(feb.index.dayofweek).median()
    w = x.loc["2020-03-01":"2020-04-20"]
    ratio = 100 * w / w.index.dayofweek.map(ref).to_numpy()
    c = pd.read_csv(RES / "opal" / "covid.csv", parse_dates=["first_alarm"])
    c = c[c["rule"] == "ar1"]
    al = pd.read_csv(RES / "opal" / "alarms.csv", parse_dates=["alarm", "start_est"])
    tau = {k: pd.Timestamp(v) for k, v in _start_est(c, al).items() if pd.notna(v)}
    ev = [("2020-03-16", "first public health order", 136), ("2020-03-23", "venues close", 121),
          ("2020-03-31", "stay-at-home order", 106)]
    fig, ax = plt.subplots(2, 1, figsize=(TW, 4.5), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [1.3, 1.6]})
    for d, lab, y in ev:
        ax[0].text(pd.Timestamp(d) + pd.Timedelta(hours=14), y, lab, color=INK, va="top", ha="left", zorder=6,
                   bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax[0].plot(ratio.index, ratio.values, color=INK, lw=1.3, marker=".", ms=4)
    for d, _, _ in ev:  # drawn last, on top of the series and the labels, so all three read green
        for a in ax:
            S.boundary(a, pd.Timestamp(d), lw=1.3, alpha=1, zorder=5)
    ax[0].set_ylim(0, 138)
    ax[0].set_yticks([0, 25, 50, 75, 100])
    ax[0].set_ylabel("tap-ons, % of a\nFebruary 2020 day")
    S.ygrid(ax[0])
    labels = []
    for k, (m, d) in enumerate(COVID_PICK):
        q = c[(c["method"] == m) & (c["detector"] == d)]
        labels.append(f"{COVID_NAME[m]} + {d}")
        if len(q) and pd.notna(q["first_alarm"].iloc[0]):
            t = q["first_alarm"].iloc[0]
            te = None if d == "EWMA" else tau.get((m, d))  # EWMA has no change-date estimator
            if te is not None and te <= t:
                ax[1].plot([te, t], [k, k], color=GREY, lw=0.8, zorder=1)
                ax[1].plot(te, k, marker="|", ms=8, mew=1.6, color=INK, zorder=2)
            S.alarm_marks(ax[1], [t], [k], ms=7)
            ax[1].text(t + pd.Timedelta(hours=20), k, f"{t:%d %b}", va="center", ha="left", color=INK, zorder=6,
                       bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        else:
            ax[1].text(ratio.index[-1], k, "no alarm", va="center", ha="right", color=MUTED)
    ax[1].set_yticks(range(len(COVID_PICK)), labels)
    for tl, md in zip(ax[1].get_yticklabels(), COVID_PICK):
        if md in RECOMMENDED:
            tl.set_color(BLUE)
            tl.set_fontweight("semibold")
    ax[1].set_ylim(len(COVID_PICK) - 0.5, -0.5)
    ax[1].tick_params(axis="y", length=0)
    ax[1].spines["left"].set_visible(False)
    ax[1].grid(axis="y", color=S.GRID, lw=0.6)
    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax[1].set_xlabel("2020 (AR(1) threshold; $\\hat{\\phi}$ = 0 in March 2020,\nso equal to the independent threshold)")
    if tau:
        hs = [Line2D([], [], color=RED, marker="v", ls="", ms=7, mec="white", mew=0.5, label="first alarm"),
              Line2D([], [], color=INK, marker="|", ls="", ms=8, mew=1.6, label="its estimated change date")]
        ax[1].legend(handles=hs, loc="upper right", ncol=1, handletextpad=0.3, borderaxespad=0.1,
                     facecolor="white", frameon=True, edgecolor="none", framealpha=1)
    save(fig, "F16_covid")


TRADE_NAMED = [("HW", "FOCuS", "Holt-Winters + FOCuS"), ("HW", "EWMA", "Holt-Winters + EWMA"), ("A", "CUSUM", "same-weekday\ndifference + CUSUM"),
               ("A", "FOCuS-r", "same-weekday difference\n+ capped FOCuS"), ("YM", "FOCuS", "fixed monitor: yearly\nmemory + FOCuS")]
# label offsets (points) per panel, chosen so labels clear the points; keys: (rule column, method, detector)
TRADE_OFF = {("frozen", "HW", "FOCuS"): (14, -14, "left"), ("frozen", "HW", "EWMA"): (-14, 12, "right"),
             ("frozen", "A", "CUSUM"): (-14, 10, "right"), ("frozen", "A", "FOCuS-r"): (14, -16, "left"),
             ("frozen", "YM", "FOCuS"): (16, -24, "left"),
             ("r0_7_dir", "HW", "FOCuS"): (-14, -20, "right"), ("r0_7_dir", "HW", "EWMA"): (-12, 14, "right"),
             ("r0_7_dir", "A", "CUSUM"): (-12, 16, "right"), ("r0_7_dir", "A", "FOCuS-r"): (14, -26, "left"),
             ("r0_7_dir", "YM", "FOCuS"): (16, -30, "left")}


def f_tradeoff(xcol="non_event_per_year"):
    """Not printed. Episodes caught against alarms a year away from any scored change, main rule (top) and
    direction-aware check (bottom); every AR(1) monitor in light grey, the named ones labelled, HW + EWMA in blue."""
    s = pd.read_csv(RES / "opal" / "scores.csv")
    rv = pd.read_csv(RES / "opal" / "rule_variants.csv")
    s = s.merge(rv[["method", "detector", "rule", "frozen", "r0_7_dir"]], on=["method", "detector", "rule"])
    s = s[s.rule == "ar1"].reset_index(drop=True)
    xl = {"non_event_per_year": "alarms a year away from any scored change",
          "unexplained_per_year": "unexplained alarms a year"}[xcol]
    key = {(m, d) for m, d, _ in TRADE_NAMED}
    rng = np.random.default_rng(3)
    jit = rng.uniform(-0.13, 0.13, len(s))
    ymax = int(max(s.frozen.max(), s.r0_7_dir.max())) + 1
    fig, axs = plt.subplots(2, 1, figsize=(TW, 5.0), sharex=True, layout="constrained")
    for ax, col, t in zip(axs, ("frozen", "r0_7_dir"), ("main rule", "direction-aware check")):
        bg = [i for i, r in s.iterrows() if (r.method, r.detector) not in key]
        ax.scatter(s.loc[bg, xcol], s.loc[bg, col] + jit[bg], s=20, color=LIGHT, edgecolors="white", linewidths=0.4, zorder=2)
        for m, d, lab in TRADE_NAMED:
            r = s[(s.method == m) & (s.detector == d)].iloc[0]
            c = BLUE if (m, d) in RECOMMENDED else INK
            dx, dy, ha = TRADE_OFF[(col, m, d)]
            ax.scatter(r[xcol], r[col], s=34, color=c, edgecolors="white", linewidths=0.6, zorder=4)
            ax.annotate(lab, (r[xcol], r[col]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=INK,
                        linespacing=1.0, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=3))
        ax.set_ylim(-0.6, ymax + 0.4)
        ax.set_yticks(range(0, ymax + 1, 2))
        ax.set_ylabel(f"episodes caught (of 15),\n{t}")
        S.ygrid(ax)
    axs[1].text(0.01, 0.98, "grey: every other forecast and detector", transform=axs[1].transAxes, ha="left",
                va="top", color=MUTED)
    axs[1].set_xlim(0, None)
    axs[1].set_xlabel(xl)
    save(fig, "F21_tradeoff")


# ------------------------------------------------------------------------------------------- electricity (Ch 7)
def f_electricity(adj="REGT_p", det="FOCuS", rule="ar1", start="2020-01-01"):
    """Not printed (Figure 7.1 is drawn by scripts/thesis_figures.py). Demand and forecast errors; scored lockdown periods shaded green; alarms as red triangles on the
    forecast-error panel only, each labelled with its date."""
    E = RES / "electricity"
    a = pd.read_csv(E / "adjusted.csv", index_col=0, parse_dates=True).loc[start:]
    al = pd.read_csv(E / "alarms.csv", parse_dates=["alarm"])
    al = al[(al["adj"] == adj) & (al["det"] == det) & (al["rule"] == rule)]
    x = pd.read_csv(EVE / "nsw_demand_daily.csv", parse_dates=["date_nem"]).set_index("date_nem")["mean_mw"] / 1000
    x = x.loc[start:]
    sc = pd.read_csv(EVE / "events_electricity_scored_FROZEN.csv", parse_dates=["date"])
    spans = []
    for on, off in [("covid_onset", "covid_easing"), ("delta_onset", "delta_end")]:
        d0 = sc.loc[sc["episode"] == on, "date"].min()
        d1 = sc.loc[sc["episode"] == off, "date"].max()
        if pd.notna(d0) and pd.notna(d1):
            spans.append((d0, d1))
    fig, ax = plt.subplots(2, 1, figsize=(TW, 3.5), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [1, 1.1]})
    for q in ax:
        for d0, d1 in spans:
            q.axvspan(d0, d1, color=GREEN_L, lw=0, zorder=0)
        for d in sc["date"]:
            S.boundary(q, d, lw=0.8)
    ax[0].plot(x.index, x.values, color=INK, lw=0.45)
    ax[0].set_ylabel("daily mean\ndemand (GW)")
    for (d0, d1), lab in zip(spans, ["COVID-19\nonset", "2021\nlockdown"]):
        ax[0].text(d1 + pd.Timedelta(days=12), x.max() * 1.21, lab, color=MUTED, va="top", ha="left", linespacing=1.0)
    ax[0].set_ylim(None, x.max() * 1.22)
    ax[0].set_yticks([6, 8, 10])
    s = a[adj]
    ax[1].plot(s.index, s.values, color=GREY, lw=0.45)
    ax[1].axhline(0, color=MUTED, lw=0.6, ls=":")
    top = np.nanmax(s.values) * 1.05
    ax[1].set_ylim(np.nanmin(s.values) * 1.05, top * 1.45)
    S.alarm_marks(ax[1], al["alarm"], [top * 1.12] * len(al), ms=7)
    for d in al["alarm"]:
        ax[1].text(d, top * 1.12 + 0.012, f"{d:%d %b}", ha="center", va="bottom", color=INK)
    ax[1].set_ylabel("forecast error (log)")
    for q in ax:
        S.ygrid(q)
    yearaxis(ax[1])
    ax[1].set_xlim(x.index[0] - pd.Timedelta(days=15), x.index[-1] + pd.Timedelta(days=15))
    save(fig, f"F19_electricity_{adj}_{det}")


def f_dashboard():
    """Not printed. Screenshot of the live page (live/dashboard.html, regenerated by update.py --no-refresh),
    cropped to status, the 120-day chart and the alarm log; 'How it works' and the credits are hidden."""
    import shutil
    import subprocess
    import tempfile
    from PIL import Image, ImageChops
    page = HERE / "live" / "dashboard.html"
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not page.exists() or not chrome.exists():
        print("  F20: dashboard.html or Chrome missing; skipped")
        return
    html = page.read_text()
    cut = html.find("<h2>How it works")
    if cut > 0:
        html = html[:cut] + "</body></html>"
    # print view: narrower page so 15-16 px text prints at >= 9 pt at 0.9 \textwidth; the note under the chart
    # is left to the caption
    html = html.replace("</style>", "body{max-width:660px !important;margin:12px auto !important}"
                        ".small,table{font-size:15px !important}img+p.small{display:none}</style>")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "p.html").write_text(html)
    png = tmp / "p.png"
    subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--screenshot={png}", "--window-size=700,1400", (tmp / "p.html").as_uri()],
                   check=True, capture_output=True, timeout=60)
    im = Image.open(png).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if bbox:
        pad = 24
        im = im.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, im.width),
                      min(bbox[3] + pad, im.height)))
    im.save(FIG / "F20_dashboard.png", optimize=True)
    w_in = 0.9 * TW
    print(f"  F20_dashboard: {im.width}x{im.height} px; printed {w_in:.2f} x {w_in * im.height / im.width:.2f} in; "
          f"15 px text prints at {15 * 2 * w_in * 72 / im.width:.1f} pt")
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------- methods (Ch 4)
def f_pipeline():
    """Not printed (Figure 4.1 is drawn by scripts/thesis_figures.py). The monitoring procedure in one row, with the cap as a dashed side branch."""
    steps = ["daily\ncount", "forecast\nfrom past\ndays", "forecast\nerror\n(log gap)", "divide by\nrunning\nscale",
             "detector\nstatistic", "threshold\nfor the\nestimated ϕ", "alarm;\nrelearn\n28 days"]
    W, H = TW - 0.04, 1.75
    n = len(steps)
    gap = 0.15
    bw = (W - 0.3 - (n - 1) * gap) / n
    bh, y0 = 0.62, 0.62
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    arrow = dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=8, shrinkA=0, shrinkB=0)
    xs = [0.02 + k * (bw + gap) for k in range(n)]
    for k, (x, t) in enumerate(zip(xs, steps)):
        last = k == n - 1
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#fbeeed" if last else PALE, ec=RED if last else GREY, lw=0.9))
        ax.text(x + bw / 2, y0 + bh / 2, t, ha="center", va="center", linespacing=1.05, color=INK)
        if k < n - 1:
            ax.annotate("", xy=(x + bw + gap - 0.02, y0 + bh / 2), xytext=(x + bw + 0.02, y0 + bh / 2), arrowprops=arrow)
    # dashed side branch between "divide by running scale" (3) and "detector statistic" (4)
    cx0, cx1 = xs[3] + bw / 2, xs[4] + bw / 2
    bx0, bx1 = xs[3] - 0.07, xs[4] + bw + 0.07
    cy, ch = 0.04, 0.42
    ax.add_patch(FancyBboxPatch((bx0, cy), bx1 - bx0, ch, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=GREY, lw=0.9, ls=(0, (3, 2))))
    ax.text((bx0 + bx1) / 2, cy + ch / 2, "cap each day at ±3\n(capped detectors only)", ha="center", va="center",
            color=INK, linespacing=1.05)
    dash = dict(arrow, ls=(0, (3, 2)), color=MUTED)
    ax.annotate("", xy=(cx0, cy + ch), xytext=(cx0, y0), arrowprops=dash)
    ax.annotate("", xy=(cx1, y0), xytext=(cx1, cy + ch), arrowprops=dash)
    # bracket over the past-only part; note under the alarm box
    yb = y0 + bh + 0.1
    x_a, x_b = xs[0], xs[5] + bw
    ax.plot([x_a, x_a, x_b, x_b], [yb, yb + 0.07, yb + 0.07, yb], color=MUTED, lw=0.8)
    ax.text((x_a + x_b) / 2, yb + 0.12, "uses only days before today", ha="center", va="bottom", color=MUTED)
    ax.text(xs[6] + bw / 2, y0 - 0.08, "then compared\nwith holiday and\nevent lists", ha="center", va="top",
            color=MUTED, linespacing=1.05)
    save(fig, "F22_pipeline")


ALL = {"F02": f_opal_series, "F03": f_rhythm, "F04": f_adjust_covid, "F05": f_detectors_toy, "F06": f_thresholds,
       "F07": f_sim_fa, "F07i": f_sim_fa_indep, "F08": f_sim_mean, "F09": f_sim_var, "F10": f_sim_tornado,
       "F13": f_opal_grid_both, "F13o": lambda: f_opal_grid("frozen", "F13_opal_grid_original", recommended={("A", "CUSUM")}), "F14": f_timeline, "F15": f_testbed, "F16": f_covid,
       "F19": lambda: f_electricity("REGT_p", "FOCuS", "ar1"), "F20": f_dashboard,
       "F21": f_tradeoff, "F22": f_pipeline}


if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        try:
            ALL[n]()
            print("ok", n)
        except Exception as e:  # keep going so one missing input does not stop the rest
            import traceback
            traceback.print_exc()
            print("skip", n, repr(e)[:200])
