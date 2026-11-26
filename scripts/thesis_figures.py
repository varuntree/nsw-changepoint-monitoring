"""Eight of the fifteen figures printed in the thesis (Figures 3.1, 4.1, 4.3, 4.4, 6.2, 6.4, 6.5 and 7.1), and two
figures that are not printed (F04, F21). Figures 5.1, 5.2 and 5.3 come from scripts/figures.py, and Figures 4.2, 6.1,
6.3 and 8.1 from scripts/fig_worked_example.py, fig_three_cases.py, fig_episode_catches.py and fig_quiet_or_sensitive.py.

These are redrawn forms of figures in scripts/figures.py, with one message each, plain names for the methods
(scripts/figure_names.py), "forecast error" for the standardised one-step error, and the two scoring rules named
"main rule" and "direction-aware check". They read results/ and use the data helpers of scripts/figures.py.

Output: figures/<name>.pdf (what the thesis includes) and figures/<name>.png (a copy for viewing).
Usage:  python scripts/thesis_figures.py [keys...]     keys: F02 F04 F13 F14 F16 F19 F21 F22 F23 F24 (default: all)
Every figure is drawn at its printed width, so a font size here is the printed size (>= 9 pt).
"""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent      # scripts/
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

import figstyle as S  # noqa: E402
import figures as F  # noqa: E402  (data helpers and the grid panel; its FIG folder is not written to)
import figure_names as N  # noqa: E402
from cpd.data import opal_daily  # noqa: E402
from figstyle import BLUE, GREEN, GREEN_L, GREY, INK, LIGHT, MUTED, PALE, RED, TW  # noqa: E402

S.setup()
RES = ROOT / "results"
FIG = ROOT / "figures"
PRE = FIG                                   # PNG copies next to the PDFs
OUT = Path(tempfile.mkdtemp())              # side tables of the plotted values (not kept)
END = F.END
SIZES = {}


def save(fig, name, print_w=TW):
    fig.canvas.draw()
    (pt, txt), scale, w = S.print_pt(fig, print_w)
    h = fig.get_tightbbox(fig.canvas.get_renderer()).height * scale
    flag = "  <-- below 9 pt" if pt < S.MIN_PT - 0.05 else ""
    print(f"  {name}: printed {print_w / TW:.2f} textwidth x {h:.2f} in (x{scale:.2f}); smallest text {pt:.1f} pt "
          f"({txt[:24]!r}){flag}")
    SIZES[name] = (print_w / TW, h, pt)
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(PRE / f"{name}.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- Fig 3.1
def f_opal_series():
    """COVID-19 cut train travel by about four fifths; most scored changes fall in 2020 to 2022."""
    x = opal_daily("train") / 1e6
    sc = F.scored_opal()
    nb, ne = sc["date"].nunique(), sc["episode"].nunique()
    med = x.rolling(28, center=True, min_periods=7).median()
    fig, ax = plt.subplots(figsize=(TW, 2.4), layout="constrained")
    for d in sc["date"].unique():
        S.boundary(ax, d, lw=1.2, ls=(0, (4, 2)), alpha=1, zorder=2.5)
    ax.plot(x.index, x.values, color=GREY, lw=0.35, zorder=2)
    ax.plot(med.index, med.values, color=INK, lw=1.4, zorder=3)
    ax.set_xlim(pd.Timestamp("2019-12-20"), END + pd.Timedelta(days=4))
    ax.set_ylim(0, 1.98)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_ylabel("train tap-ons a day (millions)")
    F.yearaxis(ax)
    S.ygrid(ax)
    box = dict(facecolor="white", edgecolor="none", pad=0.6)
    ax.annotate("daily count", (pd.Timestamp("2023-09-02"), x.loc["2023-09-02"]),
                xytext=(pd.Timestamp("2023-02-15"), 0.2), color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=1))
    ax.text(pd.Timestamp("2023-06-15"), med.loc["2023-06-15"] + 0.12, "28-day median", color=INK, ha="center",
            va="bottom")
    arrow = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=1)
    ax.annotate("COVID-19 restrictions\n(March 2020)", (pd.Timestamp("2020-03-23"), 1.28),
                xytext=(pd.Timestamp("2020-05-25"), 1.94), ha="left", va="top", color=INK, linespacing=1.05,
                bbox=box, zorder=4, arrowprops=arrow)
    ax.annotate("2021 lockdown", (pd.Timestamp("2021-06-26"), 1.02), xytext=(pd.Timestamp("2021-08-05"), 1.42),
                ha="left", va="top", color=INK, bbox=box, zorder=4, arrowprops=arrow)
    ax.annotate("2025 industrial\naction", (pd.Timestamp("2025-01-15"), 1.42),
                xytext=(pd.Timestamp("2025-06-20"), 1.94), ha="left", va="top", color=INK, linespacing=1.05,
                bbox=box, zorder=4, arrowprops=arrow)
    ax.text(pd.Timestamp("2022-10-10"), 1.94, f"green dashed lines:\nthe {nb} dated boundaries\nof the {ne} scored episodes",
            color=GREEN, ha="left", va="top", linespacing=1.05, bbox=box, zorder=4)
    save(fig, "F02_opal_series")


# ------------------------------------------------------------------------------------------- not printed
def f_adjust_covid():
    """A past-only forecast turns a lasting drop into a dip that fades; a hindsight adjustment keeps the step."""
    adj = pd.read_csv(RES / "opal" / "adjusted.csv", parse_dates=["date"]).set_index("date")
    w = slice("2020-02-01", "2020-05-31")
    feb = slice("2020-02-01", "2020-02-29")
    panels = [("raw", "no forecast", True), ("A", N.FORECAST["A"], False), ("HW", "Holt-Winters", False),
              ("STL", "retrospective STL (hindsight)", True)]
    fig, ax = plt.subplots(2, 2, figsize=(TW, 3.1), sharex=True, sharey=True, layout="constrained")
    for a, (m, lab, level) in zip(ax.flat, panels):
        s = adj[m].loc[w]
        if level:                                   # level series: shown relative to their February 2020 median
            s = s - adj[m].loc[feb].median()
        for d in ("2020-03-16", "2020-03-23"):
            S.boundary(a, pd.Timestamp(d), lw=1.0)
        a.axhline(0, color=GREY, lw=0.7, ls=":")
        a.plot(s.index, s.values, color=INK, lw=1.1)
        a.set_title(lab, fontweight="normal")
        S.ygrid(a)
    ax[0, 0].set_ylim(-2.75, 1.75)
    ax[0, 0].set_yticks([-2, -1, 0, 1])
    ax[0, 0].set_ylabel("log tap-ons, relative\nto February 2020")
    ax[0, 1].set_ylabel(N.SURPRISE)
    ax[1, 0].set_ylabel(N.SURPRISE)
    ax[1, 1].set_ylabel("adjusted log tap-ons,\nrelative to February")
    for a in ax[:, 1]:
        a.yaxis.set_label_position("right")
    ax[0, 0].text(pd.Timestamp("2020-03-14"), 1.7, "16 Mar", color=GREEN, va="top", ha="right")
    ax[0, 0].text(pd.Timestamp("2020-03-25"), 1.7, "23 Mar", color=GREEN, va="top", ha="left")
    for a in ax[-1]:
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        a.set_xlabel("2020")
    save(fig, "F04_adjustments_covid")


# ------------------------------------------------------------------------------------------- Fig 6.2
GRID_ORDER = ["A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG"]      # "no forecast" row dropped


def f_opal_grid():
    """Main rule: one monitor catches 12 of 15. Check: none with the AR(1) threshold catches more than 7."""
    F.GRID_ORDER = GRID_ORDER
    F.GRID_ROW = {m: N.FORECAST[m] for m in GRID_ORDER}
    sf, sd = F._grid_data("frozen"), F._grid_data("r0_7_dir")
    sf, sd = sf[sf.method != "raw"], sd[sd.method != "raw"]
    vmax = int(max(sf["episodes_caught"].max(), sd["episodes_caught"].max()))
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(TW, 4.55), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.07])
    for k, (s, col) in enumerate([(sf, "frozen"), (sd, "r0_7_dir")]):
        a = fig.add_subplot(gs[k, 0])
        F._grid_panel(a, s, norm, F.RECOMMENDED, compact=True, show_cols=k == 0)
        a.lines[1].remove()                        # the rule above the dropped "no forecast" row
        a.set_ylabel(N.RULE[col], fontweight="semibold", labelpad=6)
    cax = fig.add_subplot(gs[2, 0])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=F.SHADE), cax=cax, orientation="horizontal")
    cb.set_ticks(range(0, vmax + 1, 2))
    cb.outline.set_visible(False)
    cb.set_label("episodes caught of 15 (AR(1) threshold); * beats chance on its own")
    hs = [Patch(fc="white", ec=LIGHT, hatch="////", label="fewer than 10 alarms"),
          Patch(fc="white", ec=BLUE, lw=2.0, label="Holt-Winters + EWMA, the steadiest monitor")]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, handlelength=2.2)
    save(fig, "F13_opal_grid")


# ------------------------------------------------------------------------------------------- not printed
TRADE = [("A", "CUSUM", N.monitor("A", "CUSUM"), ("data", 3.3, 15.3, "center")),
         ("HW", "EWMA", "Holt-Winters + EWMA", ("data", 3.68, 9.75, "right")),
         ("HW", "FOCuS", "Holt-Winters + FOCuS", ("data", 5.9, 13.3, "center")),
         ("A", "FOCuS-r", N.monitor("A", "FOCuS-r").replace(" + ", "\n+ "), ("off", 0, 32, "center")),
         ("YM", "FOCuS", "fixed monitor:\nyearly memory + FOCuS", ("data", 7.15, 1.2, "right"))]


def f_tradeoff():
    """No monitor catches many episodes without also raising several alarms a year away from any scored change."""
    s = pd.read_csv(RES / "opal" / "scores.csv")
    rv = pd.read_csv(RES / "opal" / "rule_variants.csv")
    s = s.merge(rv[["method", "detector", "rule", "frozen", "r0_7_dir"]], on=["method", "detector", "rule"])
    s = s[s.rule == "ar1"].reset_index(drop=True)
    xcol = "non_event_per_year"
    key = {(m, d) for m, d, _, _ in TRADE}
    rng = np.random.default_rng(3)
    jit = rng.uniform(-0.13, 0.13, len(s))
    W = 0.75 * TW
    fig, ax = plt.subplots(figsize=(W, 2.95), layout="constrained")
    bg = [i for i, r in s.iterrows() if (r.method, r.detector) not in key]
    ax.scatter(s.loc[bg, xcol], s.loc[bg, "frozen"] + jit[bg], s=18, color=LIGHT, edgecolors="white", linewidths=0.4,
               zorder=2)
    rows = []
    for m, d, lab, (how, dx, dy, ha) in TRADE:
        r = s[(s.method == m) & (s.detector == d)].iloc[0]
        c = BLUE if (m, d) in F.RECOMMENDED else INK
        ax.plot([r[xcol]] * 2, [r["r0_7_dir"], r["frozen"]], color=c, lw=0.9, zorder=3)
        ax.scatter(r[xcol], r["r0_7_dir"], s=30, facecolors="white", edgecolors=c, linewidths=1.1, zorder=4)
        ax.scatter(r[xcol], r["frozen"], s=34, color=c, edgecolors="white", linewidths=0.6, zorder=5)
        ax.annotate(lab, (r[xcol], r["frozen"]), xytext=(dx, dy), textcoords="offset points" if how == "off" else "data",
                    ha=ha, va="center",
                    color=INK, linespacing=1.0, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=3))
        rows.append(dict(method=m, detector=d, non_event_per_year=r[xcol], main=r["frozen"], check=r["r0_7_dir"]))
    pd.DataFrame(rows).to_csv(OUT / "tradeoff_labelled.csv", index=False)
    ax.set_ylim(-0.6, 16.3)
    ax.set_yticks(range(0, 16, 3))
    ax.set_xlim(0, None)
    ax.set_ylabel("episodes caught (of 15)")
    ax.set_xlabel(N.NON_EVENT)
    S.ygrid(ax)
    hs = [Line2D([], [], marker="o", ls="", ms=6, color=INK, mec="white", label="main rule"),
          Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=INK, mew=1.1, label="direction-aware check"),
          Line2D([], [], marker="o", ls="", ms=5, color=LIGHT, mec="white", label="other monitors (main rule)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=2, handletextpad=0.1, columnspacing=1.0)
    save(fig, "F21_tradeoff", print_w=W)


# ------------------------------------------------------------------------------------------- Fig 6.4
def f_timeline(method="A", det="CUSUM", rule="ar1"):
    """Of the 36 alarms of one monitor, 11 are near a scored change, 12 in school holidays, 8 unexplained."""
    from cpd.score import classify, day_classes
    al = pd.read_csv(RES / "opal" / "alarms.csv", parse_dates=["alarm"])
    al = al[(al["method"] == method) & (al["detector"] == det) & (al["rule"] == rule)]
    x = opal_daily("train") / 1e6
    cls = day_classes(x.index)
    lab = classify([x.index.get_loc(d) for d in al["alarm"]], cls)
    n = Counter(lab)
    pd.DataFrame([dict(cls=k, n=v) for k, v in n.items()] + [dict(cls="all", n=len(al))]).to_csv(
        OUT / "timeline_counts.csv", index=False)
    rows = [(k, name, c) for k, name, c in F.TL_CLASSES if n[k]]
    fig, ax = plt.subplots(2, 1, figsize=(TW, 2.75), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [0.42, 1.3]})
    for d in F.scored_opal()["date"].unique():
        S.boundary(ax[0], d, lw=1.1, ls=(0, (4, 2)), alpha=1, zorder=3)
    ax[0].plot(x.index, x.values, color=GREY, lw=0.3)
    ax[0].set_xlim(x.index[0] - pd.Timedelta(days=20), END + pd.Timedelta(days=20))
    ax[0].set_ylim(0, 1.55)
    ax[0].set_yticks([0, 1])
    ax[0].set_ylabel("tap-ons\n(millions)")
    ax[0].set_title(f"{len(al)} alarms of {N.monitor(method, det)}, {N.THRESHOLD[rule]}", fontweight="normal")
    for i, (k, name, c) in enumerate(rows):
        ds = [d for d, l in zip(al["alarm"], lab) if l == k]
        ax[1].axhline(i, color=S.GRID, lw=0.6, zorder=0)
        ax[1].plot(ds, [i] * len(ds), "|", color=c if c != GREY else MUTED, ms=11, mew=2.0)
    ax[1].set_yticks(range(len(rows)), [f"{name} ({n[k]})" for k, name, _ in rows])
    ax[1].set_ylim(len(rows) - 0.5, -0.5)
    ax[1].tick_params(axis="y", length=0)
    ax[1].spines["left"].set_visible(False)
    F.yearaxis(ax[1])
    save(fig, "F14_timeline_A_CUSUM")


# ------------------------------------------------------------------------------------------- Fig 6.5
def f_covid():
    """Most monitors alarmed within two days of the first public health order of 16 March 2020."""
    x = opal_daily("train")
    feb = x.loc["2020-02-03":"2020-02-29"]
    ref = feb.groupby(feb.index.dayofweek).median()
    w = x.loc["2020-03-01":"2020-04-20"]
    ratio = 100 * w / w.index.dayofweek.map(ref).to_numpy()
    c = pd.read_csv(RES / "opal" / "covid.csv", parse_dates=["first_alarm"])
    c = c[c["rule"] == "ar1"]
    al = pd.read_csv(RES / "opal" / "alarms.csv", parse_dates=["alarm", "start_est"])
    tau = {k: pd.Timestamp(v) for k, v in F._start_est(c, al).items() if pd.notna(v)}
    first = {(m, d): c[(c["method"] == m) & (c["detector"] == d)]["first_alarm"].iloc[0] for m, d in F.COVID_PICK}
    pick = sorted(F.COVID_PICK, key=lambda md: (first[md], F.COVID_PICK.index(md)))      # by alarm date
    pd.DataFrame([dict(method=m, detector=d, first_alarm=first[(m, d)].date(),
                       start_est=(tau.get((m, d)).date() if (m, d) in tau and d != "EWMA" else ""))
                  for m, d in pick]).to_csv(OUT / "covid_rows.csv", index=False)
    ev = [("2020-03-16", "first public health order", 158), ("2020-03-23", "venues close", 131),
          ("2020-03-31", "stay-at-home order", 104)]
    fig, ax = plt.subplots(2, 1, figsize=(TW, 3.3), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [1.05, 1.7]})
    for d, lab, y in ev:
        ax[0].text(pd.Timestamp(d) + pd.Timedelta(hours=14), y, lab, color=INK, va="top", ha="left", zorder=6,
                   bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax[0].plot(ratio.index, ratio.values, color=INK, lw=1.3, marker=".", ms=4)
    for d, _, _ in ev:
        for a in ax:
            S.boundary(a, pd.Timestamp(d), lw=1.3, alpha=1, zorder=5)
    ax[0].set_ylim(0, 160)
    ax[0].set_yticks([0, 50, 100])
    ax[0].set_ylabel("tap-ons, % of a\nFebruary 2020 day")
    S.ygrid(ax[0])
    for k, (m, d) in enumerate(pick):
        t = first[(m, d)]
        te = None if d == "EWMA" else tau.get((m, d))
        if te is not None and te <= t:
            ax[1].plot([te, t], [k, k], color=GREY, lw=0.8, zorder=1)
            ax[1].plot(te, k, marker="|", ms=8, mew=1.6, color=INK, zorder=2)
        S.alarm_marks(ax[1], [t], [k], ms=7)
        ax[1].text(t + pd.Timedelta(hours=20), k, f"{t:%-d %b}", va="center", ha="left", color=INK, zorder=6,
                   bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    ax[1].set_yticks(range(len(pick)), [N.monitor(m, d) for m, d in pick])
    for tl, md in zip(ax[1].get_yticklabels(), pick):
        if md in F.RECOMMENDED:
            tl.set_color(BLUE)
            tl.set_fontweight("semibold")
    ax[1].set_ylim(len(pick) - 0.5, -0.5)
    ax[1].tick_params(axis="y", length=0)
    ax[1].spines["left"].set_visible(False)
    ax[1].grid(axis="y", color=S.GRID, lw=0.6)
    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax[1].set_xlabel("2020")
    hs = [Line2D([], [], color=RED, marker="v", ls="", ms=7, mec="white", mew=0.5, label="first alarm"),
          Line2D([], [], color=INK, marker="|", ls="", ms=8, mew=1.6, label="its estimated change date")]
    ax[1].legend(handles=hs, loc="upper right", ncol=1, handletextpad=0.3, borderaxespad=0.1, facecolor="white",
                 frameon=True, edgecolor="none", framealpha=1)
    save(fig, "F16_covid")


# ------------------------------------------------------------------------------------------- Fig 7.1
def f_electricity(adj="REGT_p", det="FOCuS", rule="ar1", start="2020-01-01"):
    """Five alarms in almost seven years, four of them at Christmas and none near the lockdowns."""
    E = RES / "electricity"
    a = pd.read_csv(E / "adjusted.csv", index_col=0, parse_dates=True).loc[start:]
    al = pd.read_csv(E / "alarms.csv", parse_dates=["alarm"])
    al = al[(al["adj"] == adj) & (al["det"] == det) & (al["rule"] == rule)]
    x = pd.read_csv(F.EVE / "nsw_demand_daily.csv", parse_dates=["date_nem"]).set_index("date_nem")["mean_mw"] / 1000
    x = x.loc[start:]
    sc = pd.read_csv(F.EVE / "events_electricity_scored_FROZEN.csv", parse_dates=["date"])
    spans = []
    for on, off in [("covid_onset", "covid_easing"), ("delta_onset", "delta_end")]:
        d0 = sc.loc[sc["episode"] == on, "date"].min()
        d1 = sc.loc[sc["episode"] == off, "date"].max()
        if pd.notna(d0) and pd.notna(d1):
            spans.append((d0, d1))
    fig, ax = plt.subplots(2, 1, figsize=(TW, 3.0), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [0.85, 1.15]})
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
    ax[1].set_ylim(np.nanmin(s.values) * 1.05, top * 1.6)
    S.alarm_marks(ax[1], al["alarm"], [top * 1.12] * len(al), ms=7)
    for d in al["alarm"]:
        ax[1].text(d, top * 1.12 + 0.014, f"{d:%-d %b}\n{d:%Y}", ha="center", va="bottom", color=INK, linespacing=1.0)
    ax[1].set_ylabel(N.SURPRISE)
    ax[1].set_yticks([-0.2, -0.1, 0, 0.1])
    # second scale: the same forecast errors as a percentage of the forecast (an error of -0.1 is about 10% below)
    r = ax[1].twinx()
    r.set_ylim(ax[1].get_ylim())
    pc = [-20, -10, 0, 10]
    r.set_yticks([np.log(1 + p / 100) for p in pc], [f"{p:+d}%".replace("-", "−").replace("+0%", "0%") for p in pc])
    r.set_ylabel("demand against\nthe forecast")
    r.spines["right"].set_visible(True)
    r.spines["right"].set_color(GREY)
    r.spines["top"].set_visible(False)
    for q in ax:
        S.ygrid(q)
    F.yearaxis(ax[1])
    ax[1].set_xlim(x.index[0] - pd.Timedelta(days=15), x.index[-1] + pd.Timedelta(days=15))
    al[["alarm"]].to_csv(OUT / "electricity_alarms.csv", index=False)
    save(fig, f"F19_electricity_{adj}_{det}")


# ------------------------------------------------------------------------------------------- Fig 4.1
def f_pipeline():
    """Every method goes through the same four steps each day; only steps 1 and 3 differ between methods."""
    steps = [("1  Forecast", "a normal day,\nfrom earlier\ndays only", True),
             ("2  Forecast error", "observed minus\nforecast (log scale),\ndivided by its\nusual size", False),
             ("3  Detector", "adds up the\nrecent forecast\nerrors", True),
             ("4  Alarm", "if the detector is\nabove its threshold\n(set for a 5% chance\nof a false alarm in\na normal year)", False)]
    W, H = TW - 0.06, 1.84
    n, gap = len(steps), 0.2
    bw = (W - 0.04 - (n - 1) * gap) / n
    bh, y0 = 1.14, 0.33
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    arrow = dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=8, shrinkA=0, shrinkB=0)
    xs = [0.02 + k * (bw + gap) for k in range(n)]
    for k, (x, (head, body, differs)) in enumerate(zip(xs, steps)):
        last = k == n - 1
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#fbeeed" if last else PALE, ec=RED if last else GREY, lw=0.9))
        ax.text(x + 0.07, y0 + bh - 0.07, head, ha="left", va="top", color=INK, fontweight="semibold")
        ax.text(x + 0.07, y0 + bh - 0.28, body, ha="left", va="top", color=INK, linespacing=1.08)
        if differs:
            ax.text(x + bw / 2, y0 + bh + 0.05, "differs by method", ha="center", va="bottom", color=BLUE)
        if k < n - 1:
            ax.annotate("", xy=(x + bw + gap - 0.02, y0 + bh / 2), xytext=(x + bw + 0.02, y0 + bh / 2), arrowprops=arrow)
    # return arrow from 4 to 1
    xa, xb, yr = xs[3] + bw / 2, xs[0] + bw / 2, 0.13
    ax.plot([xa, xa, xb], [y0, yr, yr], color=MUTED, lw=0.9)
    ax.annotate("", xy=(xb, y0), xytext=(xb, yr - 0.004), arrowprops=dict(arrow, color=MUTED))
    ax.text((xa + xb) / 2, yr - 0.03, "after an alarm, relearn for 28 days", ha="center", va="top", color=MUTED)
    save(fig, "F22_pipeline")


# ------------------------------------------------------------------------------------------- Fig 4.3
ALARMS = [(2, "down", "1"), (-4, "down", "2"), (6, "up", "3")]
RULES = [("main rule:\nwithin 7 days either side,\nany direction", (-7, 7), lambda d, w: -7 <= d <= 7),
         ("direction-aware check:\n0 to 7 days after,\nsame direction as the change", (0, 7),
          lambda d, w: 0 <= d <= 7 and w == "down")]


def f_rules():
    """Main rule: any alarm within 7 days either side. Check: only 0 to 7 days after, in the right direction."""
    fig, axes = plt.subplots(3, 1, figsize=(TW, 2.8), sharex=True, gridspec_kw={"height_ratios": [1.0, 1, 1]})
    fig.subplots_adjust(left=0.36, right=0.985, top=0.98, bottom=0.2, hspace=0.12)
    lo, hi = -11.5, 12.5
    ax = axes[0]
    ax.plot([lo, 0, 0, hi], [1, 1, 0.35, 0.35], color=INK, lw=1.6, solid_joinstyle="miter")
    ax.set_ylim(0, 1.6)
    ax.text(0.3, 1.5, "boundary: travel drops", color=GREEN, va="top", ha="left")
    fig.text(0.012, 0.86, "daily tap-ons\n(schematic)", color=INK, va="center", ha="left", linespacing=1.15)
    for ax, (label, (a, b), ok), yl in zip(axes[1:], RULES, [0.565, 0.325]):
        ax.axvspan(a, b, color=GREEN_L, lw=0)
        ax.set_ylim(0, 1)
        fig.text(0.012, yl, label, color=INK, va="center", ha="left", linespacing=1.15)
        for d, way, num in ALARMS:
            counts = ok(d, way)
            ax.plot(d, 0.3, marker="v" if way == "down" else "^", ms=9, mfc=RED if counts else "white", mec=RED,
                    mew=1.3, ls="", zorder=3, clip_on=False)
            ax.text(d + (1.6 if num == "3" else 0), 0.74, f"{num}: {'counts' if counts else 'not counted'}", color=INK if counts else MUTED,
                    va="center", ha="center", bbox=dict(facecolor="none", edgecolor="none", pad=0))
    for ax in axes:
        ax.axvline(0, color=GREEN, lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        for sp in ("left", "top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
    axes[-1].set_xticks([-7, 0, 7])
    axes[-1].set_xlabel("days from the boundary")
    hs = [Line2D([], [], marker="v", ls="", ms=8, mfc=RED, mec=RED, label="alarm: travel moved down"),
          Line2D([], [], marker="^", ls="", ms=8, mfc=RED, mec=RED, label="alarm: travel moved up"),
          Line2D([], [], marker="v", ls="", ms=8, mfc="white", mec=RED, mew=1.3, label="open: not counted")]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.045))
    save(fig, "F23_scoring_rules")


# ------------------------------------------------------------------------------------------- Fig 4.4
def f_inserted_drop(start="2025-09-02", adj="HW", det="CUSUM-r", rule="ar1", drop=20):
    """A change of known size is put into otherwise unchanged data; the monitor is scored on the copy with the
    change against the same days without it. Uses the test bed's own code (scripts/testbed.py)."""
    import testbed as T
    from cpd import calibrate, monitor
    from cpd import detectors as D
    x = opal_daily("train")
    dates = x.index
    xv = x.to_numpy().copy()
    tau = dates.get_loc(pd.Timestamp(start))
    taus, _, _ = T.eligible(x)
    assert tau in set(taus), "start day is not an eligible test-bed start"
    t0, t1 = tau - T.PRE, min(len(xv), tau + T.POST)
    xm = T.insert(xv, tau, "drop", drop, None)
    tabs = calibrate.load(T.TABLE, "surprise")
    out = {}
    for key, xx in (("none", xv), ("drop", xm)):
        r, kind = T.adjust_one(adj, xx, dates)
        res = monitor.run(r, kind, det, rule, tabs, t0, t1, method=adj)
        # the first segment, as monitor.run computes it
        blind = np.zeros(len(r), dtype=np.bool_)
        dl = monitor.delevel(r, False)
        ph = monitor.phi_hat(dl, blind, t0)
        h = tabs[det](ph if rule == "ar1" else 0.0)
        s = monitor.running_scale(dl, blind, t0, t1)
        path = D.stat_path(det, r[t0:t1] / s, ph)
        assert abs(h - res.hs[0]) < 1e-12
        cross = np.flatnonzero(path > h)
        assert ([t0 + int(cross[0])] if len(cross) else []) == list(res.alarms[:1])
        out[key] = dict(r=r[t0:t1], path=path, h=h, phi=ph, alarms=list(res.alarms))
    assert not out["none"]["alarms"], "the unmodified copy alarms"
    assert len(out["drop"]["alarms"]) >= 1 and out["drop"]["alarms"][0] >= tau
    ta = out["drop"]["alarms"][0]
    pairs = pd.read_csv(RES / "testbed" / "pairs.csv")
    q = pairs[(pairs.tau == tau) & (pairs.adj == adj) & (pairs.det == det) & (pairs.rule == rule)].set_index("setting")
    assert q.loc[f"drop{drop}", "first_post"] == ta - tau and q.loc["none0", "n_post"] == 0 and q.loc["none0", "pre_alarms"] == 0
    d = dates[t0:t1]
    facts = dict(start_date=start, start_weekday=dates[tau].day_name(), drop_pct=drop, forecast=adj, detector=det,
                 rule=rule, window_from=str(d[0].date()), window_to=str(d[-1].date()), days_before=T.PRE,
                 days_after=T.POST, alarm_date=str(dates[ta].date()), days_after_start=int(ta - tau),
                 delay_days=int(ta - tau + 1), threshold=round(float(out["drop"]["h"]), 2),
                 phi_hat=round(float(out["drop"]["phi"]), 3), alarms_unmodified=len(out["none"]["alarms"]),
                 max_stat_unmodified=round(float(np.nanmax(out["none"]["path"])), 2),
                 stat_at_alarm=round(float(out["drop"]["path"][ta - t0]), 2), screened=bool(q["screened"].iloc[0]))
    pd.DataFrame([facts]).to_csv(OUT / "inserted_drop.csv", index=False)
    print("  inserted drop:", facts)

    fig, ax = plt.subplots(3, 1, figsize=(TW, 3.6), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [1.15, 0.9, 1.0]})
    stop = ta - t0 + 1                               # the copy with the drop is monitored until its alarm
    for a in ax:
        a.axvline(dates[tau], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
        S.ygrid(a)
    k = 1e3
    ax[0].plot(d, xv[t0:t1] / k, color=GREY, lw=1.1, label="the same days, nothing inserted")
    ax[0].plot(d[tau - t0:], xm[tau:t1] / k, color=INK, lw=1.1, label=f"with a lasting {drop}% drop")
    ax[0].set_ylabel("tap-ons a day\n(thousands)")
    ax[0].set_ylim(0, None)
    ax[0].text(dates[tau] + pd.Timedelta(days=1.5), 0.04, f"{drop}% drop inserted from this day", color=INK,
               transform=ax[0].get_xaxis_transform(), va="bottom", ha="left",
               bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax[0].legend(loc="lower left", ncol=1, borderaxespad=0.2, labelspacing=0.2, facecolor="white", frameon=True,
                 edgecolor="none", framealpha=0.9)
    ax[1].axhline(0, color=MUTED, lw=0.6, ls=":")
    ax[1].plot(d, out["none"]["r"], color=GREY, lw=1.1)
    ax[1].plot(d[tau - t0:], out["drop"]["r"][tau - t0:], color=INK, lw=1.1)
    ax[1].set_ylabel("Holt-Winters log\nforecast error")
    ax[2].plot(d, out["none"]["path"], color=GREY, lw=1.1)
    ax[2].plot(d[tau - t0:stop], out["drop"]["path"][tau - t0:stop], color=INK, lw=1.3)
    h = out["drop"]["h"]
    ax[2].axhline(h, color=RED, lw=0.9, ls=(0, (4, 2)))
    ax[2].text(d[2], h * 1.04, "threshold", color=RED, va="bottom", ha="left")
    ymax = max(h * 1.42, float(out["drop"]["path"][ta - t0]) * 1.3)
    ax[2].set_ylim(0, ymax)
    S.alarm_marks(ax[2], [dates[ta]], [out["drop"]["path"][ta - t0] + 0.085 * ymax], ms=7)
    ax[2].text(dates[ta] + pd.Timedelta(days=2), out["drop"]["path"][ta - t0] + 0.085 * ymax,
               f"alarm, {dates[ta]:%-d %b}", color=INK, va="center", ha="left")
    ax[2].set_ylabel("capped CUSUM\nstatistic")
    # bracket: the 60 days in which an alarm counts
    yb = -0.34
    tr = ax[2].get_xaxis_transform()
    xa, xb = dates[tau], d[-1]
    ax[2].plot([xa, xa, xb, xb], [yb + 0.05, yb, yb, yb + 0.05], color=MUTED, lw=0.8, transform=tr, clip_on=False)
    ax[2].text(xa + (xb - xa) / 2, yb - 0.04, "an alarm within 60 days counts as detected", color=MUTED,
               transform=tr, ha="center", va="top")
    ax[2].xaxis.set_major_locator(mdates.MonthLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax[2].set_xlim(d[0] - pd.Timedelta(days=1), d[-1] + pd.Timedelta(days=1))
    fig.get_layout_engine().set(h_pad=0.03)
    save(fig, "F24_inserted_drop")


ALL = {"F02": f_opal_series, "F04": f_adjust_covid, "F13": f_opal_grid, "F14": f_timeline, "F16": f_covid,
       "F19": f_electricity, "F21": f_tradeoff, "F22": f_pipeline, "F23": f_rules, "F24": f_inserted_drop}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(ALL)):
        ALL[n]()
        print("ok", n)
    if SIZES:
        pd.DataFrame([dict(figure=k, width_textwidth=round(v[0], 2), height_in=round(v[1], 2), smallest_pt=round(v[2], 1))
                      for k, v in SIZES.items()]).to_csv(OUT / ("sizes.csv" if len(sys.argv) == 1 else "sizes_last.csv"),
                                                         index=False)
