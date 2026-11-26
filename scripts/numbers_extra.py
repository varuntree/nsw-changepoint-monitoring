"""Numbers the thesis quotes that are derived from result files rather than read straight from one cell.
Every value is recomputed here from results/ (and the frozen event list), so that each number in the
text traces to a CSV written by a script of this repository. Run after `make results`.

Outputs (results/extra/):
  generator_effects.csv    weekday and calendar effects of the simulation generator as percentages
                           (Saturday/Sunday against Monday, public holiday, Christmas, school holiday)
  surprise_spread.csv      robust SD (1.4826 x MAD) and lag-one correlation of the past-only surprises,
                           Opal and electricity, 1 January 2020 onwards
  holiday_sd.csv           a public holiday's log effect in noise SD: simulation sigma against the real
                           robust SDs of the Opal surprises
  opal_level.csv           observed days, median tap-ons, and the COVID-19 fall (April 2020 median and the
                           lowest 28-day median against the February 2020 median)
  elec_testbed_summary.csv electricity test bed: detection within 60 days with and without the inserted
                           drop, and the gain, by drop size, adjustment and detector
  elec_covid.csv           electricity: median surprise 23 March to 30 April 2020 against the robust SD
  echo_alarms.csv          A + CUSUM (AR(1)): each alarm, with the lowest A surprise 6 to 8 days before
                           and whether it is an upward echo of a fall of more than 14%
  extra_catches.csv        A + CUSUM: episodes caught under the pre-registered rule but not the
                           direction-aware rule, with the alarm that catches them
  caught_depth.csv         A + CUSUM: depth below the same weekday a week earlier on the first two days of
                           each boundary of the episodes it caught (direction-aware rule)
  relearn_ratio.csv        alarms with a 14-day against a 28-day relearn (Table 6.x monitors)
  quoted.csv               one row per number quoted in the thesis: value, the text, and whether they agree
                           (includes the corrected numbers, rows q2(...), checked
                           against the corrected wording; run shares_extra.py first)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))
R = HERE / "results"
OUT = R / "extra"
OUT.mkdir(exist_ok=True)
FROZEN = HERE / "data" / "opal" / "events_scored_FROZEN.csv"
START = "2020-01-01"
quoted = []


def q(id_, where, text, value, shown, ok, file, column):
    """One quoted number: `shown` is the value rounded as the text rounds it; `ok` whether the text holds."""
    quoted.append(dict(id=id_, where=where, text=text, value=value, computed_as_text=shown,
                       match=bool(ok), file=f"extra/{file}", column=column))


def rmad(x):
    x = pd.Series(x).dropna()
    return 1.4826 * float(np.median(np.abs(x - x.median())))


def lag1(x):
    x = pd.Series(x)
    return float(x.corr(x.shift(1)))


def pct(logv):
    return 100 * (np.exp(logv) - 1)


# ------------------------------------------------------------------ generator effects (Ch 1, 3, 4, 5, 8)
g = json.loads((R / "sim" / "generator.json").read_text())
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ge = [dict(effect=f"weekday_{d}", log_effect=w, pct=pct(w)) for d, w in zip(days, g["weekday"])]
ge += [dict(effect=k, log_effect=g[k], pct=pct(g[k])) for k in ("pub", "xmas", "school", "metro")]
ge += [dict(effect="sigma_robust", log_effect=g["sigma"], pct=np.nan),
       dict(effect="sd_raw", log_effect=g["sd_raw"], pct=np.nan)]
ge = pd.DataFrame(ge)
ge.to_csv(OUT / "generator_effects.csv", index=False)
gp = ge.set_index("effect").pct
sat, sun = gp["weekday_Sat"], gp["weekday_Sun"]
q("sat_vs_mon", "ch1 §1.2", "a Saturday about a third below a Monday", sat, f"{-sat:.1f}%", 30 <= -sat <= 36.7,
  "generator_effects.csv", "pct (effect=weekday_Sat)")
q("sat_vs_mon", "ch3 §3.1", "a Saturday is about 33% below a Monday", sat, f"{-sat:.0f}%", round(-sat) == 33,
  "generator_effects.csv", "pct (effect=weekday_Sat)")
q("sun_vs_mon", "ch3 §3.1", "a Sunday about 48% below", sun, f"{-sun:.0f}%", round(-sun) == 48,
  "generator_effects.csv", "pct (effect=weekday_Sun)")
for k, t, txt in [("pub", 42, "public holidays -42%; ch5/ch8 'about 42%'"), ("xmas", 25, "Christmas -25%"),
                  ("school", 7, "school holidays -7%")]:
    q(f"{k}_pct", "ch4 §4.7, ch5 §5.1", txt, gp[k], f"{-gp[k]:.0f}%", round(-gp[k]) == t, "generator_effects.csv",
      f"pct (effect={k})")
q("sigma_robust", "ch8 §8.3", "robust SD of the fitted residuals 0.049", g["sigma"], f"{g['sigma']:.3f}",
  f"{g['sigma']:.3f}" == "0.049", "generator_effects.csv", "log_effect (effect=sigma_robust)")
q("sd_raw", "ch8 §8.3", "against an ordinary one of 0.097", g["sd_raw"], f"{g['sd_raw']:.3f}",
  f"{g['sd_raw']:.3f}" == "0.097", "generator_effects.csv", "log_effect (effect=sd_raw)")

# ------------------------------------------------------------------ surprise spread and dependence (Ch 7)
op = pd.read_csv(R / "opal" / "adjusted.csv", parse_dates=["date"]).set_index("date")
el = pd.read_csv(R / "electricity" / "adjusted.csv", parse_dates=["date_nem"]).set_index("date_nem")
sp = []
for stream, df, adjs in [("opal", op, ["A", "B", "HW", "YM", "MSTL_p", "REG_p"]),
                         ("electricity", el, ["A", "B", "HW", "YM", "REG_p", "REGT_p"])]:
    x = df[START:]
    for a in adjs:
        sp.append(dict(stream=stream, adj=a, start=START, n=int(x[a].notna().sum()), robust_sd=rmad(x[a]),
                       lag1=lag1(x[a])))
sp = pd.DataFrame(sp)
sp.to_csv(OUT / "surprise_spread.csv", index=False)
S = sp.set_index(["stream", "adj"])
e3 = S.loc["electricity"].loc[["A", "B", "HW"]]
o3 = S.loc["opal"].loc[["A", "B", "HW"]]
q("elec_sd_ABHW", "ch7 §7.2", "about 8% of daily demand for A, B and Holt-Winters", e3.robust_sd.mean(),
  f"{100 * e3.robust_sd.min():.1f}-{100 * e3.robust_sd.max():.1f}%", round(100 * e3.robust_sd.mean()) == 8,
  "surprise_spread.csv", "robust_sd (stream=electricity, adj=A/B/HW)")
q("elec_vs_opal_sd", "ch7 §7.2", "about as noisy as Opal's", o3.robust_sd.mean(),
  f"Opal {100 * o3.robust_sd.min():.1f}-{100 * o3.robust_sd.max():.1f}%",
  e3.robust_sd.max() <= o3.robust_sd.max(), "surprise_spread.csv", "robust_sd (stream=opal, adj=A/B/HW)")
q("elec_lag1_ABHW", "ch7 §7.2", "lag-one correlation of about 0.7", e3.lag1.mean(),
  f"{e3.lag1.min():.2f}-{e3.lag1.max():.2f}", round(e3.lag1.mean(), 1) == 0.7, "surprise_spread.csv",
  "lag1 (stream=electricity, adj=A/B/HW)")
q("elec_vs_opal_lag1", "ch7 §7.2", "at least as dependent (as Opal)", o3.lag1.mean(),
  f"Opal {o3.lag1.min():.2f}-{o3.lag1.max():.2f}", e3.lag1.min() >= o3.lag1.min(), "surprise_spread.csv",
  "lag1 (stream=opal, adj=A/B/HW)")
for a, t_sd, t_l in [("REG_p", "6.0", "0.71"), ("REGT_p", "4.0", "0.59")]:
    r = S.loc[("electricity", a)]
    q(f"elec_{a}_sd", "ch7 §7.2", f"robust SD {t_sd}% ({a})", r.robust_sd, f"{100 * r.robust_sd:.1f}%",
      f"{100 * r.robust_sd:.1f}" == t_sd, "surprise_spread.csv", f"robust_sd (stream=electricity, adj={a})")
    q(f"elec_{a}_lag1", "ch7 §7.2", f"lag-one {t_l} ({a})", r.lag1, f"{r.lag1:.2f}", f"{r.lag1:.2f}" == t_l,
      "surprise_spread.csv", f"lag1 (stream=electricity, adj={a})")

# ------------------------------------------------------------------ holiday size in noise SD (Ch 5, Ch 8)
hs = [dict(source="simulation", adj="generator", sd=g["sigma"], pub_log=g["pub"], pub_in_sd=-g["pub"] / g["sigma"])]
hs += [dict(source="opal_real", adj=a, sd=S.loc[("opal", a)].robust_sd, pub_log=g["pub"],
            pub_in_sd=-g["pub"] / S.loc[("opal", a)].robust_sd) for a in ("A", "B", "HW")]
hs = pd.DataFrame(hs)
hs.to_csv(OUT / "holiday_sd.csv", index=False)
sim_sd = hs.pub_in_sd.iloc[0]
real = hs[hs.source == "opal_real"].pub_in_sd
q("pub_sd_sim", "ch5 §5.1, ch8 §8.3", "a public holiday is about 11 SD", sim_sd, f"{sim_sd:.1f}", round(sim_sd) == 11,
  "holiday_sd.csv", "pub_in_sd (source=simulation)")
q("pub_sd_real", "ch8 §8.3", "against about 5 to 7 on real surprises", real.mean(), f"{real.min():.1f}-{real.max():.1f}",
  round(real.min()) == 5 and round(real.max()) == 7, "holiday_sd.csv", "pub_in_sd (source=opal_real)")

# ------------------------------------------------------------------ Opal level and the COVID fall (Ch 3)
raw = np.exp(op["raw"])
feb, apr = raw["2020-02"].median(), raw["2020-04"].median()
m28 = raw.rolling(28, min_periods=20).median()["2020-03":"2020-12"]
lvl = pd.DataFrame([dict(observed_days=int(raw.notna().sum()), median_taps=raw.median(), feb2020_median=feb,
                         apr2020_median=apr, drop_apr_vs_feb=1 - apr / feb,
                         min_28d_median_2020=m28.min(), min_28d_median_date=m28.idxmin().date(),
                         drop_min28_vs_feb=1 - m28.min() / feb)])
lvl.to_csv(OUT / "opal_level.csv", index=False)
L = lvl.iloc[0]
q("covid_drop", "ch3 Fig 3.1 caption", "COVID-19 cut travel by about four fifths", L.drop_apr_vs_feb,
  f"{100 * L.drop_apr_vs_feb:.1f}% (28-day min {100 * L.drop_min28_vs_feb:.1f}%)", 0.75 <= L.drop_apr_vs_feb <= 0.85,
  "opal_level.csv", "drop_apr_vs_feb, drop_min28_vs_feb")
q("observed_days", "ch3 §3.1", "2,459 observed days", L.observed_days, f"{L.observed_days:,}", L.observed_days == 2459,
  "opal_level.csv", "observed_days")
q("median_taps", "ch3 §3.1", "median of 730,300 train tap-ons", L.median_taps, f"{round(L.median_taps, -2):,.0f}",
  round(L.median_taps, -2) == 730300, "opal_level.csv", "median_taps")

# ------------------------------------------------------------------ electricity test bed (Ch 7, Ch 8)
tb = pd.read_csv(R / "electricity" / "testbed.csv")
tb["det60"] = (tb.first_post >= 0) & (tb.first_post < 60)
p = tb.pivot_table(index=["size", "adj", "det"], columns="which", values="det60", aggfunc="mean")
n = tb.groupby(["size", "adj", "det"]).tau.nunique().rename("n_start_days")
ts = pd.concat([n, p.rename(columns={"drop": "detect_drop", "none": "background"})], axis=1).reset_index()
ts["gain"] = ts.detect_drop - ts.background
ts.to_csv(OUT / "elec_testbed_summary.csv", index=False)
T = ts.set_index(["size", "adj", "det"])
UNCH = ["A", "B", "HW", "YM", "REG_p"]   # the Opal procedure unchanged (no temperature)
u10 = ts[(ts["size"] == 10) & ts.adj.isin(UNCH)]
u5 = ts[(ts["size"] == 5) & ts.adj.isin(UNCH)]
f2 = lambda v: f"{v:.2f}"  # noqa: E731
q("elec_n_start", "ch7 (method)", "175 start days", int(ts.n_start_days.min()), str(int(ts.n_start_days.min())),
  ts.n_start_days.min() == ts.n_start_days.max(), "elec_testbed_summary.csv", "n_start_days")
q("elec_unch10", "ch7 §7.3", "10% drop, unchanged procedure: gain 0.02 to 0.21", u10.gain.max(),
  f"{f2(u10.gain.min())}-{f2(u10.gain.max())}", f2(u10.gain.min()) == "0.02" and f2(u10.gain.max()) == "0.21",
  "elec_testbed_summary.csv", "gain (size=10, adj in A,B,HW,YM,REG_p, all det)")
for (a, d, t, txt) in [("YM", "FOCuS", "0.16", "yearly memory with FOCuS, gain 0.16"),
                       ("REGT_p", "CUSUM", "0.51", "with temperature, CUSUM 0.51"),
                       ("REGT_p", "FOCuS", "0.55", "with temperature, FOCuS 0.55"),
                       ("REGT_p", "FOCuS-r", "0.50", "capped FOCuS gained 0.50 for a 10% drop")]:
    v = T.loc[(10, a, d)].gain
    q(f"elec_{a}_{d}_10", "ch7 §7.3", txt, v, f2(v), f2(v) == t,
      "elec_testbed_summary.csv", f"gain (size=10, adj={a}, det={d})")
bg = T.loc[(10, "REGT_p")].loc[["CUSUM", "FOCuS"]].background
q("elec_REGT_bg", "ch7 §7.3", "background rate stayed at 12 to 13%", bg.mean(),
  f"{100 * bg.min():.0f}-{100 * bg.max():.0f}%", round(100 * bg.min()) == 12 and round(100 * bg.max()) == 13,
  "elec_testbed_summary.csv", "background (adj=REGT_p, det=CUSUM/FOCuS)")
r5 = T.loc[(5, "REGT_p")].loc[["CUSUM", "FOCuS"]].gain
q("elec_5pct", "ch7 §7.3", "5% drop: gain rose from at most 0.13 to 0.17", r5.max(),
  f"{f2(u5.gain.max())} -> {f2(r5.max())}", f2(u5.gain.max()) == "0.13" and f2(r5.max()) == "0.17",
  "elec_testbed_summary.csv", "gain (size=5; unchanged adjs max; REGT_p CUSUM/FOCuS)")

# electricity COVID fall against the surprise spread (Ch 7 'smaller than one robust SD')
ec = []
for a in ["A", "B", "HW", "REG_p", "REGT_p"]:
    med = float(el[a]["2020-03-23":"2020-04-30"].median())
    ec.append(dict(adj=a, median_surprise_23mar_30apr2020=med, robust_sd=S.loc[("electricity", a)].robust_sd,
                   ratio=abs(med) / S.loc[("electricity", a)].robust_sd))
ec = pd.DataFrame(ec)
ec.to_csv(OUT / "elec_covid.csv", index=False)
q("elec_covid_lt_sd", "ch7 §7.3", "the COVID-19 fall is smaller than one robust SD of the surprises", ec.ratio.max(),
  f"|median|/SD {ec.ratio.min():.2f}-{ec.ratio.max():.2f}", (ec.ratio < 1).all(), "elec_covid.csv", "ratio")

# ------------------------------------------------------------------ A + CUSUM echoes and depth (Ch 6)
al = pd.read_csv(R / "opal" / "alarms.csv", parse_dates=["alarm"])
ac = al[(al.method == "A") & (al.detector == "CUSUM") & (al.rule == "ar1")].sort_values("alarm")
FALL = np.log(1 - 0.14)
ech = []
for _, r in ac.iterrows():
    w = op.A[r.alarm - pd.Timedelta(days=8):r.alarm - pd.Timedelta(days=6)]
    lo = float(w.min()) if len(w.dropna()) else np.nan
    ech.append(dict(alarm=r.alarm.date(), direction=int(r.direction), min_A_6to8_before=lo,
                    min_pct_6to8_before=pct(lo), echo=bool(r.direction == 1 and lo < FALL)))
ech = pd.DataFrame(ech)
ech.to_csv(OUT / "echo_alarms.csv", index=False)
q("echo_count", "ch6 §6.1.2", "Eight of the 36 alarms of A with CUSUM are upward echoes", int(ech.echo.sum()),
  f"{int(ech.echo.sum())} of {len(ech)}", ech.echo.sum() == 8 and len(ech) == 36, "echo_alarms.csv", "echo (sum)")

ev = pd.read_csv(FROZEN, parse_dates=["date"])
rv = pd.read_csv(R / "opal" / "rule_variants.csv")
ra = rv[(rv.method == "A") & (rv.detector == "CUSUM") & (rv.rule == "ar1")].iloc[0]
head = set(ra.episodes_headline.split(";"))
orig = set(ra.episodes_pm7_nodir.split(";"))      # +-7 days, nearest-boundary crediting (column renamed 24 Oct)
ex = []
for e in sorted(orig - head):
    for _, b in ev[ev.episode == e].iterrows():
        near = ac[(ac.alarm >= b.date - pd.Timedelta(days=7)) & (ac.alarm <= b.date + pd.Timedelta(days=7))]
        for _, r in near.iterrows():
            off = (r.alarm - b.date).days
            hit = ech[ech.alarm == r.alarm.date()].iloc[0]
            kind = "echo" if hit.echo else ("before" if off < 0 else "after, wrong direction")
            ex.append(dict(episode=e, boundary=b.date.date(), alarm=r.alarm.date(), offset_days=off,
                           direction=int(r.direction), kind=kind))
ex = pd.DataFrame(ex)
ex.to_csv(OUT / "extra_catches.csv", index=False)
per = ex.groupby("episode").kind.agg(lambda k: "echo" if "echo" in set(k) else ("before" if "before" in set(k) else k.iloc[0]))
bef = ex[ex.kind == "before"].groupby("episode").offset_days.max()   # nearest alarm before each episode
q("extra_catches", "ch6 §6.1.2", "six extra catches: two echoes in 2025, four 2-4 days before a dated order",
  len(per), f"{len(per)} extra; echoes {sorted(per[per == 'echo'].index)}; before {int((per == 'before').sum())} "
  f"({-bef.max()}-{-bef.min()} days)",
  len(per) == 6 and sorted(per[per == "echo"].index) == ["strike_2025_01", "strike_2025_02"] and (per == "before").sum() == 4,
  "extra_catches.csv", "kind by episode")

dp = []
for e in sorted(head):
    for _, b in ev[ev.episode == e].iterrows():
        two = op.A[b.date:b.date + pd.Timedelta(days=1)]
        hit = ac[(ac.alarm >= b.date) & (ac.alarm <= b.date + pd.Timedelta(days=7)) & (ac.direction == -1)]
        dp.append(dict(episode=e, boundary=b.date.date(), announced=b.announced_before, day0_pct=pct(two.iloc[0]),
                       day1_pct=pct(two.iloc[1]), worst_pct=pct(two.min()), caught_here=len(hit) > 0))
dp = pd.DataFrame(dp)
dp.to_csv(OUT / "caught_depth.csv", index=False)
un = dp[dp.announced == "no"]
q("caught_depth", "ch6 §6.4", "travel was 31 to 45% below the same weekday a week earlier", un.worst_pct.min(),
  f"unannounced catches {-un.worst_pct.max():.0f}-{-un.worst_pct.min():.0f}%",
  round(-un.worst_pct.max()) == 31 and round(-un.worst_pct.min()) == 45, "caught_depth.csv",
  "worst_pct (announced=no)")
ch = dp[dp.caught_here]
q("caught_depth_all", "ch6 §6.4", "same sentence, all five caught episodes at the boundary caught", ch.worst_pct.min(),
  f"all catches {-ch.worst_pct.max():.0f}-{-ch.worst_pct.min():.0f}%",
  round(-ch.worst_pct.max()) == 31 and round(-ch.worst_pct.min()) == 45, "caught_depth.csv",
  "worst_pct (caught_here)")

# ------------------------------------------------------------------ relearn: alarms with 14 against 28 days (Ch 6)
se = pd.read_csv(R / "opal" / "sensitivity.csv")
sc = pd.read_csv(R / "opal" / "scores.csv")
rr = []
for m, d in [("HW", "FOCuS"), ("HW", "EWMA"), ("A", "CUSUM")]:
    a14 = int(se[(se.method == m) & (se.detector == d) & (se.variant == "relearn=14")].n_alarms.iloc[0])
    a28 = int(sc[(sc.method == m) & (sc.detector == d) & (sc.rule == "ar1")].n_alarms.iloc[0])
    rr.append(dict(method=m, detector=d, alarms_28=a28, alarms_14=a14, ratio=a14 / a28))
rr = pd.DataFrame(rr)
tot = rr.alarms_14.sum() / rr.alarms_28.sum()
rr = pd.concat([rr, pd.DataFrame([dict(method="all three", detector="", alarms_28=rr.alarms_28.sum(),
                                       alarms_14=rr.alarms_14.sum(), ratio=tot)])], ignore_index=True)
rr.to_csv(OUT / "relearn_ratio.csv", index=False)
q("relearn_more_alarms", "ch6 §6.7", "a 14-day relearn raised about 40% more alarms", tot - 1,
  f"+{100 * (tot - 1):.0f}% (per monitor {', '.join(f'+{100 * (x - 1):.0f}%' for x in rr.ratio[:3])})",
  round(10 * (tot - 1)) == 4, "relearn_ratio.csv", "ratio (method=all three)")

# ------------------------------------------------------------------ corrected numbers
def q2(id_, where, text, value, shown, ok, file, column):
    quoted.append(dict(id=id_, where=where, text=text, value=value, computed_as_text=shown, match=bool(ok),
                       file=file, column=column))


fa = pd.read_csv(R / "sim" / "summary" / "fa.csv")
dt = pd.read_csv(R / "sim" / "summary" / "detect.csv")
dl = pd.read_csv(R / "sim" / "summary" / "delay.csv")
va = pd.read_csv(R / "sim" / "summary" / "variance.csv")
rv = pd.read_csv(R / "opal" / "rule_variants.csv")
ho = pd.read_csv(R / "opal" / "heldout.csv")
cv = pd.read_csv(R / "opal" / "covid.csv")
vd = pd.read_csv(R / "calib" / "validation.csv")
bo = pd.read_csv(R / "opal" / "bootstrap.csv")
bh = pd.read_csv(R / "opal" / "bootstrap_heldout.csv")
# the original calm-stretch set (A, B, Holt-Winters x six detectors); later rows extend it for opal_budget.py
ORIG_BOOT = ["CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r", "FOCuS-r"]
bo = bo[bo.method.isin(["A", "B", "HW"]) & bo.detector.isin(ORIG_BOOT)]
bh = bh[bh.method.isin(["A", "B", "HW"]) & bh.detector.isin(ORIG_BOOT)]
sx = pd.read_csv(R / "extra" / "shares_detsets.csv")
past = ["A", "B", "HW", "YM", "MSTL_p", "REG_p"]
f4 = fa[(fa.phi == 0.4) & (fa.noise == "normal")]
ym1 = f4[(f4.calendar == 1) & (f4.adj == "YM") & (f4.rule == "ar1")].set_index("det").fa
q2("M1_capped_ym", "ch1 l.53", "capped CUSUM 2.0%, capped FOCuS 8.1%, capped EWMA 57%", ym1["CUSUM-r"],
   f"{100*ym1['CUSUM-r']:.1f} / {100*ym1['FOCuS-r']:.1f} / {100*ym1['EWMA-r']:.0f}%",
   round(100 * ym1["CUSUM-r"], 1) == 2.0 and round(100 * ym1["FOCuS-r"], 1) == 8.1 and round(100 * ym1["EWMA-r"]) == 57,
   "sim/summary/fa.csv", "fa (phi=0.4,normal,calendar=1,adj=YM,rule=ar1)")
# M2: sensitivity shifts
sens = se.merge(sc[sc.rule == "ar1"][["method", "detector", "episodes_caught"]], on=["method", "detector"], suffixes=("", "_base"))
sens["shift"] = (sens.episodes_caught - sens.episodes_caught_base).abs()
t63 = sens[sens.apply(lambda r: (r.method, r.detector) in [("HW", "FOCuS"), ("HW", "EWMA"), ("A", "CUSUM")], axis=1)]
q2("M2_sens_three", "ch6 l.53; tab:opal-sens", "counts move by up to three (three monitors)", t63["shift"].max(),
   f"{t63['shift'].max():.0f}", t63["shift"].max() == 3, "opal/sensitivity.csv", "|episodes - scores.csv ar1| for HW+FOCuS, HW+EWMA, A+CUSUM")
q2("M2_sens_all", "ch6 l.53", "across all sensitivity runs by up to four", sens["shift"].max(),
   f"{sens['shift'].max():.0f}", sens["shift"].max() == 4, "opal/sensitivity.csv", "max |shift| over all rows")
r = rv[(rv.method == "STL") & (rv.detector == "EWMA") & (rv.rule == "indep")].iloc[0]
q2("M3_retro_best_orig", "ch6 l.99", "retrospective STL with EWMA, independent threshold, 10 (original rule)", r.pm7_nodir,
   f"{r.pm7_nodir:.0f}", r.pm7_nodir == 10 and r.pm7_nodir == rv[rv.method.isin(["STL", "MSTL", "REG"])].pm7_nodir.max(),
   "opal/rule_variants.csv", "pm7_nodir (STL, EWMA, indep; max of retrospective)")
h = sx[(sx.phi == 0.4) & (sx.noise == "normal") & (sx.calendar == 1) & (sx.detset == "CEFP") & (sx.outcome == "detect60_logit_matched_drops")].iloc[0]
q2("M4_holiday_bars", "Fig 5.3, §5.4", "holiday detection: forecast 43, detector 41, interaction 16 (%)", h.adj,
   f"{h.adj:.0f} / {h.det:.0f} / {h.adjxdet:.0f}", (round(h.adj), round(h.det), round(h.adjxdet)) == (43, 41, 16),
   "extra/shares_detsets.csv", "adj/det/adjxdet (phi 0.4, normal, calendar 1, CEFP, drops)")
d4 = dl[(dl.rule == "oracle") & (dl.setting == "true pattern, no holidays") & (dl.shift_sd == 4)].set_index("det")
q2("M5_fastest", "ch5 l.32, tab:delay", "RMD 2.6 (FOCuS-AR), 3.3 (FOCuS), 4.4 (EWMA) at 4 SD", d4.rmd["FOCuS-AR"],
   f"{d4.rmd['FOCuS-AR']:.1f} / {d4.rmd['FOCuS']:.1f} / {d4.rmd['EWMA']:.1f}",
   (round(d4.rmd["FOCuS-AR"], 1), round(d4.rmd["FOCuS"], 1), round(d4.rmd["EWMA"], 1)) == (2.6, 3.3, 4.4), "sim/summary/delay.csv", "rmd")
q2("M6_capping_speed", "ch5 l.47", "within a week: capped CUSUM 38% vs 93%; capped FOCuS 99%; capped EWMA 90% vs 98%", d4.F7["CUSUM-r"],
   f"{100*d4.F7['CUSUM-r']:.0f} / {100*d4.F7['CUSUM']:.0f} / {100*d4.F7['FOCuS-r']:.0f} / {100*d4.F7['EWMA-r']:.0f} / {100*d4.F7['EWMA']:.0f}",
   [round(100 * d4.F7[k]) for k in ("CUSUM-r", "CUSUM", "FOCuS-r", "EWMA-r", "EWMA")] == [38, 93, 99, 90, 98], "sim/summary/delay.csv", "F7")
hi = ho[(ho.rule == "indep")].set_index(["method", "detector"])
q2("M8_heldout_indep", "ch6 l.53, ch8 l.15", "median of four with EWMA, independent threshold: all three held-out (p<0.001), 6.3 non-event/yr",
   hi.loc[("B", "EWMA")].episodes_caught, f"{hi.loc[('B','EWMA')].episodes_caught:.0f}, p={hi.loc[('B','EWMA')].p_episodes:.3f}, {hi.loc[('B','EWMA')].non_event_per_year:.1f}",
   hi.loc[("B", "EWMA")].episodes_caught == 3 and hi.loc[("B", "EWMA")].p_episodes < 0.001 and round(hi.loc[("B", "EWMA")].non_event_per_year, 1) == 6.3,
   "opal/heldout.csv", "rule=indep, B, EWMA")
ha = ho[(ho.rule == "ar1") & ~ho.method.isin(["MSTL", "MSTL_p"])]
q2("M8_heldout_ar1", "ch6 l.53", "no AR(1) monitor outside MSTL caught more than one (smallest p = 0.08)", ha.episodes_caught.max(),
   f"max {ha.episodes_caught.max():.0f}, min p {ha.p_episodes.min():.3f}", ha.episodes_caught.max() == 1 and round(ha.p_episodes.min(), 2) == 0.08,
   "opal/heldout.csv", "rule=ar1, not MSTL/MSTL_p")
dy = dt[(dt.phi == 0.4) & (dt.noise == "normal") & (dt.calendar == 1) & (dt.rule == "oracle") & (dt.change == "mean") & (dt["size"] == -4.0) & (dt.adj == "YM")].set_index("det").detect60
q2("M12_ym_power", "ch5 l.41", "yearly memory, 4 SD: capped CUSUM 99, capped FOCuS 97, CPM 88, CUSUM 61 (%)", dy["CUSUM-r"],
   " / ".join(f"{100*dy[k]:.0f}" for k in ("CUSUM-r", "FOCuS-r", "CPM", "CUSUM")),
   [round(100 * dy[k]) for k in ("CUSUM-r", "FOCuS-r", "CPM", "CUSUM")] == [99, 97, 88, 61], "sim/summary/detect.csv", "detect60 (YM, cal 1, -4, oracle)")
vv = vd[(vd.kind == "surprise") & (vd.phi == 0.4) & (vd.rule == "indep")].set_index("det").fa
mean4 = vv[["CUSUM", "EWMA", "FOCuS", "CPM"]]
q2("M13_indep_validation", "ch5 l.11", "independent thresholds at phi=0.4: mean detectors 60-78%, FOCuS-AR 5.3%, variance detectors 10-13%",
   mean4.min(), f"{100*mean4.min():.0f}-{100*mean4.max():.0f} / {100*vv['FOCuS-AR']:.1f} / {100*vv[['CUSUM-var','EWMA-var','FOCuS-var']].min():.0f}-{100*vv[['CUSUM-var','EWMA-var','FOCuS-var']].max():.0f}",
   round(100 * vv["FOCuS-AR"], 1) == 5.3, "calib/validation.csv", "fa (surprise, phi 0.4, indep)")
cp = cv[(cv.detector == "CPM") & cv.method.isin(["A", "B", "HW", "YM"])]
q2("M14_cpm_covid", "ch6 l.81", "CPM alarmed 7 to 16 days after the first order (A, B, HW, YM)", cp.days_from_16mar.min(),
   f"{cp.days_from_16mar.min():.0f}-{cp.days_from_16mar.max():.0f}", (cp.days_from_16mar.min(), cp.days_from_16mar.max()) == (7, 16),
   "opal/covid.csv", "days_from_16mar (CPM, A/B/HW/YM, both rules)")
# the quoted 15-48% after A needs the capped CUSUM and FOCuS in the set (CUSUM, EWMA, FOCuS, CPM alone: 15-40%)
M15D = ["CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r", "FOCuS-r"]
fi = f4[(f4.calendar == 0) & (f4.rule == "indep") & f4.adj.isin(["B", "HW", "YM"]) & f4.det.isin(M15D)]
fa_ = f4[(f4.calendar == 0) & (f4.rule == "indep") & (f4.adj == "A") & f4.det.isin(M15D)]
q2("M15_indep_fa", "ch1 l.13", "independent thresholds: 52 to 100% after most forecasts; after A 15 to 48% (CUSUM, EWMA, FOCuS, CPM, capped CUSUM and FOCuS)", fi.fa.min(),
   f"{100*fi.fa.min():.0f}-{100*fi.fa.max():.0f}; A {100*fa_.fa.min():.0f}-{100*fa_.fa.max():.0f}",
   (round(100 * fi.fa.min()), round(100 * fi.fa.max()), round(100 * fa_.fa.min()), round(100 * fa_.fa.max())) == (52, 100, 15, 48),
   "sim/summary/fa.csv", "fa (phi 0.4, normal, cal 0, indep, B/HW/YM and A; CUSUM/EWMA/FOCuS/CPM/CUSUM-r/FOCuS-r)")
c4 = cv[cv.method.isin(["A", "B", "HW", "YM"]) & cv.detector.isin(["CUSUM", "EWMA", "FOCuS", "CPM", "FOCuS-AR", "CUSUM-r", "EWMA-r", "FOCuS-r"])]
rg = cv[(cv.method == "REG_p") & (cv.detector == "CUSUM") & (cv.rule == "ar1")].iloc[0]
q2("S4d_start_gap", "ch6 l.71", "estimated change dates up to 12 days before the alarm (A, B, HW, YM)", c4.days_alarm_after_start_est.max(),
   f"{c4.days_alarm_after_start_est.max():.0f}; REG_p+CUSUM alarm {rg.first_alarm}, start {rg.start_est}",
   c4.days_alarm_after_start_est.max() == 12, "opal/covid.csv", "days_alarm_after_start_est")
u = sc[(sc.rule == "ar1") & sc.method.isin(past) & sc.detector.isin(["CUSUM", "EWMA", "FOCuS", "FOCuS-AR"])].non_event_per_year
cpmr = sc[(sc.rule == "ar1") & sc.method.isin(past) & (sc.detector == "CPM")].non_event_per_year
q2("S4e_non_event", "ch6 l.19, ch8 l.13", "2.9 to 6.8 (median 5.0); CPM 0.5-1.4", u.min(),
   f"{u.min():.1f}-{u.max():.1f} (median {u.median():.1f}); CPM {cpmr.min():.1f}-{cpmr.max():.1f}",
   (round(u.min(), 1), round(u.max(), 1), round(u.median(), 1), round(cpmr.min(), 1), round(cpmr.max(), 1)) == (2.9, 6.8, 5.0, 0.5, 1.4),
   "opal/scores.csv", "non_event_per_year (ar1, past-only)")
g = f4[(f4.calendar == 0) & (f4.rule == "ar1") & f4.det.isin(["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"])]
gy, ga = g[g.adj == "YM"].fa, g[g.adj == "A"].fa
q2("S4f_ar1_target", "ch8 l.11", "AR(1) threshold: yearly memory 1.7-4.7%, A 0-4.6%", gy.min(),
   f"YM {100*gy.min():.1f}-{100*gy.max():.1f}; A {100*ga.min():.1f}-{100*ga.max():.1f}",
   (round(100 * gy.min(), 1), round(100 * gy.max(), 1), round(100 * ga.max(), 1)) == (1.7, 4.7, 4.6), "sim/summary/fa.csv", "fa (phi 0.4, normal, cal 0, ar1, all 8 mean detectors)")
q2("W11_boot_heldout", "abstract, ch6 l.23", "bootstrap thresholds: at most 0.6 non-event/yr held out, 1 of 3; at most 4 of 15 in 2020-26",
   bh.non_event_per_year.max(), f"{bh.non_event_per_year.max():.2f}; {bh.episodes_caught.max():.0f} of 3; {bo.episodes_caught.max():.0f} of 15",
   round(bh.non_event_per_year.max(), 1) == 0.6 and bh.episodes_caught.max() == 1 and bo.episodes_caught.max() == 4,
   "opal/bootstrap_heldout.csv; opal/bootstrap.csv", "non_event_per_year, episodes_caught")
t5 = dt[(dt.phi == 0.4) & (dt.noise == "t5") & (dt.calendar == 1) & (dt.rule == "oracle") & (dt.adj == "oracle") & (dt["size"] == -1.0) & (dt.det == "FOCuS-AR")].iloc[0]
q2("MF16_focusar_t5", "ch5 §5.5", "FOCuS-AR fell to 1.9%", t5.detect60, f"{100*t5.detect60:.1f}%", round(100 * t5.detect60, 1) == 1.9,
   "sim/summary/detect.csv", "detect60 (t5, cal 1, oracle adj, -1 SD, oracle rule)")
o = fa[(fa.rule == "oracle") & fa.noise.isin(["normal", "t5"]) & ~fa.det.isin(["CUSUM-k0.25", "CUSUM-k1", "EWMA-l0.05", "EWMA-l0.2"])]
inr = ((o.fa >= 0.04) & (o.fa <= 0.06)).mean()
w = o.loc[o.fa.idxmin()]
q2("MF17_split_half", "ch4 l.262", "98.5% of the 462 cells within 4-6%; worst YM + capped EWMA with holidays, 0.9%", inr,
   f"{100*inr:.1f}% of {len(o)}; min {100*w.fa:.2f}% ({w.adj}, {w.det}, cal {w.calendar})",
   round(100 * inr, 1) == 98.5 and len(o) == 462, "sim/summary/fa.csv", "rule=oracle, noise != wide")
r = sc[(sc.method == "YM") & (sc.detector == "CUSUM") & (sc.rule == "indep")].iloc[0]
rr_ = rv[(rv.method == "YM") & (rv.detector == "CUSUM") & (rv.rule == "indep")].iloc[0]
q2("YM_CUSUM_indep_row", "Table 6.1", "44 | 20 | 9 (4) | 9 (1) | 5.3 | 2.6 | 187", r.n_alarms,
   f"{r.n_alarms:.0f} | {100*r.share_real:.0f} (chance {100*r.null_share_real:.0f}) | {rr_.pm7_nodir:.0f} ({rr_.pm7_nodir_null:.0f}) | {rr_.r0_7_dir:.0f} ({rr_.r0_7_dir_null:.0f}) | "
   f"{r.non_event_per_year:.1f} | {r.unexplained_per_year:.1f} | {r.blind_days_per_year:.0f}",
   (r.n_alarms, round(100 * r.share_real), rr_.pm7_nodir, rr_.r0_7_dir, round(r.non_event_per_year, 1), round(r.unexplained_per_year, 1), round(r.blind_days_per_year)) == (44, 20, 9, 9, 5.3, 2.6, 187),
   "opal/scores.csv; opal/rule_variants.csv", "YM, CUSUM, indep")
v5 = va[(va.noise == "t5") & (va.adj == "oracle") & (va.rule == "oracle")].set_index("det").detect60
q2("t5_variance_col", "tab:variance", "x2, t5: var CUSUM 12, var EWMA 6, var FOCuS 12, CUSUM 8, EWMA 7, FOCuS 5, FOCuS-AR 4", v5["CUSUM-var"],
   " / ".join(f"{100*v5[k]:.0f}" for k in ("CUSUM-var", "EWMA-var", "FOCuS-var", "CUSUM", "EWMA", "FOCuS", "FOCuS-AR")),
   [round(100 * v5[k]) for k in ("CUSUM-var", "EWMA-var", "FOCuS-var", "CUSUM", "EWMA", "FOCuS", "FOCuS-AR")] == [12, 6, 12, 8, 7, 5, 4],
   "sim/summary/variance.csv", "detect60 (t5, oracle adj, oracle rule)")
ft5 = fa[(fa.phi == 0.4) & (fa.noise == "t5") & (fa.adj == "oracle") & (fa.rule == "ar1") & fa.det.isin(["CUSUM-var", "EWMA-var", "FOCuS-var"])].fa
q2("t5_variance_fa", "tab:variance caption", "variance detectors' AR(1) false alarms under t5: about 60%", ft5.mean(),
   f"{100*ft5.min():.0f}-{100*ft5.max():.0f}%", 0.55 < ft5.min() and ft5.max() < 0.65, "sim/summary/fa.csv", "fa (t5, oracle adj, ar1, variance detectors)")
fv = fa[(fa.phi == 0.4) & (fa.noise == "normal") & (fa.calendar == 0) & (fa.rule == "ar1") & fa.adj.isin(["A", "B", "HW", "YM"]) & fa.det.isin(["CUSUM-var", "EWMA-var", "FOCuS-var"])]
q2("var_fa_noholiday", "§5.1", "variance detectors hold their target without holidays (4.6 to 8.6% after a past-only forecast)", fv.fa.min(),
   f"{100*fv.fa.min():.1f}-{100*fv.fa.max():.1f}%", (round(100 * fv.fa.min(), 1), round(100 * fv.fa.max(), 1)) == (4.6, 8.6),
   "sim/summary/fa.csv", "fa (phi 0.4, normal, cal 0, ar1, A/B/HW/YM, variance detectors)")
far = dl[(dl.rule == "oracle") & (dl.det == "FOCuS-AR")].set_index(["setting", "shift_sd"])
vals = [far.loc[("true pattern, no holidays", 2.0)], far.loc[("true pattern, no holidays", 4.0)], far.loc[("Holt-Winters, no holidays", 2.0)],
        far.loc[("Holt-Winters, no holidays", 4.0)], far.loc[("yearly memory, holidays", 4.0)]]
q2("B2_focusar_delay_row", "tab:delay", "FOCuS-AR: 12.6 0.24 2.6 0.99 | 49.5 0.07 5.1 0.85 | 60.8 0.00", vals[1].rmd,
   " ".join(f"{v.rmd:.1f} {v.F7:.2f}" for v in vals),
   [(round(v.rmd, 1), round(v.F7, 2)) for v in vals] == [(12.6, 0.24), (2.6, 0.99), (49.5, 0.07), (5.1, 0.85), (60.8, 0.0)],
   "sim/summary/delay.csv", "rmd, F7 (oracle rule, FOCuS-AR)")

# ------------------------------------------------------------------ further corrected numbers
gj = json.loads((R / "sim" / "generator.json").read_text())
ar = pd.read_csv(R / "calib" / "arl0.csv")
op = pd.read_csv(R / "sim" / "summary" / "offline.csv")
tu = pd.read_csv(R / "sim" / "summary" / "tuning.csv")
vvm = pd.read_csv(R / "opal" / "variance_vs_mean.csv")
MEAN8 = ["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"]
x = dt[(dt.phi == 0.4) & (dt.noise == "normal") & (dt.calendar == 0) & (dt.change == "mean") & (dt["size"] == -1.0)
       & (dt.rule == "oracle") & dt.adj.isin(["A", "B", "HW", "YM"]) & dt.det.isin(MEAN8)].detect60
q2("N1_past_only_1sd", "ch5 l.39", "1 to 11% of runs", x.min(), f"{100*x.min():.1f}-{100*x.max():.1f}%",
   (round(100 * x.min()), round(100 * x.max())) == (1, 11), "sim/summary/detect.csv", "detect60 (phi .4, cal 0, -1 SD, oracle rule, A/B/HW/YM, 8 mean detectors)")
s_ = ar[(ar.kind == "simulated") & ~ar.det.astype(str).str.endswith("-var")].arl0_days
q2("N2_arl0_range", "ch5 l.11", "6,300 to 7,800 days", s_.min(), f"{s_.min():,.0f}-{s_.max():,.0f}",
   (round(s_.min(), -2), round(s_.max(), -2)) == (6300, 7800), "calib/arl0.csv", "arl0_days (simulated, non-variance detectors)")
vv3 = vd[(vd.kind == "surprise") & (vd.phi == 0.4) & (vd.rule == "indep") & vd.det.isin(["CUSUM-var", "EWMA-var", "FOCuS-var"])].fa
q2("N3_var_indep", "ch5 l.11", "variance detectors 10 to 14% (the maximum is 13.48%, so '10 to 13%' is the correct rounding)", vv3.min(), f"{100*vv3.min():.2f}-{100*vv3.max():.2f}%",
   (round(100 * vv3.min()), round(100 * vv3.max())) == (10, 14), "calib/validation.csv", "fa (surprise, phi .4, indep, variance)")
fv4 = fa[(fa.phi == 0.4) & (fa.noise == "normal") & (fa.calendar == 0) & (fa.rule == "ar1") & fa.det.isin(["CUSUM-var", "EWMA-var", "FOCuS-var"])]
g1, g2 = fv4[fv4.adj.isin(["HW", "YM"])].fa, fv4[fv4.adj.isin(["A", "B"])].fa
q2("N4_var_ar1_by_forecast", "ch5 l.11", "4.6 to 5.8% after HW and YM; 6.0 to 8.6% after A and B", g1.min(),
   f"{100*g1.min():.1f}-{100*g1.max():.1f}; {100*g2.min():.1f}-{100*g2.max():.1f}",
   (round(100 * g1.min(), 1), round(100 * g1.max(), 1), round(100 * g2.min(), 1), round(100 * g2.max(), 1)) == (4.6, 5.8, 6.0, 8.6),
   "sim/summary/fa.csv", "fa (phi .4, normal, cal 0, ar1, variance detectors)")
y5 = va[(va.noise == "normal") & (va.calendar == 1) & (va.adj == "YM") & (va.rule == "oracle")]
t52 = y5[y5.det.isin(["CUSUM-var", "EWMA-var", "FOCuS-var", "CUSUM", "EWMA", "FOCuS", "FOCuS-AR"])]
q2("N5_ym_var_holidays", "ch5 l.61", "at most 3.5% (capped FOCuS, tripling); Table 5.2 detectors at most 1.3%", y5.detect60.max(),
   f"{100*y5.detect60.max():.1f}% ({y5.loc[y5.detect60.idxmax(), 'det']}, x{y5.loc[y5.detect60.idxmax(), 'size']:g}); {100*t52.detect60.max():.1f}%",
   round(100 * y5.detect60.max(), 1) == 3.5 and round(100 * t52.detect60.max(), 1) == 1.3, "sim/summary/variance.csv", "detect60 (cal 1, YM, oracle)")
y7 = dl[(dl.rule == "oracle") & (dl.setting == "yearly memory, holidays") & (dl.shift_sd == 4)].set_index("det")
q2("N7_ym_speed", "ch5 l.47", "capped FOCuS most within a week (35%); capped CUSUM and capped FOCuS lowest RMD (12.5, 12.9)", y7.F7["FOCuS-r"],
   f"F7 {100*y7.F7['FOCuS-r']:.0f}%; RMD {y7.rmd['CUSUM-r']:.1f} / {y7.rmd['FOCuS-r']:.1f}",
   round(100 * y7.F7["FOCuS-r"]) == 35 and (round(y7.rmd["CUSUM-r"], 1), round(y7.rmd["FOCuS-r"], 1)) == (12.5, 12.9),
   "sim/summary/delay.csv", "F7, rmd (yearly memory, holidays, 4 SD, oracle)")
w8 = vvm[(vvm.kind == "summary") & (vvm.rule == "ar1")].share_within_1d_of_mean
q2("N8_var_within1d", "opal_variance.tex caption", "60 to 76% at AR(1)", w8.min(), f"{100*w8.min():.0f}-{100*w8.max():.0f}%",
   (round(100 * w8.min()), round(100 * w8.max())) == (60, 76), "opal/variance_vs_mean.csv", "share_within_1d_of_mean (summary, ar1)")
r14 = rv[rv.r0_14_dir == rv[rv.rule == "ar1"].r0_14_dir.max()]
q2("N9_window14_best", "ch6 l.112", "yearly memory with EWMA and yearly memory with CUSUM (indep) catch the most, 10 each", rv.r0_14_dir.max(),
   "; ".join(f"{a}+{b} {c} {d}" for a, b, c, d in zip(r14.method, r14.detector, r14.rule, r14.r0_14_dir)),
   set(zip(r14.method, r14.detector, r14.rule)) == {("YM", "EWMA", "ar1"), ("YM", "CUSUM", "indep")} and rv.r0_14_dir.max() == 10,
   "opal/rule_variants.csv", "r0_14_dir (max)")
o10 = op[(op.change == "var") & (op.phi == 0.4) & (op.noise == "normal") & (op["size"] == 2.0)].detect7
o10b = op[(op.change == "var") & (op.noise == "normal")]
q2("N10_offline_var", "appB l.19", "offline variance doubling at most 3.5% (phi 0.4)", o10.max(),
   f"{100*o10.max():.1f}% (all phi, x2: {100*o10b[o10b['size'] == 2.0].detect7.max():.1f}%; x3: {100*o10b[o10b['size'] == 3.0].detect7.max():.1f}%)",
   round(100 * o10.max(), 1) == 3.5, "sim/summary/offline.csv", "detect7 (var x2, phi .4, normal)")
v11 = va[(va.phi == 0.4) & (va.noise == "normal") & (va.calendar == 0) & (va["size"] == 2.0) & (va.adj == "oracle") & (va.rule == "oracle")].set_index("det").detect60
sp, mn = v11[["CUSUM-var", "EWMA-var", "FOCuS-var"]], v11[["CUSUM", "EWMA", "FOCuS"]]
r_lo, r_hi = sp.min() / mn.max(), sp.max() / mn.min()
a_lo, a_hi = sp.min() / v11["FOCuS-AR"], sp.max() / v11["FOCuS-AR"]
q2("N11_var_ratio", "variance.tex caption, ch5 l.77, l.99, ch8 l.9", "1.6 to 2.8 times CUSUM/EWMA/FOCuS; 1.3 to 1.7 times FOCuS-AR", r_lo,
   f"{r_lo:.1f}-{r_hi:.1f}; {a_lo:.1f}-{a_hi:.1f}", (round(r_lo, 1), round(r_hi, 1), round(a_lo, 1), round(a_hi, 1)) == (1.6, 2.8, 1.3, 1.7),
   "sim/summary/variance.csv", "detect60 (phi .4, cal 0, x2, true pattern, matched): spread / mean detectors")
hs = abs(gj["pub"]) / gj["sd_raw"]
q2("N12_holiday_sd", "ch5 l.23", "about 5.7 SD", hs, f"{hs:.2f}", round(hs, 1) == 5.7, "sim/generator.json", "|pub| / sd_raw")
fi13 = f4[(f4.calendar == 0) & (f4.rule == "indep")].set_index(["adj", "det"]).fa
q2("N13_figB1", "Fig B.1 caption", "CPM raw 3.6%; FOCuS-AR after A and YM 4.6, 4.7%; capped CUSUM, EWMA raw 8.2, 6.8%", fi13[("raw", "CPM")],
   f"{100*fi13[('raw','CPM')]:.1f}; {100*fi13[('A','FOCuS-AR')]:.1f}, {100*fi13[('YM','FOCuS-AR')]:.1f}; {100*fi13[('raw','CUSUM-r')]:.1f}, {100*fi13[('raw','EWMA-r')]:.1f}",
   [round(100 * fi13[k], 1) for k in (("raw", "CPM"), ("A", "FOCuS-AR"), ("YM", "FOCuS-AR"), ("raw", "CUSUM-r"), ("raw", "EWMA-r"))] == [3.6, 4.6, 4.7, 8.2, 6.8],
   "sim/summary/fa.csv", "fa (phi .4, normal, cal 0, indep)")
v14 = vd[(vd.rule == "ar1") & (vd.phi > 0) & vd.det.isin(["CUSUM", "EWMA", "FOCuS", "FOCuS-AR", "CPM", "CUSUM-r", "EWMA-r", "FOCuS-r"])].fa
q2("N14_validation_median", "ch5 l.11 / App D", "median 5.5% for the mean detectors when phi > 0", v14.median(), f"{100*v14.median():.1f}%",
   abs(100 * v14.median() - 5.5) <= 0.05, "calib/validation.csv", "fa (ar1, phi > 0, mean detectors)")
s16 = sc[sc.detector != "FOCuS-AR"].pivot_table(index=["method", "detector"], columns="rule", values="non_event_per_year")
lower = int((s16.ar1 < s16.indep).sum())
sub = s16.loc[[(m, d) for m in ["A", "B", "HW", "YM", "MSTL_p", "REG_p", "STL", "MSTL", "REG"] for d in ["CUSUM", "EWMA", "FOCuS", "CPM"]]]
q2("N16_ar1_cut", "ch6 l.23, ch8 l.11", "fewer non-event alarms in 68 of 70; median 4.9 -> 2.7 (45%); uncapped CEFP 5.6 -> 4.3", lower,
   f"{lower} of {len(s16)}; {s16.indep.median():.1f} -> {s16.ar1.median():.1f} ({100*(1-s16.ar1.median()/s16.indep.median()):.0f}%); {sub.indep.median():.1f} -> {sub.ar1.median():.1f}",
   lower == 68 and len(s16) == 70 and (round(s16.indep.median(), 1), round(s16.ar1.median(), 1), round(sub.indep.median(), 1), round(sub.ar1.median(), 1)) == (4.9, 2.7, 5.6, 4.3),
   "opal/scores.csv", "non_event_per_year, both rules, 70 monitors without FOCuS-AR; subset 9 non-raw x CUSUM/EWMA/FOCuS/CPM")
h17 = ho[(ho.rule == "ar1") & ho.method.isin(["A", "B", "HW"]) & ho.detector.isin(["CUSUM", "EWMA", "FOCuS"])]
b17 = bh[bh.detector.isin(["CUSUM", "EWMA", "FOCuS"])]
q2("N17_heldout_counts", "ch6 l.27", "one alarm each in 21 months, against 9 to 15 with AR(1); same 1 of 3; all <= 0.91 non-event over 2020-26", b17.n_alarms.max(),
   f"boot {b17.n_alarms.min():.0f}-{b17.n_alarms.max():.0f} alarms; AR(1) {h17.n_alarms.min():.0f}-{h17.n_alarms.max():.0f}; caught {b17.episodes_caught.max():.0f}; 2020-26 max {bo.non_event_per_year.max():.2f}",
   b17.n_alarms.max() == 1 and (h17.n_alarms.min(), h17.n_alarms.max()) == (9, 15) and round(bo.non_event_per_year.max(), 2) == 0.91,
   "opal/bootstrap_heldout.csv; opal/heldout.csv; opal/bootstrap.csv", "n_alarms, episodes_caught, non_event_per_year (A/B/HW x CUSUM/EWMA/FOCuS)")
# tuning
t1 = tu[(tu.adj == "oracle") & (tu.rule == "oracle") & (tu["size"] == -1.0)].set_index("det").detect60
q2("A2_tuning_1sd", "§5.2, Table B.3", "1 SD, true pattern, matched: CUSUM k .25/.5/1 and EWMA lambda .05/.1/.2", t1["EWMA-l0.05"],
   " / ".join(f"{100*t1[k]:.0f}" for k in ("CUSUM-k0.25", "CUSUM", "CUSUM-k1", "EWMA-l0.05", "EWMA", "EWMA-l0.2")), True,
   "sim/summary/tuning.csv", "detect60 (oracle adj, oracle rule, -1 SD)")
t2 = tu[(tu.adj == "oracle") & (tu.rule == "oracle") & (tu["size"] == -2.0)].set_index("det")
q2("A2_tuning_2sd_rmd", "§5.2, Table B.3", "2 SD restricted mean delay by tuning", t2.rmd["EWMA-l0.05"],
   " / ".join(f"{t2.rmd[k]:.1f}" for k in ("CUSUM-k0.25", "CUSUM", "CUSUM-k1", "EWMA-l0.05", "EWMA", "EWMA-l0.2")), True,
   "sim/summary/tuning.csv", "rmd (oracle adj, oracle rule, -2 SD)")
# real holiday scale and % drop per SD
for noise, sig in (("normal", gj["sigma"]), ("wide", gj["sd_raw"])):
    q2(f"A4_pct_per_sd_{noise}", "Table B.1", f"1 SD drop in {noise} noise, %", 100 * (1 - np.exp(-sig)), f"{100*(1-np.exp(-sig)):.1f}%", True,
       "sim/generator.json", "sigma" if noise == "normal" else "sd_raw")
wd = dt[(dt.noise == "wide") & (dt.rule == "oracle") & dt.adj.isin(["A", "B", "HW", "YM"]) & dt.det.isin(["CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r", "FOCuS-r"])]
for (sz, adj), g in wd.groupby(["size", "adj"]):
    g = g.set_index("det").detect60
    q2(f"A4_wide_{adj}_{-sz:g}SD", "Table B.1 (real holiday scale)", f"detection within 60 days, {adj}, {-sz:g} SD (= {100*(1-np.exp(sz*gj['sd_raw'])):.0f}% drop)", g["CUSUM"],
       " / ".join(f"{k} {100*g[k]:.0f}" for k in ("CUSUM", "EWMA", "FOCuS", "CPM", "CUSUM-r", "FOCuS-r")), True,
       "sim/summary/detect.csv", f"detect60 (noise wide, oracle rule, size {sz:g}, {adj})")
# dashboard build stamp
import re
html = (HERE / "live" / "dashboard.html").read_text()
mm = re.search(r"built [0-9]{1,2} [A-Za-z]{3} [0-9]{4} [0-9:]{5}", html)
q2("D3_dashboard_stamp", "ch7 §7.4", "dashboard build stamp", mm.group(0) if mm else "", mm.group(0) if mm else "missing", bool(mm),
   "live/dashboard.html", "'built ...' stamp")

qq = pd.DataFrame(quoted)
qq.to_csv(OUT / "quoted_checks.csv", index=False)       # the sentence checks
bad = qq[~qq.match]
print(f"{len(qq)} sentence checks, {len(bad)} differ from the text" + ("" if bad.empty else ": " + ", ".join(bad.id)))

# ------------------------------------------------------------------ derived values (complex quantities for the ledger)
DER = []


def der(id_, value, how):
    DER.append(dict(id=id_, value=float(value), how=how))


der("holiday_sd_real", abs(gj["pub"]) / gj["sd_raw"], "generator.json |pub| / sd_raw")
der("pct_per_sd_normal", 100 * (1 - np.exp(-gj["sigma"])), "100 (1 - exp(-sigma))")
der("pct_per_sd_wide", 100 * (1 - np.exp(-gj["sd_raw"])), "100 (1 - exp(-sd_raw))")
der("var_ratio_lo", r_lo, "variance.csv: min spread / max of CUSUM, EWMA, FOCuS (x2, phi .4, cal 0, oracle)")
der("var_ratio_hi", r_hi, "max spread / min of CUSUM, EWMA, FOCuS")
der("var_ratio_ar_lo", a_lo, "min spread / FOCuS-AR")
der("var_ratio_ar_hi", a_hi, "max spread / FOCuS-AR")
der("ar1_pairs_lower", lower, "scores.csv: monitors (no FOCuS-AR) with fewer non-event alarms under AR(1) than indep")
der("ar1_pairs_n", len(s16), "pairs")
der("ar1_median_indep", s16.indep.median(), "median non-event, indep")
der("ar1_median_ar1", s16.ar1.median(), "median non-event, AR(1)")
der("ar1_cut_pct", 100 * (1 - s16.ar1.median() / s16.indep.median()), "% cut of the median")
der("ar1_sub_indep", sub.indep.median(), "36 pairs: median indep")
der("ar1_sub_ar1", sub.ar1.median(), "36 pairs: median AR(1)")
fco = pd.read_csv(R / "opal" / "focus_check_opal.csv")
der("focusfix_changed", int((fco.alarms_in_common < fco[["alarms_package", "alarms_exact"]].max(axis=1)).sum()), "monitors whose alarms changed")
der("focusfix_n", len(fco), "FOCuS-family Opal monitors")
der("focusfix_kept_median", 100 * (fco.alarms_in_common / fco.alarms_exact).median(), "median % of exact alarms also in the package run")
der("focusfix_frozen_moved3", int(((fco.frozen_exact - fco.frozen_package).abs() >= 3).sum()), "frozen counts moved by >= 3")
der("focusfix_dir_moved3", int(((fco.dir_exact - fco.dir_package).abs() >= 3).sum()), "direction-aware counts moved by >= 3")
der("focusfix_frozen_maxmove", (fco.frozen_exact - fco.frozen_package).abs().max(), "max frozen move")
der("focusfix_dir_maxmove", (fco.dir_exact - fco.dir_package).abs().max(), "max direction-aware move")
tab = json.loads((R / "calib" / "h_table.json").read_text())
for ph in ("0.0", "0.2", "0.4", "0.6"):
    h = tab["surprise"]["FOCuS-AR"][ph]
    der(f"focusar_h_llr_phi{ph}", h / (1 - float(ph) ** 2), "FOCuS-AR threshold on the LLR scale h/(1-phi^2)")
der("focusar_trigger_phi0.4", np.sqrt(tab["surprise"]["FOCuS-AR"]["0.4"]), "single-day trigger sqrt(h)")
der("focus_trigger_phi0.4", np.sqrt(tab["surprise"]["FOCuS"]["0.4"]), "single-day trigger sqrt(h)")
kur = {}
adjd = pd.read_csv(R / "opal" / "adjusted.csv", parse_dates=["date"]).set_index("date")
from scipy.stats import kurtosis  # noqa: E402
for m_ in ("A", "B", "HW"):
    v_ = adjd.loc["2023-01-01":"2025-12-31", m_].dropna()
    der(f"kurtosis_{m_}", kurtosis(v_), "excess kurtosis of the surprises 2023-25")
from cpd.data import opal_daily  # noqa: E402
_x = opal_daily("train")
der("opal_days", len(_x.loc["2020-01-01":"2026-09-28"]), "days 1 Jan 2020 - 28 Sep 2026")
der("opal_missing_days", int(_x.loc["2020-01-01":"2026-09-28"].isna().sum()), "missing or incomplete days")
_e = pd.read_csv(HERE / "data" / "electricity" / "nsw_demand_daily.csv", parse_dates=["date_nem"])
der("elec_days", int(_e["complete"].sum()), "complete electricity days (2019-01-01 to 2026-10-22)")
der("generator_phi", gj["phi"], "generator lag-one autocorrelation")
der("ar1_var_ratio_phi0.40", 1.4 / 0.6, "(1 + phi) / (1 - phi) at phi = 0.40")
der("ar1_sd_ratio_phi0.40", np.sqrt(1.4 / 0.6), "sqrt of the variance ratio at phi = 0.40")
from cpd.adjust import READER_OFF  # noqa: E402
READER_OFF_DAYS = list(READER_OFF)
der("reader_off_days", len(READER_OFF), "reader-off fare-free days masked in MSTL")
der("generator_kurtosis", gj["excess_kurtosis"], "generator residuals, excess kurtosis")
# the surprise kurtosis without the reader-off days (and, stricter, without the 7 days after each,
# which carry their echo through the same-weekday reference and the forecasts' memory)
_ro_idx = pd.DatetimeIndex(READER_OFF_DAYS)
_ro_wk = pd.DatetimeIndex(sorted({d + pd.Timedelta(days=k) for d in _ro_idx for k in range(8)}))
for m_ in ("A", "B", "HW"):
    v_ = adjd.loc["2023-01-01":"2025-12-31", m_].dropna()
    der(f"kurtosis_{m_}_no_reader_off", kurtosis(v_[~v_.index.isin(_ro_idx)]), "as kurtosis_*, reader-off days removed")
    der(f"kurtosis_{m_}_no_reader_off_week", kurtosis(v_[~v_.index.isin(_ro_wk)]), "reader-off days and the 7 days after removed")
der("reader_off_days_2023_25", int(((_ro_idx >= "2023-01-01") & (_ro_idx <= "2025-12-31")).sum()), "reader-off days inside 2023-25")
# FOCuS-AR with resampled noise, two-SD drop, true pattern removed, matched rate (Table 5.2 cell)
_dt = pd.read_csv(R / "sim" / "summary" / "detect.csv")
der("boot_focusar_2sd_pct", 100 * _dt[(_dt.noise == "boot") & (_dt.adj == "oracle") & (_dt.change == "mean") & (_dt["size"] == -2.0)
                                      & (_dt.det == "FOCuS-AR") & (_dt.rule == "oracle")].detect60.iloc[0],
    "FOCuS-AR, resampled noise, 2 SD drop, true pattern, matched rate: % found within 60 days")
# the detectors run in the electricity test bed
pd.DataFrame({"det": sorted(pd.read_csv(R / "electricity" / "testbed.csv", usecols=["det"]).det.unique())}).to_csv(
    OUT / "elec_testbed_detectors.csv", index=False)
der("generator_sigma", gj["sigma"], "generator robust sigma")
der("mc_se_fa", 100 * np.sqrt(0.05 * 0.95 / 10000), "Monte Carlo SE of a 5% rate, 10,000 runs, percentage points")
der("mc_se_det", 100 * np.sqrt(0.25 / 10000), "largest Monte Carlo SE of a detection rate, 10,000 runs")
der("pub_pct", 100 * (1 - np.exp(gj["pub"])), "public-holiday effect, % fall")
der("xmas_pct", 100 * (1 - np.exp(gj["xmas"])), "Christmas-period effect, % fall")
der("holiday_sd_sim", abs(gj["pub"]) / gj["sigma"], "public holiday in simulated SD units")
der("pct_4sd_normal", 100 * (1 - np.exp(-4 * gj["sigma"])), "% fall of a 4 SD drop")
_vt = va[(va.noise == "t5") & (va.adj == "oracle") & (va.rule == "oracle") & (va["size"] == 2.0)].set_index("det").detect60
der("t5_spread_ratio", (_vt["CUSUM-var"] + _vt["FOCuS-var"]) / 2 / _vt["CUSUM"], "t5, x2: mean of variance CUSUM and FOCuS over CUSUM")
der("generator_sd_raw", gj["sd_raw"], "generator ordinary SD")
der("generator_n_used", gj["n_used"], "days used in the generator fit")
der("generator_n_days", gj["n_used"] + gj["n_excluded"], "days 2023-2025")
_pf = pd.read_csv(OUT / "phi_fallback.csv")
_later = [sum(d_ > "2020-06-01" for d_ in str(x).split(";")) for x in _pf[(_pf.method == "B") & _pf.detector.isin(["CUSUM", "EWMA", "FOCuS"])].phi0_starts]
der("phi0_later_B_min", min(_later), "later phi_hat = 0 segments, median of four with CUSUM, EWMA, FOCuS")
der("phi0_later_B_max", max(_later), "same, max")
_ro = []
for d_ in READER_OFF_DAYS:
    if d_ in _x.index and not np.isnan(_x[d_]):
        prev = _x.reindex([d_ - pd.Timedelta(days=7 * k) for k in (1, 2, 3, 4)]).median()
        _ro.append(100 * (1 - _x[d_] / prev))
der("reader_off_min_fall_pct", min(_ro), "smallest fall on a reader-off day against the median of the 4 previous same weekdays (all 99.3-99.98%)")
_o = fa[(fa.rule == "oracle") & fa.noise.isin(["normal", "t5"]) & ~fa.det.isin(["CUSUM-k0.25", "CUSUM-k1", "EWMA-l0.05", "EWMA-l0.2"])]
der("splithalf_share_in_band", 100 * ((_o.fa >= 0.04) & (_o.fa <= 0.06)).mean(), "% of matched cells within 4-6% on odd batches")
der("splithalf_cells", len(_o), "cells")
der("splithalf_min_pct", 100 * _o.fa.min(), "worst cell, %")
# later period (2025 to Sep 2026) with and without the 15 Jan 2025 boundary
from cpd import score as SC  # noqa: E402
_sc = SC.scored()
_late = _sc[_sc.date >= "2025-01-01"]
_al = pd.read_csv(R / "opal" / "alarms.csv", parse_dates=["alarm"])
_rows = []
for (m_, d_, r_), g_ in _al.groupby(["method", "detector", "rule"]):
    g_ = g_[g_.alarm >= "2025-01-01"]
    ad_, dv_ = pd.DatetimeIndex(g_.alarm), g_.direction.to_numpy()
    for lab, sc_ in (("with", _late), ("without_strike_2025_01", _late[_late.episode != "strike_2025_01"])):
        _rows.append(dict(method=m_, detector=d_, rule=r_, scored=lab, episodes=sc_.episode.nunique(),
                          dir=SC.episodes_caught(ad_, dv_, sc=sc_)[0],
                          frozen=SC.episodes_caught(ad_, None, 7, 7, sc=sc_, credit="all")[0]))
pd.DataFrame(_rows).to_csv(OUT / "later_no_dq.csv", index=False)

# ch6 values for the ledger
_rv = pd.read_csv(R / "opal" / "rule_variants.csv")
_ex = pd.read_csv(OUT / "extra_catches.csv")
_near = _ex[_ex.kind == "before"].groupby("episode").offset_days.max().abs()
der("extra_before_min_days", _near.min(), "A+CUSUM extra catches before a boundary: nearest offset per episode, min")
der("extra_before_max_days", _near.max(), "same, max")
_a = _rv[(_rv.method == "A") & (_rv.detector == "CUSUM") & (_rv.rule == "ar1")].iloc[0]
_fx = set(_a.episodes_frozen.split(";")) - set(_a.episodes_pm7_nodir.split(";"))
der("a_cusum_frozen_extra_n", len(_fx), "episodes the frozen rule adds over the nearest-boundary rule (A+CUSUM, AR(1))")
_aa = _al[(_al.method == "A") & (_al.detector == "CUSUM") & (_al.rule == "ar1")].alarm
_nb = []
for ep_ in _fx:
    for b_ in _sc[_sc.episode == ep_].date:
        for t_ in _aa[(_aa - pd.Timestamp(b_)).dt.days.abs() <= 7]:
            _nb.append(_sc[(_sc.date - t_).dt.days.abs() <= 7].episode.nunique())
der("a_cusum_frozen_extra_bounds", max(_nb), "episodes with a boundary within 7 days of that alarm")
der("focusfix_frozen_minmove3", (fco.frozen_exact - fco.frozen_package).abs()[lambda v: v >= 3].min(),
    "smallest frozen move among the counts that moved by >= 3")
_oc = pd.read_csv(R / "opal" / "offline_changes.csv", parse_dates=["change_date"])
_gap = _oc.groupby(["adjustment", "method"]).change_date.agg(lambda d: (d.max() - d.min()).days / (len(d) - 1) / 7)
der("offline_gap_weeks_min", _gap.min(), "offline segmentation: mean weeks between marked changes, min over 9 runs")
der("offline_gap_weeks_max", _gap.max(), "same, max")
_cv = pd.read_csv(R / "opal" / "covid.csv")
_pick = [("HW", "FOCuS"), ("HW", "EWMA"), ("A", "CUSUM"), ("A", "FOCuS"), ("B", "EWMA"), ("YM", "CUSUM"),
         ("A", "CPM"), ("STL", "CUSUM"), ("MSTL_p", "CUSUM")]  # figures.COVID_PICK (Figure 6.4)
_cp = _cv[(_cv.rule == "ar1") & _cv.apply(lambda r: (r.method, r.detector) in _pick, axis=1)]
der("covid_pick_max_gap_weeks", _cp.days_alarm_after_start_est.max() / 7, "Figure 6.4 monitors: longest alarm - change date, weeks")
_vs = pd.read_csv(R / "opal" / "variance_scores.csv")
_vs = _vs[(_vs.rule == "ar1") & _vs.method.isin(["HW", "YM"])]
der("var_school_hwym_min", 100 * (_vs.n_school / _vs.n_alarms).min(), "variance detectors, HW/YM, AR(1): % alarms in school holidays, min")
der("var_school_hwym_max", 100 * (_vs.n_school / _vs.n_alarms).max(), "same, max")
_va = pd.read_csv(R / "opal" / "variance_alarms.csv")
_va = _va[(_va.rule == "ar1") & _va.method.isin(["HW", "YM"])].groupby(["method", "detector"]).within_1d_of_mean.mean()
der("var_within1d_hwym_min", 100 * _va.min(), "variance detectors, HW/YM, AR(1): % alarms within a day of a mean alarm, min")
der("var_within1d_hwym_max", 100 * _va.max(), "same, max")
_se = pd.read_csv(R / "opal" / "sensitivity.csv")


def _base(m, d, rule="ar1", col="r0_7_dir"):
    return _rv[(_rv.method == m) & (_rv.detector == d) & (_rv.rule == rule)][col].iloc[0]


_k1 = [_se[(_se.method == m) & (_se.detector == "CUSUM") & (_se.variant == "k=1.0")].episodes_caught.iloc[0] - _base(m, "CUSUM")
       for m in ("YM", "STL")]
_mv = []
for m_, d_ in (("HW", "FOCuS"), ("HW", "EWMA"), ("A", "CUSUM")):  # the monitors of tab:opal-sens
    b_ = _base(m_, d_)
    _mv += [abs(v - b_) for v in _se[(_se.method == m_) & (_se.detector == d_)].episodes_caught]
    _mv += [abs(_base(m_, d_, col=c) - b_) for c in ("r0_3_dir", "r0_14_dir")]
der("sens_max_move_tab", max(_mv), "tab:opal-sens monitors: largest move in episodes over the sensitivity runs and windows")
der("sens_k1_add_ym_stl", min(_k1), "episodes added by k=1 for YM+CUSUM and STL+CUSUM (both equal)")
_hw = list(_se[(_se.method == "HW") & (_se.detector == "EWMA")].episodes_caught) + \
    [_base("HW", "EWMA", col=c) for c in ("r0_7_dir", "r0_3_dir", "r0_14_dir")]
der("hw_ewma_runs_min", min(_hw), "HW+EWMA: episodes over the sensitivity runs and the 3/7/14-day windows, min")
der("hw_ewma_runs_max", max(_hw), "same, max")
_RC = ["frozen", "pm7_nodir", "pm7_dir", "r0_7_nodir", "r0_7_dir"]  # the rule versions of tab:rule-variants
for m_, d_, k_ in (("HW", "EWMA", "hw_ewma"), ("A", "CUSUM", "a_cusum")):
    v_ = [_base(m_, d_, col=c) for c in _RC]
    der(f"{k_}_rules_min", min(v_), "episodes over the five rule versions, min")
    der(f"{k_}_rules_max", max(v_), "same, max")
der("frozen_leaders_n", (_rv.frozen == _rv.frozen.max()).sum(), "monitors at the frozen-rule maximum")
der("frozen_leaders_indep_n", ((_rv.frozen == _rv.frozen.max()) & (_rv.rule == "indep")).sum(), "of them with the independent threshold")
_retro = _rv.method.isin(["STL", "MSTL", "REG"])
der("frozen_best_past", _rv[_rv.method.isin(["A", "B", "HW", "YM", "MSTL_p", "REG_p"])].frozen.max(), "best past-only, frozen")
der("frozen_best_retro", _rv[_retro].frozen.max(), "best retrospective, frozen")
# simulated false alarms and thresholds before the exact FOCuS (package focus_cpt): kept copies of the values
# computed with the package, results/extra/fa_package_focus.csv and h_package_focus.csv
_hpre = OUT / "h_package_focus.csv"
_hp = pd.read_csv(_hpre)
_hn = pd.DataFrame([dict(kind=k, det=d, phi=float(p), h=v) for k in ("surprise", "level")
                    for d in ("FOCuS", "FOCuS-r", "FOCuS-AR") for p, v in tab[k][d].items()])
_hm = _hp.merge(_hn, on=["kind", "det", "phi"], suffixes=("_pkg", "_exact"))
for k_ in ("surprise", "level"):
    _x = _hm[_hm.kind == k_]
    der(f"focusfix_h_move_{k_}", 100 * (_x.h_exact / _x.h_pkg - 1).abs().max(), f"largest % change of a FOCuS-family threshold ({k_})")
_fc = pd.read_csv(R / "calib" / "focus_check.csv").groupby("detector").share_equal.mean()
der("focus_cpt_lower_pct_FOCuS", 100 * (1 - _fc["FOCuS"]), "focus_cpt below the exact statistic: % of days, mean of 50 series")
der("focus_cpt_lower_pct_FOCuS-AR", 100 * (1 - _fc["FOCuS-AR"]), "same, FOCuS-AR")
for k_ in (0.5, 1.0, 2.0, 4.0):   # Table D.1: a mean drop of k SD as a % change in daily travel
    der(f"drop_pct_sd{k_:g}", 100 * (np.exp(-k_ * gj["sigma"]) - 1), "100 (exp(-k sigma) - 1)")
der("generator_trend_pct", 100 * (np.exp(gj["trend_per_year"]) - 1), "fitted trend, % a year")
der("generator_trend", gj["trend_per_year"], "fitted trend, log scale a year")
der("generator_n_excluded", gj["n_excluded"], "days removed before the generator fit")
# ch8: survivors that lose an episode without the 15 Jan 2025 boundary
_s1 = pd.read_csv(R / "opal" / "bothrules_survivors.csv")
_s0 = pd.read_csv(R / "opal" / "bothrules_survivors_no_dq.csv")
_sm = _s1[_s1.survives].merge(_s0, on=["method", "detector", "rule"], suffixes=("", "_nodq"))
der("nodq_survivors_lose", int((_sm.min_count_nodq < _sm.min_count).sum()), "both-rules survivors whose count falls without 15 Jan 2025")
der("nodq_survivors_hold", int(_sm.survives_nodq.sum()), "of them still holding")

# ch7 values: the electricity monitor (temperature regression with FOCuS, AR(1))
_es = pd.read_csv(R / "electricity" / "scores.csv")
_e1 = _es[(_es.adj == "REGT_p") & (_es.det == "FOCuS") & (_es.rule == "ar1")].iloc[0]
der("elec_years", _e1.n_alarms / _e1.alarms_per_year, "years monitored (n_alarms / alarms_per_year)")
_ea = pd.read_csv(R / "electricity" / "alarms.csv", parse_dates=["alarm"])
_ea = _ea[(_ea.adj == "REGT_p") & (_ea.det == "FOCuS") & (_ea.rule == "ar1")].alarm
der("elec_xmas_alarms", int(((_ea.dt.month == 12) & _ea.dt.day.isin([25, 26])).sum()), "alarms on 25 or 26 December")
_ead = pd.read_csv(R / "electricity" / "adjusted.csv", parse_dates=["date_nem"]).set_index("date_nem")["REGT_p"]
_wk = _ead.loc["2023-11-24":"2023-11-29"]
_wk = 100 * (np.exp(_wk[_wk.index.dayofweek < 5]) - 1)
der("elec_nov23_min_pct", _wk.min(), "working days 24-29 Nov 2023: surprise of the temperature regression, % above forecast, min")
der("elec_nov23_max_pct", _wk.max(), "same, max")
_rc = pd.read_csv(R / "electricity" / "rule_credits.csv")
_rc = _rc[(_rc.rule == "ar1") & (_rc.adj == "HW")].days_from_boundary.abs()
der("elec_hw_credit_min_days", _rc.min(), "HW AR(1) frozen-rule credits: days before the lockdown order, min")
der("elec_hw_credit_max_days", _rc.max(), "same, max")
pd.DataFrame(DER).to_csv(OUT / "derived.csv", index=False)
# the threshold table as a flat file, so ledger rows can select from it
pd.DataFrame([dict(kind=k, det=d, phi=float(ph), h=v) for k, dd in tab.items() if k != "meta" for d, m_ in dd.items()
              for ph, v in m_.items()]).to_csv(R / "calib" / "h_table_flat.csv", index=False)

# ------------------------------------------------------------------ the ledger: quoted.csv from the spec
import check_numbers as CN  # noqa: E402

spec = pd.read_csv(HERE / "quoted_spec.csv", dtype=str, keep_default_na=False)
# chapter-by-chapter rows written against the thesis text (ledger/*.csv, same columns)
for _f in sorted((HERE / "ledger").glob("*.csv")):
    spec = pd.concat([spec, pd.read_csv(_f, dtype=str, keep_default_na=False)], ignore_index=True)

led = []
for _, r_ in spec.iterrows():
    row = r_.to_dict()
    try:
        row["printed"] = CN.recompute(row, R)
        row["value"] = row["printed"]
    except Exception as e:  # noqa: BLE001
        row["printed"], row["value"], row["note"] = "", "", f"ERROR {e!r}"[:200]
    led.append(row)
cols = ["id", "tex", "anchor", "printed", "value", "source", "select", "column", "transform", "rounding", "note"]
pd.DataFrame(led, columns=cols).to_csv(OUT / "quoted.csv", index=False)
print(f"{len(led)} ledger rows written to results/extra/quoted.csv")
