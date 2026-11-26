"""Ground truth for the browser detector check: insert drops with the full Python pipeline (forecast refitted on
the changed series, then cpd.monitor.run) and record the day the alarm fires. Reads ../cache and ../data.json only.
    python checks/check_truth.py   ->  checks/truth.json   (then: node checks/check_js.mjs)"""
import json
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd
import build_data as B

E, A, monitor, calibrate = B.E, B.A, B.monitor, B.calibrate
data = json.loads((HERE.parent / "data.json").read_text())
tabs = calibrate.load(B.TABLE, "surprise")
PCTS, DAYS = [2, 3, 5, 7, 10, 15, 20, 30], [5, 10, 14, 21, 30, 45, 60]
out = {}

# electricity
S = data["series"]["elec"]
cfg = json.loads(B.CONFIG.read_text())
x = E.load().iloc[:S["n"]]
assert str(x.index[-1].date()) == S["last_day"]
Z = E.temps(x.index)
v = x.to_numpy()
r0 = E.adj_regt_past(v, x.index, Z)
rows = []
for p in PCTS:
    for d in DAYS:
        tau = len(v) - d
        y = v.copy(); y[tau:] *= (1 - p / 100)
        r = E.adj_regt_past(y, x.index, Z, t_from=tau, base=r0)
        res = monitor.run(r, "surprise", cfg["det"], cfg["rule"], tabs, S["t_start"], method=cfg["adj"])
        hit = [a for a in res.alarms if a >= tau]
        rows.append({"pct": p, "days": d, "after": int(hit[0] - tau + 1) if hit else None})
out["elec"] = rows

# opal
S = data["series"]["opal"]
m = B.OPAL_MONITOR
x = B.opal_daily("train")
v = x.to_numpy()
rows = []
for p in PCTS:
    for d in DAYS:
        tau = len(v) - d
        y = v.copy(); y[tau:] *= (1 - p / 100)
        r = A.adj_HW(y, x.index)
        res = monitor.run(r, "surprise", m["det"], m["rule"], tabs, S["t_start"], method=m["adj"])
        hit = [a for a in res.alarms if a >= tau]
        rows.append({"pct": p, "days": d, "after": int(hit[0] - tau + 1) if hit else None})
out["opal"] = rows
(HERE / "truth.json").write_text(json.dumps(out, indent=0))
print("wrote truth.json:", {k: sum(r["after"] is not None for r in v) for k, v in out.items()}, "of", len(PCTS) * len(DAYS), "cases flagged")
