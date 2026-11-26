#!/usr/bin/env python3
"""Build tidy Opal datasets from the raw daily files (Transport for NSW Open Data Hub, "Opal Tap Data", CC BY 4.0).
Raw file: Opal_Patronage_YYYYMMDD.txt, pipe-separated:
  trip_origin_date|mode_name|ti_region|tap_hour|Tap_Ons|Tap_Offs   (counts rounded to 100; "<100" = fewer than 100)
Outputs (in this folder):
  opal_hourly_nsw.csv   date,hour,mode,tap_ons,tap_offs,lt100_on,lt100_off   (region "All - NSW", every mode, every hour)
  opal_hourly_train_regions.csv   date,hour,region,tap_ons,tap_offs,lt100_on,lt100_off   (train only, all 10 regions)
  opal_daily_nsw.csv    date,mode,tap_ons,tap_offs,hours_lt100             (daily sums of the hourly NSW series)
"<100" is kept as 0 in the counts and flagged in lt100_* so it can be treated differently later.
usage: python3 build_opal.py [raw_dir]   (default: ./raw)"""
import csv, glob, os, sys, collections
raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
out = os.path.dirname(os.path.abspath(__file__))
def num(v):
    v = (v or "").strip()
    if v.startswith("<"): return 0, 1
    return (int(float(v)) if v else 0), 0
nsw, trg = [], []
daily = collections.defaultdict(lambda: [0, 0, 0])
files = sorted(glob.glob(os.path.join(raw, "Opal_Patronage_*.txt")))
for f in files:
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="|"):
            d, m, reg, h = r["trip_origin_date"], r["mode_name"].strip().lower(), r["ti_region"].strip(), int(r["tap_hour"])
            on, lon = num(r["Tap_Ons"]); off, loff = num(r["Tap_Offs"])
            if reg == "All - NSW":
                nsw.append((d, h, m, on, off, lon, loff))
                a = daily[(d, m)]; a[0] += on; a[1] += off; a[2] += lon
            if m == "train":
                trg.append((d, h, reg, on, off, lon, loff))
nsw.sort(); trg.sort()
with open(os.path.join(out, "opal_hourly_nsw.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["date", "hour", "mode", "tap_ons", "tap_offs", "lt100_on", "lt100_off"]); w.writerows(nsw)
with open(os.path.join(out, "opal_hourly_train_regions.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["date", "hour", "region", "tap_ons", "tap_offs", "lt100_on", "lt100_off"]); w.writerows(trg)
with open(os.path.join(out, "opal_daily_nsw.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["date", "mode", "tap_ons", "tap_offs", "hours_lt100"])
    for (d, m), (on, off, l) in sorted(daily.items()): w.writerow([d, m, on, off, l])
print(f"{len(files)} files -> {len(nsw)} NSW hourly rows, {len(trg)} train-by-region hourly rows, {len(daily)} daily rows")
