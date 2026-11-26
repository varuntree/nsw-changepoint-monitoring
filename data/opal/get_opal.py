#!/usr/bin/env python3
"""Download the Transport for NSW "Opal Tap Data" daily files (formerly Opal Patronage) into ./raw, then rebuild the CSVs.
Source: https://opendata.transport.nsw.gov.au/data/dataset/opal-patronage  (Creative Commons Attribution; credit Transport for NSW)
File URL (verified 30 Sep 2026): https://opendata-tpa.transport.nsw.gov.au/Opal_Patronage/YYYY-MM/Opal_Patronage_YYYYMMDD.txt
The server requires the header  Referer: https://opendata.transport.nsw.gov.au/   (as the dataset metadata says).
Files appear with a lag (the newest was 2 days old on 30 Sep 2026); a 404 for recent days is normal.
usage: python3 get_opal.py [--since YYYY-MM-DD] [--refresh-days N]
  default: fetch every day missing from ./raw from 2020-01-01 to today, and re-fetch the last 4 days
  (the publisher revises recent files: 15-minute refreshes for 2 days, then a full refresh on day 4)."""
import datetime as dt, os, subprocess, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "raw")
URL = "https://opendata-tpa.transport.nsw.gov.au/Opal_Patronage/{ym}/Opal_Patronage_{ymd}.txt"
HDR = {"Referer": "https://opendata.transport.nsw.gov.au/", "User-Agent": "DATA5925-Group5/1.0"}
def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
since = dt.date.fromisoformat(arg("--since", "2020-01-01")); refresh = int(arg("--refresh-days", "4"))
today = dt.date.today(); os.makedirs(RAW, exist_ok=True)
got = miss = 0
d = since
while d <= today:
    f = os.path.join(RAW, f"Opal_Patronage_{d:%Y%m%d}.txt")
    if not os.path.exists(f) or (today - d).days <= refresh:
        req = urllib.request.Request(URL.format(ym=f"{d:%Y-%m}", ymd=f"{d:%Y%m%d}"), headers=HDR)
        try:
            data = urllib.request.urlopen(req, timeout=30).read()
            if data.startswith(b"trip_origin_date"):
                open(f, "wb").write(data); got += 1
        except Exception:
            miss += 1
    d += dt.timedelta(days=1)
print(f"downloaded/refreshed {got} files, {miss} not available yet")
subprocess.run([sys.executable, os.path.join(HERE, "build_opal.py"), RAW], check=True)
