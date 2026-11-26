# NSW electricity demand and Sydney temperature

Used in the live application chapter and by the dashboard (`live/`). Licences: see `../LICENSES.md`.

| File | What |
|---|---|
| `nsw_demand_daily.csv` | Daily mean of AEMO TOTALDEMAND (MW), region NSW1, by NEM day, from 1 Jan 2019, with max/min half-hour MW, completeness counts and energy (MWh, complete days only). |
| `nsw_demand_30min.csv.gz` | Half-hourly mean TOTALDEMAND, labelled by interval end (NEM time), with the simple mean of the five-minute prices. |
| `sydney_temp_hourly.csv.gz` | Hourly 2 m temperature at Sydney (-33.87, 151.21) and Penrith (-33.75, 150.69), Open-Meteo archive, ECMWF IFS model, UTC+10. |
| `sydney_temp_daily_sydney.csv`, `sydney_temp_daily_penrith.csv` | Daily mean, max, min, hours, cooling degree days (base 24 C) and heating degree days (base 18 C). |
| `events_electricity.csv` | 53 dated events with source URLs, confidence and when each was announced: COVID orders and lockdowns, heat and cold peaks, market events, plant closures, policy and calendar effects. Checked 22 Oct 2026. |
| `events_electricity_scored_FROZEN.csv`, `events_electricity_scored_RULES.md` | The scored onsets and ends (COVID 2020, Greater Sydney lockdown 2021), frozen on 22 Oct 2026 before any electricity monitoring run. Every other event is an explanation only. |
| `events_scored_FROZEN.csv`, `events_verified.csv` | Working copies that `scripts/electricity.py` makes so that it can reuse the Opal scorer. The first is identical to the frozen list; the second keeps the dated events (31 days or shorter) as explanations. They are recreated if deleted. |

**Refresh.** `make data-electricity` runs `python live/data_fetch.py all`. It downloads the AEMO monthly files
(`https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_YYYYMM_NSW1.csv`) into `raw/`, rebuilds the
two demand files, and rebuilds the temperature files from the Open-Meteo archive. `python live/update.py` does a
smaller refresh (the current month and the new days of temperature) before it runs the monitor.

**Time conventions.**

- AEMO SETTLEMENTDATE is NEM time (UTC+10, no daylight saving) and marks the end of each interval.
- Intervals are 30 minutes up to 1 Oct 2021 00:00 and five minutes after that.
- A NEM day D covers the intervals that start in [D 00:00, D+1 00:00).
- AEMO rewrites the current-month file once a day, just after midnight NEM time, so the daily monitor runs after
  00:10 NEM time.

**What TOTALDEMAND is.** It is the demand met by scheduled and semi-scheduled generation plus imports. It excludes
rooftop PV and battery charging, so it falls as rooftop solar grows. It is not the "operational demand" in AEMO's
quarterly reports, so compare its levels with those reports only roughly.
