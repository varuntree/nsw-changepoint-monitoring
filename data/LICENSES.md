# Data sources, licences and attribution

The code in this repository is under the MIT licence (`LICENSE`). The data are not ours; each source keeps its own
terms, summarised below with the files that come from it. Where a licence allows redistribution with attribution,
the processed files used by the analysis are included so that every result can be rebuilt without downloading
anything. Large raw downloads are left out and are recreated by the fetch scripts.

| Source | Licence / terms | In this repository | Not included (fetch script) |
|---|---|---|---|
| Transport for NSW, Opal Tap Data (formerly Opal Patronage) | Creative Commons Attribution 4.0 (CC BY 4.0) | `opal/opal_daily_nsw.csv` (daily NSW tap-ons and tap-offs by mode, built from the official daily files) | raw daily files, about 70 MB, and the two hourly CSVs: `make data-opal` (`opal/get_opal.py`, `opal/build_opal.py`) |
| Australian Energy Market Operator (AEMO), Aggregated price and demand data, region NSW1 | AEMO copyright permissions: general permission to use AEMO material for any purpose with accurate and appropriate attribution | `electricity/nsw_demand_daily.csv`, `electricity/nsw_demand_30min.csv.gz` (derived daily and half-hourly series) | 94 monthly CSVs, about 26 MB: `make data-electricity` (`../live/data_fetch.py`) |
| Open-Meteo.com, Historical Weather API and Forecast API (ECMWF IFS model) | CC BY 4.0 (Open-Meteo licence page); free API for non-commercial use | `electricity/sydney_temp_hourly.csv.gz`, `electricity/sydney_temp_daily_sydney.csv`, `electricity/sydney_temp_daily_penrith.csv` (hourly 2 m temperature and daily aggregates, degree days derived by us) | none |
| Bureau of Meteorology (BoM) | BoM copyright | No BoM data. Some rows of the event lists cite BoM web pages as their source (link only). | none |
| NSW Department of Education, school holiday calendar | public calendar | the holiday dates are typed into `cpd/score.py` | none |
| R package `cpm` 2.3 (G. J. Ross, CRAN) | GPL-3 | `cpm/cpm23_MWARL500.csv`: the package's published Mann-Whitney thresholds (ARL 500, startup 20), extracted unchanged, used only to check our CPM kernel (`scripts/cpm_hd_check.py`) | none |
| `holidays` Python package (NSW public holidays) | MIT | installed from `requirements.txt` | none |
| Event lists compiled by the authors (`opal/events_verified.csv`, `opal/events_scored_FROZEN.csv`, `electricity/events_electricity.csv`, `electricity/events_electricity_scored_FROZEN.csv`) | CC BY 4.0 (the authors); each row links to its public source | included | none |

The results (`../results/`), figures (`../figures/`), tables (`../tables/`, `../repo-tables/`), appendix material
(`../appendix/`) and the live page's data (`../dashboard/data.json`) are derived from the sources above. They are
released under CC BY 4.0 with the same attributions. The fonts in `../dashboard/fonts/` (Inter, Geist Mono) are
under the SIL Open Font License 1.1; their licence texts are in that folder.

## Required attribution

- **Opal:** "Source: Transport for NSW, Opal Tap Data, Open Data Hub
  (https://opendata.transport.nsw.gov.au/data/dataset/opal-patronage), licensed under CC BY 4.0." The data licence
  page (https://opendata.transport.nsw.gov.au/datalicence) allows redistribution and reuse on condition that
  Transport for NSW is attributed as the source.
- **AEMO:** "Source: Australian Energy Market Operator (AEMO), Aggregated price and demand data, NSW1, 2019-2026,
  accessed 22 October 2026." Terms: https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions. AEMO
  gives its information without warranty and may update it.
- **Open-Meteo:** "Weather data by Open-Meteo.com (https://open-meteo.com/), CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/); ECMWF IFS 9 km model. Daily aggregates and degree days derived
  by the authors."

## Further retrospective adjustments and independently written code

Eight further retrospective seasonal adjustments of the Opal series (STL, MSTL, MSTL with the free week smoothed,
Fourier regression, GAM, classical decomposition, Prophet, TBATS) were fitted in exploratory work; their scored
results are in `../results/opal/retro_extra.csv`. An EWMA and CPM, and a PELT, Binary Segmentation and WBS, were also
written independently from the same definitions and used to check our implementation; the comparisons are in
`../results/extra/crosscheck_ewma_cpm*.csv` and `../results/extra/crosscheck_offline*.csv`. The adjusted series and
the independent code are not included. When they are placed in `independent/` (`independent/ewma_cpm/`,
`independent/offline/`), `make results` re-runs `scripts/retro_extra.py`, `scripts/crosscheck_ewma_cpm.py` and
`scripts/crosscheck_offline.py` on them.

## Changes to the data

- Opal: daily sums of the hourly "All - NSW" rows; counts below 100 are stored as 0 and flagged in `hours_lt100`.
  3-4 June 2021 have no published file and are missing; 31 July and 1 August 2025 have fewer than 20 reported
  hours and are set to missing in `cpd/data.py`.
- AEMO: five-minute values averaged to half-hours, half-hours averaged to NEM days (UTC+10, no daylight saving).
- Open-Meteo: hourly values at a fixed UTC+10 offset; daily mean, maximum and minimum from hours 00:00-23:00.
