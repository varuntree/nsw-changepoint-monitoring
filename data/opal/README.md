# NSW Opal train trips

**Source:** Transport for NSW Open Data Hub, "Opal Tap Data" (formerly "Opal Patronage"),
https://opendata.transport.nsw.gov.au/data/dataset/opal-patronage. Licence: CC BY 4.0, credit "Transport for NSW"
(see `../LICENSES.md`). No API key is needed. The download needs the header
`Referer: https://opendata.transport.nsw.gov.au/`, and `get_opal.py` sends it.

The source has one file per day of tap-ons and tap-offs for every hour, mode (train, which includes Metro; bus;
ferry; light rail) and region. Counts are rounded to the nearest 100, and cells below 100 are shown as `<100`.
Counts are taps, not unique people.

| File | What |
|---|---|
| `opal_daily_nsw.csv` | date, mode, tap_ons, tap_offs, hours_lt100. Region "All - NSW", daily sums of the hourly rows, 1 Jan 2020 to 28 Sep 2026 (11,557 rows). The analysis uses `mode == "train"`, tap_ons. |
| `events_verified.csv` | 71 dated events, each with its source (URL and title), the date it was checked (22 Oct 2026), its kind (disruption, policy, structural, fault), direction and confidence. |
| `events_scored_FROZEN.csv` | The 18 scored boundaries (15 episodes) taken from `events_verified.csv` before any monitor was scored. Not edited since. |
| `events_scored_RULES.md` | How the scored list was chosen and how an alarm is credited to an episode, with the dated amendment note. |
| `get_opal.py` | Downloads missing or recent daily files into `raw/`, then runs `build_opal.py` (`make data-opal`). |
| `build_opal.py` | Builds `opal_daily_nsw.csv` and two hourly CSVs (`opal_hourly_nsw.csv`, `opal_hourly_train_regions.csv`) from `raw/`. |

**Caveats from the publisher:**

- Data may be incomplete in two periods: 28 Jan to 5 Feb 2025, and 28 Jan to 12 Feb 2026.
- Bus taps are under-reported from 18 Jun to 2 Jul 2026 (industrial action).
- Recent days can be revised for up to four days.
- 3 and 4 June 2021 have no file.

These periods are listed as "fault" rows in `events_verified.csv`. Publisher documentation: "Opal Patronage
documentation v2.0" on the dataset page.

Running `get_opal.py` today downloads more recent days than the thesis used. The analysis stops at 28 Sep 2026
(`cpd/data.py`), so its results do not change.
