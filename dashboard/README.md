# Live page

The live page of the thesis, **https://varuntree.github.io/nsw-normal-day-monitor/**, is built from this folder.
It is served by GitHub Pages from the separate public repository
[varuntree/nsw-normal-day-monitor](https://github.com/varuntree/nsw-normal-day-monitor), which holds only the built
page (`index.html`, `data.json`, `fonts/`).

| File | What |
|---|---|
| `build_data.py` | Refreshes the AEMO demand and Open-Meteo temperatures (into `dashboard/cache/`), runs the electricity monitor of Chapter 7 (configuration in `config.json`: past-only regression forecast with weekday, time of year and temperature; FOCuS; AR(1) threshold) and the Opal monitor (Holt-Winters with EWMA, AR(1) threshold, on the fixed Opal file), and writes `data.json`. It imports the thesis code (`cpd/`, `scripts/electricity.py`, `results/calib/h_table.json`) and only reads `data/`. |
| `fetch.py` | The downloads (same sources and time rules as `live/data_fetch.py`). |
| `index.html` | The page (plain JavaScript, no external scripts). |
| `data.json` | The last build (a snapshot; the daily job rebuilds it). |
| `fonts/` | Inter and Geist Mono (SIL Open Font License; licence texts included). |
| `checks/` | `check_js.mjs` runs the detector code shipped in `index.html` and compares its alarms with the Python alarms in `data.json`; `check_truth.py` adds inserted-drop cases for the "Try it" panel; `shots.sh` takes screenshots with headless Chrome. |
| `requirements.txt` | The packages the daily job needs. |

`build_data.py` stops if its day-by-day rebuild of the monitor disagrees with `cpd.monitor.run`, or if its forecast
differs from `scripts/electricity.py`. `live/update.py` runs the same electricity monitor and writes the simpler
page `live/dashboard.html` (used for the screenshot figure).

## Run it

```bash
python dashboard/build_data.py              # add --no-refresh to skip the downloads
node dashboard/checks/check_js.mjs          # must end with PASS
python -m http.server 8000 --directory dashboard    # then open http://localhost:8000
```

## Daily job

`.github/workflows/dashboard.yml` runs `build_data.py` every day at 14:40 UTC (00:40 NEM time, just after AEMO
rewrites the current month) and again at 20:40 UTC in case a feed was late, runs `check_js.mjs`, and pushes
`index.html`, `data.json` and `fonts/` to `varuntree/nsw-normal-day-monitor`, whose Pages site is the live page.

To let it publish, create an SSH key pair, add the public key to `varuntree/nsw-normal-day-monitor` under
Settings, Deploy keys (with write access), and add the private key to this repository under Settings, Secrets and
variables, Actions, as `MONITOR_DEPLOY_KEY`. Without the secret the job still builds and checks the page and keeps
it as a workflow artifact, but publishes nothing.

- If a download fails, the page is built from the saved data and shows the older "last complete day".
- AEMO sometimes refuses requests from cloud servers. If the job fails with HTTP 403, run `build_data.py` on
  another machine and push the two files to `varuntree/nsw-normal-day-monitor` by hand.
- GitHub turns off scheduled workflows after 60 days with no repository activity. Any push, or a manual run from
  the Actions tab, turns it back on.
- URL options: `?tab=opal`, `?range=120|365|all`, `?drop=10&days=20`, `?focus=tryit`.
