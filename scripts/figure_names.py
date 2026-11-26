"""Names used in the figures of scripts/thesis_figures.py: one name per thing. No letters A/B, no "-r", no "robust"."""
FORECAST = {  # full name
    "raw": "no forecast", "oracle": "perfect forecast (true pattern removed)",
    "A": "same-weekday difference", "B": "median of four",
    "HW": "Holt-Winters", "YM": "yearly memory",
    "MSTL_p": "MSTL refitted on the past", "REG_p": "regression refitted on the past",
    "STL": "retrospective STL", "MSTL": "retrospective MSTL", "REG": "retrospective regression",
    "REGT_p": "regression with temperature, refitted on the past"}
FORECAST_SHORT = {  # axis ticks and table rows
    "raw": "none", "oracle": "perfect forecast", "A": "same-weekday difference", "B": "median of four",
    "HW": "Holt-Winters", "YM": "yearly memory", "MSTL_p": "MSTL (past)", "REG_p": "regression (past)",
    "STL": "STL (hindsight)", "MSTL": "MSTL (hindsight)", "REG": "regression (hindsight)"}
DETECTOR = {"CUSUM": "CUSUM", "EWMA": "EWMA", "FOCuS": "FOCuS", "FOCuS-AR": "FOCuS-AR", "CPM": "CPM",
            "CUSUM-r": "capped CUSUM", "EWMA-r": "capped EWMA", "FOCuS-r": "capped FOCuS"}
THRESHOLD = {"ar1": "AR(1) threshold", "indep": "independent threshold"}
RULE = {"frozen": "main rule", "r0_7_dir": "direction-aware check"}
RULE_SHORT = {"frozen": "main", "r0_7_dir": "check"}
SURPRISE = "forecast error (log scale)"
NON_EVENT = "alarms a year away from any scored change"
FIXED = "the fixed monitor"
CAP_GLOSS = "capped: each day limited to ±3"


def monitor(m, d, short=False):
    return f"{(FORECAST_SHORT if short else FORECAST)[m]} + {DETECTOR[d]}"
