"""Production forecast matrix — the Operational Forecast Engine's single
source of truth for which (dataset, variable, horizon) combinations get
automatically, continuously forecast.

This does not define any new model — it only states which of the
already-trained production models (registry.DATASETS / models/*/metrics.
json) the engine keeps a live job running against at all times. Kp is
special-cased to the literal string "interval" rather than one of
HORIZONS, matching predict.predict_kp_interval's own contract (NOAA's
next official 3-hour window, not a fixed hourly horizon) — see
orchestrator.run_forecast_cycle for how that's resolved into the
horizon=1 placeholder start_job/prediction_panel already use for Kp.
"""

from swdss.models.registry import HORIZONS

PRODUCTION_MATRIX = [
    ("solar_wind", "speed", HORIZONS),
    ("solar_wind", "density", HORIZONS),
    ("solar_wind", "temperature", HORIZONS),
    ("imf", "bt", HORIZONS),
    ("imf", "bx_gsm", HORIZONS),
    ("imf", "by_gsm", HORIZONS),
    ("imf", "bz_gsm", HORIZONS),
    ("analytics", "dst", HORIZONS),
    ("analytics", "kp", "interval"),
    ("ae", "ae", HORIZONS),
]
