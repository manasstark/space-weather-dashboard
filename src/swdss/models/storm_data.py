"""Historical storm data acquisition — shared by the Storm Backtest and
Storm Learning research tools (dashboard/lib/storm_lab.py).

Why this exists: every accuracy number this project has ever reported
(R², skill scores, confidence calibration) was computed on live NOAA data
from 2026, which happens to have been geomagnetically quiet the entire
time. That means the engine has never actually been checked against a
real storm. This module pulls real historical data for named, independently
documented storms from NASA's OMNI2 archive — the same public source
already used by scripts/refresh/01_download_parse_2026.py — so both tools
can evaluate against genuine storm conditions instead of hoping.

Column layout and fill values below are copied from
scripts/refresh/01_download_parse_2026.py (verified there against NASA's
own OMNI2 format spec: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text)
and generalized here to accept any year, not just 2026.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests

from swdss.paths import RAW_DIR

OMNI_URL_TEMPLATE = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
STORM_CACHE_DIR = RAW_DIR / "storm_omni"

COL = {
    "bt": 8,
    "bx_gsm": 12,
    "by_gsm": 15,
    "bz_gsm": 16,
    "temperature": 22,
    "density": 23,
    "speed": 24,
    "kp": 38,
    "dst": 40,
    "ae": 41,
}

FILL = {
    "bt": 999.9,
    "bx_gsm": 999.9,
    "by_gsm": 999.9,
    "bz_gsm": 999.9,
    "temperature": 9999999.0,
    "density": 999.9,
    "speed": 9999.0,
    "kp": 99,
    "dst": 99999,
    "ae": 9999,
}

# Named, independently-documented storms across the last ~10 years plus one
# just outside it (St. Patrick's Day 2015) kept anyway for its significance.
# `in_training_range` flags whether the storm's dates fall inside the
# 2023-Jun2026 window the current production models were trained on
# (data/features/training/*.csv) — True means a result here is weaker
# evidence (the model may have already seen this exact event), False means
# it's a genuine blind test the model has never encountered in any form.
NAMED_STORMS = {
    "gannon_2024": {
        "label": "May 2024 \"Gannon Storm\"",
        "peak_date": "2024-05-11",
        "window_start": "2024-05-09",
        "window_end": "2024-05-12",
        "years": [2024],
        "dst_min_nT": -412,
        "g_scale": "G5 Extreme",
        "notes": "Strongest geomagnetic storm since 2003; aurora visible at very low latitudes.",
        "in_training_range": True,
    },
    "october_2024": {
        "label": "October 2024 storm",
        "peak_date": "2024-10-11",
        "window_start": "2024-10-09",
        "window_end": "2024-10-13",
        "years": [2024],
        "dst_min_nT": -355,
        "g_scale": "G4-G5",
        "notes": "Second major storm of 2024, days after the Gannon storm.",
        "in_training_range": True,
    },
    "april_2023": {
        "label": "April 2023 storm",
        "peak_date": "2023-04-24",
        "window_start": "2023-04-22",
        "window_end": "2023-04-25",
        "years": [2023],
        "dst_min_nT": -213,
        "g_scale": "G4 Severe",
        "notes": "Early in Solar Cycle 25's ramp-up toward maximum.",
        "in_training_range": True,
    },
    "september_2017": {
        "label": "September 2017 storm",
        "peak_date": "2017-09-08",
        "window_start": "2017-09-06",
        "window_end": "2017-09-09",
        "years": [2017],
        "dst_min_nT": -142,
        "g_scale": "G4 Severe",
        "notes": "Tied to the X9.3 flare, the largest of Solar Cycle 24.",
        "in_training_range": False,
    },
    "august_2018": {
        "label": "August 2018 storm",
        "peak_date": "2018-08-26",
        "window_start": "2018-08-24",
        "window_end": "2018-08-27",
        "years": [2018],
        "dst_min_nT": -174,
        "g_scale": "G3-G4",
        "notes": "Cycle 24 decline phase.",
        "in_training_range": False,
    },
    "st_patricks_2015": {
        "label": "St. Patrick's Day storm (2015)",
        "peak_date": "2015-03-17",
        "window_start": "2015-03-16",
        "window_end": "2015-03-18",
        "years": [2015],
        "dst_min_nT": -223,
        "g_scale": "G4 Severe",
        "notes": "11 years back, just outside a strict 10-year window, kept for its significance.",
        "in_training_range": False,
    },
}


def _parse_omni_text(text: str, year: int) -> pd.DataFrame:
    records = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 42:
            continue
        row_year, doy, hour = int(parts[0]), int(parts[1]), int(parts[2])
        if row_year != year:
            continue

        dt = datetime(row_year, 1, 1) + timedelta(days=doy - 1, hours=hour)
        row: dict = {"datetime": dt}
        for var, col in COL.items():
            try:
                val = float(parts[col])
            except (ValueError, IndexError):
                val = float("nan")
            if val == FILL[var]:
                val = float("nan")
            row[var] = val
        records.append(row)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def download_omni_year(year: int, force: bool = False) -> pd.DataFrame:
    """Downloads (or reads a locally cached copy of) one full year of
    NASA OMNI2 hourly data. Cached under data/raw/storm_omni/ so re-running
    a backtest against the same storm never re-downloads.
    """
    cache_path = STORM_CACHE_DIR / f"omni2_{year}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, parse_dates=["datetime"])

    url = OMNI_URL_TEMPLATE.format(year=year)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    df = _parse_omni_text(response.text, year)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_storm_window(storm_key: str, lookback_hours: int = 48) -> pd.DataFrame:
    """Returns raw OMNI2 rows covering a named storm's window, extended
    `lookback_hours` earlier than window_start so the 24h lag/rolling
    features swdss.models.features needs have real history to compute from
    before the first in-window prediction — the same reason the earlier
    "100 values before the storm" idea was worth keeping, just used to
    build features for the existing model rather than to train a new one.
    """
    storm = NAMED_STORMS[storm_key]
    frames = [download_omni_year(year) for year in storm["years"]]
    df = pd.concat(frames).drop_duplicates(subset="datetime").sort_values("datetime")

    start = pd.Timestamp(storm["window_start"]) - pd.Timedelta(hours=lookback_hours)
    end = pd.Timestamp(storm["window_end"]) + pd.Timedelta(hours=24)
    window = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
    return window.reset_index(drop=True)


def build_base_df(dataset_key: str, omni_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Turns a raw OMNI2 pull into the same shape swdss.models.train._load_base_df
    builds from a training CSV: select this dataset's feature variables,
    apply its scale_factors (e.g. Kp/10), and add the derived physics
    features (VBz/Ey/Dynamic Pressure) — identical logic, different source,
    so a trained model sees the exact same feature semantics either way.

    Returns (base_df, feature_vars) where feature_vars includes the derived
    physics columns — the same two-value contract train._load_base_df uses,
    since build_feature_frame needs the full list to compute lag/rolling/
    change features for the derived columns too, not just the raw ones.
    """
    from swdss.models.features import add_derived_physics_features
    from swdss.models.registry import DATASETS

    config = DATASETS[dataset_key]
    base = omni_df.set_index("datetime").sort_index()
    feature_vars = config.feature_variables or config.variables
    base_df = base[[c for c in feature_vars if c in base.columns]].copy()

    for column, factor in (config.scale_factors or {}).items():
        if column in base_df.columns:
            base_df[column] = base_df[column] / factor

    derived_cols = add_derived_physics_features(base_df)
    return base_df, feature_vars + derived_cols


def build_context_frame(omni_df: pd.DataFrame) -> pd.DataFrame:
    """A small Kp/Dst/AE frame (Kp scaled to its natural 0-9 range) used
    purely to tag each historical hour's real activity regime via
    swdss.engine.outlook.classify_activity_regime — independent of which
    dataset/variable is actually being backtested, since Solar Wind/IMF
    alone don't carry Kp/Dst/AE columns.
    """
    context = omni_df.set_index("datetime").sort_index()[["kp", "dst", "ae"]].copy()
    context["kp"] = context["kp"] / 10.0
    return context


def _kp_next_interval_map(base_df: pd.DataFrame) -> tuple:
    """Shared by build_target_series/build_persistence_series: one Kp value
    per 3h block (00-03, 03-06, ... UTC), keyed by that block's own start.
    """
    block_start = base_df.index.floor("3h")
    block_kp = base_df["kp"].groupby(block_start).first()
    return block_start, block_kp


def build_target_series(variable: str, horizon, base_df: pd.DataFrame) -> pd.Series:
    """The same target definition production training uses. Every ordinary
    variable/horizon is a simple `.shift(-horizon)` — but Kp on the
    "analytics"/"experimental" datasets follows NOAA's real 3-hour
    publishing cadence instead (see swdss.models.train.train_kp_interval_model):
    the target is always "the next official interval's eventual Kp,"
    identical for every hourly row inside the same current block, with the
    lead time naturally varying 1-3h row by row. Reproduced here exactly so
    Storm Backtest/Learning score the interval model against the same
    target it was actually trained on, not a fixed-horizon substitute.
    """
    if variable == "kp" and horizon == "interval":
        block_start, block_kp = _kp_next_interval_map(base_df)
        next_block_start = pd.Series(block_start + pd.Timedelta(hours=3), index=base_df.index)
        return next_block_start.map(block_kp)
    return base_df[variable].shift(-horizon)


def build_persistence_series(variable: str, horizon, base_df: pd.DataFrame) -> pd.Series:
    """The "value known at issuance persists to target" baseline — the
    same definition swdss.engine.skill uses, generalized for Kp's block
    cadence: the persistence anchor is this row's OWN current-block Kp,
    not a shifted value.
    """
    if variable == "kp" and horizon == "interval":
        block_start, block_kp = _kp_next_interval_map(base_df)
        return pd.Series(block_start, index=base_df.index).map(block_kp)
    return base_df[variable]
