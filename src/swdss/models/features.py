"""Feature engineering shared by training (notebooks/Solar_Wind.ipynb,
notebooks/IMF.ipynb) and live inference (predict.py).

This must stay byte-for-byte identical to the logic the notebooks used to
produce solar_wind_features.csv / imf_features.csv, or trained models will
see different features at inference time than they were trained on.
"""

import pandas as pd

LAGS = [1, 3, 6, 12, 24]
ROLLING_WINDOW = 24


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int] = LAGS) -> list[str]:
    created = []
    for column in columns:
        for lag in lags:
            name = f"{column}_lag{lag}h"
            df[name] = df[column].shift(lag)
            created.append(name)
    return created


def add_rolling_features(df: pd.DataFrame, columns: list[str], window: int = ROLLING_WINDOW) -> list[str]:
    created = []
    for column in columns:
        mean_name = f"{column}_{window}h"
        std_name = f"{column}_{window}h_std"
        df[mean_name] = df[column].rolling(window).mean()
        df[std_name] = df[column].rolling(window).std()
        created.extend([mean_name, std_name])
    return created


def add_change_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    created = []
    for column in columns:
        name = f"{column}_change"
        df[name] = df[column].diff()
        created.append(name)
    return created


def add_derived_physics_features(df: pd.DataFrame) -> list[str]:
    """Adds Sun-Earth coupling features in memory — never as separate
    datasets — so training and live inference always compute them
    identically from the same merged Solar Wind + IMF (+ geomagnetic)
    frame, with zero risk of train/serve drift.

    - VBz = Speed x min(Bz, 0): geoeffective coupling function (e.g.
      Burton et al. 1975). Southward IMF (negative Bz) drives dayside
      reconnection; energy injection into the ring current scales with
      how fast the solar wind is moving it in. Positive (northward, non-
      geoeffective) Bz is clipped to 0, so VBz spikes more negative
      exactly when conditions are most likely to drive a storm.
    - Ey = -Speed x Bz x 1e-3 (mV/m): the interplanetary dawn-dusk
      electric field. Southward Bz makes this positive — the convention
      used across space weather literature for "geoeffective E-field".
    - Dynamic Pressure = 1.6726e-6 x Density x Speed^2 (nPa): solar wind
      ram pressure on the magnetopause — same formula already used by
      the dashboard's own Heliosphere > Dynamic Pressure panel.

    Each is only added when its required inputs are present — a no-op
    for datasets that don't have all of them, e.g. standalone Solar Wind
    (no Bz) or standalone IMF (no Speed/Density).
    """
    created = []
    has_speed = "speed" in df.columns
    has_bz = "bz_gsm" in df.columns
    has_density = "density" in df.columns

    if has_speed and has_bz:
        df["vbz"] = df["speed"] * df["bz_gsm"].clip(upper=0)
        created.append("vbz")
        df["ey"] = -df["speed"] * df["bz_gsm"] * 1e-3
        created.append("ey")

    if has_speed and has_density:
        df["dynamic_pressure"] = 1.6726e-6 * df["density"] * df["speed"] ** 2
        created.append("dynamic_pressure")

    return created


def build_feature_frame(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Adds lag/rolling/change features for `columns` onto a copy of df.

    Returns (frame, feature_columns) where feature_columns is the base
    variables followed by all derived features, in a fixed order.
    """
    frame = df.copy()
    lag_cols = add_lag_features(frame, columns)
    rolling_cols = add_rolling_features(frame, columns)
    change_cols = add_change_features(frame, columns)
    feature_columns = list(columns) + lag_cols + rolling_cols + change_cols
    return frame, feature_columns
