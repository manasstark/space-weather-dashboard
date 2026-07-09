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
    new_cols = {}
    for column in columns:
        for lag in lags:
            name = f"{column}_lag{lag}h"
            new_cols[name] = df[column].shift(lag)
    df[list(new_cols.keys())] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols.keys())


def add_rolling_features(df: pd.DataFrame, columns: list[str], window: int = ROLLING_WINDOW) -> list[str]:
    new_cols = {}
    for column in columns:
        new_cols[f"{column}_{window}h"] = df[column].rolling(window).mean()
        new_cols[f"{column}_{window}h_std"] = df[column].rolling(window).std()
    df[list(new_cols.keys())] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols.keys())


def add_change_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    new_cols = {f"{column}_change": df[column].diff() for column in columns}
    df[list(new_cols.keys())] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols.keys())


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

    All new columns are collected into one dict and joined via pd.concat
    in a single operation — avoids the fragmentation that builds up when
    columns are added incrementally across multiple calls.
    """
    base = df.copy()
    new_cols: dict[str, pd.Series] = {}

    lag_names: list[str] = []
    for column in columns:
        for lag in LAGS:
            name = f"{column}_lag{lag}h"
            new_cols[name] = base[column].shift(lag)
            lag_names.append(name)

    rolling_names: list[str] = []
    for column in columns:
        mean_name = f"{column}_{ROLLING_WINDOW}h"
        std_name = f"{column}_{ROLLING_WINDOW}h_std"
        new_cols[mean_name] = base[column].rolling(ROLLING_WINDOW).mean()
        new_cols[std_name] = base[column].rolling(ROLLING_WINDOW).std()
        rolling_names += [mean_name, std_name]

    change_names: list[str] = []
    for column in columns:
        name = f"{column}_change"
        new_cols[name] = base[column].diff()
        change_names.append(name)

    frame = pd.concat([base, pd.DataFrame(new_cols, index=base.index)], axis=1)
    feature_columns = list(columns) + lag_names + rolling_names + change_names
    return frame, feature_columns
