"""Feature engineering shared by training (notebooks/Solar_Wind.ipynb,
notebooks/IMF.ipynb) and live inference (predict.py).

This must stay byte-for-byte identical to the logic the notebooks used to
produce solar_wind_features.csv / imf_features.csv, or trained models will
see different features at inference time than they were trained on.
"""

import pandas as pd

from swdss.physics.core import add_derived_physics_features as _physics_add_derived_physics_features

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
    """Adds Sun-Earth coupling features (VBz, Ey, Dynamic Pressure) in
    memory — never as separate datasets — so training and live inference
    always compute them identically from the same merged Solar Wind +
    IMF (+ geomagnetic) frame, with zero risk of train/serve drift.

    Delegates to swdss.physics.core — the project's Physics Engine and
    now the single canonical implementation of these three formulas.
    This wrapper exists purely for backward compatibility: every existing
    caller (train.py, predict.py, experimental.py, the v2 refresh
    scripts) imports add_derived_physics_features from THIS module by
    name, so the import path stays stable while the formula's actual
    home moves to swdss.physics.core. Verified byte-identical against
    the pre-migration inline implementation across every production
    dataset (analytics, ae, experimental, solar_wind, imf) before this
    delegation was wired in — see the Physics Engine migration's Phase 0
    verification.
    """
    return _physics_add_derived_physics_features(df)


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
