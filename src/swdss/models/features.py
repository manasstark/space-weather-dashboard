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
