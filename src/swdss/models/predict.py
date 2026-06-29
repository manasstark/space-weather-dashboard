"""Live inference for the dashboard's Prediction tabs.

Loads the latest minute-level NOAA observations, reproduces the exact
feature engineering used during training (swdss.models.features), loads
the model+metrics chosen by training for the requested (variable, horizon),
and returns a single prediction record ready for display.
"""

import json
from functools import lru_cache

import joblib
import pandas as pd

from swdss.models.features import build_feature_frame
from swdss.models.registry import DATASETS, metrics_path, model_path, raw_column_for


@lru_cache(maxsize=8)
def _load_metrics(dataset: str) -> dict:
    with open(metrics_path(dataset)) as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _load_model(dataset: str, variable: str, horizon: int):
    return joblib.load(model_path(dataset, variable, horizon))


def load_live_features(dataset: str) -> pd.DataFrame:
    """Resamples the live minute-level processed parquet to hourly means,
    renames columns to match training names, interpolates gaps the same
    way training did, and applies the identical lag/rolling/change feature
    engineering used to build the training CSVs.
    """
    config = DATASETS[dataset]
    raw = pd.read_parquet(config.processed_parquet)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    raw = raw.rename(columns=config.raw_column_map)
    raw = raw.set_index("timestamp_utc")

    hourly = raw[list(config.variables)].resample("1h").mean()
    hourly.index = hourly.index.tz_localize(None)
    hourly = hourly.interpolate(method="time")

    frame, feature_columns = build_feature_frame(hourly, config.variables)
    frame.attrs["feature_columns"] = feature_columns
    return frame


def latest_minute_observation(dataset: str, variable: str) -> tuple:
    """Returns (timestamp, value) for the most recent raw minute-level NOAA
    reading of `variable`, straight from the processed parquet — no
    resampling. Used to detect "a new NOAA minute has arrived" and to show
    the instantaneous reading in the live prediction-job terminal log.
    Returns (None, None) if no data is available yet.
    """
    config = DATASETS[dataset]
    raw_col = raw_column_for(dataset, variable)
    df = pd.read_parquet(config.processed_parquet, columns=["timestamp_utc", raw_col])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", raw_col]).sort_values("timestamp_utc")
    if df.empty:
        return None, None
    row = df.iloc[-1]
    return row["timestamp_utc"], float(row[raw_col])


def predict(dataset: str, variable: str, horizon: int) -> dict:
    metrics_doc = _load_metrics(dataset)
    key = f"{variable}_{horizon}h"
    if key not in metrics_doc:
        raise ValueError(f"No trained model for {dataset}/{key}. Run swdss.models.train first.")
    meta = metrics_doc[key]
    feature_columns = meta["feature_columns"]

    frame = load_live_features(dataset)

    usable = frame.dropna(subset=feature_columns)
    if usable.empty:
        raise ValueError(f"Not enough live history to build features for {dataset}/{variable}.")

    latest = usable.iloc[-1]
    observed_at = usable.index[-1]

    model = _load_model(dataset, variable, horizon)
    X = latest[feature_columns].to_frame().T
    predicted_value = float(model.predict(X)[0])

    current_value = float(latest[variable])
    change = predicted_value - current_value
    if abs(change) < 1e-9:
        trend = "Stable"
    elif change > 0:
        trend = "Increasing"
    else:
        trend = "Decreasing"

    recent_series = frame[variable].dropna().tail(48)

    return {
        "dataset": dataset,
        "variable": variable,
        "horizon": horizon,
        "current_value": current_value,
        "predicted_value": predicted_value,
        "change": change,
        "trend": trend,
        "model_name": meta["algorithm"],
        "metrics": {"r2": meta["r2"], "mae": meta["mae"], "rmse": meta["rmse"]},
        "observed_at": observed_at,
        "predicted_for": observed_at + pd.Timedelta(hours=horizon),
        "recent_series": recent_series,
    }
