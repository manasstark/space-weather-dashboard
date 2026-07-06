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

from swdss.models.features import add_derived_physics_features, build_feature_frame
from swdss.models.registry import (
    DATASETS,
    kp_interval_model_path,
    metrics_path,
    model_path,
    owning_source_config,
    raw_column_for,
)


@lru_cache(maxsize=8)
def _load_metrics(dataset: str) -> dict:
    with open(metrics_path(dataset)) as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _load_model(dataset: str, variable: str, horizon: int):
    return joblib.load(model_path(dataset, variable, horizon))


@lru_cache(maxsize=4)
def _load_kp_interval_model(dataset: str):
    return joblib.load(kp_interval_model_path(dataset))


def _load_single_source_raw(config) -> pd.DataFrame:
    raw = pd.read_parquet(config.processed_parquet)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    raw = raw.rename(columns=config.raw_column_map)
    raw = raw.set_index("timestamp_utc")
    return raw


def _load_multi_source_raw(config) -> pd.DataFrame:
    """Merges live data from several other DATASETS entries (e.g. Solar
    Wind + IMF + Kp + Dst for the combined Analytics predictor) into one
    timestamp-indexed frame, each contributing via its own raw_column_map.
    Native cadences differ (minute-level Solar Wind/IMF vs hourly Kp/Dst);
    that's resolved by the caller's subsequent hourly resample, same as
    any single-source dataset.
    """
    frames = []
    for source_name in config.source_datasets:
        source_config = DATASETS[source_name]
        part = _load_single_source_raw(source_config)
        cols = list((source_config.raw_column_map or {}).values())
        frames.append(part[cols])
    return pd.concat(frames, axis=1)


def generate_predicted_ae(feature_frame: pd.DataFrame) -> pd.Series:
    """Runs the frozen, already-trained AE 1h model (models/ae/ae_1h.joblib
    — read-only, never retrained here) across every row of `feature_frame`
    that has its full set of required feature columns, producing one
    Predicted AE value per timestamp.

    This is the SAME function used both to build the one-time historical
    "predicted_ae" training column (see swdss.models.experimental) and to
    generate the live cascade's Predicted AE step (see the "experimental"
    branch in load_live_features below) — training and live inference can
    never see different generation logic this way.
    """
    meta = _load_metrics("ae")["ae_1h"]
    feature_columns = meta["feature_columns"]
    model = _load_model("ae", "ae", 1)

    predicted = pd.Series(index=feature_frame.index, dtype=float)
    usable = feature_frame.dropna(subset=feature_columns)
    if not usable.empty:
        predicted.loc[usable.index] = model.predict(usable[feature_columns])
    return predicted


def _load_experimental_hourly() -> pd.DataFrame:
    """The experimental cascade's own Solar Wind + IMF + Kp + Dst hourly
    frame, with a "predicted_ae" column merged in — generated fresh from
    the frozen AE model rather than read from any stored source, since
    there's nothing to read: predicted_ae only exists as a model output.
    """
    config = DATASETS["experimental"]
    raw = _load_multi_source_raw(config)
    base_vars = [v for v in config.feature_variables if v != "predicted_ae"]

    hourly = raw[base_vars].resample("1h").mean()
    hourly.index = hourly.index.tz_localize(None)
    hourly = hourly.interpolate(method="time")

    ae_frame = load_live_features("ae")
    predicted_ae = generate_predicted_ae(ae_frame)
    hourly["predicted_ae"] = predicted_ae.reindex(hourly.index)

    return hourly


def load_live_features(dataset: str) -> pd.DataFrame:
    """Resamples live processed data to hourly means, interpolates gaps the
    same way training did, and applies the identical lag/rolling/change
    feature engineering used to build the training CSVs. For multi-source
    datasets, merges every contributing source first.
    """
    config = DATASETS[dataset]
    feature_vars = config.feature_variables or config.variables

    if dataset == "experimental":
        hourly = _load_experimental_hourly()
    else:
        if config.source_datasets:
            raw = _load_multi_source_raw(config)
        else:
            raw = _load_single_source_raw(config)

        hourly = raw[feature_vars].resample("1h").mean()
        hourly.index = hourly.index.tz_localize(None)

        # Variables with no continuously-updating live feed (e.g. AE) can't
        # be time-interpolated past their last real observation — there's
        # nothing ahead to interpolate toward. Forward-fill those instead,
        # carrying the last known value forward; every other variable keeps
        # the normal time-interpolation for genuine live gaps.
        static_vars = [v for v in (config.static_variables or []) if v in hourly.columns]
        interp_vars = [v for v in feature_vars if v not in static_vars]
        hourly[interp_vars] = hourly[interp_vars].interpolate(method="time")
        if static_vars:
            hourly[static_vars] = hourly[static_vars].ffill()

    derived_cols = add_derived_physics_features(hourly)
    feature_vars = feature_vars + derived_cols

    frame, feature_columns = build_feature_frame(hourly, feature_vars)
    frame.attrs["feature_columns"] = feature_columns
    return frame


def latest_minute_observation(dataset: str, variable: str) -> tuple:
    """Returns (timestamp, value) for the most recent raw minute-level NOAA
    reading of `variable`, straight from the owning source's processed
    parquet — no resampling. Used to detect "a new NOAA minute has
    arrived" and to show the instantaneous reading in the live
    prediction-job terminal log. Returns (None, None) if no data yet.
    """
    config = owning_source_config(dataset, variable)
    raw_col = raw_column_for(dataset, variable)
    df = pd.read_parquet(config.processed_parquet, columns=["timestamp_utc", raw_col])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", raw_col]).sort_values("timestamp_utc")
    if df.empty:
        return None, None
    row = df.iloc[-1]
    return row["timestamp_utc"], float(row[raw_col])


def resolve_static_actual(dataset: str, variable: str, target_hour) -> float:
    """Ground-truth lookup for a `static_variables` entry (e.g. AE), which
    has no continuously-updating live feed — load_live_features forward-
    fills its last known value indefinitely, and that forward-filled value
    must never be reported as if it were the real observation for a given
    target hour. This instead reads the variable's own raw historical
    source directly (pre-ffill) and returns None unless that source has
    genuinely been refreshed to cover target_hour.
    """
    config = owning_source_config(dataset, variable)
    raw_col = raw_column_for(dataset, variable)
    raw = _load_single_source_raw(config)
    hourly = raw[raw_col].resample("1h").mean()
    hourly.index = hourly.index.tz_localize(None)

    ts = pd.Timestamp(target_hour)
    ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    ts = ts.floor("h")
    if ts not in hourly.index:
        return None
    value = hourly.loc[ts]
    return None if pd.isna(value) else float(value)


def predict_kp_interval(dataset: str = "analytics") -> dict:
    """Kp on the Analytics page follows NOAA's real publishing cadence:
    a new official value only every 3 hours (00, 03, 06, ... UTC). This
    always targets the NEXT such boundary after the latest available
    hourly bucket, using the single interval-aware model — never an
    arbitrary hourly horizon.
    """
    metrics_doc = _load_metrics(dataset)
    if "kp_interval" not in metrics_doc:
        raise ValueError(f"No trained Kp interval model for {dataset}. Run swdss.models.train_kp_interval_model.")
    meta = metrics_doc["kp_interval"]
    feature_columns = meta["feature_columns"]

    frame = load_live_features(dataset)
    usable = frame.dropna(subset=feature_columns)
    if usable.empty:
        raise ValueError(f"Not enough live history to build features for {dataset}/kp.")

    latest = usable.iloc[-1]
    observed_at = usable.index[-1]

    model = _load_kp_interval_model(dataset)
    X = latest[feature_columns].to_frame().T
    predicted_value = float(model.predict(X)[0])

    current_value = float(latest["kp"])
    change = predicted_value - current_value
    if abs(change) < 1e-9:
        trend = "Stable"
    elif change > 0:
        trend = "Increasing"
    else:
        trend = "Decreasing"

    next_block_start = observed_at.floor("3h") + pd.Timedelta(hours=3)
    recent_series = frame["kp"].dropna().tail(48)

    return {
        "dataset": dataset,
        "variable": "kp",
        "horizon": "interval",
        "current_value": current_value,
        "predicted_value": predicted_value,
        "change": change,
        "trend": trend,
        "model_name": meta["algorithm"],
        "metrics": {"r2": meta["r2"], "mae": meta["mae"], "rmse": meta["rmse"]},
        "observed_at": observed_at,
        "predicted_for": next_block_start,
        "recent_series": recent_series,
    }


def predict(dataset: str, variable: str, horizon: int) -> dict:
    # Both the production "analytics" and experimental cascade Kp models
    # follow NOAA's real 3-hour publishing cadence, not an arbitrary
    # hourly horizon — see predict_kp_interval.
    if dataset in ("analytics", "experimental") and variable == "kp":
        return predict_kp_interval(dataset)

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
