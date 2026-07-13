"""IMF Research Laboratory — experimental Bz/Bt/Bx/By model comparison,
completely isolated from the production IMF predictor
(swdss.models.predict / swdss.models.train / swdss.models.jobs).

Key data-contract difference from production, worth stating up front:
this pipeline trains on RAW MINUTE-LEVEL Solar Wind + IMF data, not the
HOURLY-resampled frame production uses. Bz's physics (southward
duration, clock-angle evolution, LSTM/GRU sequence windows) genuinely
lives at minute timescales — resampling to hourly first would destroy
exactly the information these features and sequence models exist to
capture. See imf_physics_features.py for the full feature set built on
top of this minute-level frame.

Production safety: nothing in this module is ever imported by predict.py
or jobs.py, and nothing here writes to models/<dataset>/ (the production
model directories) or metrics.json (the production metrics file).
Training-run artifacts live under models/imf_research/<run_id>/, tracked
in their own JSON registry (RUNS_REGISTRY_PATH) — a completely separate
namespace. "Promoting" a run only sets a label on its registry entry; it
never touches the production path. A human engineer must manually wire a
promoted model into predict.py themselves for it to ever affect a live
forecast.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from swdss.models.features import add_change_features, add_lag_features, add_rolling_features
from swdss.models.imf_physics_features import (
    add_all_imf_physics_features,
    add_minute_change_features,
    add_minute_lag_features,
    add_minute_rolling_features,
)
from swdss.models.registry import HORIZONS
from swdss.paths import DATA_DIR, MODELS_DIR

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# TensorFlow is deliberately NEVER imported at module level here (or
# anywhere else in this process) — only checked for via find_spec, which
# locates the package without executing it. Empirically confirmed:
# having scikit-learn/XGBoost/LightGBM/CatBoost AND TensorFlow imported
# in the same process makes TF's model.fit hang indefinitely on real
# training work — reproduced even with plain synthetic data, and
# unaffected by import order, OMP_NUM_THREADS=1, KMP_DUPLICATE_LIB_OK, or
# explicit tf.config.threading limits. Keras training instead runs in a
# fully separate subprocess (imf_research_keras_worker.py) that imports
# only TensorFlow, sidestepping the conflict entirely. See that module's
# docstring for the full story.
KERAS_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


RESEARCH_MODELS_DIR = MODELS_DIR / "imf_research"
RUNS_REGISTRY_PATH = DATA_DIR / "predictions" / "imf_research_runs.json"

TEST_FRACTION = 0.2

TARGET_OPTIONS = {
    "Bz": "bz_gsm",
    "Bt": "bt",
    "Bx": "bx_gsm",
    "By": "by_gsm",
    "Solar Wind Speed": "speed",
    "Density": "density",
    "Temperature": "temperature",
}
DEFAULT_TARGET = "Bz"
BASELINE_COLUMNS = ["speed", "density", "temperature", "bx_gsm", "by_gsm", "bz_gsm", "bt"]
SOLAR_WIND_BASE_COLUMNS = ["speed", "density", "temperature"]
IMF_BASE_COLUMNS = ["bt", "bx_gsm", "by_gsm", "bz_gsm"]

# Forecast Granularity/Horizon — the axis this module was missing before:
# "Minute" targets shift(-horizon) minutes on the minute-native frame
# (horizon in MINUTE_HORIZONS); "Hourly" targets shift(-horizon) hours on
# an hourly-resampled frame built the same way production's is (horizon
# in HOURLY_HORIZONS, reused directly from swdss.models.registry.HORIZONS
# so "Hourly, 1h" is the exact same target definition production's
# bz_gsm_1h model was trained against — see train_research_model).
GRANULARITY_OPTIONS = ["Minute", "Hourly"]
DEFAULT_GRANULARITY = "Minute"
MINUTE_HORIZONS = [1, 5, 15, 30, 60]
HOURLY_HORIZONS = HORIZONS
DEFAULT_HORIZON = 1

TABULAR_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso",
    "ElasticNet",
    "Random Forest",
    "XGBoost",
]
if LIGHTGBM_AVAILABLE:
    TABULAR_MODELS.append("LightGBM")
if CATBOOST_AVAILABLE:
    TABULAR_MODELS.append("CatBoost")
TABULAR_MODELS += ["SVR", "MLP"]

SEQUENCE_MODELS = ["LSTM", "GRU"] if KERAS_AVAILABLE else []

# Registered but not trainable yet — surfaced in the UI as disabled
# entries so the model selector's shape never has to change when one of
# these is eventually implemented; see module docstring's "Future
# Expansion" contract.
FUTURE_MODELS = ["Transformer", "Temporal Convolution Network (TCN)", "Physics-Informed Neural Network"]

ALL_TRAINABLE_MODELS = TABULAR_MODELS + SEQUENCE_MODELS

SEQUENCE_LENGTH_OPTIONS = [30, 60, 120]
DEFAULT_SEQUENCE_LENGTH = 60

# ---- Experiment 3: Solar Wind Feature-Set Options -------------------------
# Each entry defines which BASE columns feed into the lag/rolling/change
# pipeline for an Hourly-granularity Bz forecast experiment.  Physics
# features (Ey, VBz, dynamic pressure, clock angle) are handled separately
# in add_hourly_physics_features() — they are only available when at least
# speed + bz_gsm (for Ey/VBz) or speed + density (for dynamic pressure) are
# included in the base set, so the builder checks column availability.
FEATURE_SET_OPTIONS = {
    "IMF Only": IMF_BASE_COLUMNS,
    "IMF + Speed": IMF_BASE_COLUMNS + ["speed"],
    "IMF + Speed + Density": IMF_BASE_COLUMNS + ["speed", "density"],
    "IMF + All Solar Wind": list(BASELINE_COLUMNS),  # bt/bx/by/bz + speed/density/temperature
}
DEFAULT_FEATURE_SET = "IMF + All Solar Wind"


# ---- Experiment 4 & 5: Hourly Dynamics + Physics feature builders ----------

_HOURLY_SHORT_WINDOWS = [3, 6, 12]  # hours — in addition to the standard 24h


def add_hourly_dynamics_features(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Short-term dynamics features for Hourly-granularity experiments.

    Adds rolling min/max over 3/6/12h windows, linear slope (OLS over last
    6 hours), and second-difference acceleration.  All computed in-place and
    returned as a name list — same calling convention as the existing
    add_*_features family so callers don't need special cases.

    These are the Experiment-4 features: shorter rolling windows, volatility,
    rolling min/max, slope/gradient/acceleration — computed at HOURLY cadence
    (matching the production pipeline's native resolution) so results are
    directly comparable to a baseline Hourly run.
    """
    created = []
    for col in columns:
        for window in _HOURLY_SHORT_WINDOWS:
            mn = f"{col}_{window}h_min"
            mx = f"{col}_{window}h_max"
            std = f"{col}_{window}h_std"
            frame[mn] = frame[col].rolling(window, min_periods=1).min()
            frame[mx] = frame[col].rolling(window, min_periods=1).max()
            frame[std] = frame[col].rolling(window, min_periods=1).std()
            created += [mn, mx, std]
        # Linear slope: OLS gradient over the last 6h (coefficient of time)
        slope_name = f"{col}_slope6h"
        frame[slope_name] = (
            frame[col]
            .rolling(6, min_periods=2)
            .apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 2 else np.nan, raw=True)
        )
        created.append(slope_name)
        # Acceleration: second difference (Δ² = difference of difference)
        accel_name = f"{col}_accel"
        frame[accel_name] = frame[col].diff().diff()
        created.append(accel_name)
    return created


def add_hourly_physics_features(frame: pd.DataFrame) -> list[str]:
    """Physics-derived features computable from hourly data, via
    swdss.physics — the Physics Engine and this project's single
    canonical implementation of these formulas.

    Ey, VBz, Dynamic Pressure, Clock Angle, and Clock Angle Rate can all be
    computed from hourly data because they are point-in-time quantities (not
    "streak" counters that need minute-level resolution like southward_duration
    or integrated_southward_bz).  Column availability is checked — each
    feature is only added when its required inputs are present.

    southward_hours_24h/strong_southward_hours_24h are deliberately NOT
    swdss.physics.core.southward_duration_series/strong_southward_duration_series
    (those are consecutive-STREAK counts) — these are a rolling COUNT of
    southward hours within the trailing 24h window regardless of
    consecutiveness, a distinct quantity kept local to preserve this
    caller's exact existing values.

    Note: integrated_southward_bz_24h now uses the engine's default
    rolling-window NaN behavior (requires the full 24h window before
    producing a value) rather than this function's previous min_periods=1
    (produced a value from as little as 1 hour of data) — a minor,
    documented behavior change affecting only the first 23 rows of the
    ~3-year historical dataset.
    """
    from swdss.physics import core as physics_core
    from swdss.physics import geometry as physics_geometry

    created = []
    has_speed = "speed" in frame.columns
    has_bz = "bz_gsm" in frame.columns
    has_density = "density" in frame.columns
    has_by = "by_gsm" in frame.columns

    if has_speed and has_bz:
        frame["ey_h"] = physics_core.ey_series(frame["speed"], frame["bz_gsm"])
        frame["vbz_h"] = physics_core.vbz_series(frame["speed"], frame["bz_gsm"])
        created += ["ey_h", "vbz_h"]
    if has_speed and has_density:
        frame["dyn_pressure_h"] = physics_core.dynamic_pressure_series(frame["density"], frame["speed"])
        created.append("dyn_pressure_h")
    if has_by and has_bz:
        frame["clock_angle_h"] = physics_geometry.clock_angle_series(frame["by_gsm"], frame["bz_gsm"])
        frame["clock_angle_rate_h"] = physics_geometry.clock_angle_rate_series(frame["clock_angle_h"])
        created += ["clock_angle_h", "clock_angle_rate_h"]
    if has_bz:
        # Rolling count of southward hours in last 24h — hourly analog of
        # minute-level southward_duration_min. See docstring: NOT the
        # engine's consecutive-streak Southward Duration.
        frame["southward_hours_24h"] = (frame["bz_gsm"] < 0).rolling(24, min_periods=1).sum()
        frame["strong_southward_hours_24h"] = (frame["bz_gsm"] < -5).rolling(24, min_periods=1).sum()
        frame["integrated_southward_bz_24h"] = physics_core.integrated_southward_bz_series(frame["bz_gsm"], 24)
        created += ["southward_hours_24h", "strong_southward_hours_24h", "integrated_southward_bz_24h"]
    return created


# ---- Persistence Benchmark (Experiment 2) ---------------------------------

_BENCHMARK_KEY = "__persistence_benchmark__"


def compute_persistence_benchmark(
    target_label: str = "Bz",
    horizon: int = 1,
    granularity: str = "Hourly",
) -> dict:
    """Persistence forecast: predict next value = last observed value.

    This is the "naive" lower-bound that any useful ML model must beat.
    Stores the result permanently in the runs registry under a fixed
    synthetic run_id so it is always retrievable without rerunning, and
    so the UI can display it alongside real model runs.

    Returns the benchmark record (same schema as a train_research_model
    run record, with model_type="Persistence Baseline").
    """
    if target_label not in TARGET_OPTIONS:
        raise ValueError(f"Unknown target: {target_label}")
    target_col = TARGET_OPTIONS[target_label]
    valid_horizons = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    if horizon not in valid_horizons:
        raise ValueError(f"Horizon {horizon} not valid for {granularity} — expected one of {valid_horizons}.")

    # Use the same frame production trains on so the test split is
    # identical to a real model run at the same granularity/horizon.
    frame, _ = load_research_frame(granularity)
    frame = frame.copy()
    frame["__target__"] = frame[target_col].shift(-horizon)
    frame = frame.dropna(subset=[target_col, "__target__"])

    split_idx = int(len(frame) * (1 - TEST_FRACTION))
    y_test = frame["__target__"].iloc[split_idx:].to_numpy()
    # Persistence: predict current value → next value
    y_pred = frame[target_col].iloc[split_idx:].to_numpy()

    metrics = compute_metrics(y_test, y_pred)
    sample_n = min(300, len(y_test))

    fixed_id = f"persistence_{target_label}_{granularity}_{horizon}"
    record = {
        "run_id": fixed_id,
        "target": target_label,
        "target_column": target_col,
        "model_type": "Persistence Baseline",
        "granularity": granularity,
        "horizon": horizon,
        "horizon_label": f"{horizon}{'m' if granularity == 'Minute' else 'h'}",
        "sequence_length": None,
        "hyperparams": {},
        "metrics": metrics,
        "feature_columns": [target_col],
        "feature_importance": None,
        "loss_history": None,
        "prediction_sample": {
            "y_true": [float(v) for v in y_test[-sample_n:]],
            "y_pred": [float(v) for v in y_pred[-sample_n:]],
        },
        "model_path": None,
        "n_train_samples": split_idx,
        "n_test_samples": len(y_test),
        "train_period": None,
        "test_period": None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "promoted": False,
        "notes": "Persistence baseline: next_value = last_observed_value.",
        "experiment_tag": "persistence_benchmark",
    }

    # Upsert — replace if already exists so re-running refreshes the record.
    runs = _load_runs()
    runs = [r for r in runs if r["run_id"] != fixed_id]
    runs.append(record)
    _save_runs(runs)
    return record


# ---- Extended train_research_model with feature_set + dynamics + physics --

def train_research_model_exp(
    target_label: str,
    model_type: str,
    granularity: str = DEFAULT_GRANULARITY,
    horizon: int = DEFAULT_HORIZON,
    sequence_length: int = None,
    hyperparams: dict = None,
    feature_set: str = DEFAULT_FEATURE_SET,
    add_dynamics: bool = False,
    add_physics: bool = False,
    experiment_tag: str = "",
) -> dict:
    """Extended training entry-point for the 8-experiment optimization study.

    Identical to train_research_model() but adds three control axes:

    - feature_set: which base columns to use (one of FEATURE_SET_OPTIONS).
      Only honoured for Hourly granularity — Minute always uses the full
      minute-native physics feature set.

    - add_dynamics: if True, adds Experiment-4 short-term dynamics features
      (short rolling min/max/std, 6h slope, acceleration) on top of the
      standard lag/rolling/change set.  Hourly only.

    - add_physics: if True, adds Experiment-5 hourly physics features
      (Ey, VBz, Dynamic Pressure, Clock Angle, southward hours).  Hourly
      only.

    - experiment_tag: free-text label stored in the run record so the UI
      can group/filter runs by experiment number.

    All other behaviour, the production-safety contract, and the registry
    namespace are identical to train_research_model().
    """
    if target_label not in TARGET_OPTIONS:
        raise ValueError(f"Unknown target: {target_label}")
    if model_type not in ALL_TRAINABLE_MODELS:
        raise ValueError(f"'{model_type}' is not trainable yet — see FUTURE_MODELS.")
    valid_horizons = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    if horizon not in valid_horizons:
        raise ValueError(f"Horizon {horizon} is not valid for {granularity} granularity.")
    target_col = TARGET_OPTIONS[target_label]

    # --- Build feature frame ------------------------------------------------
    if granularity == "Minute":
        # Minute granularity always uses the full physics feature set
        base_frame, physics_cols = _load_minute_base_frame()
        feature_source_cols = list(BASELINE_COLUMNS) + physics_cols
        from swdss.models.imf_physics_features import (
            add_minute_change_features as _amcf,
            add_minute_lag_features as _amlf,
            add_minute_rolling_features as _amrf,
        )
        lag_cols = _amlf(base_frame, feature_source_cols)
        rolling_cols = _amrf(base_frame, feature_source_cols)
        change_cols = _amcf(base_frame, feature_source_cols)
        frame = base_frame
        feature_columns = feature_source_cols + lag_cols + rolling_cols + change_cols
    else:
        # Hourly: use selected feature set, optionally extend with dynamics/physics
        base_cols = list(FEATURE_SET_OPTIONS.get(feature_set, BASELINE_COLUMNS))
        frame = _load_hourly_historical_base_frame()
        # Only keep columns that are actually present (some feature sets need SW)
        base_cols = [c for c in base_cols if c in frame.columns]

        lag_cols = add_lag_features(frame, base_cols)
        rolling_cols = add_rolling_features(frame, base_cols)
        change_cols = add_change_features(frame, base_cols)
        feature_columns = base_cols + lag_cols + rolling_cols + change_cols

        dyn_cols: list[str] = []
        if add_dynamics:
            dyn_cols = add_hourly_dynamics_features(frame, base_cols)
            feature_columns += dyn_cols

        phys_cols: list[str] = []
        if add_physics:
            phys_cols = add_hourly_physics_features(frame)
            feature_columns += phys_cols

    # --- Train (mirrors train_research_model logic exactly) -----------------
    frame = frame.copy()
    frame["__target__"] = frame[target_col].shift(-horizon)
    frame = frame.dropna(subset=feature_columns + ["__target__"])

    min_rows = 200 if granularity == "Minute" else 60
    if len(frame) < min_rows:
        raise ValueError(f"Not enough {granularity.lower()}-level history (need {min_rows}+ clean rows).")

    run_id = str(uuid.uuid4())
    model_dir = RESEARCH_MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    unit = "m" if granularity == "Minute" else "h"

    needs_scaling = model_type in SCALE_SENSITIVE_MODELS
    X_full = frame[feature_columns]
    if needs_scaling:
        row_split = int(len(frame) * (1 - TEST_FRACTION))
        scaler = StandardScaler()
        scaler.fit(X_full.iloc[:row_split])
        X_full = pd.DataFrame(scaler.transform(X_full), index=X_full.index, columns=feature_columns)
        joblib.dump(scaler, model_dir / "scaler.joblib")

    y = frame["__target__"]
    split_idx = int(len(X_full) * (1 - TEST_FRACTION))
    X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = _build_tabular_model(model_type, hyperparams)
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_sec = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    preds = np.asarray(model.predict(X_test))
    inference_time_sec = time.perf_counter() - infer_start
    inference_time_ms_per_sample = (inference_time_sec / len(X_test)) * 1000 if len(X_test) else None

    metrics = compute_metrics(y_test.to_numpy(), preds)
    train_preds = np.asarray(model.predict(X_train))
    train_r2 = float(r2_score(y_train.to_numpy(), train_preds)) if len(X_train) else None

    if hasattr(model, "feature_importances_"):
        pairs = list(zip(feature_columns, (float(v) for v in model.feature_importances_)))
        feature_importance = sorted(pairs, key=lambda kv: -abs(kv[1]))[:30]
    elif hasattr(model, "coef_"):
        pairs = list(zip(feature_columns, (float(v) for v in np.ravel(model.coef_))))
        feature_importance = sorted(pairs, key=lambda kv: -abs(kv[1]))[:30]
    else:
        feature_importance = None

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    model_size_kb = model_path.stat().st_size / 1024

    sample_n = min(300, len(y_test))
    prediction_sample = {
        "y_true": [float(v) for v in np.asarray(y_test)[-sample_n:]],
        "y_pred": [float(v) for v in preds[-sample_n:]],
    }
    train_index, test_index = X_train.index, X_test.index

    run_record = {
        "run_id": run_id,
        "target": target_label,
        "target_column": target_col,
        "model_type": model_type,
        "granularity": granularity,
        "horizon": horizon,
        "horizon_label": f"{horizon}{unit}",
        "sequence_length": None,
        "hyperparams": hyperparams or {},
        "metrics": metrics,
        "feature_columns": feature_columns,
        "feature_importance": feature_importance,
        "loss_history": None,
        "prediction_sample": prediction_sample,
        "model_path": str(model_path),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "train_period": [str(train_index[0]), str(train_index[-1])] if len(train_index) else None,
        "test_period": [str(test_index[0]), str(test_index[-1])] if len(test_index) else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "promoted": False,
        "notes": "",
        "experiment_tag": experiment_tag,
        "feature_set": feature_set if granularity == "Hourly" else "Minute (full physics)",
        "add_dynamics": add_dynamics,
        "add_physics": add_physics,
        "training_time_sec": round(training_time_sec, 4),
        "inference_time_ms_per_sample": round(inference_time_ms_per_sample, 5) if inference_time_ms_per_sample is not None else None,
        "model_size_kb": round(model_size_kb, 2),
        "train_r2": train_r2,
    }
    _append_run(run_record)
    return run_record


# ---- Real Promotion Workflow -----------------------------------------------

def promote_to_production(run_id: str, notes: str = "") -> dict:
    """Archives the current production Bz 1h model and installs a research
    run in its place.

    Steps performed:
    1. Validates run exists, is Hourly granularity, Bz target, 1h horizon,
       and uses a tabular (joblib) model — not LSTM/GRU.
    2. Archives the current bz_gsm_1h.joblib to
       models/imf/archive/bz_gsm_1h_<timestamp>.joblib.
    3. Copies the research model's model.joblib → models/imf/bz_gsm_1h.joblib.
    4. Updates models/imf/metrics.json with the research run's metrics and
       feature_columns, under the "bz_gsm_1h" key.
    5. Marks the run promoted=True, promoted_to_production=True in the
       registry.

    Returns a summary dict with archive_path, new_model_path, old_metrics,
    new_metrics for display in the UI.

    Raises ValueError if any pre-condition fails so the caller can show a
    clear error rather than silently corrupting production state.
    """
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found in the research registry.")
    if run.get("granularity") != "Hourly":
        raise ValueError("Only Hourly-granularity runs can be promoted — production trains on hourly data.")
    if run.get("target") != "Bz":
        raise ValueError("Only Bz-target runs can be promoted to the Bz production model.")
    if run.get("horizon") != 1:
        raise ValueError("Only 1h-horizon runs match the production bz_gsm_1h model — other horizons have separate files.")
    if run.get("model_type") in SEQUENCE_MODELS:
        raise ValueError("LSTM/GRU promotion is not supported — loading a Keras model into the prediction process causes the same process-conflict hang training does.")

    production_dir = MODELS_DIR / "imf"
    prod_model_path = production_dir / "bz_gsm_1h.joblib"
    prod_metrics_path = production_dir / "metrics.json"

    # Step 2: archive existing model
    archive_dir = production_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"bz_gsm_1h_{ts}.joblib"
    if prod_model_path.exists():
        shutil.copy2(prod_model_path, archive_path)
    else:
        archive_path = None

    # Step 3: copy research model → production
    shutil.copy2(Path(run["model_path"]), prod_model_path)

    # Step 4: update metrics.json
    old_metrics = None
    metrics_data = {}
    if prod_metrics_path.exists():
        with open(prod_metrics_path) as f:
            metrics_data = json.load(f)
        old_metrics = metrics_data.get("bz_gsm_1h")

    metrics_data["bz_gsm_1h"] = {
        "variable": "bz_gsm",
        "horizon": 1,
        "algorithm": run["model_type"],
        "r2": run["metrics"]["r2"],
        "mae": run["metrics"]["mae"],
        "rmse": run["metrics"]["rmse"],
        "bias": run["metrics"]["bias"],
        "n_train": run["n_train_samples"],
        "n_test": run["n_test_samples"],
        "n_samples": run["n_train_samples"] + run["n_test_samples"],
        "feature_columns": run["feature_columns"],
        "model_path": str(prod_model_path),
        "trained_at": run["trained_at"],
        "promoted_from_research": run_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promoted_notes": notes,
        "training_csv": str(MODELS_DIR.parent / "data" / "features" / "training_v2" / "imf_features.csv"),
    }
    with open(prod_metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Step 5: mark run promoted in registry
    runs = _load_runs()
    for r in runs:
        if r["run_id"] == run_id:
            r["promoted"] = True
            r["promoted_at"] = datetime.now(timezone.utc).isoformat()
            r["promoted_to_production"] = True
            if notes:
                r["notes"] = notes
    _save_runs(runs)

    return {
        "run_id": run_id,
        "archive_path": str(archive_path) if archive_path else None,
        "new_model_path": str(prod_model_path),
        "old_metrics": old_metrics,
        "new_metrics": run["metrics"],
        "model_type": run["model_type"],
        "feature_count": len(run["feature_columns"]),
        "promoted_at": metrics_data["bz_gsm_1h"]["promoted_at"],
    }


HYPERPARAM_SCHEMA = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0}},
    "Lasso": {"alpha": {"type": "float", "default": 0.1, "min": 0.001, "max": 10.0}},
    "ElasticNet": {
        "alpha": {"type": "float", "default": 0.1, "min": 0.001, "max": 10.0},
        "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    },
    "SVR": {
        "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0},
        "epsilon": {"type": "float", "default": 0.1, "min": 0.001, "max": 5.0},
    },
    "MLP": {
        "hidden_layer_size": {"type": "int", "default": 64, "min": 8, "max": 256},
        "max_iter": {"type": "int", "default": 300, "min": 50, "max": 2000},
        "alpha": {"type": "float", "default": 0.0001, "min": 0.00001, "max": 0.1},
    },
    "Random Forest": {
        "n_estimators": {"type": "int", "default": 200, "min": 20, "max": 1000},
        "max_depth": {"type": "int", "default": 10, "min": 2, "max": 40},
    },
    "XGBoost": {
        "n_estimators": {"type": "int", "default": 300, "min": 20, "max": 1000},
        "max_depth": {"type": "int", "default": 6, "min": 2, "max": 20},
        "learning_rate": {"type": "float", "default": 0.05, "min": 0.001, "max": 0.5},
    },
    "LightGBM": {
        "n_estimators": {"type": "int", "default": 300, "min": 20, "max": 1000},
        "num_leaves": {"type": "int", "default": 31, "min": 4, "max": 256},
        "learning_rate": {"type": "float", "default": 0.05, "min": 0.001, "max": 0.5},
    },
    "CatBoost": {
        "iterations": {"type": "int", "default": 300, "min": 20, "max": 1000},
        "depth": {"type": "int", "default": 6, "min": 2, "max": 16},
        "learning_rate": {"type": "float", "default": 0.05, "min": 0.001, "max": 0.5},
    },
    "LSTM": {
        "units": {"type": "int", "default": 64, "min": 8, "max": 256},
        "epochs": {"type": "int", "default": 15, "min": 1, "max": 100},
        "batch_size": {"type": "int", "default": 64, "min": 8, "max": 512},
        "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.6},
    },
    "GRU": {
        "units": {"type": "int", "default": 64, "min": 8, "max": 256},
        "epochs": {"type": "int", "default": 15, "min": 1, "max": 100},
        "batch_size": {"type": "int", "default": 64, "min": 8, "max": 512},
        "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.6},
    },
}


def _load_minute_base_frame() -> tuple:
    """Raw Solar Wind + IMF merged on timestamp, minute-native, with the
    physics feature set added — but NOT yet lag/rolling/change'd. Shared
    by both granularities below: physics features (southward duration,
    clock angle, ...) are only meaningful computed at minute resolution
    (they're literally defined in terms of consecutive minutes), so they
    are always derived here first, then either used directly (Minute
    granularity) or aggregated to hourly (Hourly granularity) — never
    recomputed from already-aggregated data, which would be a different,
    wrong quantity (e.g. you cannot recover "consecutive minutes
    southward" from an hourly mean of Bz).
    """
    from swdss.models.registry import DATASETS

    imf_raw = pd.read_parquet(DATASETS["imf"].processed_parquet)
    imf_raw = imf_raw.rename(columns={"bx": "bx_gsm", "by": "by_gsm", "bz": "bz_gsm"})
    sw_raw = pd.read_parquet(DATASETS["solar_wind"].processed_parquet)
    sw_raw = sw_raw.rename(columns={"solar_wind_speed": "speed", "proton_density": "density"})

    imf_raw["timestamp_utc"] = pd.to_datetime(imf_raw["timestamp_utc"], utc=True)
    sw_raw["timestamp_utc"] = pd.to_datetime(sw_raw["timestamp_utc"], utc=True)

    frame = pd.merge(sw_raw, imf_raw, on="timestamp_utc", how="inner").set_index("timestamp_utc").sort_index()
    physics_cols = add_all_imf_physics_features(frame)
    return frame, physics_cols


def _load_hourly_historical_base_frame() -> pd.DataFrame:
    """The SAME historical data production trains on — `imf_features.csv`
    + `solar_wind_features.csv` (~3 years, ~26,000 hourly rows each) —
    stripped back to just the raw base columns (the CSVs already ship
    pre-computed lag/rolling/change columns from the original notebook
    pipeline; those are ignored here and recomputed via
    swdss.models.features, so the two pipelines are guaranteed to use
    identical feature-engineering code, not just similar-looking CSVs).

    This exists because resampling the live minute-level parquet (only
    ~7 days deep) to hourly gives ~140 usable rows after lag/rolling
    windows eat the first day — nowhere near enough to be statistically
    comparable to production's 26,000-row models. Reading the same
    historical CSVs production itself was trained on is what makes an
    "Hourly, 1h" experimental run and production's own bz_gsm_1h model
    genuinely comparable, not just aimed at the same target definition.
    """
    from swdss.models.registry import DATASETS

    imf = pd.read_csv(DATASETS["imf"].training_csv)[["datetime"] + IMF_BASE_COLUMNS]
    sw = pd.read_csv(DATASETS["solar_wind"].training_csv)[["datetime"] + SOLAR_WIND_BASE_COLUMNS]
    imf["datetime"] = pd.to_datetime(imf["datetime"])
    sw["datetime"] = pd.to_datetime(sw["datetime"])

    frame = pd.merge(sw, imf, on="datetime", how="inner").sort_values("datetime").set_index("datetime")
    return frame


def load_research_frame(granularity: str = DEFAULT_GRANULARITY) -> tuple:
    """Builds the IMF research frame at the requested granularity.
    Returns (frame, feature_columns).

    - "Minute" (default, backward-compatible with every existing caller):
      live minute-level parquet (~7 days), minute-scale lag/rolling/rate-
      of-change features ([1,5,15,30,60] minutes / 60-minute rolling
      window — see imf_physics_features.py) on top of baseline + physics
      columns. Physics features live here because they're only
      meaningful at minute resolution (they're literally defined in
      terms of consecutive minutes) — see _load_minute_base_frame.

    - "Hourly": production's own 3-year historical CSVs (see
      _load_hourly_historical_base_frame), HOUR-scale lag/rolling/change
      features via swdss.models.features — the identical functions
      swdss.models.train uses — reused directly, not reimplemented. NO
      physics features at this granularity: they require minute-level
      data, and no 3-year minute-level archive exists (only the live
      dataset's ~7-day rolling window does) — computing them from
      already-hourly-aggregated data would produce a different, wrong
      quantity (e.g. "consecutive minutes southward" cannot be recovered
      from an hourly mean of Bz). This is a real, acknowledged asymmetry:
      the "does adding physics features help" question can only be asked
      within Minute granularity (same data, same period, physics
      features on vs. off), not across granularities.
    """
    if granularity == "Minute":
        base_frame, physics_cols = _load_minute_base_frame()
        feature_source_cols = list(BASELINE_COLUMNS) + physics_cols
        lag_cols = add_minute_lag_features(base_frame, feature_source_cols)
        rolling_cols = add_minute_rolling_features(base_frame, feature_source_cols)
        change_cols = add_minute_change_features(base_frame, feature_source_cols)
        frame = base_frame
    elif granularity == "Hourly":
        frame = _load_hourly_historical_base_frame()
        feature_source_cols = list(BASELINE_COLUMNS)
        # Reusing production's own hour-scale feature functions directly
        # (LAGS=[1,3,6,12,24]h, ROLLING_WINDOW=24h) — not reimplemented.
        lag_cols = add_lag_features(frame, feature_source_cols)
        rolling_cols = add_rolling_features(frame, feature_source_cols)
        change_cols = add_change_features(frame, feature_source_cols)
    else:
        raise ValueError(f"Unknown granularity: {granularity!r} — expected 'Minute' or 'Hourly'.")

    feature_columns = feature_source_cols + lag_cols + rolling_cols + change_cols
    return frame, feature_columns


def build_sequences(frame: pd.DataFrame, feature_columns: list[str], target: pd.Series, seq_len: int) -> tuple:
    """Windows the (already feature-engineered) frame into overlapping
    `seq_len`-minute sequences for LSTM/GRU — never single rows, per the
    "Sequence Models" requirement. Row i's sequence is minutes
    [i-seq_len, i), predicting the target at row i.
    """
    values = frame[feature_columns].to_numpy(dtype="float32")
    targets = target.to_numpy(dtype="float32")
    X = np.stack([values[i - seq_len : i] for i in range(seq_len, len(frame))])
    y = targets[seq_len:]
    return X, y


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    nonzero = y_true != 0  # MAPE undefined at 0 — Bz/By cross zero constantly
    mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else None
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape,
        "bias": float(np.mean(errors)),
    }


# Models sensitive enough to feature scale that training on this
# pipeline's raw, wildly-different-magnitude columns (temperature ~1e5,
# Bz ~1-10 nT) would be meaningless without standardizing first — same
# lesson learned the hard way with LSTM/GRU (see train_research_model).
# Tree/boosting models and plain OLS-family linear models are scale-
# invariant or scale-robust and deliberately excluded from this set.
SCALE_SENSITIVE_MODELS = {"SVR", "MLP", "LSTM", "GRU"}


def _build_tabular_model(model_type: str, hyperparams: dict):
    hp = hyperparams or {}
    if model_type == "Linear Regression":
        return LinearRegression()
    if model_type == "Ridge Regression":
        return Ridge(alpha=hp.get("alpha", 1.0))
    if model_type == "Lasso":
        return Lasso(alpha=hp.get("alpha", 0.1))
    if model_type == "ElasticNet":
        return ElasticNet(alpha=hp.get("alpha", 0.1), l1_ratio=hp.get("l1_ratio", 0.5))
    if model_type == "SVR":
        return SVR(C=hp.get("C", 1.0), epsilon=hp.get("epsilon", 0.1))
    if model_type == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(hp.get("hidden_layer_size", 64),),
            max_iter=hp.get("max_iter", 300),
            alpha=hp.get("alpha", 0.0001),
            random_state=42,
        )
    if model_type == "Random Forest":
        return RandomForestRegressor(
            n_estimators=hp.get("n_estimators", 200), max_depth=hp.get("max_depth", 10), n_jobs=-1, random_state=42
        )
    if model_type == "XGBoost":
        return XGBRegressor(
            n_estimators=hp.get("n_estimators", 300),
            max_depth=hp.get("max_depth", 6),
            learning_rate=hp.get("learning_rate", 0.05),
            n_jobs=-1,
            random_state=42,
        )
    if model_type == "LightGBM":
        return LGBMRegressor(
            n_estimators=hp.get("n_estimators", 300),
            num_leaves=hp.get("num_leaves", 31),
            learning_rate=hp.get("learning_rate", 0.05),
            random_state=42,
            verbosity=-1,
        )
    if model_type == "CatBoost":
        return CatBoostRegressor(
            iterations=hp.get("iterations", 300),
            depth=hp.get("depth", 6),
            learning_rate=hp.get("learning_rate", 0.05),
            random_state=42,
            verbose=False,
        )
    raise ValueError(f"Unknown tabular model type: {model_type}")


def _run_keras_worker(X_train, y_train, X_test, y_test, meta: dict) -> dict:
    """Hands Keras training off to imf_research_keras_worker.py in a
    brand-new subprocess — see that module's docstring for why this is
    necessary (a real, reproducible hang when TensorFlow shares a process
    with scikit-learn/XGBoost/LightGBM/CatBoost). PYTHONPATH is passed
    explicitly since the subprocess doesn't inherit the dashboard's own
    runtime sys.path.insert — only real environment variables.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.npz"
        meta_path = Path(tmpdir) / "meta.json"
        output_path = Path(tmpdir) / "output.json"

        np.savez(
            input_path,
            X_train=X_train.astype("float32"),
            y_train=y_train.astype("float32"),
            X_test=X_test.astype("float32"),
            y_test=y_test.astype("float32"),
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        src_dir = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, "-m", "swdss.models.imf_research_keras_worker", str(input_path), str(meta_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Sequence model training subprocess failed:\n{result.stderr[-3000:]}")

        with open(output_path) as f:
            return json.load(f)


def train_research_model(
    target_label: str,
    model_type: str,
    granularity: str = DEFAULT_GRANULARITY,
    horizon: int = DEFAULT_HORIZON,
    sequence_length: int = None,
    hyperparams: dict = None,
    experiment_tag: str = "",
) -> dict:
    """Trains one model for one (target, granularity, horizon, model_type)
    combination and records a run — never touches the production
    model/metrics files.

    `horizon` is in minutes when granularity="Minute" (must be one of
    MINUTE_HORIZONS) or hours when granularity="Hourly" (must be one of
    HOURLY_HORIZONS — the SAME list production trains against). Target is
    always `frame[target_col].shift(-horizon)` on whichever frame
    load_research_frame(granularity) returns — see that function for why
    "Hourly, 1h" is built the identical way production's own 1h models
    are, not merely a similar-looking parallel implementation.

    `sequence_length` (LSTM/GRU only) is an independent axis from
    `horizon`: it's how far BACK the model looks (input window), while
    `horizon` is how far AHEAD it predicts (target lead time). Its units
    match `granularity` (minutes or hours) too.
    """
    if target_label not in TARGET_OPTIONS:
        raise ValueError(f"Unknown target: {target_label}")
    if model_type not in ALL_TRAINABLE_MODELS:
        raise ValueError(f"'{model_type}' is not trainable yet — see FUTURE_MODELS.")
    valid_horizons = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    if horizon not in valid_horizons:
        raise ValueError(f"Horizon {horizon} is not valid for {granularity} granularity — expected one of {valid_horizons}.")
    target_col = TARGET_OPTIONS[target_label]

    frame, feature_columns = load_research_frame(granularity)
    frame = frame.copy()
    frame["__target__"] = frame[target_col].shift(-horizon)
    frame = frame.dropna(subset=feature_columns + ["__target__"])

    min_rows = 200 if granularity == "Minute" else 60
    if len(frame) < min_rows:
        raise ValueError(f"Not enough {granularity.lower()}-level history to train a research model yet (need {min_rows}+ clean rows).")

    run_id = str(uuid.uuid4())
    model_dir = RESEARCH_MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    unit = "m" if granularity == "Minute" else "h"

    if model_type in SEQUENCE_MODELS:
        if not KERAS_AVAILABLE:
            raise ValueError("TensorFlow/Keras is not installed — sequence models are unavailable.")
        seq_len = sequence_length or DEFAULT_SEQUENCE_LENGTH

        # Neural nets are far more sensitive to feature scale than the
        # tree/linear models below — raw features here span wildly
        # different magnitudes (temperature ~1e5, Bz ~1-10 nT,
        # southward_duration in the hundreds of minutes). Standardizing
        # every feature column is standard practice for this reason;
        # fit only on the training rows so no test-set statistics leak
        # into the transform.
        row_split = int(len(frame) * (1 - TEST_FRACTION))
        scaler = StandardScaler()
        scaler.fit(frame[feature_columns].iloc[:row_split])
        scaled_frame = frame.copy()
        scaled_frame[feature_columns] = scaler.transform(frame[feature_columns])

        X, y = build_sequences(scaled_frame, feature_columns, scaled_frame["__target__"], seq_len)
        if len(X) < 100:
            raise ValueError(f"Not enough {granularity.lower()}-level history to build sequences of this length yet.")
        split_idx = int(len(X) * (1 - TEST_FRACTION))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        train_index = frame.index[seq_len : seq_len + len(X_train)]
        test_index = frame.index[seq_len + len(X_train) :]

        hp = hyperparams or {}
        model_path = model_dir / "model.keras"
        worker_meta = {
            "model_type": model_type,
            "seq_len": seq_len,
            "n_features": len(feature_columns),
            "units": hp.get("units", 64),
            "dropout": hp.get("dropout", 0.2),
            "epochs": hp.get("epochs", 15),
            "batch_size": hp.get("batch_size", 64),
            "model_path": str(model_path),
        }
        fit_start = time.perf_counter()
        worker_result = _run_keras_worker(X_train, y_train, X_test, y_test, worker_meta)
        # The worker subprocess trains AND predicts before returning — wall
        # time here covers both. Not split further since doing so would
        # require a second round-trip subprocess call just for timing.
        training_time_sec = time.perf_counter() - fit_start
        inference_time_ms_per_sample = None  # not separable from the subprocess call above
        preds = np.array(worker_result["preds"], dtype="float32")
        metrics = compute_metrics(y_test, preds)
        loss_history = {"loss": worker_result["loss"], "val_loss": worker_result["val_loss"]}
        feature_importance = None  # not supported for sequence models
        train_r2 = None  # worker does not return train-set predictions

        # The scaler is part of this model's inference contract — anyone
        # loading it back must apply the identical transform to new
        # features first, or predictions will be meaningless.
        joblib.dump(scaler, model_dir / "scaler.joblib")

        n_train, n_test = len(X_train), len(X_test)
        y_true_out, y_pred_out = y_test, preds
        model_size_kb = model_path.stat().st_size / 1024 if model_path.exists() else None
    else:
        needs_scaling = model_type in SCALE_SENSITIVE_MODELS
        X_full = frame[feature_columns]
        if needs_scaling:
            row_split = int(len(frame) * (1 - TEST_FRACTION))
            scaler = StandardScaler()
            scaler.fit(X_full.iloc[:row_split])
            X_full = pd.DataFrame(scaler.transform(X_full), index=X_full.index, columns=feature_columns)

        y = frame["__target__"]
        split_idx = int(len(X_full) * (1 - TEST_FRACTION))
        X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        train_index, test_index = X_train.index, X_test.index

        model = _build_tabular_model(model_type, hyperparams)
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time_sec = time.perf_counter() - fit_start

        infer_start = time.perf_counter()
        preds = np.asarray(model.predict(X_test))
        inference_time_sec = time.perf_counter() - infer_start
        inference_time_ms_per_sample = (inference_time_sec / len(X_test)) * 1000 if len(X_test) else None

        metrics = compute_metrics(y_test.to_numpy(), preds)
        loss_history = None
        train_preds = np.asarray(model.predict(X_train))
        train_r2 = float(r2_score(y_train.to_numpy(), train_preds)) if len(X_train) else None

        if hasattr(model, "feature_importances_"):
            pairs = list(zip(feature_columns, (float(v) for v in model.feature_importances_)))
            feature_importance = sorted(pairs, key=lambda kv: -abs(kv[1]))[:20]
        elif hasattr(model, "coef_"):
            pairs = list(zip(feature_columns, (float(v) for v in np.ravel(model.coef_))))
            feature_importance = sorted(pairs, key=lambda kv: -abs(kv[1]))[:20]
        else:
            feature_importance = None

        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        if needs_scaling:
            joblib.dump(scaler, model_dir / "scaler.joblib")
        model_size_kb = model_path.stat().st_size / 1024

        n_train, n_test = len(X_train), len(X_test)
        y_true_out, y_pred_out = y_test.to_numpy(), preds

    sample_n = min(300, len(y_true_out))
    prediction_sample = {
        "y_true": [float(v) for v in np.asarray(y_true_out)[-sample_n:]],
        "y_pred": [float(v) for v in np.asarray(y_pred_out)[-sample_n:]],
    }

    run_record = {
        "run_id": run_id,
        "target": target_label,
        "target_column": target_col,
        "model_type": model_type,
        "granularity": granularity,
        "horizon": horizon,
        "horizon_label": f"{horizon}{unit}",
        "sequence_length": sequence_length if model_type in SEQUENCE_MODELS else None,
        "hyperparams": hyperparams or {},
        "metrics": metrics,
        "feature_columns": feature_columns,
        "feature_importance": feature_importance,
        "loss_history": loss_history,
        "prediction_sample": prediction_sample,
        "model_path": str(model_path),
        "n_train_samples": int(n_train),
        "n_test_samples": int(n_test),
        "train_period": [str(train_index[0]), str(train_index[-1])] if len(train_index) else None,
        "test_period": [str(test_index[0]), str(test_index[-1])] if len(test_index) else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "promoted": False,
        "notes": "",
        "experiment_tag": experiment_tag,
        "feature_set": None,
        "add_dynamics": None,
        "add_physics": None,
        "training_time_sec": round(training_time_sec, 4),
        "inference_time_ms_per_sample": round(inference_time_ms_per_sample, 5) if inference_time_ms_per_sample is not None else None,
        "model_size_kb": round(model_size_kb, 2) if model_size_kb is not None else None,
        "train_r2": train_r2,
    }
    _append_run(run_record)
    return run_record


def train_horizon_sweep(
    target_label: str,
    model_type: str,
    granularity: str = DEFAULT_GRANULARITY,
    hyperparams: dict = None,
    reuse_existing: bool = True,
) -> list[dict]:
    """Trains (or reuses) one run per horizon in the chosen granularity's
    full horizon list, for the Horizon Analysis tab — "how does skill
    decay with lead time?" answered directly, without a researcher having
    to manually click Train five or ten times. When `reuse_existing` is
    True, the most recently trained matching run (same target,
    granularity, model_type, horizon — ignoring sequence_length, since
    the sweep only exercises non-sequence model types for now) is reused
    instead of retraining. Returns one run dict per horizon, in
    ascending horizon order.
    """
    horizons = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    results = []
    for horizon in horizons:
        existing = None
        if reuse_existing:
            candidates = [
                r
                for r in list_runs(target_label)
                if r.get("granularity", "Minute") == granularity
                and r.get("horizon", 1) == horizon
                and r["model_type"] == model_type
            ]
            existing = candidates[0] if candidates else None  # list_runs is newest-first
        run = existing or train_research_model(target_label, model_type, granularity=granularity, horizon=horizon, hyperparams=hyperparams)
        results.append(run)
    return results


# ---------------------------------------------------------------- registry


def _load_runs() -> list[dict]:
    if not RUNS_REGISTRY_PATH.exists():
        return []
    with open(RUNS_REGISTRY_PATH) as f:
        return json.load(f)


def _save_runs(runs: list[dict]) -> None:
    RUNS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_REGISTRY_PATH, "w") as f:
        json.dump(runs, f, indent=2)


def _append_run(run_record: dict) -> None:
    runs = _load_runs()
    runs.append(run_record)
    _save_runs(runs)


def list_runs(target_label: str = None) -> list[dict]:
    runs = _load_runs()
    if target_label:
        runs = [r for r in runs if r["target"] == target_label]
    return sorted(runs, key=lambda r: r["trained_at"], reverse=True)


def get_run(run_id: str) -> dict:
    for r in _load_runs():
        if r["run_id"] == run_id:
            return r
    return None


def delete_run(run_id: str) -> bool:
    runs = _load_runs()
    remaining = [r for r in runs if r["run_id"] != run_id]
    if len(remaining) == len(runs):
        return False
    shutil.rmtree(RESEARCH_MODELS_DIR / run_id, ignore_errors=True)
    _save_runs(remaining)
    return True


def promote_run(run_id: str, notes: str = "") -> bool:
    """Marks a run 'promoted' — a label for a researcher's own tracking,
    nothing more. Never touches production model files, predict.py, or
    jobs.py; a promoted model only becomes live if a human engineer
    manually wires it into the production path themselves later. See
    module docstring's Production Safety note.
    """
    runs = _load_runs()
    found = False
    for r in runs:
        if r["run_id"] == run_id:
            r["promoted"] = True
            r["promoted_at"] = datetime.now(timezone.utc).isoformat()
            if notes:
                r["notes"] = notes
            found = True
    if found:
        _save_runs(runs)
    return found


def load_trained_model(run_id: str):
    """Loads a run's trained tabular model back into the current process.

    LSTM/GRU runs are deliberately NOT supported here: loading a Keras
    model (even just for `.predict`, no training) into a process that
    also has scikit-learn/XGBoost/LightGBM/CatBoost loaded was
    empirically confirmed to hang exactly like training did — this isn't
    just a training-time issue. Re-running a saved sequence model would
    need the same subprocess isolation _run_keras_worker uses for
    training; that's not built yet since no current UI flow calls this
    for a sequence-model run. Failing loudly here beats silently hanging.
    """
    run = get_run(run_id)
    if run is None:
        return None
    if run["model_type"] in SEQUENCE_MODELS:
        raise NotImplementedError(
            "Loading a saved LSTM/GRU model back into this process isn't supported — it reproduces the "
            "same TensorFlow/scikit-learn process conflict training does. The saved artifact "
            f"({run['model_path']}) is still on disk and can be loaded in a fresh, isolated process."
        )
    return joblib.load(Path(run["model_path"]))


def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    """Rule-based comparison between two training runs (e.g. baseline vs
    experimental architecture), in the same Supported/Not Supported/
    Inconclusive spirit as swdss.models.hypothesis — adapted for offline
    training-run metrics rather than live verified predictions. There's
    no repeated-prediction sample size to derive a confidence LEVEL from
    here (each run has exactly one held-out test-set score), so this
    intentionally never claims a confidence percentage, only a verdict.
    """
    a, b = get_run(run_id_a), get_run(run_id_b)
    if a is None or b is None:
        raise ValueError("One or both runs not found.")

    delta_r2 = b["metrics"]["r2"] - a["metrics"]["r2"]
    delta_mae = b["metrics"]["mae"] - a["metrics"]["mae"]  # negative = improvement

    if delta_r2 >= 0.01 and delta_mae < 0:
        verdict = "Supported"
        explanation = (
            f"{b['model_type']} improved R² by {delta_r2:+.4f} and reduced MAE by "
            f"{abs(delta_mae):.4f} versus {a['model_type']}."
        )
    elif delta_r2 <= -0.01 or delta_mae > 0:
        verdict = "Not Supported"
        explanation = (
            f"{b['model_type']} did not outperform {a['model_type']} "
            f"(ΔR²={delta_r2:+.4f}, ΔMAE={delta_mae:+.4f})."
        )
    else:
        verdict = "Inconclusive"
        explanation = (
            f"Difference is too small to draw a conclusion (ΔR²={delta_r2:+.4f}, "
            f"ΔMAE={delta_mae:+.4f}) — a single held-out test split isn't strong evidence either way."
        )

    return {
        "baseline": a,
        "experimental": b,
        "delta_r2": delta_r2,
        "delta_mae": delta_mae,
        "verdict": verdict,
        "explanation": explanation,
    }


# ============================================================================
# AutoML Orchestration Layer — "Run Complete Optimization Study"
# ============================================================================
# Everything below orchestrates the SAME experiment functions the manual
# Exp 1-8 tabs call (train_research_model_exp, compute_persistence_benchmark)
# into one end-to-end pipeline — it does not introduce a new training path.
# Every candidate trained here is indistinguishable from one trained by hand
# in the manual tabs and lands in the exact same runs registry; this section
# only adds a structured feature/model search grid, timing-aware evaluation,
# SHAP-based interpretation, a leaderboard, a promotion-criteria checker, and
# a separate study-history registry on top.
#
# Promotion stays manual: run_complete_optimization_study() only ever trains
# and evaluates. promote_to_production() (above) is the sole function that
# writes to models/imf/ — this layer only recommends and gates the button.

STUDIES_REGISTRY_PATH = DATA_DIR / "predictions" / "imf_optimization_studies.json"

# Experiment 3-5's structured feature search grid — NOT a random search.
# Each entry names a distinct hypothesis about what Bz needs: progressively
# adding solar-wind context, then dynamics, then physics, then both
# together. Matches the study's own Exp 3/4/5 tabs exactly, so a result
# found here is reproducible by hand in those tabs.
FEATURE_SEARCH_GRID = [
    {"name": "Raw IMF", "feature_set": "IMF Only", "add_dynamics": False, "add_physics": False},
    {"name": "Raw IMF + Speed", "feature_set": "IMF + Speed", "add_dynamics": False, "add_physics": False},
    {"name": "Raw IMF + Speed + Density", "feature_set": "IMF + Speed + Density", "add_dynamics": False, "add_physics": False},
    {"name": "Raw IMF + Speed + Density + Temperature", "feature_set": "IMF + All Solar Wind", "add_dynamics": False, "add_physics": False},
    {"name": "Raw IMF + Dynamics", "feature_set": "IMF Only", "add_dynamics": True, "add_physics": False},
    {"name": "Raw IMF + Physics", "feature_set": "IMF Only", "add_dynamics": False, "add_physics": True},
    {"name": "Raw IMF + Solar Wind + Dynamics", "feature_set": "IMF + All Solar Wind", "add_dynamics": True, "add_physics": False},
    {"name": "Raw IMF + Solar Wind + Physics", "feature_set": "IMF + All Solar Wind", "add_dynamics": False, "add_physics": True},
    {"name": "All Features", "feature_set": "IMF + All Solar Wind", "add_dynamics": True, "add_physics": True},
]

# Probe model used during feature search (Steps 3-5): captures nonlinear
# feature value (unlike a pure linear probe) while still being far cheaper
# than tuning every model type across all 9 combinations. The winning
# combination is then re-tested by every real model in Step 6.
FEATURE_SEARCH_PROBE_MODEL = "Random Forest"

# Promotion-criteria thresholds — see check_promotion_criteria().
OVERFIT_R2_GAP_THRESHOLD = 0.15          # train_r2 - test_r2 above this = overfitting risk
MAX_REASONABLE_INFERENCE_MS = 50.0       # per-sample inference time ceiling for interactive use

SHAP_SUPPORTED_MODELS = {
    "Linear Regression", "Ridge Regression", "Lasso", "ElasticNet",
    "Random Forest", "XGBoost", "LightGBM", "CatBoost",
}


def _load_studies() -> list[dict]:
    if not STUDIES_REGISTRY_PATH.exists():
        return []
    with open(STUDIES_REGISTRY_PATH) as f:
        return json.load(f)


def _save_studies(studies: list[dict]) -> None:
    STUDIES_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDIES_REGISTRY_PATH, "w") as f:
        json.dump(studies, f, indent=2)


def _append_study(study: dict) -> None:
    studies = _load_studies()
    studies.append(study)
    _save_studies(studies)


def list_studies() -> list[dict]:
    return sorted(_load_studies(), key=lambda s: s["started_at"], reverse=True)


def get_study(study_id: str) -> dict:
    for s in _load_studies():
        if s["study_id"] == study_id:
            return s
    return None


def _update_study(study_id: str, **fields) -> bool:
    studies = _load_studies()
    found = False
    for s in studies:
        if s["study_id"] == study_id:
            s.update(fields)
            found = True
    if found:
        _save_studies(studies)
    return found


def get_production_bz_metrics() -> dict:
    """Reads the CURRENT production bz_gsm_1h entry straight from
    models/imf/metrics.json — always live, never cached, so a Production
    Comparison always reflects whatever is actually deployed right now
    (including immediately after a promotion from this same study).
    Returns None if no production Bz 1h model exists yet.
    """
    metrics_path = MODELS_DIR / "imf" / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get("bz_gsm_1h")


def _rebuild_test_frame_for_run(run: dict) -> tuple:
    """Deterministically rebuilds (X_train, X_test) for a stored run so
    SHAP can be computed after the fact, without persisting raw feature
    matrices in the JSON registry (which would bloat it hugely). Reruns
    the exact same frame-building steps train_research_model_exp used,
    keyed off fields the run record already stores (granularity,
    feature_set, add_dynamics, add_physics, target_column, horizon,
    feature_columns) — same inputs in, same split, same columns out.
    """
    granularity = run.get("granularity", "Hourly")
    horizon = run["horizon"]
    target_col = run["target_column"]
    feature_columns = run["feature_columns"]

    if granularity == "Minute":
        base_frame, physics_cols = _load_minute_base_frame()
        from swdss.models.imf_physics_features import (
            add_minute_change_features as _amcf,
            add_minute_lag_features as _amlf,
            add_minute_rolling_features as _amrf,
        )
        feature_source_cols = list(BASELINE_COLUMNS) + physics_cols
        _amlf(base_frame, feature_source_cols)
        _amrf(base_frame, feature_source_cols)
        _amcf(base_frame, feature_source_cols)
        frame = base_frame
    else:
        feature_set = run.get("feature_set") or DEFAULT_FEATURE_SET
        base_cols = list(FEATURE_SET_OPTIONS.get(feature_set, BASELINE_COLUMNS))
        frame = _load_hourly_historical_base_frame()
        base_cols = [c for c in base_cols if c in frame.columns]
        add_lag_features(frame, base_cols)
        add_rolling_features(frame, base_cols)
        add_change_features(frame, base_cols)
        if run.get("add_dynamics"):
            add_hourly_dynamics_features(frame, base_cols)
        if run.get("add_physics"):
            add_hourly_physics_features(frame)

    frame = frame.copy()
    frame["__target__"] = frame[target_col].shift(-horizon)
    frame = frame.dropna(subset=feature_columns + ["__target__"])
    split_idx = int(len(frame) * (1 - TEST_FRACTION))
    X_train = frame[feature_columns].iloc[:split_idx]
    X_test = frame[feature_columns].iloc[split_idx:]
    return X_train, X_test


def compute_shap_importance(run_id: str, max_background: int = 100, max_explain: int = 200) -> dict:
    """SHAP-based feature importance for a stored run (Experiment 8).

    Only supported for models with a fast native SHAP explainer (linear
    and tree/boosting families — see SHAP_SUPPORTED_MODELS). SVR/MLP would
    fall back to shap's KernelExplainer, too slow for interactive
    dashboard use at this feature-count scale, and LSTM/GRU can't be
    reloaded into this process at all (see load_trained_model's documented
    TensorFlow/scikit-learn process conflict). Both return a clear
    `skipped_reason` instead of hanging or erroring.

    Returns {"run_id", "supported": bool, "skipped_reason": str|None,
    "shap_importance": [(feature, mean_abs_shap), ...] | None}.
    """
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found.")
    model_type = run["model_type"]

    if model_type not in SHAP_SUPPORTED_MODELS:
        return {
            "run_id": run_id,
            "supported": False,
            "skipped_reason": (
                f"SHAP skipped for {model_type} — no fast native explainer available for this model "
                "family in this dashboard (KernelExplainer would be too slow for interactive use, and "
                "LSTM/GRU cannot be reloaded into this process). Use a linear or tree-based model for "
                "SHAP analysis."
            ),
            "shap_importance": None,
        }

    import shap  # local import — heavy dependency, only needed here

    model = load_trained_model(run_id)
    X_train, X_test = _rebuild_test_frame_for_run(run)
    background = X_train.sample(n=min(max_background, len(X_train)), random_state=42) if len(X_train) else X_train
    explain_rows = X_test.sample(n=min(max_explain, len(X_test)), random_state=42) if len(X_test) else X_test

    # Passing the raw estimator (not model.predict) lets shap auto-select
    # the fast native explainer for each family — TreeExplainer for
    # RF/XGBoost/LightGBM/CatBoost, LinearExplainer for the linear family —
    # rather than falling back to the much slower model-agnostic path.
    explainer = shap.Explainer(model, background)
    shap_values = explainer(explain_rows)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    pairs = list(zip(run["feature_columns"], (float(v) for v in mean_abs)))
    ranked = sorted(pairs, key=lambda kv: -kv[1])[:30]

    return {
        "run_id": run_id,
        "supported": True,
        "skipped_reason": None,
        "shap_importance": ranked,
    }


def check_promotion_criteria(candidate_run: dict, production_metrics: dict) -> dict:
    """Evaluates the promotion checklist from the study spec against one
    candidate run. Returns {"eligible": bool, "checklist": [...]} where
    each checklist item is {"criterion", "passed", "detail"} — the UI
    renders this directly as a pass/fail list next to the Promote button.

    `production_metrics` is get_production_bz_metrics()'s return value —
    None means there is no production Bz model on disk yet (first-ever
    deploy), in which case every comparison criterion trivially passes
    (there is nothing to be worse than).
    """
    checklist = []

    def _check(name, passed, detail):
        checklist.append({"criterion": name, "passed": bool(passed), "detail": detail})

    # --- Structural compatibility -------------------------------------
    _check(
        "Same prediction target (Bz)",
        candidate_run.get("target") == "Bz",
        f"Candidate target: {candidate_run.get('target')}",
    )
    _check(
        "Same forecast horizon (1h)",
        candidate_run.get("granularity") == "Hourly" and candidate_run.get("horizon") == 1,
        f"Candidate: {candidate_run.get('granularity')} +{candidate_run.get('horizon')}",
    )
    _check(
        f"Same training methodology (Hourly historical CSV, {TEST_FRACTION:.0%} test split)",
        candidate_run.get("granularity") == "Hourly",
        "Candidate uses the same load_research_frame('Hourly') pipeline and TEST_FRACTION split as production.",
    )
    is_sequence = candidate_run.get("model_type") in SEQUENCE_MODELS
    _check(
        "Compatible with the production pipeline (not LSTM/GRU)",
        not is_sequence,
        (
            "LSTM/GRU models cannot be reloaded by predict.py in-process — see load_trained_model's "
            "documented TensorFlow/scikit-learn process conflict."
            if is_sequence
            else f"{candidate_run.get('model_type')} loads via plain joblib, same as production."
        ),
    )

    cand_metrics = candidate_run.get("metrics", {})

    # --- Performance vs. current production ----------------------------
    if production_metrics is None:
        _check("Better R² than current production", True, "No production model currently on disk — nothing to compare against.")
        _check("Lower MAE than current production", True, "No production model currently on disk — nothing to compare against.")
        _check("Lower RMSE than current production", True, "No production model currently on disk — nothing to compare against.")
    else:
        prod_r2 = production_metrics.get("r2", float("-inf"))
        prod_mae = production_metrics.get("mae", float("inf"))
        prod_rmse = production_metrics.get("rmse", float("inf"))
        _check(
            "Better R² than current production",
            cand_metrics.get("r2", float("-inf")) > prod_r2,
            f"Candidate R²={cand_metrics.get('r2', float('nan')):.4f} vs. production R²={prod_r2:.4f}",
        )
        _check(
            "Lower MAE than current production",
            cand_metrics.get("mae", float("inf")) < prod_mae,
            f"Candidate MAE={cand_metrics.get('mae', float('nan')):.4f} vs. production MAE={prod_mae:.4f}",
        )
        _check(
            "Lower RMSE than current production",
            cand_metrics.get("rmse", float("inf")) < prod_rmse,
            f"Candidate RMSE={cand_metrics.get('rmse', float('nan')):.4f} vs. production RMSE={prod_rmse:.4f}",
        )

    # --- Overfitting check ------------------------------------------------
    train_r2 = candidate_run.get("train_r2")
    test_r2 = cand_metrics.get("r2")
    if train_r2 is None or test_r2 is None:
        _check("No obvious overfitting", True, "Train-set R² unavailable for this model type — skipped, review manually.")
    else:
        gap = train_r2 - test_r2
        _check(
            "No obvious overfitting",
            gap <= OVERFIT_R2_GAP_THRESHOLD,
            f"Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, gap={gap:.4f} "
            f"(threshold {OVERFIT_R2_GAP_THRESHOLD:.2f}).",
        )

    # --- Inference speed ---------------------------------------------------
    infer_ms = candidate_run.get("inference_time_ms_per_sample")
    if infer_ms is None:
        _check(
            "Reasonable inference time",
            True,
            "Per-sample inference time unavailable for this model type (subprocess-timed) — review manually.",
        )
    else:
        _check(
            "Reasonable inference time",
            infer_ms <= MAX_REASONABLE_INFERENCE_MS,
            f"{infer_ms:.4f} ms/sample (ceiling {MAX_REASONABLE_INFERENCE_MS:.0f} ms/sample).",
        )

    eligible = all(item["passed"] for item in checklist)
    return {"eligible": eligible, "checklist": checklist}


def run_complete_optimization_study(progress_cb=None) -> dict:
    """AutoML orchestrator — runs the full 10-step pipeline end-to-end with
    zero user interaction beyond this one call. Every candidate it trains
    uses the exact same functions the manual Exp 1-8 tabs call
    (train_research_model_exp, compute_persistence_benchmark) — this
    function only sequences them, tracks the best result at each stage,
    and packages a final study record. It NEVER calls
    promote_to_production(); promotion is always a separate, explicit,
    human-confirmed action in the UI.

    `progress_cb`, if given, is called as progress_cb(step_number,
    total_steps, message) after each major step completes — the caller
    (dashboard UI) uses this to drive a live st.status() panel.
    """
    def _p(step, total, msg):
        if progress_cb:
            progress_cb(step, total, msg)

    total_steps = 10
    study_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # Step 1 — Reproduce production baseline
    baseline = train_research_model_exp(
        "Bz", "Linear Regression", granularity="Hourly", horizon=1,
        feature_set="IMF + All Solar Wind", add_dynamics=False, add_physics=False,
        experiment_tag=f"auto_{study_id}_baseline",
    )
    _p(1, total_steps, f"Baseline reproduced — R²={baseline['metrics']['r2']:.4f}")

    # Step 2 — Persistence benchmark
    persistence = compute_persistence_benchmark("Bz", horizon=1, granularity="Hourly")
    _p(2, total_steps, f"Persistence benchmark — R²={persistence['metrics']['r2']:.4f}")

    # Steps 3-5 — Structured solar-wind / dynamics / physics feature search
    feature_search_results = []
    for combo in FEATURE_SEARCH_GRID:
        run = train_research_model_exp(
            "Bz", FEATURE_SEARCH_PROBE_MODEL, granularity="Hourly", horizon=1,
            feature_set=combo["feature_set"], add_dynamics=combo["add_dynamics"], add_physics=combo["add_physics"],
            experiment_tag=f"auto_{study_id}_feature_search",
        )
        feature_search_results.append({
            "name": combo["name"],
            "feature_set": combo["feature_set"],
            "add_dynamics": combo["add_dynamics"],
            "add_physics": combo["add_physics"],
            "run_id": run["run_id"],
            "r2": run["metrics"]["r2"],
            "mae": run["metrics"]["mae"],
            "rmse": run["metrics"]["rmse"],
            "feature_count": len(run["feature_columns"]),
        })
    feature_search_results.sort(key=lambda r: -r["r2"])
    best_combo = feature_search_results[0]
    _p(
        5, total_steps,
        f"Feature search complete — best: '{best_combo['name']}' (R²={best_combo['r2']:.4f}, "
        f"{best_combo['feature_count']} features)",
    )

    # Step 6 — Run every supported ML model on the winning feature combination
    model_search_results = []
    for model_type in TABULAR_MODELS:
        try:
            run = train_research_model_exp(
                "Bz", model_type, granularity="Hourly", horizon=1,
                feature_set=best_combo["feature_set"], add_dynamics=best_combo["add_dynamics"],
                add_physics=best_combo["add_physics"], experiment_tag=f"auto_{study_id}_model_search",
            )
            model_search_results.append(run)
        except Exception as exc:
            model_search_results.append({"run_id": None, "model_type": model_type, "error": str(exc)})

    if KERAS_AVAILABLE and SEQUENCE_MODELS:
        # Sequence models train on the standard Hourly baseline feature set —
        # train_research_model_exp's configurable feature search doesn't
        # extend to LSTM/GRU (see its docstring); a documented asymmetry,
        # not an oversight.
        for model_type in SEQUENCE_MODELS:
            try:
                run = train_research_model(
                    "Bz", model_type, granularity="Hourly", horizon=1,
                    experiment_tag=f"auto_{study_id}_model_search",
                )
                model_search_results.append(run)
            except Exception as exc:
                model_search_results.append({"run_id": None, "model_type": model_type, "error": str(exc)})

    valid_candidates = [r for r in model_search_results if r.get("run_id")]
    if not valid_candidates:
        raise RuntimeError("Model search produced no successful candidates — cannot complete the study.")
    valid_candidates.sort(key=lambda r: -r["metrics"]["r2"])
    winner = valid_candidates[0]
    _p(6, total_steps, f"Model search complete — winner: {winner['model_type']} (R²={winner['metrics']['r2']:.4f})")

    # Step 7 — Feature importance (already computed for tree/linear winners;
    # if the winner has none — e.g. SVR/MLP — fall back to the best
    # candidate that does, so Exp 7 always has something to show).
    fi_source = winner
    if not fi_source.get("feature_importance"):
        with_fi = [r for r in valid_candidates if r.get("feature_importance")]
        fi_source = with_fi[0] if with_fi else None
    feature_importance = fi_source["feature_importance"] if fi_source else None
    feature_importance_source_run_id = fi_source["run_id"] if fi_source else None
    _p(7, total_steps, "Feature importance extracted." if feature_importance else "Feature importance unavailable for all candidates.")

    # Step 8 — SHAP analysis on the winner (or nearest SHAP-supported candidate)
    shap_source_run_id = None
    for candidate in valid_candidates:
        if candidate["model_type"] in SHAP_SUPPORTED_MODELS:
            shap_source_run_id = candidate["run_id"]
            break
    if shap_source_run_id:
        try:
            shap_result = compute_shap_importance(shap_source_run_id)
        except Exception as exc:
            shap_result = {"run_id": shap_source_run_id, "supported": False, "skipped_reason": f"SHAP failed: {exc}", "shap_importance": None}
    else:
        shap_result = {"run_id": None, "supported": False, "skipped_reason": "No SHAP-supported candidate in this study's model search.", "shap_importance": None}
    _p(8, total_steps, "SHAP analysis complete." if shap_result.get("supported") else f"SHAP skipped: {shap_result.get('skipped_reason')}")

    # Step 9 — Leaderboard (every model-search candidate, ranked)
    leaderboard = []
    for rank, r in enumerate(valid_candidates, start=1):
        is_seq = r["model_type"] in SEQUENCE_MODELS
        feat_name = (
            "Hourly baseline (fixed — sequence models skip feature search)"
            if is_seq else best_combo["name"]
        )
        leaderboard.append({
            "rank": rank,
            "run_id": r["run_id"],
            "model_type": r["model_type"],
            "feature_set_name": feat_name,
            "r2": r["metrics"]["r2"],
            "mae": r["metrics"]["mae"],
            "rmse": r["metrics"]["rmse"],
            "mape": r["metrics"].get("mape"),
            "bias": r["metrics"]["bias"],
            "training_time_sec": r.get("training_time_sec"),
            "inference_time_ms_per_sample": r.get("inference_time_ms_per_sample"),
            "model_size_kb": r.get("model_size_kb"),
            "feature_count": len(r.get("feature_columns", [])),
        })
    failed_candidates = [r for r in model_search_results if not r.get("run_id")]
    _p(9, total_steps, f"Leaderboard generated — {len(leaderboard)} ranked candidates.")

    # Step 10 — Production comparison + recommendation
    production_metrics = get_production_bz_metrics()
    promotion_check = check_promotion_criteria(winner, production_metrics)
    if production_metrics is not None:
        production_comparison = {
            "current": {
                "algorithm": production_metrics.get("algorithm"),
                "r2": production_metrics.get("r2"),
                "mae": production_metrics.get("mae"),
                "rmse": production_metrics.get("rmse"),
            },
            "candidate": {
                "algorithm": winner["model_type"],
                "r2": winner["metrics"]["r2"],
                "mae": winner["metrics"]["mae"],
                "rmse": winner["metrics"]["rmse"],
            },
            "delta_r2": winner["metrics"]["r2"] - production_metrics.get("r2", 0),
            "delta_mae": winner["metrics"]["mae"] - production_metrics.get("mae", 0),
            "delta_rmse": winner["metrics"]["rmse"] - production_metrics.get("rmse", 0),
        }
    else:
        production_comparison = {
            "current": None,
            "candidate": {
                "algorithm": winner["model_type"],
                "r2": winner["metrics"]["r2"],
                "mae": winner["metrics"]["mae"],
                "rmse": winner["metrics"]["rmse"],
            },
            "delta_r2": None, "delta_mae": None, "delta_rmse": None,
        }
    recommendation = "Promote" if promotion_check["eligible"] else "Keep Current Production"
    _p(10, total_steps, f"Recommendation: {recommendation}")

    completed_at = datetime.now(timezone.utc).isoformat()
    study = {
        "study_id": study_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "target": "Bz",
        "granularity": "Hourly",
        "horizon": 1,
        "training_dataset": "data/features/training_v2/{imf,solar_wind}_features.csv",
        "baseline_run_id": baseline["run_id"],
        "baseline_metrics": baseline["metrics"],
        "persistence_run_id": persistence["run_id"],
        "persistence_metrics": persistence["metrics"],
        "feature_search_results": feature_search_results,
        "best_feature_combo": best_combo,
        "models_tested": [r.get("model_type") for r in model_search_results],
        "feature_sets_tested": [c["name"] for c in FEATURE_SEARCH_GRID],
        "failed_candidates": [{"model_type": r.get("model_type"), "error": r.get("error")} for r in failed_candidates],
        "leaderboard": leaderboard,
        "winner_run_id": winner["run_id"],
        "winner_model_type": winner["model_type"],
        "winner_metrics": winner["metrics"],
        "feature_importance": feature_importance,
        "feature_importance_source_run_id": feature_importance_source_run_id,
        "shap_result": shap_result,
        "production_comparison": production_comparison,
        "promotion_check": promotion_check,
        "recommendation": recommendation,
        "promotion_status": "pending",
        "promoted_run_id": None,
        "promoted_at": None,
    }
    _append_study(study)
    return study


def mark_study_promoted(study_id: str, run_id: str) -> bool:
    """Records that a study's winning candidate was promoted — called by
    the UI immediately after a successful promote_to_production() so the
    study's own history entry reflects what actually happened, distinct
    from the run-level "promoted" flag already set on the run record
    itself by promote_to_production().
    """
    return _update_study(
        study_id,
        promotion_status="promoted",
        promoted_run_id=run_id,
        promoted_at=datetime.now(timezone.utc).isoformat(),
    )


def mark_study_rejected(study_id: str, notes: str = "") -> bool:
    """Records that a human reviewed a completed study and chose to keep
    current production rather than promote the winning candidate."""
    return _update_study(
        study_id,
        promotion_status="rejected",
        rejection_notes=notes,
        rejected_at=datetime.now(timezone.utc).isoformat(),
    )
