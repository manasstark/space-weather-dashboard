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
        worker_result = _run_keras_worker(X_train, y_train, X_test, y_test, worker_meta)
        preds = np.array(worker_result["preds"], dtype="float32")
        metrics = compute_metrics(y_test, preds)
        loss_history = {"loss": worker_result["loss"], "val_loss": worker_result["val_loss"]}
        feature_importance = None  # not supported for sequence models

        # The scaler is part of this model's inference contract — anyone
        # loading it back must apply the identical transform to new
        # features first, or predictions will be meaningless.
        joblib.dump(scaler, model_dir / "scaler.joblib")

        n_train, n_test = len(X_train), len(X_test)
        y_true_out, y_pred_out = y_test, preds
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
        model.fit(X_train, y_train)
        preds = np.asarray(model.predict(X_test))
        metrics = compute_metrics(y_test.to_numpy(), preds)
        loss_history = None

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
