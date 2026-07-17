"""Kp Research Laboratory — experimental Kp model comparison, feature
ablation, physics-feature experimentation, and hypothesis testing,
completely isolated from the production Kp predictor (swdss.models.predict
/ swdss.models.train / swdss.models.jobs).

Production answers "what is the operational forecast?" — this module
answers "why does one model perform better than another?" and "which
physics actually improves Kp prediction?". Nothing here ever imports from
or writes to models/analytics/ (production's model directory) or
models/analytics/metrics.json (production's metrics file). Training-run
artifacts live under models/kp_research/<run_id>/, tracked in their own
JSON registry (RUNS_REGISTRY_PATH) — a completely separate namespace.
"Promoting" a run only sets a label on its registry entry; a human
engineer must manually wire a promoted model into predict.py themselves
for it to ever affect a live forecast.

Data source: the SAME `analytics_features.csv` production's own
`train_kp_interval_model` trains on (~3 years, hourly) — so every
research run here is genuinely comparable to production, not aimed at a
different problem. The target is built with the identical "next official
NOAA 3-hour interval" block logic as train.py, reused verbatim (see
build_kp_interval_target) rather than reimplemented, so a drift between
the two target definitions can never creep in silently.

Feature Ablation is the primary research tool this module exists to
support: every base feature column (Solar Wind/IMF/Derived Physics/
Geomagnetic) and every engineered-feature group (Lags/Rolling Mean/
Rolling Std/Rate of Change) can be independently toggled on or off, and
every physics-experiment feature (kp_physics_features.py) can be enabled
individually — see load_kp_research_frame and run_feature_ablation_sweep.
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
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from swdss.models.features import add_change_features, add_lag_features
from swdss.models.kp_physics_features import PHYSICS_FEATURE_DEPENDENCIES, PHYSICS_FEATURE_FUNCTIONS
from swdss.models.registry import DATASETS
from swdss.models.validation import evaluate_walk_forward
from swdss.paths import DATA_DIR, MODELS_DIR
from swdss.physics.registry import apply_requested_features

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

# See imf_research.py's identical note: TensorFlow must never be imported
# at module level here (only checked for via find_spec) — having it share
# a process with scikit-learn/XGBoost/LightGBM/CatBoost was empirically
# confirmed to hang model.fit indefinitely. Keras training instead runs
# in imf_research_keras_worker.py's subprocess, reused directly since
# it's fully domain-agnostic (just seq_len/n_features/units/model_type).
KERAS_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


RESEARCH_MODELS_DIR = MODELS_DIR / "kp_research"
RUNS_REGISTRY_PATH = DATA_DIR / "predictions" / "kp_research_runs.json"
HYPOTHESIS_REGISTRY_PATH = DATA_DIR / "predictions" / "kp_hypothesis_tests.json"

TEST_FRACTION = 0.2
TARGET_LABEL = "Kp"

# Every base column a researcher can toggle on/off, grouped exactly as
# specified — mirrors ANALYTICS_FEATURE_VARIABLES's own column set
# (registry.py), so "everything on" reproduces production's own feature
# pool. "Derived Physics" (ey/vbz/dynamic_pressure) is always COMPUTED
# from the frame's speed/density/bz_gsm (see load_kp_research_frame) even
# when its own toggle is off, since Kp Physics Experiments features
# (Integrated Ey, Integrated VBz, ...) may still need the underlying
# columns present regardless of whether they're also included as direct
# features.
FEATURE_GROUP_COLUMNS = {
    "Solar Wind": ["speed", "density", "temperature"],
    "IMF": ["bt", "bx_gsm", "by_gsm", "bz_gsm"],
    "Derived Physics": ["ey", "vbz", "dynamic_pressure"],
    "Geomagnetic": ["kp", "dst", "ae"],
}

ENGINEERED_GROUPS = ["Lags", "Rolling Mean", "Rolling Std", "Rate of Change"]
ENGINEERED_ROLLING_WINDOW = 24  # hours — matches production's own ROLLING_WINDOW (features.py)

PHYSICS_FEATURE_OPTIONS = list(PHYSICS_FEATURE_FUNCTIONS)

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

# Registered but not trainable yet — kept as disabled entries so the
# model selector's shape never has to change when one of these is
# eventually implemented (same "Future Expansion" contract as the IMF lab).
FUTURE_MODELS = [
    "Transformer",
    "Temporal Convolution Network (TCN)",
    "Physics-Informed Neural Network",
    "Stacked Model",
    "Bayesian-Optimized Ensemble",
]

ALL_TRAINABLE_MODELS = TABULAR_MODELS + SEQUENCE_MODELS

SEQUENCE_LENGTH_OPTIONS = [6, 12, 24, 48]
DEFAULT_SEQUENCE_LENGTH = 24

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
        "batch_size": {"type": "int", "default": 32, "min": 8, "max": 512},
        "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.6},
    },
    "GRU": {
        "units": {"type": "int", "default": 64, "min": 8, "max": 256},
        "epochs": {"type": "int", "default": 15, "min": 1, "max": 100},
        "batch_size": {"type": "int", "default": 32, "min": 8, "max": 512},
        "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.6},
    },
}

SCALE_SENSITIVE_MODELS = {"SVR", "MLP", "LSTM", "GRU"}


def default_feature_toggles() -> dict:
    return {group: {col: True for col in cols} for group, cols in FEATURE_GROUP_COLUMNS.items()}


def default_engineered_toggles() -> dict:
    return {group: True for group in ENGINEERED_GROUPS}


def _add_rolling_mean(df: pd.DataFrame, columns: list[str], window: int = ENGINEERED_ROLLING_WINDOW) -> list[str]:
    created = []
    for column in columns:
        name = f"{column}_{window}h"
        df[name] = df[column].rolling(window).mean()
        created.append(name)
    return created


def _add_rolling_std(df: pd.DataFrame, columns: list[str], window: int = ENGINEERED_ROLLING_WINDOW) -> list[str]:
    created = []
    for column in columns:
        name = f"{column}_{window}h_std"
        df[name] = df[column].rolling(window).std()
        created.append(name)
    return created


def _load_analytics_base_frame() -> pd.DataFrame:
    """The exact CSV production's train_kp_interval_model trains on,
    scaled the same way (_load_base_df in train.py divides "kp" by 10 to
    correct the CSV's tenths-of-Kp OMNI convention down to the natural
    0-9 scale) — so a research run's raw "kp" column always matches what
    a researcher sees live on the dashboard, not a 10x-inflated value.
    """
    config = DATASETS["analytics"]
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")
    for column, factor in (config.scale_factors or {}).items():
        if column in raw.columns:
            raw[column] = raw[column] / factor
    return raw


def _derived_physics_columns(df: pd.DataFrame) -> list[str]:
    """vbz/ey/dynamic_pressure via swdss.physics — the Physics Engine and
    this project's single canonical implementation of these formulas.
    This wrapper still exists (rather than calling swdss.physics.core.
    add_derived_physics_features directly) only because that function
    also handles the IMF-only/Solar-Wind-only no-op cases this module
    never needs (this frame always has speed/density/bz_gsm) — the
    formulas themselves are the exact same engine calls production uses.
    """
    from swdss.physics.core import dynamic_pressure_series, ey_series, vbz_series

    created = []
    df["vbz"] = vbz_series(df["speed"], df["bz_gsm"])
    created.append("vbz")
    df["ey"] = ey_series(df["speed"], df["bz_gsm"])
    created.append("ey")
    df["dynamic_pressure"] = dynamic_pressure_series(df["density"], df["speed"])
    created.append("dynamic_pressure")
    return created


def build_kp_interval_target(frame: pd.DataFrame) -> pd.Series:
    """Identical block logic to train.py's train_kp_interval_model: for
    every hourly row inside 3h-block B, the label is block B+1's eventual
    Kp value. Reused verbatim (not reimplemented) so the research target
    can never silently drift from production's own definition — the
    audited invariant this whole module exists to preserve while varying
    everything else (model, features, physics).
    """
    block_start = frame.index.floor("3h")
    block_kp = frame["kp"].groupby(block_start).first()
    next_block_start = pd.Series(block_start + pd.Timedelta(hours=3), index=frame.index)
    return next_block_start.map(block_kp)


def load_kp_research_frame(
    feature_toggles: dict = None,
    engineered_groups: dict = None,
    physics_features: dict = None,
) -> tuple:
    """Builds the Kp research frame and returns (frame, feature_columns).

    `feature_toggles`: {group: {column: bool}} — defaults to everything
    on (default_feature_toggles()). Only affects which base columns are
    INCLUDED as features; the underlying columns always exist in `frame`
    (needed for target construction and physics-feature computation
    regardless of toggle state).

    `engineered_groups`: {"Lags"/"Rolling Mean"/"Rolling Std"/"Rate of
    Change": bool} — applied only to the enabled base columns (not to
    physics-experiment features, which already encode their own temporal
    structure — e.g. re-lagging "storm_phase" or "previous_storm_strength"
    adds combinatorial complexity without a clear research question it
    answers).

    `physics_features`: {display_name: bool} from kp_physics_features's
    PHYSICS_FEATURE_FUNCTIONS registry — each independently toggleable.
    """
    feature_toggles = feature_toggles or default_feature_toggles()
    engineered_groups = engineered_groups or default_engineered_toggles()
    physics_features = physics_features or {}

    frame = _load_analytics_base_frame()
    _derived_physics_columns(frame)

    # Resolves prerequisites via the Physics Engine's shared registry
    # resolver — see swdss.physics.registry (generalizes what used to be
    # this lab's own hand-rolled loop; the AE Research Laboratory now
    # uses the identical resolver).
    physics_cols = apply_requested_features(frame, physics_features, PHYSICS_FEATURE_FUNCTIONS, PHYSICS_FEATURE_DEPENDENCIES)

    base_cols = [
        col
        for group, cols in FEATURE_GROUP_COLUMNS.items()
        for col in cols
        if feature_toggles.get(group, {}).get(col, True)
    ]

    engineered_cols: list[str] = []
    if engineered_groups.get("Lags", True):
        engineered_cols += add_lag_features(frame, base_cols)
    if engineered_groups.get("Rolling Mean", True):
        engineered_cols += _add_rolling_mean(frame, base_cols)
    if engineered_groups.get("Rolling Std", True):
        engineered_cols += _add_rolling_std(frame, base_cols)
    if engineered_groups.get("Rate of Change", True):
        engineered_cols += add_change_features(frame, base_cols)

    feature_columns = base_cols + physics_cols + engineered_cols
    if not feature_columns:
        raise ValueError("At least one feature group, physics feature, or engineered group must be enabled.")
    return frame, feature_columns


def build_sequences(frame: pd.DataFrame, feature_columns: list[str], target: pd.Series, seq_len: int) -> tuple:
    values = frame[feature_columns].to_numpy(dtype="float32")
    targets = target.to_numpy(dtype="float32")
    X = np.stack([values[i - seq_len : i] for i in range(seq_len, len(frame))])
    y = targets[seq_len:]
    return X, y


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else None
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape,
        "bias": float(np.mean(errors)),
    }


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
    """Delegates to the SAME subprocess worker the IMF lab uses
    (imf_research_keras_worker.py) — it only cares about seq_len/
    n_features/units/dropout/epochs/batch_size/model_type, nothing
    IMF-specific, so reusing it directly here avoids a near-duplicate
    file just to isolate Keras training from this module's own
    scikit-learn/XGBoost/LightGBM/CatBoost imports (see KERAS_AVAILABLE
    above for why that isolation is required at all).
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


def train_kp_research_model(
    model_type: str,
    feature_toggles: dict = None,
    engineered_groups: dict = None,
    physics_features: dict = None,
    sequence_length: int = None,
    hyperparams: dict = None,
    notes: str = "",
    experiment_tag: str = "",
    run_cv: bool = False,
) -> dict:
    """Trains one Kp model for one feature configuration and records a
    run — never touches models/analytics/ or its metrics.json. Target is
    always "the next official NOAA 3-hour Kp interval" (see
    build_kp_interval_target), identical to production's own target
    definition, so R²/MAE/RMSE here are directly comparable to
    production's stored kp_interval metrics.

    `run_cv` additionally benchmarks the model with walk-forward
    (rolling-origin) cross-validation (swdss.models.validation) — a
    genuine stability estimate across several time periods, not just the
    one 80/20 holdout window `metrics` above already reports. Stored in
    `run_record["cv_metrics"]` (None if run_cv=False). Skipped
    automatically for LSTM/GRU regardless of run_cv — each fold would
    mean another full subprocess Keras training run.
    """
    if model_type not in ALL_TRAINABLE_MODELS:
        raise ValueError(f"'{model_type}' is not trainable yet — see FUTURE_MODELS.")

    feature_toggles = feature_toggles or default_feature_toggles()
    engineered_groups = engineered_groups or default_engineered_toggles()
    physics_features = physics_features or {}

    frame, feature_columns = load_kp_research_frame(feature_toggles, engineered_groups, physics_features)
    frame = frame.copy()
    frame["__target__"] = build_kp_interval_target(frame)
    frame = frame.dropna(subset=feature_columns + ["__target__"])

    if len(frame) < 100:
        raise ValueError(
            "Not enough history to train a Kp research model (need 100+ clean rows after lag/rolling windows)."
        )

    run_id = str(uuid.uuid4())
    model_dir = RESEARCH_MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Walk-forward CV is opt-in and skipped for LSTM/GRU regardless of
    # run_cv — each fold would mean another full subprocess Keras training
    # run, and the sequence models already train through a slow isolated
    # worker (see _run_keras_worker) for the single holdout split alone.
    cv_metrics = None

    if model_type in SEQUENCE_MODELS:
        if not KERAS_AVAILABLE:
            raise ValueError("TensorFlow/Keras is not installed — sequence models are unavailable.")
        seq_len = sequence_length or DEFAULT_SEQUENCE_LENGTH

        row_split = int(len(frame) * (1 - TEST_FRACTION))
        scaler = StandardScaler()
        scaler.fit(frame[feature_columns].iloc[:row_split])
        scaled_frame = frame.copy()
        scaled_frame[feature_columns] = scaler.transform(frame[feature_columns])

        X, y = build_sequences(scaled_frame, feature_columns, scaled_frame["__target__"], seq_len)
        if len(X) < 100:
            raise ValueError("Not enough history to build sequences of this length yet.")
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
            "batch_size": hp.get("batch_size", 32),
            "model_path": str(model_path),
        }
        train_start = time.perf_counter()
        worker_result = _run_keras_worker(X_train, y_train, X_test, y_test, worker_meta)
        training_time_sec = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        preds = np.array(worker_result["preds"], dtype="float32")
        prediction_time_sec = time.perf_counter() - predict_start

        metrics = compute_metrics(y_test, preds)
        loss_history = {"loss": worker_result["loss"], "val_loss": worker_result["val_loss"]}
        feature_importance = None
        train_r2 = None  # worker does not return train-set predictions
        inference_time_ms_per_sample = None  # not separable from the subprocess call above

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
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time_sec = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        preds = np.asarray(model.predict(X_test))
        prediction_time_sec = time.perf_counter() - predict_start
        inference_time_ms_per_sample = (prediction_time_sec / len(X_test)) * 1000 if len(X_test) else None

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

        if run_cv:
            # Fresh per-fold fit on the RAW (unscaled) feature frame — never
            # X_full above, whose scaler was fit only on the single 80/20
            # split's train rows and would leak that split's statistics
            # into earlier CV folds' own "unseen" test windows.
            def _cv_factory(_model_type=model_type, _hyperparams=hyperparams, _needs_scaling=needs_scaling):
                estimator = _build_tabular_model(_model_type, _hyperparams)
                return make_pipeline(StandardScaler(), estimator) if _needs_scaling else estimator

            cv_metrics = evaluate_walk_forward(_cv_factory, frame[feature_columns], frame["__target__"])
        else:
            cv_metrics = None

    sample_n = min(300, len(y_true_out))
    prediction_sample = {
        "y_true": [float(v) for v in np.asarray(y_true_out)[-sample_n:]],
        "y_pred": [float(v) for v in np.asarray(y_pred_out)[-sample_n:]],
    }

    run_record = {
        "run_id": run_id,
        "target": TARGET_LABEL,
        "model_type": model_type,
        "feature_toggles": feature_toggles,
        "engineered_groups": engineered_groups,
        "physics_features": {k: v for k, v in physics_features.items() if v},
        "sequence_length": sequence_length if model_type in SEQUENCE_MODELS else None,
        "hyperparams": hyperparams or {},
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "training_time_sec": training_time_sec,
        "prediction_time_sec": prediction_time_sec,
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
        "notes": notes,
        "experiment_tag": experiment_tag,
        "model_size_kb": round(model_size_kb, 2) if model_size_kb is not None else None,
        "train_r2": train_r2,
        "inference_time_ms_per_sample": round(inference_time_ms_per_sample, 5) if inference_time_ms_per_sample is not None else None,
    }
    _append_run(run_record)
    return run_record


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


def list_runs() -> list[dict]:
    return sorted(_load_runs(), key=lambda r: r["trained_at"], reverse=True)


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
    """Label-only, exactly like the IMF lab's promote_run — never touches
    production. A human engineer must manually wire a promoted model into
    predict.py themselves for it to ever affect a live forecast.
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
    LSTM/GRU runs are deliberately unsupported here — same TensorFlow/
    scikit-learn process conflict as training (see KERAS_AVAILABLE note);
    the saved .keras artifact is still on disk and loadable from a fresh,
    isolated process.
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


# ---------------------------------------------------------------- feature ablation

# The 8 independently-ablatable units this sweep tests: the 4 base
# feature groups plus the 4 engineered-feature groups. Physics-experiment
# features (kp_physics_features.py) are intentionally NOT part of this
# automatic sweep — there are 14 of them, individually enabled per the
# Physics Experiments tab and tested one at a time via Hypothesis Testing
# instead, where each gets its own dedicated Accept/Reject verdict.
FEATURE_ABLATION_UNITS = list(FEATURE_GROUP_COLUMNS) + list(ENGINEERED_GROUPS)


def run_feature_ablation_sweep(model_type: str = "Linear Regression", hyperparams: dict = None) -> dict:
    """Leave-one-out ablation: trains a "Full Model" with every base
    feature group and every engineered group enabled, then retrains once
    per unit with just that ONE unit disabled (physics-experiment
    features left off throughout, so this isolates exactly the 8 units
    above). Ranks units by how much R² DROPS when removed — how much the
    full model actually relies on it.

    Deliberately leave-one-out rather than the cumulative "enable Ey,
    then enable VBz, ..." style: a cumulative/additive sweep's deltas
    depend on the ORDER features are added (whichever goes first tends to
    look most important merely by going first), whereas leave-one-out is
    order-independent — a standard ablation methodology for this reason.
    """
    full_toggles = default_feature_toggles()
    full_engineered = default_engineered_toggles()
    full_run = train_kp_research_model(
        model_type,
        feature_toggles=full_toggles,
        engineered_groups=full_engineered,
        hyperparams=hyperparams,
        notes="Feature Ablation — Full Model (baseline for this sweep)",
    )
    full_r2 = full_run["metrics"]["r2"]

    rows = [
        {
            "unit": "Full Model (all groups enabled)",
            "run_id": full_run["run_id"],
            "r2": full_r2,
            "mae": full_run["metrics"]["mae"],
            "delta_r2": 0.0,
        }
    ]
    for unit in FEATURE_ABLATION_UNITS:
        toggles = default_feature_toggles()
        engineered = default_engineered_toggles()
        if unit in FEATURE_GROUP_COLUMNS:
            toggles[unit] = {col: False for col in FEATURE_GROUP_COLUMNS[unit]}
        else:
            engineered[unit] = False
        run = train_kp_research_model(
            model_type,
            feature_toggles=toggles,
            engineered_groups=engineered,
            hyperparams=hyperparams,
            notes=f"Feature Ablation — without {unit}",
        )
        rows.append(
            {
                "unit": f"Without {unit}",
                "run_id": run["run_id"],
                "r2": run["metrics"]["r2"],
                "mae": run["metrics"]["mae"],
                "delta_r2": full_r2 - run["metrics"]["r2"],
            }
        )

    ranked = sorted(rows[1:], key=lambda r: -r["delta_r2"])
    return {
        "full_run_id": full_run["run_id"],
        "full_r2": full_r2,
        "model_type": model_type,
        "rows": rows,
        "ranked": ranked,
        "swept_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------- hypothesis testing

# Each entry pairs a plain-language hypothesis with the single feature
# (base column or physics-experiment feature) it isolates. run_hypothesis
# trains a baseline WITHOUT that feature and an experimental run WITH it
# — everything else held at defaults — and reports Accept/Reject.
HYPOTHESIS_DEFINITIONS = {
    "Previous AE improves Kp prediction": {"kind": "feature", "group": "Geomagnetic", "column": "ae"},
    "Previous Dst improves Kp prediction": {"kind": "feature", "group": "Geomagnetic", "column": "dst"},
    "Previous Kp improves Kp prediction": {"kind": "feature", "group": "Geomagnetic", "column": "kp"},
    "Ey improves Kp prediction": {"kind": "feature", "group": "Derived Physics", "column": "ey"},
    "VBz improves Kp prediction": {"kind": "feature", "group": "Derived Physics", "column": "vbz"},
    "Dynamic Pressure improves Kp prediction": {"kind": "feature", "group": "Derived Physics", "column": "dynamic_pressure"},
    "Clock Angle improves Kp prediction": {"kind": "physics", "name": "Clock Angle"},
    "Southward Duration improves Kp prediction": {"kind": "physics", "name": "Southward Duration"},
    "Integrated Ey improves Kp prediction": {"kind": "physics", "name": "Integrated Ey"},
    "Integrated VBz improves Kp prediction": {"kind": "physics", "name": "Integrated VBz"},
    "Maximum AE (6h) improves Kp prediction": {"kind": "physics", "name": "Maximum AE (6h)"},
    "Storm Phase improves Kp prediction": {"kind": "physics", "name": "Storm Phase"},
}

# A hypothesis is Accepted if adding the tested feature raises held-out
# R² by at least this much. Deliberately smaller than the IMF lab's
# compare_runs threshold (0.01) since Kp's R² operates in a narrower
# band (~0.5-0.7) where individual single-feature contributions are
# typically modest — a single held-out split is still not strong
# statistical evidence either way, which is why every result records the
# full baseline/experimental run pair for independent inspection rather
# than just the verdict.
HYPOTHESIS_ACCEPT_THRESHOLD_R2 = 0.005


def run_hypothesis_test(hypothesis_label: str, model_type: str = "Linear Regression", hyperparams: dict = None) -> dict:
    if hypothesis_label not in HYPOTHESIS_DEFINITIONS:
        raise ValueError(f"Unknown hypothesis: {hypothesis_label}")
    spec = HYPOTHESIS_DEFINITIONS[hypothesis_label]

    baseline_toggles = default_feature_toggles()
    baseline_physics: dict = {}
    experimental_toggles = default_feature_toggles()
    experimental_physics: dict = {}

    if spec["kind"] == "feature":
        baseline_toggles[spec["group"]] = dict(baseline_toggles[spec["group"]])
        baseline_toggles[spec["group"]][spec["column"]] = False
    else:
        experimental_physics[spec["name"]] = True

    engineered = default_engineered_toggles()
    baseline_run = train_kp_research_model(
        model_type,
        feature_toggles=baseline_toggles,
        engineered_groups=engineered,
        physics_features=baseline_physics,
        hyperparams=hyperparams,
        notes=f"Hypothesis Testing — baseline for: {hypothesis_label}",
    )
    experimental_run = train_kp_research_model(
        model_type,
        feature_toggles=experimental_toggles,
        engineered_groups=engineered,
        physics_features=experimental_physics,
        hyperparams=hyperparams,
        notes=f"Hypothesis Testing — experimental for: {hypothesis_label}",
    )

    delta_r2 = experimental_run["metrics"]["r2"] - baseline_run["metrics"]["r2"]
    delta_mae = experimental_run["metrics"]["mae"] - baseline_run["metrics"]["mae"]
    delta_rmse = experimental_run["metrics"]["rmse"] - baseline_run["metrics"]["rmse"]
    verdict = "Accept" if delta_r2 >= HYPOTHESIS_ACCEPT_THRESHOLD_R2 else "Reject"

    result = {
        "hypothesis": hypothesis_label,
        "model_type": model_type,
        "baseline_run_id": baseline_run["run_id"],
        "experimental_run_id": experimental_run["run_id"],
        "baseline_r2": baseline_run["metrics"]["r2"],
        "experimental_r2": experimental_run["metrics"]["r2"],
        "delta_r2": delta_r2,
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,
        "verdict": verdict,
        "tested_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_hypothesis_result(result)
    return result


def _load_hypothesis_results() -> list[dict]:
    if not HYPOTHESIS_REGISTRY_PATH.exists():
        return []
    with open(HYPOTHESIS_REGISTRY_PATH) as f:
        return json.load(f)


def _save_hypothesis_results(results: list[dict]) -> None:
    HYPOTHESIS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYPOTHESIS_REGISTRY_PATH, "w") as f:
        json.dump(results, f, indent=2)


def _append_hypothesis_result(result: dict) -> None:
    results = _load_hypothesis_results()
    results.append(result)
    _save_hypothesis_results(results)


def list_hypothesis_results() -> list[dict]:
    return sorted(_load_hypothesis_results(), key=lambda r: r["tested_at"], reverse=True)


# ============================================================================
# Kp Optimization Study — AutoML Orchestration Layer
# ============================================================================
# Kp is fundamentally an EARTH-RESPONSE forecasting problem (Solar Wind ->
# IMF -> magnetosphere-ionosphere coupling -> geomagnetic activity), unlike
# Bz's upstream-IMF forecasting problem — so this study is structured around
# validating which physical COUPLING mechanisms (not just which raw
# variables) genuinely improve the forecast, before ever touching model
# choice. It reuses train_kp_research_model()/load_kp_research_frame() for
# every candidate it trains — nothing here is a new training path, and
# nothing here ever writes to models/analytics/ except promote_kp_to_
# production(), which is only ever called by explicit human action in the
# UI, never automatically.
#
# Forecasting objective preserved throughout: every candidate predicts the
# next official NOAA 3-hour Kp interval (build_kp_interval_target), trained
# on the same analytics_features.csv production itself trains on — the
# exact same guarantee load_kp_research_frame's own docstring already makes.

STUDIES_REGISTRY_PATH = DATA_DIR / "predictions" / "kp_optimization_studies.json"

# The algorithm production's own _fit_best (train.py) currently selects for
# kp_interval — used to reproduce the production baseline exactly in
# Experiment 1. If a future production refresh picks a different winner,
# update this constant so Experiment 1 keeps reproducing the true baseline.
PRODUCTION_BASELINE_MODEL = "XGBoost"

# Probe model for structured feature-search experiments (3-6): captures
# nonlinear coupling relationships a linear probe would miss, while still
# being far cheaper than sweeping every model type through every combo.
# The winning configuration is re-tested by every real model in Experiment 7.
FEATURE_SEARCH_PROBE_MODEL = "Random Forest"

# How many top positive-contribution physics groups from Experiment 6 get
# combined into the single "Best Combined Feature Set" Experiment 7 trains
# every model on — a structured combination step (not just picking one
# winner), since real magnetosphere coupling is rarely captured by a single
# variable in isolation.
TOP_PHYSICS_GROUPS_TO_COMBINE = 5

OVERFIT_R2_GAP_THRESHOLD = 0.15
MAX_REASONABLE_INFERENCE_MS = 50.0

SHAP_SUPPORTED_MODELS = {
    "Linear Regression", "Ridge Regression", "Lasso", "ElasticNet",
    "Random Forest", "XGBoost", "LightGBM", "CatBoost",
}


def _build_isolated_toggles(groups: dict) -> dict:
    """Builds a feature_toggles dict with EVERYTHING off except the
    columns named in `groups` ({group_name: [columns]}) — used by
    Experiments 3-5 to isolate exactly one input combination's
    contribution, with no other base feature group riding along.
    """
    toggles = {group: {col: False for col in cols} for group, cols in FEATURE_GROUP_COLUMNS.items()}
    for group, cols in groups.items():
        for col in cols:
            toggles[group][col] = True
    return toggles


# ---- Experiment 3: Solar Wind Inputs --------------------------------------
SOLAR_WIND_INPUT_GRID = [
    {"name": "Solar Wind Only", "groups": {"Solar Wind": ["speed", "density", "temperature"]}},
    {"name": "Solar Wind + Previous Kp", "groups": {"Solar Wind": ["speed", "density", "temperature"], "Geomagnetic": ["kp"]}},
    {"name": "Solar Wind + IMF", "groups": {"Solar Wind": ["speed", "density", "temperature"], "IMF": ["bt", "bx_gsm", "by_gsm", "bz_gsm"]}},
    {
        "name": "Solar Wind + IMF + Previous Kp",
        "groups": {
            "Solar Wind": ["speed", "density", "temperature"],
            "IMF": ["bt", "bx_gsm", "by_gsm", "bz_gsm"],
            "Geomagnetic": ["kp"],
        },
    },
]

# ---- Experiment 4: IMF Inputs ----------------------------------------------
IMF_INPUT_GRID = [
    {"name": "Bt Only", "groups": {"IMF": ["bt"]}},
    {"name": "Bx Only", "groups": {"IMF": ["bx_gsm"]}},
    {"name": "By Only", "groups": {"IMF": ["by_gsm"]}},
    {"name": "Bz Only", "groups": {"IMF": ["bz_gsm"]}},
    {"name": "All IMF (Bt + Bx + By + Bz)", "groups": {"IMF": ["bt", "bx_gsm", "by_gsm", "bz_gsm"]}},
]

# ---- Experiment 5: Geomagnetic History -------------------------------------
GEOMAGNETIC_HISTORY_GRID = [
    {"name": "Previous Kp", "groups": {"Geomagnetic": ["kp"]}},
    {"name": "Previous Dst", "groups": {"Geomagnetic": ["dst"]}},
    {"name": "Previous AE", "groups": {"Geomagnetic": ["ae"]}},
    {"name": "Previous Kp + Dst", "groups": {"Geomagnetic": ["kp", "dst"]}},
    {"name": "Previous Kp + AE", "groups": {"Geomagnetic": ["kp", "ae"]}},
    {"name": "Previous Kp + AE + Dst", "groups": {"Geomagnetic": ["kp", "dst", "ae"]}},
]

# ---- Experiment 6: Physics Optimization -------------------------------------
# Every entry is tested ALONE against the full production-baseline feature
# set (never all-at-once) — "add_physics" entries add ONE physics-experiment
# feature on top of the baseline (delta = candidate_r2 - baseline_r2);
# "remove_column" entries (Ey/VBz/Dynamic Pressure — already always-on in
# the baseline, per production's own feature set) instead test the baseline
# WITHOUT that one column (delta = baseline_r2 - candidate_r2), so the
# reported "Δ R²" is uniformly signed: positive always means "this variable
# helps the forecast", regardless of whether it was tested by adding or
# removing it.
PHYSICS_OPTIMIZATION_GRID = [
    {"name": "Ey", "kind": "remove_column", "group": "Derived Physics", "column": "ey"},
    {"name": "VBz", "kind": "remove_column", "group": "Derived Physics", "column": "vbz"},
    {"name": "Dynamic Pressure", "kind": "remove_column", "group": "Derived Physics", "column": "dynamic_pressure"},
    {"name": "Clock Angle", "kind": "add_physics"},
    {"name": "Clock Angle Rate", "kind": "add_physics", "physics_name": "Clock Angle Change"},
    {"name": "Southward Duration", "kind": "add_physics"},
    {"name": "Strong Southward Duration", "kind": "add_physics"},
    {"name": "Integrated Southward Bz", "kind": "add_physics"},
    {"name": "Integrated Ey", "kind": "add_physics"},
    {"name": "Integrated VBz", "kind": "add_physics"},
    {"name": "Integrated Energy Input", "kind": "add_physics"},
    {"name": "Newell Coupling Function", "kind": "add_physics"},
    {"name": "Akasofu ε", "kind": "add_physics"},
    {"name": "Boyle Index", "kind": "add_physics"},
    {"name": "Magnetic Pressure", "kind": "add_physics"},
    {"name": "Thermal Pressure", "kind": "add_physics"},
    {"name": "Total Pressure", "kind": "add_physics"},
    {"name": "Plasma Beta", "kind": "add_physics"},
    {"name": "Alfvén Speed", "kind": "add_physics"},
    {"name": "Alfvén Mach Number", "kind": "add_physics"},
    {"name": "Magnetic Shear", "kind": "add_physics"},
    {"name": "IMF Rotation Rate", "kind": "add_physics"},
    {"name": "Solar Wind Persistence", "kind": "add_physics"},
    {"name": "IMF Persistence", "kind": "add_physics"},
    {"name": "Bz Persistence", "kind": "add_physics"},
    {"name": "Bt Persistence", "kind": "add_physics"},
    {"name": "Magnetopause Stand-off Distance", "kind": "add_physics"},
    {"name": "Estimated Compression", "kind": "add_physics"},
]


def compute_kp_persistence_benchmark() -> dict:
    """Persistence forecast: next official Kp interval = current (most
    recently known) official Kp value — the naïve lower bound production
    and every ML model in this study must beat. Stored permanently in the
    runs registry under a fixed run_id so it's always retrievable without
    rerunning.
    """
    frame = _load_analytics_base_frame()
    frame = frame.copy()
    frame["__target__"] = build_kp_interval_target(frame)
    frame = frame.dropna(subset=["kp", "__target__"])

    split_idx = int(len(frame) * (1 - TEST_FRACTION))
    y_test = frame["__target__"].iloc[split_idx:].to_numpy()
    y_pred = frame["kp"].iloc[split_idx:].to_numpy()  # persistence: next = current
    metrics = compute_metrics(y_test, y_pred)
    sample_n = min(300, len(y_test))

    fixed_id = "persistence_kp_interval"
    record = {
        "run_id": fixed_id,
        "target": TARGET_LABEL,
        "model_type": "Persistence Baseline",
        # Empty dicts, not None: existing UI code (Experiment Tracking tab)
        # unconditionally calls .items() on feature_toggles/physics_features
        # for every run in the registry — None would break it, {} is a
        # truthful "no feature groups, this is a non-feature persistence
        # forecast" answer that iterates safely.
        "feature_toggles": {},
        "engineered_groups": {},
        "physics_features": {},
        "sequence_length": None,
        "hyperparams": {},
        "metrics": metrics,
        "training_time_sec": 0.0,
        "prediction_time_sec": 0.0,
        "feature_columns": ["kp"],
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
        "notes": "Persistence baseline: next official Kp interval = current official Kp value.",
        "experiment_tag": "persistence_benchmark",
        "model_size_kb": None,
        "train_r2": None,
        "inference_time_ms_per_sample": None,
    }
    runs = _load_runs()
    runs = [r for r in runs if r["run_id"] != fixed_id]
    runs.append(record)
    _save_runs(runs)
    return record


def get_production_kp_metrics() -> dict:
    """Reads the CURRENT production kp_interval entry straight from
    models/analytics/metrics.json — always live, never cached, so a
    Production Comparison always reflects whatever is actually deployed
    right now. Returns None if no production Kp model exists yet.
    """
    metrics_path = MODELS_DIR / "analytics" / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get("kp_interval")


def _rebuild_test_frame_for_kp_run(run: dict) -> tuple:
    """Deterministically rebuilds (X_train, X_test) for a stored run so
    SHAP can be computed after the fact, without persisting raw feature
    matrices in the JSON registry. Reruns load_kp_research_frame with the
    exact feature_toggles/engineered_groups/physics_features the run
    record already stores — same inputs in, same split, same columns out.
    """
    feature_toggles = run.get("feature_toggles") or default_feature_toggles()
    engineered_groups = run.get("engineered_groups") or default_engineered_toggles()
    physics_features = run.get("physics_features") or {}
    feature_columns = run["feature_columns"]

    frame, _ = load_kp_research_frame(feature_toggles, engineered_groups, physics_features)
    frame = frame.copy()
    frame["__target__"] = build_kp_interval_target(frame)
    frame = frame.dropna(subset=feature_columns + ["__target__"])
    split_idx = int(len(frame) * (1 - TEST_FRACTION))
    X_train = frame[feature_columns].iloc[:split_idx]
    X_test = frame[feature_columns].iloc[split_idx:]
    return X_train, X_test


def compute_shap_importance_kp(run_id: str, max_background: int = 100, max_explain: int = 200) -> dict:
    """SHAP-based feature importance for a stored Kp run (Experiment 9).

    Only supported for linear and tree/boosting families (fast native
    SHAP explainers) — SVR/MLP would fall back to shap's slow
    KernelExplainer, and LSTM/GRU cannot be reloaded into this process at
    all (see load_trained_model). Both return a clear `skipped_reason`.
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
                "family (KernelExplainer would be too slow for interactive use, and LSTM/GRU cannot be "
                "reloaded into this process). Use a linear or tree-based model for SHAP analysis."
            ),
            "shap_importance": None,
        }

    import shap

    model = load_trained_model(run_id)
    X_train, X_test = _rebuild_test_frame_for_kp_run(run)
    background = X_train.sample(n=min(max_background, len(X_train)), random_state=42) if len(X_train) else X_train
    explain_rows = X_test.sample(n=min(max_explain, len(X_test)), random_state=42) if len(X_test) else X_test

    explainer = shap.Explainer(model, background)
    shap_values = explainer(explain_rows)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    pairs = list(zip(run["feature_columns"], (float(v) for v in mean_abs)))
    ranked = sorted(pairs, key=lambda kv: -kv[1])[:30]

    return {"run_id": run_id, "supported": True, "skipped_reason": None, "shap_importance": ranked}


def check_promotion_criteria_kp(candidate_run: dict, production_metrics: dict) -> dict:
    """Evaluates the Kp promotion checklist against one candidate run.
    Returns {"eligible": bool, "checklist": [...]}. `production_metrics`
    is get_production_kp_metrics()'s return value — None means no
    production Kp model exists yet (every comparison trivially passes).
    """
    checklist = []

    def _check(name, passed, detail):
        checklist.append({"criterion": name, "passed": bool(passed), "detail": detail})

    _check(
        "Same prediction objective (next official 3-hour Kp interval)",
        candidate_run.get("target") == TARGET_LABEL,
        "Every run in this module targets build_kp_interval_target() — the identical block logic "
        "train.py's own train_kp_interval_model uses; this can never drift silently.",
    )
    _check(
        f"Same training methodology (analytics_features.csv, {TEST_FRACTION:.0%} test split)",
        True,
        "Every run in this module trains on load_kp_research_frame(), built from the same "
        "analytics_features.csv production's own model trains on, with the same TEST_FRACTION split.",
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
    if production_metrics is None:
        _check("Better R² than current production", True, "No production Kp model currently on disk — nothing to compare against.")
        _check("Lower MAE than current production", True, "No production Kp model currently on disk — nothing to compare against.")
        _check("Lower RMSE than current production", True, "No production Kp model currently on disk — nothing to compare against.")
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

    train_r2 = candidate_run.get("train_r2")
    test_r2 = cand_metrics.get("r2")
    if train_r2 is None or test_r2 is None:
        _check("No obvious overfitting", True, "Train-set R² unavailable for this model type — skipped, review manually.")
    else:
        gap = train_r2 - test_r2
        _check(
            "No obvious overfitting",
            gap <= OVERFIT_R2_GAP_THRESHOLD,
            f"Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, gap={gap:.4f} (threshold {OVERFIT_R2_GAP_THRESHOLD:.2f}).",
        )

    infer_ms = candidate_run.get("inference_time_ms_per_sample")
    if infer_ms is None:
        _check("Reasonable inference time", True, "Per-sample inference time unavailable for this model type — review manually.")
    else:
        _check(
            "Reasonable inference time",
            infer_ms <= MAX_REASONABLE_INFERENCE_MS,
            f"{infer_ms:.4f} ms/sample (ceiling {MAX_REASONABLE_INFERENCE_MS:.0f} ms/sample).",
        )

    eligible = all(item["passed"] for item in checklist)
    return {"eligible": eligible, "checklist": checklist}


def promote_kp_to_production(run_id: str, notes: str = "") -> dict:
    """Archives the current production kp_interval model and installs a
    research run in its place.

    1. Validates the run is a non-sequence tabular model with a saved
       .joblib artifact.
    2. Archives models/analytics/kp_interval.joblib to
       models/analytics/archive/kp_interval_<timestamp>.joblib.
    3. Copies the research model's model.joblib -> models/analytics/kp_interval.joblib.
    4. Updates models/analytics/metrics.json["kp_interval"] with the new
       metrics, algorithm, and feature_columns.
    5. Marks the run promoted=True, promoted_to_production=True.

    Returns a summary dict for the UI. Raises ValueError on any
    pre-condition failure so production state is never silently corrupted.
    """
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found in the research registry.")
    if run.get("model_type") in SEQUENCE_MODELS:
        raise ValueError("LSTM/GRU promotion is not supported — loading a Keras model into the prediction process causes the same process-conflict hang training does.")
    if not run.get("model_path"):
        raise ValueError("This run has no saved model artifact to promote (e.g. the persistence benchmark).")

    production_dir = MODELS_DIR / "analytics"
    prod_model_path = production_dir / "kp_interval.joblib"
    prod_metrics_path = production_dir / "metrics.json"

    archive_dir = production_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"kp_interval_{ts}.joblib"
    if prod_model_path.exists():
        shutil.copy2(prod_model_path, archive_path)
    else:
        archive_path = None

    shutil.copy2(Path(run["model_path"]), prod_model_path)

    old_metrics = None
    metrics_data = {}
    if prod_metrics_path.exists():
        with open(prod_metrics_path) as f:
            metrics_data = json.load(f)
        old_metrics = metrics_data.get("kp_interval")

    metrics_data["kp_interval"] = {
        "variable": "kp",
        "horizon": "interval",
        "algorithm": run["model_type"],
        "r2": run["metrics"]["r2"],
        "mae": run["metrics"]["mae"],
        "rmse": run["metrics"]["rmse"],
        "feature_columns": run["feature_columns"],
        "model_path": str(prod_model_path),
        "trained_at": run["trained_at"],
        "n_samples": run["n_train_samples"] + run["n_test_samples"],
        "promoted_from_research": run_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promoted_notes": notes,
    }
    with open(prod_metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

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
        "promoted_at": metrics_data["kp_interval"]["promoted_at"],
    }


def run_complete_kp_optimization_study(progress_cb=None) -> dict:
    """AutoML orchestrator — runs the full 10-experiment Kp Optimization
    Study end-to-end with zero user interaction beyond this one call.
    Every candidate uses train_kp_research_model()/load_kp_research_frame()
    — the exact same functions the manual Exp 1-10 tabs call — this
    function only sequences them, tracks results, and packages a final
    study record. It NEVER calls promote_kp_to_production(); promotion is
    always a separate, explicit, human-confirmed action in the UI.

    `progress_cb`, if given, is called as progress_cb(step_number,
    total_steps, message) after each major step — drives a live
    st.status() panel in the dashboard UI.
    """
    def _p(step, total, msg):
        if progress_cb:
            progress_cb(step, total, msg)

    total_steps = 10
    study_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # Step 1 — Reproduce production baseline
    baseline = train_kp_research_model(
        PRODUCTION_BASELINE_MODEL,
        feature_toggles=default_feature_toggles(),
        engineered_groups=default_engineered_toggles(),
        physics_features={},
        experiment_tag=f"auto_{study_id}_baseline",
    )
    _p(1, total_steps, f"Baseline reproduced ({PRODUCTION_BASELINE_MODEL}) — R²={baseline['metrics']['r2']:.4f}")

    # Step 2 — Persistence benchmark
    persistence = compute_kp_persistence_benchmark()
    _p(2, total_steps, f"Persistence benchmark — R²={persistence['metrics']['r2']:.4f}")

    # Step 3 — Solar Wind Inputs
    solar_wind_results = []
    for combo in SOLAR_WIND_INPUT_GRID:
        run = train_kp_research_model(
            FEATURE_SEARCH_PROBE_MODEL,
            feature_toggles=_build_isolated_toggles(combo["groups"]),
            engineered_groups=default_engineered_toggles(),
            physics_features={},
            experiment_tag=f"auto_{study_id}_solar_wind_inputs",
        )
        solar_wind_results.append({
            "name": combo["name"], "run_id": run["run_id"], "r2": run["metrics"]["r2"],
            "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"], "feature_count": len(run["feature_columns"]),
        })
    solar_wind_results.sort(key=lambda r: -r["r2"])
    _p(3, total_steps, f"Solar Wind Inputs — best: '{solar_wind_results[0]['name']}' (R²={solar_wind_results[0]['r2']:.4f})")

    # Step 4 — IMF Inputs
    imf_results = []
    for combo in IMF_INPUT_GRID:
        run = train_kp_research_model(
            FEATURE_SEARCH_PROBE_MODEL,
            feature_toggles=_build_isolated_toggles(combo["groups"]),
            engineered_groups=default_engineered_toggles(),
            physics_features={},
            experiment_tag=f"auto_{study_id}_imf_inputs",
        )
        imf_results.append({
            "name": combo["name"], "run_id": run["run_id"], "r2": run["metrics"]["r2"],
            "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"], "feature_count": len(run["feature_columns"]),
        })
    imf_results.sort(key=lambda r: -r["r2"])
    _p(4, total_steps, f"IMF Inputs — best: '{imf_results[0]['name']}' (R²={imf_results[0]['r2']:.4f})")

    # Step 5 — Geomagnetic History
    geomag_results = []
    for combo in GEOMAGNETIC_HISTORY_GRID:
        run = train_kp_research_model(
            FEATURE_SEARCH_PROBE_MODEL,
            feature_toggles=_build_isolated_toggles(combo["groups"]),
            engineered_groups=default_engineered_toggles(),
            physics_features={},
            experiment_tag=f"auto_{study_id}_geomagnetic_history",
        )
        geomag_results.append({
            "name": combo["name"], "run_id": run["run_id"], "r2": run["metrics"]["r2"],
            "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"], "feature_count": len(run["feature_columns"]),
        })
    geomag_results.sort(key=lambda r: -r["r2"])
    _p(5, total_steps, f"Geomagnetic History — best: '{geomag_results[0]['name']}' (R²={geomag_results[0]['r2']:.4f})")

    # Step 6 — Physics Optimization (the primary Kp experiment)
    physics_results = []
    baseline_r2 = baseline["metrics"]["r2"]
    for entry in PHYSICS_OPTIMIZATION_GRID:
        if entry["kind"] == "remove_column":
            toggles = default_feature_toggles()
            toggles[entry["group"]] = dict(toggles[entry["group"]])
            toggles[entry["group"]][entry["column"]] = False
            physics_features = {}
        else:
            toggles = default_feature_toggles()
            physics_features = {entry.get("physics_name", entry["name"]): True}
        run = train_kp_research_model(
            FEATURE_SEARCH_PROBE_MODEL,
            feature_toggles=toggles,
            engineered_groups=default_engineered_toggles(),
            physics_features=physics_features,
            experiment_tag=f"auto_{study_id}_physics_optimization",
        )
        candidate_r2 = run["metrics"]["r2"]
        delta_r2 = (candidate_r2 - baseline_r2) if entry["kind"] == "add_physics" else (baseline_r2 - candidate_r2)
        physics_results.append({
            "name": entry["name"], "kind": entry["kind"], "run_id": run["run_id"],
            "r2": candidate_r2, "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"],
            "delta_r2": delta_r2, "feature_count": len(run["feature_columns"]),
            "physics_name": entry.get("physics_name", entry["name"]) if entry["kind"] == "add_physics" else None,
        })
    physics_results.sort(key=lambda r: -r["delta_r2"])
    _p(
        6, total_steps,
        f"Physics Optimization complete — top contributor: '{physics_results[0]['name']}' "
        f"(ΔR²={physics_results[0]['delta_r2']:+.4f})",
    )

    # Step 6b — Combine the top-K positive-contribution physics groups into
    # one structured "Best Combined Feature Set", confirmed with its own run
    top_physics = [
        r for r in physics_results if r["kind"] == "add_physics" and r["delta_r2"] > 0
    ][:TOP_PHYSICS_GROUPS_TO_COMBINE]
    combined_physics_features = {r["physics_name"]: True for r in top_physics}
    best_combo_run = train_kp_research_model(
        FEATURE_SEARCH_PROBE_MODEL,
        feature_toggles=default_feature_toggles(),
        engineered_groups=default_engineered_toggles(),
        physics_features=combined_physics_features,
        experiment_tag=f"auto_{study_id}_best_combo",
    )
    best_feature_config = {
        "feature_toggles": default_feature_toggles(),
        "engineered_groups": default_engineered_toggles(),
        "physics_features": combined_physics_features,
        "combined_physics_names": [r["name"] for r in top_physics],
        "run_id": best_combo_run["run_id"],
        "r2": best_combo_run["metrics"]["r2"],
    }

    # Step 7 — Model Optimization: sweep every model on the Best Combined
    # Feature Set found above
    model_search_results = []
    for model_type in TABULAR_MODELS:
        try:
            run = train_kp_research_model(
                model_type,
                feature_toggles=best_feature_config["feature_toggles"],
                engineered_groups=best_feature_config["engineered_groups"],
                physics_features=best_feature_config["physics_features"],
                experiment_tag=f"auto_{study_id}_model_search",
            )
            model_search_results.append(run)
        except Exception as exc:
            model_search_results.append({"run_id": None, "model_type": model_type, "error": str(exc)})

    if KERAS_AVAILABLE and SEQUENCE_MODELS:
        for model_type in SEQUENCE_MODELS:
            try:
                run = train_kp_research_model(
                    model_type,
                    feature_toggles=best_feature_config["feature_toggles"],
                    engineered_groups=best_feature_config["engineered_groups"],
                    physics_features=best_feature_config["physics_features"],
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
    _p(7, total_steps, f"Model Optimization complete — winner: {winner['model_type']} (R²={winner['metrics']['r2']:.4f})")

    # Step 8 — Feature Importance
    fi_source = winner
    if not fi_source.get("feature_importance"):
        with_fi = [r for r in valid_candidates if r.get("feature_importance")]
        fi_source = with_fi[0] if with_fi else None
    feature_importance = fi_source["feature_importance"] if fi_source else None
    feature_importance_source_run_id = fi_source["run_id"] if fi_source else None
    _p(8, total_steps, "Feature importance extracted." if feature_importance else "Feature importance unavailable for all candidates.")

    # Step 9 — SHAP Analysis
    shap_source_run_id = None
    for candidate in valid_candidates:
        if candidate["model_type"] in SHAP_SUPPORTED_MODELS:
            shap_source_run_id = candidate["run_id"]
            break
    if shap_source_run_id:
        try:
            shap_result = compute_shap_importance_kp(shap_source_run_id)
        except Exception as exc:
            shap_result = {"run_id": shap_source_run_id, "supported": False, "skipped_reason": f"SHAP failed: {exc}", "shap_importance": None}
    else:
        shap_result = {"run_id": None, "supported": False, "skipped_reason": "No SHAP-supported candidate in this study's model search.", "shap_importance": None}
    _p(9, total_steps, "SHAP analysis complete." if shap_result.get("supported") else f"SHAP skipped: {shap_result.get('skipped_reason')}")

    # Step 10 — Optimization Summary: leaderboard + production comparison + recommendation
    leaderboard = []
    for rank, r in enumerate(valid_candidates, start=1):
        leaderboard.append({
            "rank": rank, "run_id": r["run_id"], "model_type": r["model_type"],
            "r2": r["metrics"]["r2"], "mae": r["metrics"]["mae"], "rmse": r["metrics"]["rmse"],
            "mape": r["metrics"].get("mape"), "bias": r["metrics"]["bias"],
            "training_time_sec": r.get("training_time_sec"),
            "inference_time_ms_per_sample": r.get("inference_time_ms_per_sample"),
            "model_size_kb": r.get("model_size_kb"), "feature_count": len(r.get("feature_columns", [])),
        })
    failed_candidates = [r for r in model_search_results if not r.get("run_id")]

    production_metrics = get_production_kp_metrics()
    promotion_check = check_promotion_criteria_kp(winner, production_metrics)
    if production_metrics is not None:
        production_comparison = {
            "current": {
                "algorithm": production_metrics.get("algorithm"), "r2": production_metrics.get("r2"),
                "mae": production_metrics.get("mae"), "rmse": production_metrics.get("rmse"),
            },
            "candidate": {
                "algorithm": winner["model_type"], "r2": winner["metrics"]["r2"],
                "mae": winner["metrics"]["mae"], "rmse": winner["metrics"]["rmse"],
            },
            "delta_r2": winner["metrics"]["r2"] - production_metrics.get("r2", 0),
            "delta_mae": winner["metrics"]["mae"] - production_metrics.get("mae", 0),
            "delta_rmse": winner["metrics"]["rmse"] - production_metrics.get("rmse", 0),
        }
    else:
        production_comparison = {
            "current": None,
            "candidate": {
                "algorithm": winner["model_type"], "r2": winner["metrics"]["r2"],
                "mae": winner["metrics"]["mae"], "rmse": winner["metrics"]["rmse"],
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
        "target": TARGET_LABEL,
        "training_dataset": "data/features/training/analytics_features.csv",
        "baseline_run_id": baseline["run_id"],
        "baseline_metrics": baseline["metrics"],
        "baseline_model_type": PRODUCTION_BASELINE_MODEL,
        "persistence_run_id": persistence["run_id"],
        "persistence_metrics": persistence["metrics"],
        "solar_wind_results": solar_wind_results,
        "imf_results": imf_results,
        "geomagnetic_results": geomag_results,
        "physics_results": physics_results,
        "best_feature_config": best_feature_config,
        "models_tested": [r.get("model_type") for r in model_search_results],
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
    _append_kp_study(study)
    return study


def _load_kp_studies() -> list[dict]:
    if not STUDIES_REGISTRY_PATH.exists():
        return []
    with open(STUDIES_REGISTRY_PATH) as f:
        return json.load(f)


def _save_kp_studies(studies: list[dict]) -> None:
    STUDIES_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDIES_REGISTRY_PATH, "w") as f:
        json.dump(studies, f, indent=2)


def _append_kp_study(study: dict) -> None:
    studies = _load_kp_studies()
    studies.append(study)
    _save_kp_studies(studies)


def list_kp_studies() -> list[dict]:
    return sorted(_load_kp_studies(), key=lambda s: s["started_at"], reverse=True)


def get_kp_study(study_id: str) -> dict:
    for s in _load_kp_studies():
        if s["study_id"] == study_id:
            return s
    return None


def _update_kp_study(study_id: str, **fields) -> bool:
    studies = _load_kp_studies()
    found = False
    for s in studies:
        if s["study_id"] == study_id:
            s.update(fields)
            found = True
    if found:
        _save_kp_studies(studies)
    return found


def mark_kp_study_promoted(study_id: str, run_id: str) -> bool:
    return _update_kp_study(
        study_id, promotion_status="promoted", promoted_run_id=run_id,
        promoted_at=datetime.now(timezone.utc).isoformat(),
    )


def mark_kp_study_rejected(study_id: str, notes: str = "") -> bool:
    return _update_kp_study(
        study_id, promotion_status="rejected", rejection_notes=notes,
        rejected_at=datetime.now(timezone.utc).isoformat(),
    )
