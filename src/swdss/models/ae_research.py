"""AE Research Laboratory — experimental AE model comparison, feature
ablation, physics-feature experimentation, and hypothesis testing,
completely isolated from the production AE predictor (swdss.models.predict
/ swdss.models.train / swdss.models.jobs).

Production answers "what is the operational AE forecast?" — this module
answers "how can AE prediction be improved?", "what physics governs AE?",
and "which ML model performs best?". Nothing here ever imports from or
writes to models/ae/ (production's model directory) or its metrics.json.
Training-run artifacts live under models/ae_research/<run_id>/, tracked in
their own JSON registry (RUNS_REGISTRY_PATH) — a completely separate
namespace. "Promoting" a run only sets a label on its registry entry; a
human engineer must manually wire a promoted model into predict.py
themselves for it to ever affect a live forecast.

Data source: the SAME `ae_analytics_features.csv` production's own
train_dataset("ae") trains on (~3 years, hourly, confirmed 26,305 rows) —
so research runs here are genuinely comparable to production's own
ae_1h/ae_3h models (R²=0.7436 / R²=0.3978 respectively).

Horizon note: AE has NO minute-level ground truth anywhere in this
codebase — data/processed/ae/ae_processed.parquet was checked directly
and is itself hourly (26,304 rows / 3 years, one row per hour). Sub-hourly
horizons (15min/30min) were deliberately dropped from this lab rather
than faked via interpolation of the hourly series, which would produce an
artificially inflated R² derived from nearby known values rather than
genuine forecast skill. HORIZON_OPTIONS is therefore [1, 2, 3] hours only.
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

from swdss.models.ae_physics_features import PHYSICS_FEATURE_FUNCTIONS, add_all_core_derived_physics
from swdss.models.features import add_change_features, add_lag_features
from swdss.models.registry import DATASETS
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

# Same note as kp_research.py / imf_research.py: TensorFlow must never be
# imported at module level here — having it share a process with
# scikit-learn/XGBoost/LightGBM/CatBoost was empirically confirmed to hang
# model.fit indefinitely. Keras training runs in the shared
# imf_research_keras_worker.py subprocess instead (fully domain-agnostic).
KERAS_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


RESEARCH_MODELS_DIR = MODELS_DIR / "ae_research"
RUNS_REGISTRY_PATH = DATA_DIR / "predictions" / "ae_research_runs.json"
HYPOTHESIS_REGISTRY_PATH = DATA_DIR / "predictions" / "ae_hypothesis_tests.json"

TEST_FRACTION = 0.2
TARGET_LABEL = "AE"

HORIZON_OPTIONS = [1, 2, 3]
DEFAULT_HORIZON = 1

# "Derived Physics" here is a full 6-column group (unlike the plain
# ey/vbz/dynamic_pressure trio elsewhere) — clock angle and southward
# duration/integration are included as core toggleable columns per the
# AE lab's own feature-group spec, computed unconditionally into the
# frame (see load_ae_research_frame) regardless of this group's toggle
# state, since the Physics Feature Experiments registry below may still
# need the underlying ey/vbz/clock_angle columns present.
FEATURE_GROUP_COLUMNS = {
    "Solar Wind": ["speed", "density", "temperature"],
    "IMF": ["bt", "bx_gsm", "by_gsm", "bz_gsm"],
    "Derived Physics": ["ey", "vbz", "dynamic_pressure", "clock_angle_deg", "southward_duration_hr", "integrated_southward_bz_24h"],
    "Geomagnetic": ["ae"],
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

FUTURE_MODELS = [
    "Transformer",
    "Temporal Convolution Network (TCN)",
    "Physics-Informed Neural Network",
    "Stacked Model",
    "Bayesian-Optimized Ensemble",
]

ALL_TRAINABLE_MODELS = TABULAR_MODELS + SEQUENCE_MODELS

SEQUENCE_LENGTH_OPTIONS = [1, 3, 6, 12, 24]
DEFAULT_SEQUENCE_LENGTH = 6

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


def _load_ae_base_frame() -> pd.DataFrame:
    """The exact CSV production's train_dataset("ae") trains on — no
    scale factor correction needed (unlike Kp's tenths-of-Kp CSV
    convention), since AE has no such unit mismatch (registry.py's "ae"
    DatasetConfig has no scale_factors entry).
    """
    config = DATASETS["ae"]
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")
    return raw


def load_ae_research_frame(
    feature_toggles: dict = None,
    engineered_groups: dict = None,
    physics_features: dict = None,
) -> tuple:
    """Builds the AE research frame and returns (frame, feature_columns).
    Same three-toggle contract as kp_research.load_kp_research_frame:
    feature_toggles gates which base columns are INCLUDED as features
    (the underlying columns always exist in `frame`), engineered_groups
    gates which of Lags/Rolling Mean/Rolling Std/Rate of Change are
    applied to the enabled base columns, and physics_features is the
    opt-in exotic-physics registry (each independently toggleable).
    """
    feature_toggles = feature_toggles or default_feature_toggles()
    engineered_groups = engineered_groups or default_engineered_toggles()
    physics_features = physics_features or {}

    frame = _load_ae_base_frame()
    add_all_core_derived_physics(frame)

    physics_cols: list[str] = []
    for name, fn in PHYSICS_FEATURE_FUNCTIONS.items():
        if physics_features.get(name, False):
            physics_cols += fn(frame)

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
    """Delegates to the SAME subprocess worker the IMF/Kp labs use
    (imf_research_keras_worker.py) — fully domain-agnostic, so reused
    directly rather than duplicated. See KERAS_AVAILABLE above for why
    this isolation is required at all.
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


def train_ae_research_model(
    model_type: str,
    horizon: int = DEFAULT_HORIZON,
    feature_toggles: dict = None,
    engineered_groups: dict = None,
    physics_features: dict = None,
    sequence_length: int = None,
    hyperparams: dict = None,
    notes: str = "",
) -> dict:
    """Trains one AE model for one (horizon, feature configuration)
    combination and records a run — never touches models/ae/ or its
    metrics.json. Target is a plain shift(-horizon) on the raw "ae"
    column (no publishing-cadence block logic — AE has no such official
    cadence, unlike Kp), matching production's own train_dataset("ae")
    target definition exactly, so horizon=1/3 results are directly
    comparable to production's own ae_1h/ae_3h metrics.
    """
    if model_type not in ALL_TRAINABLE_MODELS:
        raise ValueError(f"'{model_type}' is not trainable yet — see FUTURE_MODELS.")
    if horizon not in HORIZON_OPTIONS:
        raise ValueError(f"Horizon {horizon} is not valid — expected one of {HORIZON_OPTIONS}.")

    feature_toggles = feature_toggles or default_feature_toggles()
    engineered_groups = engineered_groups or default_engineered_toggles()
    physics_features = physics_features or {}

    frame, feature_columns = load_ae_research_frame(feature_toggles, engineered_groups, physics_features)
    frame = frame.copy()
    frame["__target__"] = frame["ae"].shift(-horizon)
    frame = frame.dropna(subset=feature_columns + ["__target__"])

    if len(frame) < 100:
        raise ValueError(
            "Not enough history to train an AE research model (need 100+ clean rows after lag/rolling windows)."
        )

    run_id = str(uuid.uuid4())
    model_dir = RESEARCH_MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

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
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time_sec = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        preds = np.asarray(model.predict(X_test))
        prediction_time_sec = time.perf_counter() - predict_start

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
        "target": TARGET_LABEL,
        "model_type": model_type,
        "horizon": horizon,
        "feature_toggles": feature_toggles,
        "engineered_groups": engineered_groups,
        "physics_features": {k: v for k, v in physics_features.items() if v},
        "sequence_length": sequence_length if model_type in SEQUENCE_MODELS else None,
        "hyperparams": hyperparams or {},
        "metrics": metrics,
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
    }
    _append_run(run_record)
    return run_record


def train_horizon_sweep(model_type: str, hyperparams: dict = None, reuse_existing: bool = True) -> list[dict]:
    """Trains (or reuses) one run per horizon in HORIZON_OPTIONS, for the
    Horizon Analysis tab — "how does AE forecast skill decay across the
    1-3h range?" answered directly. Returns runs in ascending horizon
    order.
    """
    results = []
    for horizon in HORIZON_OPTIONS:
        existing = None
        if reuse_existing:
            candidates = [
                r for r in list_runs() if r.get("horizon") == horizon and r["model_type"] == model_type
            ]
            existing = candidates[0] if candidates else None  # list_runs is newest-first
        run = existing or train_ae_research_model(model_type, horizon=horizon, hyperparams=hyperparams)
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
    """Label-only — never touches production. A human engineer must
    manually wire a promoted model into predict.py for it to ever affect
    a live forecast.
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
    scikit-learn process conflict as training.
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

FEATURE_ABLATION_UNITS = list(FEATURE_GROUP_COLUMNS) + list(ENGINEERED_GROUPS)


def run_feature_ablation_sweep(model_type: str = "Linear Regression", horizon: int = DEFAULT_HORIZON, hyperparams: dict = None) -> dict:
    """Leave-one-out ablation, identical methodology to
    kp_research.run_feature_ablation_sweep: trains a Full Model with
    every base + engineered group enabled, then retrains once per unit
    with just that ONE unit disabled, ranking by R² drop when removed.
    """
    full_toggles = default_feature_toggles()
    full_engineered = default_engineered_toggles()
    full_run = train_ae_research_model(
        model_type,
        horizon=horizon,
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
        run = train_ae_research_model(
            model_type,
            horizon=horizon,
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
        "horizon": horizon,
        "rows": rows,
        "ranked": ranked,
        "swept_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------- hypothesis testing

# Five straightforward "does adding X help" hypotheses (kind="feature"/
# "physics", same pattern as kp_research.py), plus one head-to-head
# feature-SWAP hypothesis ("Does Akasofu Epsilon outperform Ey?") — a
# direct comparison between two competing physics features rather than
# an on/off test, handled as its own "swap" kind in run_hypothesis_test.
HYPOTHESIS_DEFINITIONS = {
    "Ey improves AE prediction": {"kind": "feature", "group": "Derived Physics", "column": "ey"},
    "VBz improves AE prediction": {"kind": "feature", "group": "Derived Physics", "column": "vbz"},
    "Dynamic Pressure improves AE prediction": {"kind": "feature", "group": "Derived Physics", "column": "dynamic_pressure"},
    "Clock Angle improves AE prediction": {"kind": "feature", "group": "Derived Physics", "column": "clock_angle_deg"},
    "Newell Coupling Function improves AE prediction": {"kind": "physics", "name": "Newell Coupling Function"},
    "Akasofu Epsilon outperforms Ey": {"kind": "swap", "drop_group": "Derived Physics", "drop_column": "ey", "add_physics": "Akasofu Epsilon Parameter"},
}

HYPOTHESIS_ACCEPT_THRESHOLD_R2 = 0.005


def run_hypothesis_test(hypothesis_label: str, model_type: str = "Linear Regression", horizon: int = DEFAULT_HORIZON, hyperparams: dict = None) -> dict:
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
    elif spec["kind"] == "physics":
        experimental_physics[spec["name"]] = True
    elif spec["kind"] == "swap":
        # Baseline keeps the incumbent feature (e.g. Ey); experimental
        # drops it and adds the challenger physics feature instead — a
        # like-for-like head-to-head swap, not an additive test.
        experimental_toggles[spec["drop_group"]] = dict(experimental_toggles[spec["drop_group"]])
        experimental_toggles[spec["drop_group"]][spec["drop_column"]] = False
        experimental_physics[spec["add_physics"]] = True
    else:
        raise ValueError(f"Unknown hypothesis kind: {spec['kind']}")

    engineered = default_engineered_toggles()
    baseline_run = train_ae_research_model(
        model_type,
        horizon=horizon,
        feature_toggles=baseline_toggles,
        engineered_groups=engineered,
        physics_features=baseline_physics,
        hyperparams=hyperparams,
        notes=f"Hypothesis Testing — baseline for: {hypothesis_label}",
    )
    experimental_run = train_ae_research_model(
        model_type,
        horizon=horizon,
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
        "horizon": horizon,
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
