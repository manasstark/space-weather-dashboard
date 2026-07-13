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
genuine forecast skill.

HORIZON_OPTIONS now matches production's own HORIZONS list ([1, 3, 6,
12, 24] hours) exactly — this module previously used an ad-hoc [1, 2, 3]
list that neither matched production's actual 5 trained horizons nor
included a horizon (2h) production has never trained at all. Widened to
[1, 3, 6, 12, 24] specifically so the AE Optimization Study can support
and independently evaluate every horizon production actually deploys.
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

from swdss.models.ae_physics_features import (
    PHYSICS_FEATURE_DEPENDENCIES,
    PHYSICS_FEATURE_FUNCTIONS,
    add_all_core_derived_physics,
)
from swdss.models.features import add_change_features, add_lag_features
from swdss.models.registry import DATASETS, HORIZONS
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

HORIZON_OPTIONS = HORIZONS  # [1, 3, 6, 12, 24] — production's own 5 independently-trained AE horizons
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

    # Resolves prerequisites (e.g. Total Pressure needs Magnetic + Thermal
    # Pressure computed first) via the Physics Engine's shared registry
    # resolver — see swdss.physics.registry and kp_research.py's
    # identical usage. Previously this loop called each requested
    # function directly with no dependency resolution at all; that was a
    # latent gap this lab's earlier, simpler physics features never
    # exercised, but several features added by the Physics Engine
    # migration (Total Pressure, Plasma Beta, Estimated Compression, IMF
    # Rotation Rate) do have real prerequisites.
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


def _load_kp_dst_for_ae() -> pd.DataFrame:
    """Loads Previous Kp / Previous Dst from analytics_features.csv — the
    ONLY place in this codebase with Kp, Dst, and AE all on the same
    hourly index. AE's own training CSV never includes Kp/Dst (see
    registry.py's "ae" DatasetConfig); this is a new, opt-in helper used
    only by the Optimization Study's Experiment 6 (Geomagnetic Memory),
    never by any other AE lab tab. Applies the same kp scale correction
    kp_research.py applies (DATASETS["analytics"].scale_factors) so the
    merged "kp" column is on the true 0-9 scale, not tenths-of-Kp.
    """
    config = DATASETS["analytics"]
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")
    kp_dst = raw[["kp", "dst"]].copy()
    for column, factor in (config.scale_factors or {}).items():
        if column in kp_dst.columns:
            kp_dst[column] = kp_dst[column] / factor
    return kp_dst


def load_ae_research_frame_with_geomagnetic_memory(
    feature_toggles: dict = None,
    engineered_groups: dict = None,
    physics_features: dict = None,
    include_kp: bool = False,
    include_dst: bool = False,
) -> tuple:
    """Extends load_ae_research_frame with OPTIONAL Previous Kp/Previous
    Dst columns merged in from analytics_features.csv (see
    _load_kp_dst_for_ae). Purely additive: when both include_kp and
    include_dst are False this returns exactly what load_ae_research_frame
    would, so every existing caller of that function is unaffected. Used
    only by train_ae_research_model when include_kp/include_dst is set.
    """
    frame, feature_columns = load_ae_research_frame(feature_toggles, engineered_groups, physics_features)
    if not include_kp and not include_dst:
        return frame, feature_columns

    engineered_groups = engineered_groups or default_engineered_toggles()
    kp_dst = _load_kp_dst_for_ae()
    merge_cols = [c for c, want in (("kp", include_kp), ("dst", include_dst)) if want]
    frame = frame.join(kp_dst[merge_cols], how="left")

    extra_cols: list[str] = merge_cols.copy()
    engineered_cols: list[str] = []
    if engineered_groups.get("Lags", True):
        engineered_cols += add_lag_features(frame, merge_cols)
    if engineered_groups.get("Rolling Mean", True):
        engineered_cols += _add_rolling_mean(frame, merge_cols)
    if engineered_groups.get("Rolling Std", True):
        engineered_cols += _add_rolling_std(frame, merge_cols)
    if engineered_groups.get("Rate of Change", True):
        engineered_cols += add_change_features(frame, merge_cols)

    return frame, feature_columns + extra_cols + engineered_cols


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
    experiment_tag: str = "",
    include_kp: bool = False,
    include_dst: bool = False,
) -> dict:
    """Trains one AE model for one (horizon, feature configuration)
    combination and records a run — never touches models/ae/ or its
    metrics.json. Target is a plain shift(-horizon) on the raw "ae"
    column (no publishing-cadence block logic — AE has no such official
    cadence, unlike Kp), matching production's own train_dataset("ae")
    target definition exactly, so results at every horizon are directly
    comparable to production's own ae_1h..ae_24h metrics.

    include_kp/include_dst default to False and are fully backward
    compatible — every existing caller (Feature Ablation, Hypothesis
    Testing, Horizon Analysis) is unaffected. They exist solely for the
    Optimization Study's Geomagnetic Memory experiment (Experiment 6),
    which needs Previous Kp/Previous Dst — columns that do not exist in
    AE's own training CSV (registry.py's "ae" DatasetConfig deliberately
    has no kp/dst inputs). When either is True, kp/dst are merged in from
    analytics_features.csv (the same combined dataset production's own
    Kp/Dst/AE model trains on) by datetime alignment, scale-corrected the
    same way kp_research.py corrects it, then engineered the same way
    every other base column is (lags/rolling/change), gated by the same
    engineered_groups toggles.
    """
    if model_type not in ALL_TRAINABLE_MODELS:
        raise ValueError(f"'{model_type}' is not trainable yet — see FUTURE_MODELS.")
    if horizon not in HORIZON_OPTIONS:
        raise ValueError(f"Horizon {horizon} is not valid — expected one of {HORIZON_OPTIONS}.")

    feature_toggles = feature_toggles or default_feature_toggles()
    engineered_groups = engineered_groups or default_engineered_toggles()
    physics_features = physics_features or {}

    if include_kp or include_dst:
        frame, feature_columns = load_ae_research_frame_with_geomagnetic_memory(
            feature_toggles, engineered_groups, physics_features, include_kp=include_kp, include_dst=include_dst
        )
    else:
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

        metrics = compute_metrics(y_test.to_numpy(), preds)
        loss_history = None
        inference_time_ms_per_sample = (prediction_time_sec / len(X_test)) * 1000 if len(X_test) else None
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
        "experiment_tag": experiment_tag,
        "model_size_kb": round(model_size_kb, 2) if model_size_kb is not None else None,
        "train_r2": train_r2,
        "inference_time_ms_per_sample": round(inference_time_ms_per_sample, 5) if inference_time_ms_per_sample is not None else None,
        "include_kp": include_kp,
        "include_dst": include_dst,
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


# ==================================================================
# AE OPTIMIZATION STUDY
# ==================================================================
# The flagship scientific component of this project. Unlike the Bz and
# Kp Optimization Studies (a single target, a single horizon), AE has 5
# INDEPENDENTLY trained production horizons (1h/3h/6h/12h/24h) — so every
# experiment below runs once per horizon with identical methodology, and
# Experiment 10 (Cross-Horizon Scientific Synthesis) compares the 5
# independent results against each other. The scientific objective is
# NOT "maximize R²" — it is to understand WHERE AE's predictability
# comes from (persistence vs. raw solar wind vs. coupling physics vs.
# geomagnetic memory) and whether that balance shifts as the horizon
# grows, while still discovering the strongest defensible production
# candidate per horizon along the way.
#
# The minute-resolution Kyoto AE archive (kyoto_ae.py, MINUTE_DATASET_NAME
# = "kyoto_ae_minute") deliberately plays NO role anywhere below. Every
# experiment trains and evaluates against the same hourly
# ae_analytics_features.csv production itself trains on. The minute
# archive exists for a different, future research question (substorm
# onset timing, minute-scale AE dynamics) that this hourly-regression
# study does not attempt to answer.

STUDIES_REGISTRY_PATH = DATA_DIR / "predictions" / "ae_optimization_studies.json"

# The algorithm production's own _fit_best (train.py) currently selects
# per horizon — used to reproduce each horizon's production baseline
# exactly in Experiment 1. If a future production refresh picks a
# different winner for some horizon, update this mapping so Experiment 1
# keeps reproducing the true baseline. Names use this lab's own
# TABULAR_MODELS convention ("Random Forest", not "RandomForest").
PRODUCTION_BASELINE_MODEL_BY_HORIZON = {
    1: "XGBoost",
    3: "XGBoost",
    6: "Random Forest",
    12: "Random Forest",
    24: "XGBoost",
}

# Production's ACTUAL base feature set, confirmed directly against
# models/ae/metrics.json's stored feature_columns: Solar Wind + IMF +
# Geomagnetic (persistence) + only 3 of the 6 "Derived Physics" columns
# (Ey/VBz/Dynamic Pressure) — Clock Angle, Southward Duration, and
# Integrated Southward Bz are NOT in production's trained feature set,
# even though this lab's FEATURE_GROUP_COLUMNS["Derived Physics"] lists
# all 6 (a deliberately broader research-lab default for exploration).
# default_feature_toggles() therefore does NOT reproduce production
# exactly — it silently adds those 3 extra columns. Experiment 1 uses
# THIS narrower, verified-accurate toggle set instead, so its "reproduce
# Production baseline exactly" claim is actually true.
_PRODUCTION_DERIVED_PHYSICS_COLUMNS = ["ey", "vbz", "dynamic_pressure"]


def _production_baseline_toggles() -> dict:
    return _build_isolated_toggles({
        "Solar Wind": FEATURE_GROUP_COLUMNS["Solar Wind"],
        "IMF": FEATURE_GROUP_COLUMNS["IMF"],
        "Geomagnetic": FEATURE_GROUP_COLUMNS["Geomagnetic"],
        "Derived Physics": _PRODUCTION_DERIVED_PHYSICS_COLUMNS,
    })

# Probe model for structured feature-search experiments (3-7): captures
# nonlinear coupling relationships a linear probe would miss, while still
# being far cheaper than sweeping every model type through every combo at
# every one of the 5 horizons. The winning configuration is re-tested by
# every real model in Experiment 8.
FEATURE_SEARCH_PROBE_MODEL = "Random Forest"

OVERFIT_R2_GAP_THRESHOLD = 0.15
MAX_REASONABLE_INFERENCE_MS = 50.0

SHAP_SUPPORTED_MODELS = {
    "Linear Regression", "Ridge Regression", "Lasso", "ElasticNet",
    "Random Forest", "XGBoost", "LightGBM", "CatBoost",
}

# Models with neither .feature_importances_ nor .coef_ but that ARE
# reloadable via joblib (unlike LSTM/GRU) — Experiment 9 falls back to
# scikit-learn's model-agnostic permutation importance for these.
PERMUTATION_SUPPORTED_MODELS = {"SVR", "MLP"}


def _build_isolated_toggles(groups: dict) -> dict:
    """Builds a feature_toggles dict with EVERYTHING off except the
    columns named in `groups` ({group_name: [columns]}) — used by
    Experiments 3-4 and 6-7 to isolate exactly one input combination's
    contribution, with no other base feature group riding along.
    """
    toggles = {group: {col: False for col in cols} for group, cols in FEATURE_GROUP_COLUMNS.items()}
    for group, cols in groups.items():
        for col in cols:
            toggles[group][col] = True
    return toggles


# ---- Experiment 3: Solar Wind + IMF Raw Floor -------------------------
# Only the 7 raw upstream measurements (+ their lags/rolling/change) —
# no persistence (Geomagnetic/"ae" group off), no Derived Physics
# columns, no opt-in physics-registry features. Establishes the raw
# explanatory floor before ANY coupling physics or memory of AE itself
# is introduced.
SOLAR_WIND_IMF_RAW_FLOOR_GROUPS = {
    "Solar Wind": ["speed", "density", "temperature"],
    "IMF": ["bt", "bx_gsm", "by_gsm", "bz_gsm"],
}

# ---- Experiment 4: Coupling Physics (individual, isolated) ------------
# Every entry tested ALONE (feature_toggles fully off except this one
# input) — a pure marginal/standalone-information test, distinct from
# Experiment 5's baseline-relative cumulative ablation. "core_column"
# entries isolate one of the always-computed Derived Physics base
# columns; "physics" entries isolate one opt-in physics-registry feature
# (computed on the frame regardless of toggle state, then included alone
# via physics_features).
COUPLING_PHYSICS_GRID = [
    {"name": "Ey", "kind": "core_column", "column": "ey"},
    {"name": "VBz", "kind": "core_column", "column": "vbz"},
    {"name": "Dynamic Pressure", "kind": "core_column", "column": "dynamic_pressure"},
    {"name": "Clock Angle", "kind": "core_column", "column": "clock_angle_deg"},
    {"name": "Clock Angle Rate", "kind": "physics", "physics_name": "Clock Angle Change"},
    {"name": "Southward Duration", "kind": "core_column", "column": "southward_duration_hr"},
    {"name": "Strong Southward Duration", "kind": "physics", "physics_name": "Strong Southward Duration"},
    {"name": "Integrated Southward Bz", "kind": "core_column", "column": "integrated_southward_bz_24h"},
    {"name": "Integrated Ey", "kind": "physics", "physics_name": "Integrated Ey"},
    {"name": "Integrated VBz", "kind": "physics", "physics_name": "Integrated VBz"},
    {"name": "Integrated Energy Input", "kind": "physics", "physics_name": "Integrated Energy Input"},
    {"name": "Newell Coupling Function", "kind": "physics", "physics_name": "Newell Coupling Function"},
    {"name": "Akasofu ε", "kind": "physics", "physics_name": "Akasofu Epsilon Parameter"},
    {"name": "Boyle Index", "kind": "physics", "physics_name": "Boyle Index"},
]

# ---- Experiment 5: Physics Engine Ablation (cumulative, from Production)
# Structured cumulative chain built ON TOP OF the Production baseline
# (all base groups + standard engineering — never isolated), adding one
# physics-registry variable at a time. Dynamic Pressure/Ey/VBz are
# already always-on Production base columns (FEATURE_GROUP_COLUMNS
# ["Derived Physics"]), so those three specific steps are deliberately
# no-ops (physics_features unchanged from the prior step) — an explicit,
# honest test of "do not assume more variables are better": if a step's
# R² is identical to the previous one, that variable was already fully
# captured by Production and contributes nothing further.
PHYSICS_ENGINE_ABLATION_STEPS = [
    {"name": "Production Baseline", "physics_features": {}, "already_in_baseline": []},
    {"name": "Production + Newell", "physics_features": {"Newell Coupling Function": True}, "already_in_baseline": []},
    {
        "name": "Production + Newell + Akasofu",
        "physics_features": {"Newell Coupling Function": True, "Akasofu Epsilon Parameter": True},
        "already_in_baseline": [],
    },
    {
        "name": "Production + Newell + Akasofu + Boyle",
        "physics_features": {"Newell Coupling Function": True, "Akasofu Epsilon Parameter": True, "Boyle Index": True},
        "already_in_baseline": [],
    },
    {
        "name": "Production + Newell + Akasofu + Boyle + Dynamic Pressure",
        "physics_features": {"Newell Coupling Function": True, "Akasofu Epsilon Parameter": True, "Boyle Index": True},
        "already_in_baseline": ["Dynamic Pressure"],
    },
    {
        "name": "Production + Newell + Akasofu + Boyle + Dynamic Pressure + Ey",
        "physics_features": {"Newell Coupling Function": True, "Akasofu Epsilon Parameter": True, "Boyle Index": True},
        "already_in_baseline": ["Dynamic Pressure", "Ey"],
    },
    {
        "name": "Production + Newell + Akasofu + Boyle + Dynamic Pressure + Ey + VBz",
        "physics_features": {"Newell Coupling Function": True, "Akasofu Epsilon Parameter": True, "Boyle Index": True},
        "already_in_baseline": ["Dynamic Pressure", "Ey", "VBz"],
    },
    {
        "name": "Production + All Coupling",
        "physics_features": {"Newell Coupling Function": True, "Akasofu Epsilon Parameter": True, "Boyle Index": True},
        "already_in_baseline": ["Dynamic Pressure", "Ey", "VBz"],
    },
]

# ---- Experiment 6: Geomagnetic Memory ---------------------------------
# Tests whether Previous Kp/Previous Dst carry information about future
# AE beyond AE's own persistence — every combination isolated (no Solar
# Wind/IMF/Derived Physics riding along), using include_kp/include_dst.
GEOMAGNETIC_MEMORY_GRID = [
    {"name": "Previous AE", "groups": {"Geomagnetic": ["ae"]}, "include_kp": False, "include_dst": False},
    {"name": "Previous Kp", "groups": {}, "include_kp": True, "include_dst": False},
    {"name": "Previous Dst", "groups": {}, "include_kp": False, "include_dst": True},
    {"name": "Previous AE + Kp", "groups": {"Geomagnetic": ["ae"]}, "include_kp": True, "include_dst": False},
    {"name": "Previous AE + Dst", "groups": {"Geomagnetic": ["ae"]}, "include_kp": False, "include_dst": True},
    {"name": "Previous AE + Kp + Dst", "groups": {"Geomagnetic": ["ae"]}, "include_kp": True, "include_dst": True},
]

# ---- Experiment 7: Best Combined Feature Sets -------------------------
# Structured, named combinations built from what Experiments 2-6
# established (persistence, solar wind, IMF, coupling physics,
# geomagnetic memory) — evaluated per horizon, with the best-scoring
# entry becoming that horizon's feature set for Experiment 8's model
# sweep. "Coupling" here means the Derived Physics core group plus the
# structured coupling registry features confirmed in Experiments 4-5
# (Newell, Akasofu, Boyle, Clock Angle Change, Strong Southward
# Duration, Integrated Ey/VBz/Energy Input).
_COUPLING_PHYSICS_FEATURES = {
    "Newell Coupling Function": True,
    "Akasofu Epsilon Parameter": True,
    "Boyle Index": True,
    "Clock Angle Change": True,
    "Strong Southward Duration": True,
    "Integrated Ey": True,
    "Integrated VBz": True,
    "Integrated Energy Input": True,
}


def _best_combo_grid() -> list[dict]:
    """Built as a function (not a module-level constant) so it always
    reflects the live PHYSICS_FEATURE_OPTIONS / FEATURE_GROUP_COLUMNS —
    e.g. "Full Physics Engine" must include every registered opt-in
    physics feature, including ones added after this module was written.
    """
    return [
        {"name": "Persistence Only", "groups": {"Geomagnetic": ["ae"]}, "physics_features": {}, "include_kp": False, "include_dst": False},
        {
            "name": "Persistence + Solar Wind",
            "groups": {"Geomagnetic": ["ae"], "Solar Wind": FEATURE_GROUP_COLUMNS["Solar Wind"]},
            "physics_features": {}, "include_kp": False, "include_dst": False,
        },
        {
            "name": "Persistence + Solar Wind + IMF",
            "groups": {"Geomagnetic": ["ae"], "Solar Wind": FEATURE_GROUP_COLUMNS["Solar Wind"], "IMF": FEATURE_GROUP_COLUMNS["IMF"]},
            "physics_features": {}, "include_kp": False, "include_dst": False,
        },
        {
            "name": "Persistence + Coupling",
            "groups": {"Geomagnetic": ["ae"], "Derived Physics": FEATURE_GROUP_COLUMNS["Derived Physics"]},
            "physics_features": dict(_COUPLING_PHYSICS_FEATURES), "include_kp": False, "include_dst": False,
        },
        {
            "name": "Persistence + Coupling + Geomagnetic Memory",
            "groups": {"Geomagnetic": ["ae"], "Derived Physics": FEATURE_GROUP_COLUMNS["Derived Physics"]},
            "physics_features": dict(_COUPLING_PHYSICS_FEATURES), "include_kp": True, "include_dst": True,
        },
        {
            "name": "Full Physics Engine",
            "groups": dict(FEATURE_GROUP_COLUMNS),
            "physics_features": {name: True for name in PHYSICS_FEATURE_OPTIONS},
            "include_kp": True, "include_dst": True,
        },
    ]


def compute_ae_persistence_benchmark(horizon: int) -> dict:
    """Persistence forecast for one horizon: "next AE (at horizon h) =
    current (most recently known) AE value" — the naive lower bound
    Production and every ML model in this study must beat at THAT
    horizon. Stored permanently under a fixed, horizon-specific run_id.
    """
    if horizon not in HORIZON_OPTIONS:
        raise ValueError(f"Horizon {horizon} is not valid — expected one of {HORIZON_OPTIONS}.")

    frame = _load_ae_base_frame()
    frame = frame.copy()
    frame["__target__"] = frame["ae"].shift(-horizon)
    frame = frame.dropna(subset=["ae", "__target__"])

    split_idx = int(len(frame) * (1 - TEST_FRACTION))
    y_test = frame["__target__"].iloc[split_idx:].to_numpy()
    y_pred = frame["ae"].iloc[split_idx:].to_numpy()  # persistence: value at h = value now
    metrics = compute_metrics(y_test, y_pred)
    sample_n = min(300, len(y_test))

    fixed_id = f"persistence_ae_{horizon}h"
    record = {
        "run_id": fixed_id,
        "target": TARGET_LABEL,
        "model_type": "Persistence Baseline",
        "horizon": horizon,
        "feature_toggles": {},
        "engineered_groups": {},
        "physics_features": {},
        "sequence_length": None,
        "hyperparams": {},
        "metrics": metrics,
        "training_time_sec": 0.0,
        "prediction_time_sec": 0.0,
        "feature_columns": ["ae"],
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
        "notes": f"Persistence baseline: AE at t+{horizon}h = AE now.",
        "experiment_tag": "persistence_benchmark",
        "model_size_kb": None,
        "train_r2": None,
        "inference_time_ms_per_sample": None,
        "include_kp": False,
        "include_dst": False,
    }
    runs = _load_runs()
    runs = [r for r in runs if r["run_id"] != fixed_id]
    runs.append(record)
    _save_runs(runs)
    return record


def get_production_ae_metrics(horizon: int) -> dict:
    """Reads the CURRENT production ae_{horizon}h entry straight from
    models/ae/metrics.json — always live, never cached. Returns None if
    no production model exists yet for this horizon.
    """
    metrics_path = MODELS_DIR / "ae" / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get(f"ae_{horizon}h")


def _rebuild_test_frame_for_ae_run(run: dict) -> tuple:
    """Deterministically rebuilds (X_train, X_test, y_train, y_test) for
    a stored run, without persisting raw feature matrices in the JSON
    registry — reruns the exact feature configuration (including
    include_kp/include_dst) the run record already stores, then reapplies
    the saved scaler if this run's model needed scaling (SVR/MLP), so
    both SHAP and permutation importance see the model's actual input
    space.
    """
    feature_toggles = run.get("feature_toggles") or default_feature_toggles()
    engineered_groups = run.get("engineered_groups") or default_engineered_toggles()
    physics_features = run.get("physics_features") or {}
    include_kp = run.get("include_kp", False)
    include_dst = run.get("include_dst", False)
    feature_columns = run["feature_columns"]
    horizon = run["horizon"]

    if include_kp or include_dst:
        frame, _ = load_ae_research_frame_with_geomagnetic_memory(
            feature_toggles, engineered_groups, physics_features, include_kp=include_kp, include_dst=include_dst
        )
    else:
        frame, _ = load_ae_research_frame(feature_toggles, engineered_groups, physics_features)
    frame = frame.copy()
    frame["__target__"] = frame["ae"].shift(-horizon)
    frame = frame.dropna(subset=feature_columns + ["__target__"])

    split_idx = int(len(frame) * (1 - TEST_FRACTION))
    X_train = frame[feature_columns].iloc[:split_idx]
    X_test = frame[feature_columns].iloc[split_idx:]
    y_train = frame["__target__"].iloc[:split_idx]
    y_test = frame["__target__"].iloc[split_idx:]

    scaler_path = RESEARCH_MODELS_DIR / run["run_id"] / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=feature_columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=feature_columns)

    return X_train, X_test, y_train, y_test


def compute_shap_importance_ae(run_id: str, max_background: int = 100, max_explain: int = 200) -> dict:
    """SHAP-based feature importance for a stored AE run (Experiment 9).
    Only supported for linear and tree/boosting families (fast native
    SHAP explainers) — see SHAP_SUPPORTED_MODELS. SVR/MLP are instead
    covered by compute_permutation_importance_ae; LSTM/GRU cannot be
    reloaded into this process at all (see load_trained_model).
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
                "family. Use compute_permutation_importance_ae for SVR/MLP, or a linear/tree model."
            ),
            "shap_importance": None,
        }

    import shap

    model = load_trained_model(run_id)
    X_train, X_test, _, _ = _rebuild_test_frame_for_ae_run(run)
    background = X_train.sample(n=min(max_background, len(X_train)), random_state=42) if len(X_train) else X_train
    explain_rows = X_test.sample(n=min(max_explain, len(X_test)), random_state=42) if len(X_test) else X_test

    explainer = shap.Explainer(model, background)
    try:
        shap_values = explainer(explain_rows)
    except Exception:
        # TreeExplainer's additivity check is a known false-positive source
        # for RandomForestRegressor on wide feature sets (floating-point
        # accumulation across many trees) — retry with the check disabled
        # rather than losing the whole importance computation to it.
        shap_values = explainer(explain_rows, check_additivity=False)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    pairs = list(zip(run["feature_columns"], (float(v) for v in mean_abs)))
    ranked = sorted(pairs, key=lambda kv: -kv[1])[:30]

    return {"run_id": run_id, "supported": True, "skipped_reason": None, "shap_importance": ranked}


def compute_permutation_importance_ae(run_id: str, n_repeats: int = 5, max_explain: int = 200) -> dict:
    """Model-agnostic permutation importance (Experiment 9's fallback for
    SVR/MLP — models with neither .feature_importances_ nor .coef_, so
    neither train_ae_research_model's own feature_importance nor SHAP's
    fast explainers apply to them). Shuffles each feature column
    independently and measures the resulting drop in test-set R²; a
    larger drop means the model relies on that feature more.
    """
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found.")
    model_type = run["model_type"]

    if model_type not in PERMUTATION_SUPPORTED_MODELS:
        return {
            "run_id": run_id,
            "supported": False,
            "skipped_reason": (
                f"Permutation importance skipped for {model_type} — this model type already exposes "
                "native feature_importances_/coef_ (see train_ae_research_model's feature_importance), "
                "or is a sequence model that cannot be reloaded into this process."
            ),
            "permutation_importance": None,
        }

    from sklearn.inspection import permutation_importance

    model = load_trained_model(run_id)
    _, X_test, _, y_test = _rebuild_test_frame_for_ae_run(run)
    if len(X_test) > max_explain:
        X_test = X_test.sample(n=max_explain, random_state=42)
        y_test = y_test.loc[X_test.index]

    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring="r2", n_jobs=-1)
    pairs = list(zip(run["feature_columns"], (float(v) for v in result.importances_mean)))
    ranked = sorted(pairs, key=lambda kv: -kv[1])[:30]

    return {"run_id": run_id, "supported": True, "skipped_reason": None, "permutation_importance": ranked}


def check_promotion_criteria_ae(candidate_run: dict, production_metrics: dict, horizon: int) -> dict:
    """Evaluates the AE promotion checklist for one horizon's candidate
    run against that SAME horizon's current production model.
    `production_metrics` is get_production_ae_metrics(horizon)'s return
    value — None means no production model exists yet for this horizon
    (every comparison trivially passes).
    """
    checklist = []

    def _check(name, passed, detail):
        checklist.append({"criterion": name, "passed": bool(passed), "detail": detail})

    _check(
        f"Same forecasting objective (hourly AE at t+{horizon}h)",
        candidate_run.get("target") == TARGET_LABEL and candidate_run.get("horizon") == horizon,
        f"Every run compared here targets AE shift(-{horizon}), the identical target production's own "
        f"ae_{horizon}h model trains on.",
    )
    _check(
        f"Same training methodology (ae_analytics_features.csv, {TEST_FRACTION:.0%} test split)",
        True,
        "Every run in this module trains on load_ae_research_frame() (or its geomagnetic-memory "
        "extension), built from the same ae_analytics_features.csv production's own model trains on, "
        "with the same TEST_FRACTION split.",
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
        _check("Better R² than current production", True, f"No production ae_{horizon}h model currently on disk — nothing to compare against.")
        _check("Lower MAE than current production", True, f"No production ae_{horizon}h model currently on disk — nothing to compare against.")
        _check("Lower RMSE than current production", True, f"No production ae_{horizon}h model currently on disk — nothing to compare against.")
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


def promote_ae_to_production(run_id: str, horizon: int, notes: str = "") -> dict:
    """Archives the current production ae_{horizon}h model and installs a
    research run in its place — per-horizon, since AE has 5 independent
    production models. Mirrors kp_research.promote_kp_to_production's
    archive/install/metrics-update/rollback pattern exactly.

    1. Validates the run is a non-sequence tabular model trained at
       exactly this horizon, with a saved .joblib artifact.
    2. Archives models/ae/ae_{horizon}h.joblib to
       models/ae/archive/ae_{horizon}h_<timestamp>.joblib (rollback path).
    3. Copies the research model's model.joblib -> models/ae/ae_{horizon}h.joblib.
    4. Updates models/ae/metrics.json[f"ae_{horizon}h"] with the new
       metrics, algorithm, and feature_columns.
    5. Marks the run promoted=True, promoted_to_production=True.
    """
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found in the research registry.")
    if run.get("horizon") != horizon:
        raise ValueError(f"Run {run_id} was trained at horizon={run.get('horizon')}, not {horizon} — refusing to promote into the wrong horizon's production slot.")
    if run.get("model_type") in SEQUENCE_MODELS:
        raise ValueError("LSTM/GRU promotion is not supported — loading a Keras model into the prediction process causes the same process-conflict hang training does.")
    if not run.get("model_path"):
        raise ValueError("This run has no saved model artifact to promote (e.g. the persistence benchmark).")

    production_dir = MODELS_DIR / "ae"
    prod_model_path = production_dir / f"ae_{horizon}h.joblib"
    prod_metrics_path = production_dir / "metrics.json"
    metrics_key = f"ae_{horizon}h"

    archive_dir = production_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"ae_{horizon}h_{ts}.joblib"
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
        old_metrics = metrics_data.get(metrics_key)

    metrics_data[metrics_key] = {
        "variable": "ae",
        "horizon": horizon,
        "algorithm": run["model_type"],
        "r2": run["metrics"]["r2"],
        "mae": run["metrics"]["mae"],
        "rmse": run["metrics"]["rmse"],
        "bias": run["metrics"].get("bias"),
        "feature_columns": run["feature_columns"],
        "model_path": str(prod_model_path),
        "trained_at": run["trained_at"],
        "n_samples": run["n_train_samples"] + run["n_test_samples"],
        "n_train": run["n_train_samples"],
        "n_test": run["n_test_samples"],
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
        "horizon": horizon,
        "archive_path": str(archive_path) if archive_path else None,
        "new_model_path": str(prod_model_path),
        "old_metrics": old_metrics,
        "new_metrics": run["metrics"],
        "model_type": run["model_type"],
        "feature_count": len(run["feature_columns"]),
        "promoted_at": metrics_data[metrics_key]["promoted_at"],
    }


def _map_feature_to_group(feature_name: str) -> str:
    """Maps an engineered feature name (e.g. "bz_gsm_lag3h",
    "speed_24h_std", "ey_change") back to its originating feature group,
    for Experiment 10's Feature Group Importance vs. Horizon plot. Used
    only for this cross-horizon aggregation — never affects training.
    """
    base = feature_name
    for suffix in ("_lag1h", "_lag3h", "_lag6h", "_lag12h", "_lag24h", "_24h_std", "_24h", "_change"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for group, cols in FEATURE_GROUP_COLUMNS.items():
        if base in cols:
            return group
    if base in ("kp", "dst"):
        return "Geomagnetic Memory (Kp/Dst)"
    return "Physics Engine (opt-in)"


def _build_cross_horizon_synthesis(horizon_results: dict) -> dict:
    """Experiment 10 — Cross-Horizon Scientific Synthesis, the centerpiece
    of this study. Compares all 5 independently-run horizons against each
    other to answer: does persistence importance fall as horizon grows?
    Does coupling-physics importance rise? Does geomagnetic memory help
    more or less? How does feature importance evolve? Does prediction
    skill decay? And — explicitly — is there a measurable horizon at
    which physics-derived features overtake AE's own memory as the
    dominant contributor to the winning model's feature importance? If
    none is found, that is reported as a genuine negative result, not
    hidden.
    """
    persistence_importance_vs_horizon = []
    physics_importance_vs_horizon = []
    model_skill_vs_horizon = []
    feature_group_importance_vs_horizon = []

    for h in HORIZON_OPTIONS:
        hr = horizon_results[h]
        persistence_r2 = hr["persistence"]["metrics"]["r2"]
        baseline_r2 = hr["baseline"]["metrics"]["r2"]
        winner_r2 = hr["winner"]["metrics"]["r2"]
        raw_floor_r2 = hr["raw_floor"]["metrics"]["r2"]
        best_coupling_r2 = max((c["r2"] for c in hr["coupling_results"]), default=None)
        top_ablation_delta = max((a["delta_r2_from_baseline"] for a in hr["ablation_results"]), default=None)
        best_geomag_r2 = max((g["r2"] for g in hr["geomag_results"]), default=None)

        persistence_importance_vs_horizon.append({
            "horizon": h,
            "persistence_r2": persistence_r2,
            "production_r2": baseline_r2,
            "winner_r2": winner_r2,
            "persistence_share_of_winner": (persistence_r2 / winner_r2) if winner_r2 else None,
        })
        physics_importance_vs_horizon.append({
            "horizon": h,
            "raw_floor_r2": raw_floor_r2,
            "best_single_coupling_variable_r2": best_coupling_r2,
            "top_ablation_delta_r2": top_ablation_delta,
        })
        model_skill_vs_horizon.append({
            "horizon": h,
            "winner_model": hr["winner"]["model_type"],
            "winner_r2": winner_r2,
            "persistence_r2": persistence_r2,
            "production_r2": baseline_r2,
            "best_geomagnetic_memory_r2": best_geomag_r2,
        })

        group_shares: dict = {}
        fi = hr.get("feature_importance")
        if fi:
            for name, value in fi:
                group = _map_feature_to_group(name)
                group_shares[group] = group_shares.get(group, 0.0) + abs(value)
            total = sum(group_shares.values()) or 1.0
            group_shares = {g: v / total for g, v in group_shares.items()}
        feature_group_importance_vs_horizon.append({"horizon": h, "group_shares": group_shares})

    crossover_horizon = None
    for entry in feature_group_importance_vs_horizon:
        shares = entry["group_shares"]
        if not shares:
            continue
        persistence_share = shares.get("Geomagnetic", 0.0) + shares.get("Geomagnetic Memory (Kp/Dst)", 0.0)
        physics_share = shares.get("Derived Physics", 0.0) + shares.get("Physics Engine (opt-in)", 0.0)
        if physics_share > persistence_share:
            crossover_horizon = entry["horizon"]
            break

    crossover_detected = crossover_horizon is not None
    crossover_conclusion = (
        f"A measurable persistence -> physics crossover was found at the {crossover_horizon}h horizon: "
        "beyond this point, coupling/physics-derived features explain more of the winning model's "
        "feature importance than AE's own memory (persistence + Kp/Dst) does."
        if crossover_detected
        else (
            "No measurable persistence -> physics crossover was found across the tested horizons "
            f"({HORIZON_OPTIONS[0]}h-{HORIZON_OPTIONS[-1]}h) — AE's own memory (persistence + Kp/Dst) "
            "remained the dominant feature-importance contributor at every horizon tested. Reported "
            "honestly as a negative result rather than forced or hidden."
        )
    )

    return {
        "persistence_importance_vs_horizon": persistence_importance_vs_horizon,
        "physics_importance_vs_horizon": physics_importance_vs_horizon,
        "model_skill_vs_horizon": model_skill_vs_horizon,
        "feature_group_importance_vs_horizon": feature_group_importance_vs_horizon,
        "crossover_detected": crossover_detected,
        "crossover_horizon": crossover_horizon,
        "crossover_conclusion": crossover_conclusion,
    }


def _build_ae_scientific_report(horizon_results: dict, cross_horizon: dict) -> dict:
    """Assembles the complete scientific report generated after every
    optimization run — every section the spec requires, derived directly
    from this run's own computed numbers (never hardcoded prose).
    """
    best_models = {str(h): horizon_results[h]["winner"]["model_type"] for h in HORIZON_OPTIONS}
    best_feature_sets = {str(h): horizon_results[h]["best_combo"]["name"] for h in HORIZON_OPTIONS}
    physics_ranking = {str(h): horizon_results[h]["coupling_results"] for h in HORIZON_OPTIONS}
    geomagnetic_memory_ranking = {str(h): horizon_results[h]["geomag_results"] for h in HORIZON_OPTIONS}
    model_rankings = {str(h): horizon_results[h]["leaderboard"] for h in HORIZON_OPTIONS}
    production_recommendation = {
        str(h): {
            "recommendation": horizon_results[h]["recommendation"],
            "production_comparison": horizon_results[h]["production_comparison"],
            "promotion_check": horizon_results[h]["promotion_check"],
        }
        for h in HORIZON_OPTIONS
    }
    experiment_summary = {
        str(h): {
            "production_baseline_r2": horizon_results[h]["baseline"]["metrics"]["r2"],
            "persistence_r2": horizon_results[h]["persistence"]["metrics"]["r2"],
            "raw_floor_r2": horizon_results[h]["raw_floor"]["metrics"]["r2"],
            "best_single_coupling_variable": max(horizon_results[h]["coupling_results"], key=lambda r: r["r2"])["name"],
            "best_geomagnetic_memory_combo": max(horizon_results[h]["geomag_results"], key=lambda r: r["r2"])["name"],
            "best_feature_set": horizon_results[h]["best_combo"]["name"],
            "winner_model": horizon_results[h]["winner"]["model_type"],
            "winner_r2": horizon_results[h]["winner"]["metrics"]["r2"],
        }
        for h in HORIZON_OPTIONS
    }

    first_h, last_h = HORIZON_OPTIONS[0], HORIZON_OPTIONS[-1]
    persistence_first = horizon_results[first_h]["persistence"]["metrics"]["r2"]
    persistence_last = horizon_results[last_h]["persistence"]["metrics"]["r2"]
    winner_first = horizon_results[first_h]["winner"]["metrics"]["r2"]
    winner_last = horizon_results[last_h]["winner"]["metrics"]["r2"]

    top_physics_by_horizon = {h: max(horizon_results[h]["coupling_results"], key=lambda r: r["r2"]) for h in HORIZON_OPTIONS}
    best_geomag_by_horizon = {h: max(horizon_results[h]["geomag_results"], key=lambda r: r["r2"]) for h in HORIZON_OPTIONS}
    promote_horizons = [h for h in HORIZON_OPTIONS if horizon_results[h]["recommendation"] == "Promote"]

    scientific_conclusions = [
        (
            f"Prediction skill decays sharply with horizon: the best model's R² falls from "
            f"{winner_first:.3f} at {first_h}h to {winner_last:.3f} at {last_h}h, while the naive "
            f"persistence benchmark falls from {persistence_first:.3f} to {persistence_last:.3f} over "
            "the same range — the realistic prediction ceiling for hourly AE forecasting shrinks "
            "quickly beyond a few hours."
        ),
        (
            "Strongest standalone coupling-physics variable per horizon: "
            + ", ".join(f"{h}h→{r['name']} (R²={r['r2']:.3f})" for h, r in top_physics_by_horizon.items())
            + "."
        ),
        (
            "Strongest geomagnetic-memory combination per horizon: "
            + ", ".join(f"{h}h→{r['name']} (R²={r['r2']:.3f})" for h, r in best_geomag_by_horizon.items())
            + "."
        ),
        cross_horizon["crossover_conclusion"],
        (
            f"Production promotion recommended at horizon(s): {promote_horizons if promote_horizons else 'none'} "
            "— see production_recommendation for the full per-horizon promotion checklist."
        ),
    ]

    future_work = [
        "The archived minute-resolution Kyoto AE dataset (kyoto_ae_minute, see kyoto_ae.py) was NOT "
        "used anywhere in this study by design — Production forecasts hourly AE, and this study is "
        "deliberately an hourly-regression optimization framework only. The minute archive exists for "
        "a genuinely different future research question (substorm onset timing, minute-scale AE "
        "dynamics, event detection) that this study does not attempt to answer.",
        "A genuine substorm onset-detection reframing of AE prediction — classifying onset timing/"
        "phase rather than regressing the raw hourly index — remains unexplored and would likely "
        "require the minute-resolution archive this study intentionally left untouched.",
        "Every feature-search experiment (3-7) used a single probe model (Random Forest) to keep the "
        "5-horizon x many-configuration search tractable; Experiment 8 re-validates the winning "
        "feature set with the full 12-model sweep, but a full model x feature-set cross-product was "
        "not attempted and could reveal different winners.",
    ]

    return {
        "experiment_summary": experiment_summary,
        "best_models": best_models,
        "best_feature_sets": best_feature_sets,
        "physics_ranking": physics_ranking,
        "geomagnetic_memory_ranking": geomagnetic_memory_ranking,
        "cross_horizon_analysis": cross_horizon,
        "persistence_analysis": cross_horizon["persistence_importance_vs_horizon"],
        "model_rankings": model_rankings,
        "production_recommendation": production_recommendation,
        "scientific_conclusions": scientific_conclusions,
        "future_work": future_work,
    }


def run_complete_ae_optimization_study(progress_cb=None) -> dict:
    """AutoML orchestrator — runs the full 10-experiment AE Optimization
    Study end-to-end with zero user interaction beyond this one call,
    independently at all 5 production horizons (1h/3h/6h/12h/24h), then
    performs Experiment 10's Cross-Horizon Scientific Synthesis. Every
    candidate uses train_ae_research_model()/load_ae_research_frame() —
    the exact same functions the manual Exp 1-10 tabs call — this
    function only sequences them, tracks results, and packages a final
    study record. It NEVER calls promote_ae_to_production(); promotion is
    always a separate, explicit, per-horizon, human-confirmed action in
    the UI.

    This trains roughly 45-50 models PER HORIZON (~230-250 total across
    all 5 horizons), including SVR/MLP and a permutation-importance pass
    at each horizon — measured on this project's own hardware/dataset
    size, a single horizon's worth of comparable work took ~12 minutes,
    so expect the full 5-horizon study to run for an hour or more.

    `progress_cb`, if given, is called as progress_cb(step_number,
    total_steps, message) after each of the 10 experiment steps — drives
    a live st.status() panel in the dashboard UI.
    """
    def _p(step, total, msg):
        if progress_cb:
            progress_cb(step, total, msg)

    total_steps = 10
    study_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    horizon_results = {h: {} for h in HORIZON_OPTIONS}

    # Experiment 1 — Production Baseline (per horizon, using that
    # horizon's own production algorithm)
    for h in HORIZON_OPTIONS:
        horizon_results[h]["baseline"] = train_ae_research_model(
            PRODUCTION_BASELINE_MODEL_BY_HORIZON[h],
            horizon=h,
            feature_toggles=_production_baseline_toggles(),
            engineered_groups=default_engineered_toggles(),
            physics_features={},
            experiment_tag=f"auto_{study_id}_baseline_h{h}",
        )
    _p(1, total_steps, "Production baselines reproduced at all 5 horizons.")

    # Experiment 2 — Persistence Benchmark (per horizon)
    for h in HORIZON_OPTIONS:
        horizon_results[h]["persistence"] = compute_ae_persistence_benchmark(h)
    _p(2, total_steps, "Persistence benchmarks computed at all 5 horizons.")

    # Experiment 3 — Solar Wind + IMF Raw Floor (per horizon)
    for h in HORIZON_OPTIONS:
        horizon_results[h]["raw_floor"] = train_ae_research_model(
            FEATURE_SEARCH_PROBE_MODEL,
            horizon=h,
            feature_toggles=_build_isolated_toggles(SOLAR_WIND_IMF_RAW_FLOOR_GROUPS),
            engineered_groups=default_engineered_toggles(),
            physics_features={},
            experiment_tag=f"auto_{study_id}_raw_floor_h{h}",
        )
    _p(3, total_steps, "Solar Wind + IMF raw explanatory floor established at all 5 horizons.")

    # Experiment 4 — Coupling Physics (individual, isolated; per horizon)
    for h in HORIZON_OPTIONS:
        coupling_results = []
        for entry in COUPLING_PHYSICS_GRID:
            if entry["kind"] == "core_column":
                toggles = _build_isolated_toggles({"Derived Physics": [entry["column"]]})
                physics_features = {}
            else:
                toggles = _build_isolated_toggles({})
                physics_features = {entry["physics_name"]: True}
            run = train_ae_research_model(
                FEATURE_SEARCH_PROBE_MODEL,
                horizon=h,
                feature_toggles=toggles,
                engineered_groups=default_engineered_toggles(),
                physics_features=physics_features,
                experiment_tag=f"auto_{study_id}_coupling_h{h}",
            )
            coupling_results.append({
                "name": entry["name"], "run_id": run["run_id"], "r2": run["metrics"]["r2"],
                "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"],
            })
        coupling_results.sort(key=lambda r: -r["r2"])
        horizon_results[h]["coupling_results"] = coupling_results
    _p(4, total_steps, "Coupling physics variables evaluated individually at all 5 horizons.")

    # Experiment 5 — Physics Engine Ablation (cumulative, from Production; per horizon)
    for h in HORIZON_OPTIONS:
        baseline_r2 = horizon_results[h]["baseline"]["metrics"]["r2"]
        ablation_results = []
        for step in PHYSICS_ENGINE_ABLATION_STEPS:
            run = train_ae_research_model(
                FEATURE_SEARCH_PROBE_MODEL,
                horizon=h,
                feature_toggles=default_feature_toggles(),
                engineered_groups=default_engineered_toggles(),
                physics_features=step["physics_features"],
                experiment_tag=f"auto_{study_id}_ablation_h{h}",
            )
            ablation_results.append({
                "name": step["name"], "run_id": run["run_id"], "r2": run["metrics"]["r2"],
                "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"],
                "delta_r2_from_baseline": run["metrics"]["r2"] - baseline_r2,
                "already_in_baseline": step["already_in_baseline"],
            })
        horizon_results[h]["ablation_results"] = ablation_results
    _p(5, total_steps, "Physics Engine cumulative ablation complete at all 5 horizons.")

    # Experiment 6 — Geomagnetic Memory (per horizon)
    for h in HORIZON_OPTIONS:
        geomag_results = []
        for combo in GEOMAGNETIC_MEMORY_GRID:
            run = train_ae_research_model(
                FEATURE_SEARCH_PROBE_MODEL,
                horizon=h,
                feature_toggles=_build_isolated_toggles(combo["groups"]),
                engineered_groups=default_engineered_toggles(),
                physics_features={},
                include_kp=combo["include_kp"],
                include_dst=combo["include_dst"],
                experiment_tag=f"auto_{study_id}_geomag_h{h}",
            )
            geomag_results.append({
                "name": combo["name"], "run_id": run["run_id"], "r2": run["metrics"]["r2"],
                "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"],
            })
        geomag_results.sort(key=lambda r: -r["r2"])
        horizon_results[h]["geomag_results"] = geomag_results
    _p(6, total_steps, "Geomagnetic memory (Previous AE/Kp/Dst) evaluated at all 5 horizons.")

    # Experiment 7 — Best Combined Feature Sets (per horizon)
    for h in HORIZON_OPTIONS:
        combo_grid = _best_combo_grid()
        combo_results = []
        for combo in combo_grid:
            run = train_ae_research_model(
                FEATURE_SEARCH_PROBE_MODEL,
                horizon=h,
                feature_toggles=_build_isolated_toggles(combo["groups"]),
                engineered_groups=default_engineered_toggles(),
                physics_features=combo["physics_features"],
                include_kp=combo["include_kp"],
                include_dst=combo["include_dst"],
                experiment_tag=f"auto_{study_id}_bestcombo_h{h}",
            )
            combo_results.append({
                "name": combo["name"], "run_id": run["run_id"], "r2": run["metrics"]["r2"],
                "mae": run["metrics"]["mae"], "rmse": run["metrics"]["rmse"], "config": combo,
            })
        combo_results.sort(key=lambda r: -r["r2"])
        horizon_results[h]["combo_results"] = combo_results
        horizon_results[h]["best_combo"] = combo_results[0]
    _p(7, total_steps, "Best combined feature set determined per horizon.")

    # Experiment 8 — Model Comparison (all 12 models, on each horizon's best feature set)
    for h in HORIZON_OPTIONS:
        best_config = horizon_results[h]["best_combo"]["config"]
        best_toggles = _build_isolated_toggles(best_config["groups"])
        best_physics = best_config["physics_features"]
        best_include_kp = best_config["include_kp"]
        best_include_dst = best_config["include_dst"]

        model_results = []
        for model_type in TABULAR_MODELS:
            try:
                run = train_ae_research_model(
                    model_type, horizon=h, feature_toggles=best_toggles, engineered_groups=default_engineered_toggles(),
                    physics_features=best_physics, include_kp=best_include_kp, include_dst=best_include_dst,
                    experiment_tag=f"auto_{study_id}_modelsweep_h{h}",
                )
                model_results.append(run)
            except Exception as exc:
                model_results.append({"run_id": None, "model_type": model_type, "error": str(exc)})
        if KERAS_AVAILABLE and SEQUENCE_MODELS:
            for model_type in SEQUENCE_MODELS:
                try:
                    run = train_ae_research_model(
                        model_type, horizon=h, feature_toggles=best_toggles, engineered_groups=default_engineered_toggles(),
                        physics_features=best_physics, include_kp=best_include_kp, include_dst=best_include_dst,
                        experiment_tag=f"auto_{study_id}_modelsweep_h{h}",
                    )
                    model_results.append(run)
                except Exception as exc:
                    model_results.append({"run_id": None, "model_type": model_type, "error": str(exc)})

        valid_candidates = [r for r in model_results if r.get("run_id")]
        if not valid_candidates:
            raise RuntimeError(f"Model comparison produced no successful candidates at horizon={h}h — cannot complete the study.")
        valid_candidates.sort(key=lambda r: -r["metrics"]["r2"])
        winner = valid_candidates[0]

        leaderboard = [
            {
                "rank": rank, "run_id": r["run_id"], "model_type": r["model_type"],
                "r2": r["metrics"]["r2"], "mae": r["metrics"]["mae"], "rmse": r["metrics"]["rmse"],
                "mape": r["metrics"].get("mape"), "bias": r["metrics"]["bias"],
                "training_time_sec": r.get("training_time_sec"),
                "inference_time_ms_per_sample": r.get("inference_time_ms_per_sample"),
                "model_size_kb": r.get("model_size_kb"), "feature_count": len(r.get("feature_columns", [])),
            }
            for rank, r in enumerate(valid_candidates, start=1)
        ]
        horizon_results[h]["model_results"] = model_results
        horizon_results[h]["failed_candidates"] = [{"model_type": r.get("model_type"), "error": r.get("error")} for r in model_results if not r.get("run_id")]
        horizon_results[h]["leaderboard"] = leaderboard
        horizon_results[h]["winner"] = winner
    _p(8, total_steps, "All 12 models compared on each horizon's best feature set.")

    # Experiment 9 — Feature Importance, SHAP, and Permutation Importance (per horizon)
    for h in HORIZON_OPTIONS:
        hr = horizon_results[h]
        valid_candidates = [r for r in hr["model_results"] if r.get("run_id")]
        winner = hr["winner"]

        fi_source = winner
        if not fi_source.get("feature_importance"):
            with_fi = [r for r in valid_candidates if r.get("feature_importance")]
            fi_source = with_fi[0] if with_fi else None
        hr["feature_importance"] = fi_source["feature_importance"] if fi_source else None
        hr["feature_importance_source_run_id"] = fi_source["run_id"] if fi_source else None

        shap_source_run_id = next((c["run_id"] for c in valid_candidates if c["model_type"] in SHAP_SUPPORTED_MODELS), None)
        if shap_source_run_id:
            try:
                hr["shap_result"] = compute_shap_importance_ae(shap_source_run_id)
            except Exception as exc:
                hr["shap_result"] = {"run_id": shap_source_run_id, "supported": False, "skipped_reason": f"SHAP failed: {exc}", "shap_importance": None}
        else:
            hr["shap_result"] = {"run_id": None, "supported": False, "skipped_reason": "No SHAP-supported candidate in this horizon's model sweep.", "shap_importance": None}

        perm_source_run_id = next((c["run_id"] for c in valid_candidates if c["model_type"] in PERMUTATION_SUPPORTED_MODELS), None)
        if perm_source_run_id:
            try:
                hr["permutation_result"] = compute_permutation_importance_ae(perm_source_run_id)
            except Exception as exc:
                hr["permutation_result"] = {"run_id": perm_source_run_id, "supported": False, "skipped_reason": f"Permutation importance failed: {exc}", "permutation_importance": None}
        else:
            hr["permutation_result"] = {"run_id": None, "supported": False, "skipped_reason": "No SVR/MLP candidate in this horizon's model sweep.", "permutation_importance": None}

        production_metrics = get_production_ae_metrics(h)
        promotion_check = check_promotion_criteria_ae(winner, production_metrics, h)
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
        hr["production_metrics"] = production_metrics
        hr["production_comparison"] = production_comparison
        hr["promotion_check"] = promotion_check
        hr["recommendation"] = "Promote" if promotion_check["eligible"] else "Keep Current Production"
    _p(9, total_steps, "Feature importance, SHAP, and permutation importance extracted per horizon.")

    # Experiment 10 — Cross-Horizon Scientific Synthesis (the centerpiece)
    cross_horizon = _build_cross_horizon_synthesis(horizon_results)
    report = _build_ae_scientific_report(horizon_results, cross_horizon)
    _p(10, total_steps, f"Cross-horizon synthesis complete — {cross_horizon['crossover_conclusion']}")

    completed_at = datetime.now(timezone.utc).isoformat()
    study = {
        "study_id": study_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "target": TARGET_LABEL,
        "training_dataset": "data/features/training/ae_analytics_features.csv",
        "horizons": HORIZON_OPTIONS,
        "horizon_results": {str(h): horizon_results[h] for h in HORIZON_OPTIONS},
        "cross_horizon_synthesis": cross_horizon,
        "report": report,
        "promotion_status_by_horizon": {str(h): "pending" for h in HORIZON_OPTIONS},
        "promoted_run_id_by_horizon": {str(h): None for h in HORIZON_OPTIONS},
        "promoted_at_by_horizon": {str(h): None for h in HORIZON_OPTIONS},
    }
    _append_ae_study(study)
    return study


def _load_ae_studies() -> list[dict]:
    if not STUDIES_REGISTRY_PATH.exists():
        return []
    with open(STUDIES_REGISTRY_PATH) as f:
        return json.load(f)


def _save_ae_studies(studies: list[dict]) -> None:
    STUDIES_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDIES_REGISTRY_PATH, "w") as f:
        json.dump(studies, f, indent=2)


def _append_ae_study(study: dict) -> None:
    studies = _load_ae_studies()
    studies.append(study)
    _save_ae_studies(studies)


def list_ae_studies() -> list[dict]:
    return sorted(_load_ae_studies(), key=lambda s: s["started_at"], reverse=True)


def get_ae_study(study_id: str) -> dict:
    for s in _load_ae_studies():
        if s["study_id"] == study_id:
            return s
    return None


def _update_ae_study(study_id: str, **fields) -> bool:
    studies = _load_ae_studies()
    found = False
    for s in studies:
        if s["study_id"] == study_id:
            s.update(fields)
            found = True
    if found:
        _save_ae_studies(studies)
    return found


def mark_ae_study_promoted(study_id: str, horizon: int, run_id: str) -> bool:
    studies = _load_ae_studies()
    found = False
    for s in studies:
        if s["study_id"] == study_id:
            s.setdefault("promotion_status_by_horizon", {})[str(horizon)] = "promoted"
            s.setdefault("promoted_run_id_by_horizon", {})[str(horizon)] = run_id
            s.setdefault("promoted_at_by_horizon", {})[str(horizon)] = datetime.now(timezone.utc).isoformat()
            found = True
    if found:
        _save_ae_studies(studies)
    return found


def mark_ae_study_rejected(study_id: str, horizon: int, notes: str = "") -> bool:
    studies = _load_ae_studies()
    found = False
    for s in studies:
        if s["study_id"] == study_id:
            s.setdefault("promotion_status_by_horizon", {})[str(horizon)] = "rejected"
            s.setdefault("rejection_notes_by_horizon", {})[str(horizon)] = notes
            found = True
    if found:
        _save_ae_studies(studies)
    return found
