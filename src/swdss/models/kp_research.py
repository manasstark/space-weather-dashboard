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
from sklearn.svm import SVR
from xgboost import XGBRegressor

from swdss.models.features import add_change_features, add_lag_features
from swdss.models.kp_physics_features import PHYSICS_FEATURE_DEPENDENCIES, PHYSICS_FEATURE_FUNCTIONS
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
    """vbz/ey/dynamic_pressure, computed in memory exactly like
    swdss.models.features.add_derived_physics_features — duplicated
    (rather than imported) only because that function also handles the
    IMF-only/Solar-Wind-only no-op cases this module never needs; the
    formulas themselves must stay identical to production's.
    """
    created = []
    df["vbz"] = df["speed"] * df["bz_gsm"].clip(upper=0)
    created.append("vbz")
    df["ey"] = -df["speed"] * df["bz_gsm"] * 1e-3
    created.append("ey")
    df["dynamic_pressure"] = 1.6726e-6 * df["density"] * df["speed"] ** 2
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

    physics_cols: list[str] = []
    for name in PHYSICS_FEATURE_FUNCTIONS:
        if not physics_features.get(name, False):
            continue
        for dependency in PHYSICS_FEATURE_DEPENDENCIES.get(name, []):
            if dependency not in physics_cols and physics_features.get(dependency, False) is False:
                PHYSICS_FEATURE_FUNCTIONS[dependency](frame)
        physics_cols += PHYSICS_FEATURE_FUNCTIONS[name](frame)

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
) -> dict:
    """Trains one Kp model for one feature configuration and records a
    run — never touches models/analytics/ or its metrics.json. Target is
    always "the next official NOAA 3-hour Kp interval" (see
    build_kp_interval_target), identical to production's own target
    definition, so R²/MAE/RMSE here are directly comparable to
    production's stored kp_interval metrics.
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
