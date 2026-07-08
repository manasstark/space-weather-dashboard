"""Step 3 — Retrain every production model on the expanded v2 dataset.

Uses identical algorithms, hyperparameters, train/test split, and feature
engineering as the v1 pipeline.  Only the training data and output paths
differ.  Never touches v1 model files.

Output:
  models/{dataset}_v2/{variable}_{horizon}h.joblib
  models/{dataset}_v2/metrics.json          (includes bias)

Run from project root:
    PYTHONPATH=src venv/bin/python3 scripts/refresh/03_train_v2.py [dataset ...]

Omitting dataset arguments trains all datasets.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swdss.models.features import add_derived_physics_features, build_feature_frame
from swdss.models.registry import (
    DATASETS,
    HORIZONS,
    DatasetConfig,
)

TRAIN_V2_DIR = PROJECT_ROOT / "data" / "features" / "training_v2"
MODELS_V2_DIR = PROJECT_ROOT / "models"

TEST_FRACTION = 0.2

CANDIDATE_MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}


def v2_training_csv(dataset_key: str) -> str:
    """Map dataset key → v2 training CSV path."""
    name_map = {
        "solar_wind":   "solar_wind_features.csv",
        "imf":          "imf_features.csv",
        "kp":           "kp_features.csv",
        "dst":          "dst_features.csv",
        "ae":           "ae_analytics_features.csv",
        "analytics":    "analytics_features.csv",
        "experimental": "experimental_features.csv",
    }
    return str(TRAIN_V2_DIR / name_map[dataset_key])


def v2_model_dir(dataset_key: str) -> Path:
    path = MODELS_V2_DIR / f"{dataset_key}_v2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def v2_model_path(dataset_key: str, variable: str, horizon: int) -> Path:
    return v2_model_dir(dataset_key) / f"{variable}_{horizon}h.joblib"


def v2_kp_interval_path(dataset_key: str) -> Path:
    return v2_model_dir(dataset_key) / "kp_interval.joblib"


def v2_metrics_path(dataset_key: str) -> Path:
    return v2_model_dir(dataset_key) / "metrics.json"


def evaluate_split(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    bias = float(np.mean(preds - y_test))
    return {
        "r2":   float(r2_score(y_test, preds)),
        "mae":  float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "bias": bias,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }


def fit_best(X: pd.DataFrame, y: pd.Series) -> tuple:
    split_idx = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    best_name = None
    best_metrics = None

    for name, factory in CANDIDATE_MODELS.items():
        m = factory()
        metrics = evaluate_split(m, X_train, y_train, X_test, y_test)
        if best_metrics is None or metrics["r2"] > best_metrics["r2"]:
            best_name, best_metrics = name, metrics

    final = CANDIDATE_MODELS[best_name]()
    final.fit(X, y)
    return best_name, best_metrics, final


def load_base_df(dataset_key: str, config: DatasetConfig):
    csv_path = v2_training_csv(dataset_key)
    raw = pd.read_csv(csv_path, parse_dates=["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")

    feature_vars = config.feature_variables or config.variables
    base_df = raw[feature_vars].copy()

    for column, factor in (config.scale_factors or {}).items():
        if column in base_df.columns:
            base_df[column] = base_df[column] / factor

    derived_cols = add_derived_physics_features(base_df)
    feature_vars = feature_vars + derived_cols
    return base_df, feature_vars


def train_dataset(dataset_key: str) -> list[dict]:
    config = DATASETS[dataset_key]
    base_df, feature_vars = load_base_df(dataset_key, config)
    frame, feature_columns = build_feature_frame(base_df, feature_vars)

    results = []
    for variable in config.variables:
        if dataset_key in ("analytics", "experimental") and variable == "kp":
            continue

        for horizon in HORIZONS:
            target = base_df[variable].shift(-horizon)
            data = frame.copy()
            data["__target__"] = target
            data = data.dropna(subset=feature_columns + ["__target__"])

            X = data[feature_columns]
            y = data["__target__"]

            best_name, best_metrics, final_model = fit_best(X, y)

            path = v2_model_path(dataset_key, variable, horizon)
            joblib.dump(final_model, path)

            record = {
                "variable":        variable,
                "horizon":         horizon,
                "algorithm":       best_name,
                "r2":              best_metrics["r2"],
                "mae":             best_metrics["mae"],
                "rmse":            best_metrics["rmse"],
                "bias":            best_metrics["bias"],
                "n_train":         best_metrics["n_train"],
                "n_test":          best_metrics["n_test"],
                "n_samples":       int(len(X)),
                "feature_columns": feature_columns,
                "model_path":      str(path),
                "trained_at":      datetime.now(timezone.utc).isoformat(),
                "training_csv":    v2_training_csv(dataset_key),
            }
            results.append(record)
            print(
                f"[{dataset_key}] {variable} +{horizon}h → {best_name} "
                f"R²={best_metrics['r2']:.4f} MAE={best_metrics['mae']:.3f} "
                f"RMSE={best_metrics['rmse']:.3f} Bias={best_metrics['bias']:+.3f}"
            )

    return results


def train_kp_interval(dataset_key: str) -> dict:
    config = DATASETS[dataset_key]
    base_df, feature_vars = load_base_df(dataset_key, config)
    frame, feature_columns = build_feature_frame(base_df, feature_vars)

    block_start = base_df.index.floor("3h")
    block_kp = base_df["kp"].groupby(block_start).first()
    next_block = pd.Series(block_start + pd.Timedelta(hours=3), index=base_df.index)
    target = next_block.map(block_kp)

    data = frame.copy()
    data["__target__"] = target
    data = data.dropna(subset=feature_columns + ["__target__"])

    X = data[feature_columns]
    y = data["__target__"]

    best_name, best_metrics, final_model = fit_best(X, y)

    path = v2_kp_interval_path(dataset_key)
    joblib.dump(final_model, path)

    record = {
        "variable":        "kp",
        "horizon":         "interval",
        "algorithm":       best_name,
        "r2":              best_metrics["r2"],
        "mae":             best_metrics["mae"],
        "rmse":            best_metrics["rmse"],
        "bias":            best_metrics["bias"],
        "n_train":         best_metrics["n_train"],
        "n_test":          best_metrics["n_test"],
        "n_samples":       int(len(X)),
        "feature_columns": feature_columns,
        "model_path":      str(path),
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "training_csv":    v2_training_csv(dataset_key),
    }
    print(
        f"[{dataset_key}] kp +interval → {best_name} "
        f"R²={best_metrics['r2']:.4f} MAE={best_metrics['mae']:.3f} "
        f"RMSE={best_metrics['rmse']:.3f} Bias={best_metrics['bias']:+.3f}"
    )
    return record


def save_metrics(dataset_key: str, records: list[dict], kp_interval: dict | None = None) -> None:
    doc = {f"{r['variable']}_{r['horizon']}h": r for r in records}
    if kp_interval:
        doc["kp_interval"] = kp_interval
    path = v2_metrics_path(dataset_key)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Metrics saved → {path}")


def check_csv_exists(dataset_key: str) -> bool:
    p = Path(v2_training_csv(dataset_key))
    if not p.exists():
        print(f"ERROR: {p} not found. Run 02_build_v2_datasets.py first.")
        return False
    return True


def main() -> None:
    all_datasets = list(DATASETS.keys())
    requested = sys.argv[1:] or all_datasets

    for key in requested:
        if not check_csv_exists(key):
            sys.exit(1)

    all_results = []
    for key in requested:
        print(f"\n{'='*60}")
        print(f"Training dataset: {key}")
        print(f"{'='*60}")
        records = train_dataset(key)
        kp_interval = None
        if key in ("analytics", "experimental"):
            kp_interval = train_kp_interval(key)
        save_metrics(key, records, kp_interval)
        all_results.extend(records)

    print("\n\n=== v2 Training Summary ===")
    summary = pd.DataFrame(all_results)[["variable", "horizon", "algorithm", "r2", "mae", "rmse", "bias", "n_train", "n_test"]]
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print("\nStep 3 complete. Run 04_benchmark.py next.")


if __name__ == "__main__":
    main()
