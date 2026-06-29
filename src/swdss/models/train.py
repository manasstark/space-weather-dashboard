"""Train per-variable, per-horizon forecasting models for Solar Wind and IMF.

Usage (from project root, with src/ on PYTHONPATH):
    PYTHONPATH=src venv/bin/python3 -m swdss.models.train
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from swdss.models.features import build_feature_frame
from swdss.models.registry import DATASETS, HORIZONS, metrics_path, model_path

CANDIDATE_MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
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

TEST_FRACTION = 0.2


def evaluate_split(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }


def train_dataset(dataset_key: str) -> list[dict]:
    config = DATASETS[dataset_key]
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")

    base_df = raw[config.variables].copy()
    frame, feature_columns = build_feature_frame(base_df, config.variables)

    results = []

    for variable in config.variables:
        for horizon in HORIZONS:
            target = base_df[variable].shift(-horizon)
            data = frame.copy()
            data["__target__"] = target
            data = data.dropna(subset=feature_columns + ["__target__"])

            X = data[feature_columns]
            y = data["__target__"]

            split_idx = int(len(X) * (1 - TEST_FRACTION))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            best_name = None
            best_metrics = None
            best_model = None

            for name, factory in CANDIDATE_MODELS.items():
                model = factory()
                metrics = evaluate_split(model, X_train, y_train, X_test, y_test)
                if best_metrics is None or metrics["r2"] > best_metrics["r2"]:
                    best_name, best_metrics, best_model = name, metrics, model

            # Refit the winning algorithm on the full dataset before saving.
            final_model = CANDIDATE_MODELS[best_name]()
            final_model.fit(X, y)

            path = model_path(dataset_key, variable, horizon)
            joblib.dump(final_model, path)

            record = {
                "variable": variable,
                "horizon": horizon,
                "algorithm": best_name,
                "r2": best_metrics["r2"],
                "mae": best_metrics["mae"],
                "rmse": best_metrics["rmse"],
                "feature_columns": feature_columns,
                "model_path": str(path),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_samples": int(len(X)),
            }
            results.append(record)
            print(
                f"[{dataset_key}] {variable} +{horizon}h -> {best_name} "
                f"(R2={best_metrics['r2']:.4f} MAE={best_metrics['mae']:.3f} RMSE={best_metrics['rmse']:.3f})"
            )

    metrics_doc = {f"{r['variable']}_{r['horizon']}h": r for r in results}
    with open(metrics_path(dataset_key), "w") as f:
        json.dump(metrics_doc, f, indent=2)

    return results


def main() -> None:
    all_results = []
    for dataset_key in DATASETS:
        all_results.extend(train_dataset(dataset_key))

    print("\n=== Training Summary ===")
    summary = pd.DataFrame(all_results)[["variable", "horizon", "algorithm", "r2", "mae", "rmse"]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
