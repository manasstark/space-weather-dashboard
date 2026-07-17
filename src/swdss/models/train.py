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

from swdss.models.features import add_derived_physics_features, build_feature_frame
from swdss.models.registry import DATASETS, HORIZONS, kp_interval_model_path, metrics_path, model_path
from swdss.models.validation import evaluate_walk_forward

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
CV_FOLDS = 5


def evaluate_split(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }


def _fit_best(X, y) -> tuple:
    """Benchmarks every candidate algorithm two ways and refits the winner
    on the full dataset:

    1. The original single chronological 80/20 holdout split (`evaluate_split`)
       — kept exactly as before so `metrics["r2"/"mae"/"rmse"]` and every
       promotion/comparison codepath that already reads those keys sees
       identical values to before this change.
    2. Walk-forward (rolling-origin) cross-validation across CV_FOLDS
       consecutive folds (`swdss.models.validation.evaluate_walk_forward`)
       — a genuine stability estimate a single split can't provide, since
       the events that matter most for this domain (geomagnetic storms)
       are rare enough that a single 20% holdout window may or may not
       contain one.

    The algorithm is now SELECTED by mean walk-forward CV R² rather than
    the single-split R² — a more robust criterion, since a candidate that
    merely got lucky on one holdout window won't out-rank one that's
    consistently good across several. Returns
    (name, holdout_metrics, cv_metrics, model, all_candidates) where
    all_candidates maps every candidate name to its own holdout/cv
    metrics, for transparency about what was and wasn't selected.
    """
    split_idx = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    all_candidates = {}
    for name, factory in CANDIDATE_MODELS.items():
        holdout_metrics = evaluate_split(factory(), X_train, y_train, X_test, y_test)
        cv_metrics = evaluate_walk_forward(factory, X, y, n_folds=CV_FOLDS)
        all_candidates[name] = {"holdout": holdout_metrics, "cv": cv_metrics}

    best_name = max(all_candidates, key=lambda n: all_candidates[n]["cv"]["r2_mean"])
    best_metrics = all_candidates[best_name]["holdout"]
    best_cv = all_candidates[best_name]["cv"]

    final_model = CANDIDATE_MODELS[best_name]()
    final_model.fit(X, y)
    return best_name, best_metrics, best_cv, final_model, all_candidates


def _load_base_df(config):
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
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
    base_df, feature_vars = _load_base_df(config)
    frame, feature_columns = build_feature_frame(base_df, feature_vars)

    results = []

    for variable in config.variables:
        # Kp on both the production Analytics page and the experimental
        # cascade follows NOAA's real 3-hour cadence instead of an
        # arbitrary hourly horizon — trained separately by
        # train_kp_interval_model(), not as 5 fixed-horizon models here.
        if dataset_key in ("analytics", "experimental") and variable == "kp":
            continue

        for horizon in HORIZONS:
            target = base_df[variable].shift(-horizon)
            data = frame.copy()
            data["__target__"] = target
            data = data.dropna(subset=feature_columns + ["__target__"])

            X = data[feature_columns]
            y = data["__target__"]

            best_name, best_metrics, best_cv, final_model, all_candidates = _fit_best(X, y)

            path = model_path(dataset_key, variable, horizon)
            joblib.dump(final_model, path)

            record = {
                "variable": variable,
                "horizon": horizon,
                "algorithm": best_name,
                "r2": best_metrics["r2"],
                "mae": best_metrics["mae"],
                "rmse": best_metrics["rmse"],
                "cv_r2_mean": best_cv["r2_mean"],
                "cv_r2_std": best_cv["r2_std"],
                "cv_mae_mean": best_cv["mae_mean"],
                "cv_mae_std": best_cv["mae_std"],
                "cv_rmse_mean": best_cv["rmse_mean"],
                "cv_rmse_std": best_cv["rmse_std"],
                "cv_n_folds": best_cv["n_folds"],
                "algorithm_candidates_cv_r2": {
                    name: round(res["cv"]["r2_mean"], 4) for name, res in all_candidates.items()
                },
                "feature_columns": feature_columns,
                "model_path": str(path),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_samples": int(len(X)),
            }
            results.append(record)
            print(
                f"[{dataset_key}] {variable} +{horizon}h -> {best_name} "
                f"(R2={best_metrics['r2']:.4f} MAE={best_metrics['mae']:.3f} RMSE={best_metrics['rmse']:.3f} | "
                f"CV R2={best_cv['r2_mean']:.4f}+/-{best_cv['r2_std']:.4f})"
            )

    metrics_doc_path = metrics_path(dataset_key)
    if dataset_key in ("analytics", "experimental") and metrics_doc_path.exists():
        # Don't clobber the kp_interval entry if it was trained in a
        # previous run and this run doesn't (re)train it.
        with open(metrics_doc_path) as f:
            existing = json.load(f)
        kept = {k: v for k, v in existing.items() if k == "kp_interval"}
    else:
        kept = {}

    metrics_doc = {**kept, **{f"{r['variable']}_{r['horizon']}h": r for r in results}}
    with open(metrics_doc_path, "w") as f:
        json.dump(metrics_doc, f, indent=2)

    return results


def train_kp_interval_model(dataset: str = "analytics") -> dict:
    """Kp is only ever officially published every 3 hours (00, 03, 06, ...
    UTC). Instead of an arbitrary hourly horizon, this trains a single
    model whose target is always "the Kp value of the next official
    interval" — e.g. observing at 16:42 (inside the 15-18 UTC interval)
    targets the 18-21 UTC interval's eventual Kp, same as observing at
    17:55 in that same interval. The lead time naturally varies 1-3 hours
    row by row, which the model learns directly from the training data.

    Used for both the production "analytics" dataset and the experimental
    cascade ("experimental") — same cadence-aware target logic, only the
    feature set differs (observed AE vs. predicted_ae).
    """
    config = DATASETS[dataset]
    base_df, feature_vars = _load_base_df(config)
    frame, feature_columns = build_feature_frame(base_df, feature_vars)

    block_start = base_df.index.floor("3h")
    block_kp = base_df["kp"].groupby(block_start).first()
    next_block_start = pd.Series(block_start + pd.Timedelta(hours=3), index=base_df.index)
    target = next_block_start.map(block_kp)

    data = frame.copy()
    data["__target__"] = target
    data = data.dropna(subset=feature_columns + ["__target__"])

    X = data[feature_columns]
    y = data["__target__"]

    best_name, best_metrics, best_cv, final_model, all_candidates = _fit_best(X, y)

    path = kp_interval_model_path(dataset)
    joblib.dump(final_model, path)

    record = {
        "variable": "kp",
        "horizon": "interval",
        "algorithm": best_name,
        "r2": best_metrics["r2"],
        "mae": best_metrics["mae"],
        "rmse": best_metrics["rmse"],
        "cv_r2_mean": best_cv["r2_mean"],
        "cv_r2_std": best_cv["r2_std"],
        "cv_mae_mean": best_cv["mae_mean"],
        "cv_mae_std": best_cv["mae_std"],
        "cv_rmse_mean": best_cv["rmse_mean"],
        "cv_rmse_std": best_cv["rmse_std"],
        "cv_n_folds": best_cv["n_folds"],
        "algorithm_candidates_cv_r2": {
            name: round(res["cv"]["r2_mean"], 4) for name, res in all_candidates.items()
        },
        "feature_columns": feature_columns,
        "model_path": str(path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
    }

    metrics_doc_path = metrics_path(dataset)
    metrics_doc = {}
    if metrics_doc_path.exists():
        with open(metrics_doc_path) as f:
            metrics_doc = json.load(f)
    metrics_doc["kp_interval"] = record
    with open(metrics_doc_path, "w") as f:
        json.dump(metrics_doc, f, indent=2)

    print(
        f"[{dataset}] kp +next-interval -> {best_name} "
        f"(R2={best_metrics['r2']:.4f} MAE={best_metrics['mae']:.3f} RMSE={best_metrics['rmse']:.3f} | "
        f"CV R2={best_cv['r2_mean']:.4f}+/-{best_cv['r2_std']:.4f})"
    )
    return record


def main() -> None:
    import sys

    requested = sys.argv[1:] or list(DATASETS)

    all_results = []
    for dataset_key in requested:
        all_results.extend(train_dataset(dataset_key))

    for dataset_key in ("analytics", "experimental"):
        if dataset_key in requested:
            all_results.append(train_kp_interval_model(dataset_key))

    print("\n=== Training Summary ===")
    summary = pd.DataFrame(all_results)[["variable", "horizon", "algorithm", "r2", "mae", "rmse"]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
