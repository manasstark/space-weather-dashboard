"""Storm Backtest — runs the EXISTING, already-frozen production models
(unchanged, not retrained) against real historical storm windows.

Why this exists: every accuracy number this engine has ever reported was
computed on live 2026 data, which has been geomagnetically quiet the whole
time. That's a real gap, not a detail — a forecasting engine's hardest job
is a storm, and this one has never been checked against one. This module
answers that directly: load the production joblib model, feed it real
historical rows from a named storm (via swdss.models.storm_data), and
score what it actually produces against what actually happened.

What this deliberately does NOT do: retrain anything. If the production
model does badly here, the right response is to look at *why* (see
storm_learning.py for the "would training on storms fix it" experiment) —
not to quietly patch this module until it looks better.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from swdss.engine.outlook import classify_activity_regime
from swdss.models.features import build_feature_frame
from swdss.models.registry import kp_interval_model_path, metrics_path, model_path
from swdss.models.storm_data import (
    NAMED_STORMS,
    build_base_df,
    build_context_frame,
    build_persistence_series,
    build_target_series,
    load_storm_window,
)
from swdss.paths import DATA_DIR

RUNS_REGISTRY_PATH = DATA_DIR / "predictions" / "storm_backtest_runs.json"

# Kp on "analytics" uses the interval-cadence model (predict_kp_interval)
# rather than the standard {variable}_{horizon}h.joblib convention every
# other variable here follows — run_storm_backtest special-cases it below
# (forcing horizon="interval" regardless of what's passed) rather than
# requiring a second entry point, so every existing caller (the UI, Storm
# Learning's production-comparison arm) keeps working unchanged.
BACKTESTABLE = {
    "solar_wind": ["speed", "density", "temperature"],
    "imf": ["bt", "bx_gsm", "by_gsm", "bz_gsm"],
    "analytics": ["dst", "kp"],
    "ae": ["ae"],
}


def _load_metrics(dataset_key: str) -> dict:
    path = metrics_path(dataset_key)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def run_storm_backtest(dataset_key: str, variable: str, horizon, storm_key: str) -> dict:
    """Scores the frozen production model for (dataset_key, variable, horizon)
    against a named historical storm. Returns a dict with MAE/RMSE against
    both the actual observations and the persistence baseline, a
    Quiet/Active/Storm regime breakdown (swdss.engine.outlook), and the
    full timestamped series for plotting.

    Kp is a special case: its production model always targets NOAA's next
    official 3-hour interval, never a fixed hourly horizon, so `horizon` is
    forced to "interval" here regardless of what's passed — the same
    leniency predict_kp_interval itself doesn't need to worry about since
    it's never called with a horizon at all.
    """
    if variable == "kp":
        horizon = "interval"

    storm = NAMED_STORMS[storm_key]
    metrics_doc = _load_metrics(dataset_key)
    key = "kp_interval" if variable == "kp" else f"{variable}_{horizon}h"
    if key not in metrics_doc:
        raise ValueError(f"No trained production model for {dataset_key}/{key}.")
    meta = metrics_doc[key]
    feature_columns = meta["feature_columns"]

    omni_df = load_storm_window(storm_key, lookback_hours=48)
    base_df, feature_vars = build_base_df(dataset_key, omni_df)
    frame, _ = build_feature_frame(base_df, feature_vars)

    target = build_target_series(variable, horizon, base_df)
    data = frame.copy()
    data["__target__"] = target
    data = data.dropna(subset=feature_columns + ["__target__"])

    win_start = pd.Timestamp(storm["window_start"])
    win_end = pd.Timestamp(storm["window_end"]) + pd.Timedelta(hours=24)
    data = data[(data.index >= win_start) & (data.index <= win_end)]
    if data.empty:
        raise ValueError(
            f"No usable rows for {dataset_key}/{variable} in the {storm['label']} window "
            "after feature construction — the storm window may be too short for the model's "
            "24h lag/rolling features to fill in."
        )

    model_file = kp_interval_model_path(dataset_key) if variable == "kp" else model_path(dataset_key, variable, horizon)
    model = joblib.load(model_file)
    X = data[feature_columns]
    y_true = data["__target__"].to_numpy()
    y_pred = model.predict(X)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    model_mse = float(np.mean(errors**2))

    persistence_pred = build_persistence_series(variable, horizon, base_df).reindex(data.index).to_numpy()
    persistence_errors = y_true - persistence_pred
    persistence_mse = float(np.mean(persistence_errors**2))
    skill_score = 1.0 - (model_mse / persistence_mse) if persistence_mse > 0 else None

    production_mae = meta.get("mae")
    within_production_band = (
        float((abs_errors <= 1.5 * production_mae).mean()) if production_mae else None
    )

    context = build_context_frame(omni_df).reindex(data.index)
    regime_tags = [
        classify_activity_regime(predicted_kp=row.kp, predicted_dst=row.dst, predicted_ae=row.ae)
        for row in context.itertuples()
    ]
    regime_series = pd.Series(regime_tags, index=data.index)
    per_regime = {}
    for regime in ("Quiet", "Active", "Storm"):
        mask = (regime_series == regime).to_numpy()
        if mask.any():
            per_regime[regime] = {
                "n": int(mask.sum()),
                "mae": float(abs_errors[mask].mean()),
            }

    result = {
        "dataset": dataset_key,
        "variable": variable,
        "horizon": horizon,
        "storm_key": storm_key,
        "storm_label": storm["label"],
        "storm_g_scale": storm["g_scale"],
        "storm_dst_min_nT": storm["dst_min_nT"],
        "in_training_range": storm["in_training_range"],
        "n_points": int(len(data)),
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(model_mse)),
        "production_mae": production_mae,
        "within_production_mae_band_rate": within_production_band,
        "skill_score": skill_score,
        "persistence_mae": float(np.mean(np.abs(persistence_errors))),
        "per_regime": per_regime,
        "model_algorithm": meta.get("algorithm"),
        "timestamps": [ts.isoformat() for ts in data.index],
        "actual": [float(v) for v in y_true],
        "predicted": [float(v) for v in y_pred],
        "persistence": [float(v) for v in persistence_pred],
    }
    return result


# ==================== Run tracking (data/predictions/storm_backtest_runs.json) ====================


def _load_runs() -> list[dict]:
    if not RUNS_REGISTRY_PATH.exists():
        return []
    with open(RUNS_REGISTRY_PATH) as f:
        return json.load(f)


def _save_runs(runs: list[dict]) -> None:
    RUNS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_REGISTRY_PATH, "w") as f:
        json.dump(runs, f, indent=2)


def record_backtest_run(result: dict) -> dict:
    run = {
        "run_id": f"backtest-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "run_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in result.items() if k not in ("timestamps", "actual", "predicted", "persistence")},
    }
    runs = _load_runs()
    runs.append(run)
    _save_runs(runs)
    return run


def list_backtest_runs() -> list[dict]:
    return sorted(_load_runs(), key=lambda r: r["run_at"], reverse=True)
