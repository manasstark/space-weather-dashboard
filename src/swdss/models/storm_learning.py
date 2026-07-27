"""Storm Learning — "can a model actually learn storm behavior if it's
shown some?" A deliberately separate question from storm_backtest.py's
"does the existing production model already generalize to a storm it's
never seen?" Conflating the two was a real design mistake caught before
building this: training a brand-new model on a handful of pre-storm hours
and testing it on the same storm proves nothing (it would fail regardless
of model quality, since it's tested outside everything it was shown).

This module instead trains a NEW model (never touching the real
production artifacts) on the existing multi-year production training
corpus PLUS several real historical storms, with one storm held out
entirely for testing — never seen during training, so there's no leakage.
It then scores that new model on the held-out storm and directly compares
it against (a) the persistence baseline, and (b) the actual frozen
production model's performance on that same storm, by reusing
storm_backtest.run_storm_backtest for the production side of that
comparison rather than re-deriving it.

This trains real models (RandomForest/XGBoost among the candidates) —
it will take real seconds-to-a-minute or so, not the multi-hour runtime of
the full Optimization Studies, since there's no walk-forward CV here (a
deliberate scope cut for an interactive research tool, not an oversight —
see CANDIDATE_MODELS below).

`sample_weight_multiplier` (2026-07 addition): the first version of this
tool concatenated a few hundred storm-hour rows onto a ~27,000-row mostly-
quiet corpus and found it barely moved the fitted model — unsurprising,
since a handful of storm rows are numerically drowned out in an ordinary
loss function regardless of how physically important they are. This tests
a genuinely different hypothesis: instead of adding more storm rows, make
the ones already there count more. Every row that came from a named
training storm's own window is weighted `sample_weight_multiplier` times
higher than an ordinary quiet-corpus row when fitting (default 1.0 —
identical to the original unweighted behavior, since a uniform weight
vector changes nothing for any of these algorithms).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from swdss.models.features import add_derived_physics_features, build_feature_frame
from swdss.models.registry import DATASETS
from swdss.models.storm_backtest import run_storm_backtest
from swdss.models.storm_data import NAMED_STORMS, build_base_df, build_persistence_series, build_target_series, load_storm_window
from swdss.models.train import CANDIDATE_MODELS
from swdss.paths import DATA_DIR

RUNS_REGISTRY_PATH = DATA_DIR / "predictions" / "storm_learning_runs.json"

LEARNABLE = {
    "solar_wind": ["speed", "density", "temperature"],
    "imf": ["bt", "bx_gsm", "by_gsm", "bz_gsm"],
    "analytics": ["dst", "kp"],
    "ae": ["ae"],
}


def _load_quiet_base(dataset_key: str) -> pd.DataFrame:
    """The existing multi-year production training CSV — mostly quiet-time
    data, same source train.py itself reads."""
    config = DATASETS[dataset_key]
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")

    feature_vars = config.feature_variables or config.variables
    base_df = raw[[c for c in feature_vars if c in raw.columns]].copy()
    for column, factor in (config.scale_factors or {}).items():
        if column in base_df.columns:
            base_df[column] = base_df[column] / factor
    return base_df


def run_storm_learning_experiment(
    dataset_key: str,
    variable: str,
    horizon: int,
    held_out_storm: str,
    training_storms: list[str],
    sample_weight_multiplier: float = 1.0,
) -> dict:
    """Trains a brand-new model on quiet-time history + training_storms,
    evaluates it on held_out_storm (never seen), and compares against the
    real production model's own performance on that same storm.

    sample_weight_multiplier > 1.0 makes rows drawn from a training storm's
    own window count that many times more than an ordinary quiet-corpus row
    in the fit — a different lever than adding more storm rows (see module
    docstring). 1.0 (default) reproduces the original unweighted behavior.

    Kp is a special case: its production model always targets NOAA's next
    official 3-hour interval, never a fixed hourly horizon — `horizon` is
    forced to "interval" here regardless of what's passed, same as
    storm_backtest.run_storm_backtest.
    """
    if variable == "kp":
        horizon = "interval"

    if held_out_storm in training_storms:
        raise ValueError("held_out_storm must not also appear in training_storms — that would leak the test event into training.")

    quiet_base = _load_quiet_base(dataset_key)

    storm_frames = []
    storm_dates = set()
    for storm_key in training_storms:
        omni_df = load_storm_window(storm_key, lookback_hours=48)
        storm_base, _ = build_base_df(dataset_key, omni_df)
        storm_frames.append(storm_base)
        # Recorded BEFORE dedup, independent of which row's values survive
        # it — a storm date that also happens to already sit in the quiet
        # corpus (e.g. an in-training-range storm) must still be weighted
        # as a storm row, not silently fall back to weight 1.0 just because
        # the quiet-corpus copy of that timestamp won the dedup below.
        storm_dates.update(storm_base.index)

    combined = pd.concat([quiet_base] + storm_frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    feature_vars = DATASETS[dataset_key].feature_variables or DATASETS[dataset_key].variables
    derived_cols = add_derived_physics_features(combined)
    feature_vars_all = feature_vars + derived_cols

    frame, feature_columns = build_feature_frame(combined, feature_vars_all)
    target = build_target_series(variable, horizon, combined)
    data = frame.copy()
    data["__target__"] = target
    data = data.dropna(subset=feature_columns + ["__target__"])

    # Exclude the held-out storm's own window (with the same lookback
    # buffer) from training so its evaluation is a genuine blind test.
    held_out = NAMED_STORMS[held_out_storm]
    ho_start = pd.Timestamp(held_out["window_start"]) - pd.Timedelta(hours=48)
    ho_end = pd.Timestamp(held_out["window_end"]) + pd.Timedelta(hours=24)
    train_data = data[~((data.index >= ho_start) & (data.index <= ho_end))]

    X_train, y_train = train_data[feature_columns], train_data["__target__"]

    # Storm-window rows count sample_weight_multiplier times more than an
    # ordinary quiet-corpus row — 1.0 (uniform) is mathematically identical
    # to no weighting at all for every candidate algorithm below.
    weight_train = pd.Series(1.0, index=train_data.index)
    weight_train[train_data.index.isin(storm_dates)] = sample_weight_multiplier

    split_idx = int(len(X_train) * 0.8)
    X_tr, X_te = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_te = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    weight_tr = weight_train.iloc[:split_idx]

    candidates = {}
    for name, factory in CANDIDATE_MODELS.items():
        model = factory()
        model.fit(X_tr, y_tr, sample_weight=weight_tr)
        preds = model.predict(X_te)
        candidates[name] = {
            "r2": float(r2_score(y_te, preds)),
            "mae": float(mean_absolute_error(y_te, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, preds))),
        }
    best_name = max(candidates, key=lambda n: candidates[n]["r2"])
    final_model = CANDIDATE_MODELS[best_name]()
    final_model.fit(X_train, y_train, sample_weight=weight_train)

    # Evaluate the new model on the held-out storm — rebuild its features
    # from a fresh OMNI2 pull, exactly like storm_backtest.py does for the
    # production model, so both arms of the comparison see identical rows.
    ho_omni = load_storm_window(held_out_storm, lookback_hours=48)
    ho_base, _ = build_base_df(dataset_key, ho_omni)
    ho_frame, _ = build_feature_frame(ho_base, feature_vars_all)
    ho_target = build_target_series(variable, horizon, ho_base)
    ho_data = ho_frame.copy()
    ho_data["__target__"] = ho_target
    ho_data = ho_data.dropna(subset=feature_columns + ["__target__"])

    win_start = pd.Timestamp(held_out["window_start"])
    win_end = pd.Timestamp(held_out["window_end"]) + pd.Timedelta(hours=24)
    ho_eval = ho_data[(ho_data.index >= win_start) & (ho_data.index <= win_end)]
    if ho_eval.empty:
        raise ValueError(f"No usable rows for the held-out storm ({held_out['label']}) after feature construction.")

    X_ho = ho_eval[feature_columns]
    y_ho = ho_eval["__target__"].to_numpy()
    y_pred_new = final_model.predict(X_ho)
    new_errors = y_ho - y_pred_new
    new_mae = float(np.mean(np.abs(new_errors)))
    new_rmse = float(np.sqrt(np.mean(new_errors**2)))

    persistence_pred = build_persistence_series(variable, horizon, ho_base).reindex(ho_eval.index).to_numpy()
    persistence_mae = float(np.mean(np.abs(y_ho - persistence_pred)))

    # Direct comparison arm: the REAL, already-deployed production model,
    # scored on this exact same held-out storm.
    production_result = run_storm_backtest(dataset_key, variable, horizon, held_out_storm)

    if abs(new_mae - production_result["mae"]) < 0.05 * production_result["mae"]:
        verdict = "No meaningful difference from the current production model on this storm."
    elif new_mae < production_result["mae"]:
        verdict = "Storm-inclusive training improved storm-period accuracy over the current production model."
    else:
        verdict = "Storm-inclusive training did NOT beat the current production model on this storm — more data alone isn't the fix here."

    return {
        "dataset": dataset_key,
        "variable": variable,
        "horizon": horizon,
        "held_out_storm": held_out_storm,
        "held_out_storm_label": held_out["label"],
        "training_storms": training_storms,
        "sample_weight_multiplier": sample_weight_multiplier,
        "new_model_algorithm": best_name,
        "new_model_candidates": {name: res["r2"] for name, res in candidates.items()},
        "n_train_samples": int(len(X_train)),
        "n_eval_samples": int(len(X_ho)),
        "new_model_mae": new_mae,
        "new_model_rmse": new_rmse,
        "production_model_mae": production_result["mae"],
        "production_model_rmse": production_result["rmse"],
        "persistence_mae": persistence_mae,
        "verdict": verdict,
        "timestamps": [ts.isoformat() for ts in ho_eval.index],
        "actual": [float(v) for v in y_ho],
        "new_model_predicted": [float(v) for v in y_pred_new],
        "production_predicted": production_result["predicted"],
    }


# ==================== Run tracking (data/predictions/storm_learning_runs.json) ====================


def _load_runs() -> list[dict]:
    if not RUNS_REGISTRY_PATH.exists():
        return []
    with open(RUNS_REGISTRY_PATH) as f:
        return json.load(f)


def _save_runs(runs: list[dict]) -> None:
    RUNS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_REGISTRY_PATH, "w") as f:
        json.dump(runs, f, indent=2)


def record_learning_run(result: dict) -> dict:
    run = {
        "run_id": f"learning-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "run_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in result.items() if k not in ("timestamps", "actual", "new_model_predicted", "production_predicted")},
    }
    runs = _load_runs()
    runs.append(run)
    _save_runs(runs)
    return run


def list_learning_runs() -> list[dict]:
    return sorted(_load_runs(), key=lambda r: r["run_at"], reverse=True)
