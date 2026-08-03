"""Physics-ML hybrid Dst backtest — tests whether Burton's ring-current
injection/decay physics (swdss.physics.burton), on its own or combined
with a residual ML correction, beats the current production model
specifically during the one confirmed failure mode: storm-regime
extrapolation (see storm_backtest.py results — production's straight-line
Dst model runs 3-9x worse during storm-regime hours than quiet hours).

Three arms are compared on each held-out storm, all scored against the
exact same actual Dst values over the exact same evaluation window
storm_backtest.py already uses (window_start through window_end+24h):

  1. Pure Burton  — physics only, no ML at all.
  2. Hybrid       — Burton's physics prediction plus a residual ML model
                     trained to predict (actual Dst - Burton's Dst) on the
                     ordinary multi-year quiet-time training corpus.
  3. Production   — the real, already-deployed frozen model, via
                     storm_backtest.run_storm_backtest (not re-derived).

Burton's own free parameters (injection coefficient, decay time) are
calibrated on the training corpus with the held-out storm's window
excised first — the same leakage-avoidance discipline storm_learning.py
uses for its own held-out-storm training split. A storm outside the
2023-2026 corpus entirely (e.g. September 2017) needs no exclusion since
it was never in the corpus to begin with; the exclusion mask is a no-op
in that case, so one code path handles both.

The residual model reuses the same CANDIDATE_MODELS/feature-engineering
machinery as every other model in this project (no walk-forward CV here,
same deliberate scope cut storm_learning.py documents — this is an
interactive research tool, not a multi-hour Optimization Study).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from swdss.engine.outlook import classify_activity_regime
from swdss.models.features import build_feature_frame
from swdss.models.registry import DATASETS
from swdss.models.storm_backtest import run_storm_backtest
from swdss.models.storm_data import (
    NAMED_STORMS,
    build_context_frame,
    build_persistence_series,
    build_target_series,
    load_storm_window,
)
from swdss.models.train import CANDIDATE_MODELS
from swdss.paths import DATA_DIR
from swdss.physics.burton import (
    burton_one_step_forecast_dst,
    burton_variable_tau_one_step_forecast_dst,
    calibrate_burton_params,
    calibrate_burton_variable_tau_a,
    southward_efield_series,
    variable_tau_hours,
)
from swdss.physics.core import add_derived_physics_features

RUNS_REGISTRY_PATH = DATA_DIR / "predictions" / "storm_burton_runs.json"

# Production's Dst model is compared at its 1h horizon — the horizon that
# matches Burton's own natural 1-hour integration step, for a fair,
# apples-to-apples comparison rather than mixing step sizes.
PRODUCTION_HORIZON_HOURS = 1

MEANINGFUL_IMPROVEMENT = 0.05  # same relative-improvement bar storm_learning.py and the ensemble/retrain gates use

TAU_MODES = ("constant", "variable")


def _calibrate_and_predict_burton(train_corpus: pd.DataFrame, storm_raw: pd.DataFrame, tau_mode: str, horizon_hours: int = 1):
    """Isolates the ONE thing that differs between the constant-tau and
    variable-tau experiments — everything else (residual ML training,
    production comparison, per-regime breakdown, verdicts) stays identical
    so any difference in results is attributable to tau's treatment alone,
    not conflated with some other pipeline change.

    `horizon_hours` (default 1, matching the original single-hour design)
    is passed through as dt_hours to both the calibration fit and the
    one-step forecast — a 6/12/24h step is a coarser finite-difference
    approximation of the same continuous-time ODE, calibrated and
    evaluated consistently at that same step size rather than mixing a
    1h-fit tau against a longer forecast step.
    """
    if tau_mode == "constant":
        calib = calibrate_burton_params(train_corpus, dt_hours=horizon_hours)
        a, tau_hours = calib["a"], calib["tau_hours"]
        if tau_hours is None or tau_hours <= 0:
            raise ValueError(
                "Burton calibration produced a non-physical decay time (tau <= 0) on this training corpus — "
                "the fit could not find a stable, mean-reverting ring-current decay to integrate against."
            )
        burton_pred_train = burton_one_step_forecast_dst(train_corpus, a, tau_hours, dt_hours=horizon_hours)
        burton_pred_full = burton_one_step_forecast_dst(storm_raw, a, tau_hours, dt_hours=horizon_hours)
        calib_fields = {
            "tau_mode": "constant",
            "burton_a": a,
            "burton_tau_hours": tau_hours,
            "burton_tau_range_hours_during_storm": None,
            "burton_calibration_r2": calib["r2"],
            "burton_calibration_n_samples": calib["n_samples"],
        }
    elif tau_mode == "variable":
        calib = calibrate_burton_variable_tau_a(train_corpus, dt_hours=horizon_hours)
        a = calib["a"]
        burton_pred_train = burton_variable_tau_one_step_forecast_dst(train_corpus, a, dt_hours=horizon_hours)
        burton_pred_full = burton_variable_tau_one_step_forecast_dst(storm_raw, a, dt_hours=horizon_hours)
        vbs_storm = southward_efield_series(storm_raw["speed"], storm_raw["bz_gsm"])
        tau_storm = variable_tau_hours(vbs_storm)
        calib_fields = {
            "tau_mode": "variable",
            "burton_a": a,
            "burton_tau_hours": None,  # not a single value — see burton_tau_range_hours_during_storm
            "burton_tau_range_hours_during_storm": [float(tau_storm.min()), float(tau_storm.max())],
            "burton_calibration_r2": calib["r2"],
            "burton_calibration_n_samples": calib["n_samples"],
        }
    else:
        raise ValueError(f"tau_mode must be one of {TAU_MODES}, got {tau_mode!r}")

    return burton_pred_train, burton_pred_full, calib_fields


def _load_quiet_corpus() -> pd.DataFrame:
    config = DATASETS["analytics"]
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    return raw.sort_values("datetime").set_index("datetime")


def _exclude_storm_window(df: pd.DataFrame, storm_key: str, lookback_hours: int = 48) -> pd.DataFrame:
    storm = NAMED_STORMS[storm_key]
    start = pd.Timestamp(storm["window_start"]) - pd.Timedelta(hours=lookback_hours)
    end = pd.Timestamp(storm["window_end"]) + pd.Timedelta(hours=24)
    return df[~((df.index >= start) & (df.index <= end))]


def _per_regime_mae(abs_errors: np.ndarray, context: pd.DataFrame) -> dict:
    regime_tags = [
        classify_activity_regime(predicted_kp=row.kp, predicted_dst=row.dst, predicted_ae=row.ae)
        for row in context.itertuples()
    ]
    regime_series = pd.Series(regime_tags, index=context.index).to_numpy()
    per_regime = {}
    for regime in ("Quiet", "Active", "Storm"):
        mask = regime_series == regime
        if mask.any():
            per_regime[regime] = {"n": int(mask.sum()), "mae": float(abs_errors[mask].mean())}
    return per_regime


def run_burton_hybrid_backtest(held_out_storm: str, tau_mode: str = "constant", horizon_hours: int = PRODUCTION_HORIZON_HOURS) -> dict:
    """`horizon_hours` defaults to the original 1h design (matching
    production's dst_1h model); passing 3/6/12/24 tests the same
    physics-hybrid approach against production's own longer-horizon Dst
    models instead, per "test the physics approach at longer horizons"
    in Development Roadmap — Burton hasn't been tried past 1h before this.
    """
    if held_out_storm not in NAMED_STORMS:
        raise ValueError(f"Unknown storm key: {held_out_storm!r}")
    storm = NAMED_STORMS[held_out_storm]

    # 1. Quiet-time training corpus, held-out storm's own window excised
    #    (a no-op if this storm predates the corpus, e.g. September 2017).
    quiet_corpus = _load_quiet_corpus()
    train_corpus = _exclude_storm_window(quiet_corpus, held_out_storm)
    train_corpus = train_corpus.dropna(subset=["speed", "density", "bz_gsm", "dst"])

    # 3. Held-out storm's own OMNI2 window (raw, unscaled) — keeps every
    #    raw column (not just the four Burton itself needs) since the
    #    residual ML model's feature set spans the full ANALYTICS_FEATURE_
    #    VARIABLES list (temperature, bt, bx_gsm, by_gsm, kp too).
    omni_df = load_storm_window(held_out_storm, lookback_hours=48)
    storm_raw = omni_df.set_index("datetime").sort_index()
    storm_raw = storm_raw.dropna(subset=["speed", "density", "bz_gsm", "dst"])

    # 4. Burton's dt_hours-ahead prediction, issued fresh from real observed
    #    state at every row (not a free-running multi-hour integration —
    #    see burton_one_step_forecast_dst's docstring for why that's the
    #    fair comparison against production's own re-anchored-every-hour
    #    forecast). Indexed at issuance time t; compare against the actual
    #    value at t+1 (build_target_series), not df["dst"] at t itself.
    #    tau_mode selects constant tau (calibrated on this corpus) or
    #    O'Brien & McPherron's driving-dependent variable tau — see
    #    _calibrate_and_predict_burton for the one place that differs.
    burton_pred_train, burton_pred_full, calib_fields = _calibrate_and_predict_burton(
        train_corpus, storm_raw, tau_mode, horizon_hours=horizon_hours
    )
    actual_target_full = build_target_series("dst", horizon_hours, storm_raw)

    win_start = pd.Timestamp(storm["window_start"])
    win_end = pd.Timestamp(storm["window_end"]) + pd.Timedelta(hours=24)
    eval_mask = (storm_raw.index >= win_start) & (storm_raw.index <= win_end)

    # 5. Residual ML target on the training corpus: actual Dst at t+1 minus
    #    Burton's own t+1-ahead prediction issued at t, using the exact same
    #    one-step physics formula (deterministic given the driving series at
    #    t, so no leakage from "fitting" it — it's a closed-form projection,
    #    not a model with weights to overfit).
    actual_target_train = build_target_series("dst", horizon_hours, train_corpus)
    residual_target_train = actual_target_train - burton_pred_train

    feature_vars = DATASETS["analytics"].feature_variables or DATASETS["analytics"].variables
    train_features = train_corpus[[c for c in feature_vars if c in train_corpus.columns]].copy()
    derived_cols = add_derived_physics_features(train_features)
    frame, feature_columns = build_feature_frame(train_features, feature_vars + derived_cols)
    residual_data = frame.copy()
    residual_data["__target__"] = residual_target_train
    residual_data = residual_data.dropna(subset=feature_columns + ["__target__"])

    X_full, y_full = residual_data[feature_columns], residual_data["__target__"]
    split_idx = int(len(X_full) * 0.8)
    X_tr, X_te = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_tr, y_te = y_full.iloc[:split_idx], y_full.iloc[split_idx:]

    candidates = {}
    for name, factory in CANDIDATE_MODELS.items():
        model = factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        candidates[name] = {"r2": float(r2_score(y_te, preds)), "mae": float(mean_absolute_error(y_te, preds))}
    best_name = max(candidates, key=lambda n: candidates[n]["r2"])

    # Refit the winning residual model on the FULL training corpus (not just
    # the 80% internal split used to pick it) before scoring on the storm —
    # same convention storm_learning.py uses.
    residual_model = CANDIDATE_MODELS[best_name]()
    residual_model.fit(X_full, y_full)

    # 6. Build the same features on the storm window, predict the residual,
    #    and add it to Burton's own physics prediction for the hybrid arm.
    storm_features = storm_raw[[c for c in feature_vars if c in storm_raw.columns]].copy()
    storm_derived = add_derived_physics_features(storm_features)
    storm_frame, _ = build_feature_frame(storm_features, feature_vars + storm_derived)
    storm_frame["__burton__"] = burton_pred_full
    storm_frame["__actual__"] = actual_target_full
    storm_eval = storm_frame.loc[eval_mask].dropna(subset=feature_columns + ["__burton__", "__actual__"])
    if storm_eval.empty:
        raise ValueError(
            f"No usable rows for {storm['label']} after feature construction — the storm window may be "
            "too short for the 24h lag/rolling features to fill in."
        )

    residual_pred = residual_model.predict(storm_eval[feature_columns])
    hybrid_pred = storm_eval["__burton__"].to_numpy() + residual_pred
    burton_only_pred = storm_eval["__burton__"].to_numpy()
    actual = storm_eval["__actual__"].to_numpy()

    hybrid_abs_errors = np.abs(actual - hybrid_pred)
    burton_abs_errors = np.abs(actual - burton_only_pred)
    hybrid_mae = float(hybrid_abs_errors.mean())
    hybrid_rmse = float(np.sqrt(np.mean((actual - hybrid_pred) ** 2)))
    burton_mae = float(burton_abs_errors.mean())
    burton_rmse = float(np.sqrt(np.mean((actual - burton_only_pred) ** 2)))

    # 7. Production comparison — the real, already-deployed frozen model,
    #    reusing storm_backtest.py rather than re-deriving that logic.
    production_result = run_storm_backtest("analytics", "dst", horizon_hours, held_out_storm)

    persistence_pred = build_persistence_series("dst", horizon_hours, storm_raw).reindex(storm_eval.index)
    persistence_mae = float(np.mean(np.abs(actual - persistence_pred.to_numpy())))

    # 8. Per-regime breakdown, same Quiet/Active/Storm buckets storm_backtest
    #    uses, so results are directly comparable to those existing numbers.
    context = build_context_frame(omni_df).reindex(storm_eval.index)
    hybrid_per_regime = _per_regime_mae(hybrid_abs_errors, context)
    burton_per_regime = _per_regime_mae(burton_abs_errors, context)

    def _verdict(candidate_mae: float, candidate_name: str) -> str:
        prod_mae = production_result["mae"]
        if abs(candidate_mae - prod_mae) < MEANINGFUL_IMPROVEMENT * prod_mae:
            return f"{candidate_name}: no meaningful difference from production on this storm."
        if candidate_mae < prod_mae:
            return f"{candidate_name}: beat production on this storm."
        return f"{candidate_name}: did NOT beat production on this storm."

    result = {
        "held_out_storm": held_out_storm,
        "horizon_hours": horizon_hours,
        "held_out_storm_label": storm["label"],
        "storm_g_scale": storm["g_scale"],
        "storm_dst_min_nT": storm["dst_min_nT"],
        "in_training_range": storm["in_training_range"],
        **calib_fields,
        "residual_model_algorithm": best_name,
        "residual_model_candidates": {name: res["r2"] for name, res in candidates.items()},
        "n_eval_samples": int(len(storm_eval)),
        "burton_only_mae": burton_mae,
        "burton_only_rmse": burton_rmse,
        "burton_only_per_regime": burton_per_regime,
        "hybrid_mae": hybrid_mae,
        "hybrid_rmse": hybrid_rmse,
        "hybrid_per_regime": hybrid_per_regime,
        "production_mae": production_result["mae"],
        "production_rmse": production_result["rmse"],
        "production_per_regime": production_result["per_regime"],
        "persistence_mae": persistence_mae,
        "verdict_burton_only": _verdict(burton_mae, "Burton (physics only)"),
        "verdict_hybrid": _verdict(hybrid_mae, "Hybrid (Burton + residual ML)"),
        "timestamps": [ts.isoformat() for ts in storm_eval.index],
        "actual": [float(v) for v in actual],
        "burton_only_predicted": [float(v) for v in burton_only_pred],
        "hybrid_predicted": [float(v) for v in hybrid_pred],
        "production_predicted": production_result["predicted"],
    }
    return result


# ==================== Run tracking (data/predictions/storm_burton_runs.json) ====================


def _load_runs() -> list[dict]:
    if not RUNS_REGISTRY_PATH.exists():
        return []
    with open(RUNS_REGISTRY_PATH) as f:
        return json.load(f)


def _save_runs(runs: list[dict]) -> None:
    RUNS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_REGISTRY_PATH, "w") as f:
        json.dump(runs, f, indent=2)


def record_burton_run(result: dict) -> dict:
    run = {
        "run_id": f"burton-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "run_at": datetime.now(timezone.utc).isoformat(),
        **{
            k: v
            for k, v in result.items()
            if k not in ("timestamps", "actual", "burton_only_predicted", "hybrid_predicted", "production_predicted")
        },
    }
    runs = _load_runs()
    runs.append(run)
    _save_runs(runs)
    return run


def list_burton_runs() -> list[dict]:
    return sorted(_load_runs(), key=lambda r: r["run_at"], reverse=True)
