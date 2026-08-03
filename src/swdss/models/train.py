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

# How much better the ensemble's walk-forward CV R² must be than the best
# single candidate's before it's ever selected — matches this project's
# established discipline of never promoting a change on the strength of a
# difference that could just be noise (the same "is the difference even
# meaningful" threshold Storm Learning's verdict logic already applies
# elsewhere). A blend that's merely marginally different from its own best
# ingredient isn't worth the extra inference cost and reduced
# interpretability of shipping three models instead of one.
ENSEMBLE_MEANINGFUL_IMPROVEMENT = 0.05


class WeightedEnsembleRegressor:
    """A blend of CANDIDATE_MODELS, weighted by each candidate's own
    walk-forward CV R² (candidates with zero or negative CV R² get zero
    weight, so a genuinely bad candidate can never drag the blend down —
    it's a weighted average of the models that showed real skill, not a
    naive average of all three regardless of quality).

    Implements the same .fit()/.predict() contract every sklearn
    estimator does, so it plugs directly into evaluate_walk_forward,
    evaluate_split, predict.py's live inference, and joblib
    persistence — every existing codepath that expects a fittable/
    predictable model — with zero changes elsewhere. Falls back to the
    model-agnostic permutation explainer automatically (see
    explainability.py's _select_shap_explainer, which already returns
    None for unrecognized model types), since SHAP's TreeExplainer/
    LinearExplainer don't know how to open a custom blended object.
    """

    def __init__(self, weights: dict):
        self.weights = {name: w for name, w in weights.items() if w > 0}
        self._models = {}

    def fit(self, X, y):
        for name in self.weights:
            model = CANDIDATE_MODELS[name]()
            model.fit(X, y)
            self._models[name] = model
        return self

    def predict(self, X):
        preds = None
        for name, model in self._models.items():
            contribution = self.weights[name] * model.predict(X)
            preds = contribution if preds is None else preds + contribution
        return preds


def _ensemble_weights(all_candidates: dict) -> dict:
    positive_r2 = {name: max(res["cv"]["r2_mean"], 0.0) for name, res in all_candidates.items()}
    total = sum(positive_r2.values())
    if total <= 0:
        return {}
    return {name: r2 / total for name, r2 in positive_r2.items()}


def _load_regime_context() -> pd.DataFrame:
    """Kp/Dst/AE on their natural 0-9/nT/nT scales, indexed by datetime —
    analytics_features.csv is the only training corpus with all three on
    the same hourly index (same source/pattern swdss.models.ae_research.
    _load_kp_dst_for_ae already uses for its own geomagnetic-memory
    feature), reused here purely to tag OTHER datasets' own rows with the
    REAL historical activity regime they occurred in, not as a feature.
    """
    config = DATASETS["analytics"]
    raw = pd.read_csv(config.training_csv)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")
    context = raw[["kp", "dst", "ae"]].copy()
    for column, factor in (config.scale_factors or {}).items():
        if column in context.columns:
            context[column] = context[column] / factor
    return context


def _regime_labels_for_index(index: pd.DatetimeIndex) -> pd.Series:
    """Real (not predicted) Quiet/Active/Storm tag per row in `index`,
    reusing swdss.engine.outlook.classify_activity_regime — the same
    worst-of-Kp/Dst/AE bucketing already used for LIVE regime-conditioned
    error bands (swdss.engine.calibration), applied here at TRAINING time
    instead so a candidate's storm-period accuracy is visible before it's
    ever deployed, not only discoverable afterward via Storm Backtest.
    Rows outside analytics_features.csv's own date range (or missing
    kp/dst/ae) fall back to "Quiet" via classify_activity_regime's own
    None-handling — a conservative default, not a crash.
    """
    from swdss.engine.outlook import classify_activity_regime

    context = _load_regime_context().reindex(index)
    labels = [
        classify_activity_regime(
            predicted_kp=None if pd.isna(row.kp) else row.kp,
            predicted_dst=None if pd.isna(row.dst) else row.dst,
            predicted_ae=None if pd.isna(row.ae) else row.ae,
        )
        for row in context.itertuples()
    ]
    return pd.Series(labels, index=index)


def _storm_quiet_mae(abs_errors: np.ndarray, regime_labels: pd.Series) -> dict:
    """Per-regime MAE + sample count over one holdout split — reporting
    only, the same Quiet/Active/Storm segmentation swdss.engine.
    calibration.compute_regime_error_bands applies to live evaluated
    forecasts, computed here at training time so a candidate's real
    storm-period accuracy is reported alongside its aggregate score
    rather than assumed to track it.
    """
    labels = regime_labels.to_numpy()
    result = {}
    for regime, key in (("Quiet", "quiet"), ("Active", "active"), ("Storm", "storm")):
        mask = labels == regime
        if mask.any():
            result[f"{key}_mae"] = float(abs_errors[mask].mean())
            result[f"{key}_n"] = int(mask.sum())
        else:
            result[f"{key}_mae"] = None
            result[f"{key}_n"] = 0
    return result


def evaluate_split(model, X_train, y_train, X_test, y_test, regime_labels: pd.Series = None) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    result = {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    if regime_labels is not None:
        abs_errors = np.abs(y_test.to_numpy() - preds)
        result.update(_storm_quiet_mae(abs_errors, regime_labels))
    return result


def _fit_best(X, y) -> tuple:
    """Benchmarks every candidate algorithm two ways, then ALSO benchmarks
    a validated weighted ensemble of them, and refits the overall winner
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

    The single-model algorithm is selected by mean walk-forward CV R²,
    same as before. A WeightedEnsembleRegressor (see above) is then built
    from the candidates' own CV performance and evaluated the identical
    way — it only ever replaces the single-model winner if its CV R² beats
    it by ENSEMBLE_MEANINGFUL_IMPROVEMENT, the same "prove it before you
    trust it" bar every other new capability in this project has to clear
    (F10.7's harmonic-vs-naive gate, the SHARP flare/CME TSS>0 gate). No
    ensemble slot in PRODUCTION_MATRIX ever ships on the strength of "it's
    fancier," only on a demonstrated, real improvement.

    Returns (name, holdout_metrics, cv_metrics, model, all_candidates)
    where all_candidates maps every candidate name (plus "Ensemble" when
    one was built) to its own holdout/cv metrics, for transparency about
    what was and wasn't selected — same contract as before, one possible
    extra key. Each candidate's holdout metrics also now report
    storm/active/quiet MAE (see _storm_quiet_mae) — reporting only, this
    does NOT change which candidate wins: selection is still purely by
    mean walk-forward CV R², same as before. Surfacing per-regime holdout
    accuracy here is the "explicitly report it" half of storm-aware model
    selection; actually weighting selection by it is a deliberately
    separate, not-yet-taken step (see Development Roadmap).
    """
    split_idx = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    regime_labels = _regime_labels_for_index(X_test.index)

    all_candidates = {}
    for name, factory in CANDIDATE_MODELS.items():
        holdout_metrics = evaluate_split(factory(), X_train, y_train, X_test, y_test, regime_labels)
        cv_metrics = evaluate_walk_forward(factory, X, y, n_folds=CV_FOLDS)
        all_candidates[name] = {"holdout": holdout_metrics, "cv": cv_metrics}

    best_name = max(all_candidates, key=lambda n: all_candidates[n]["cv"]["r2_mean"])
    best_cv_r2 = all_candidates[best_name]["cv"]["r2_mean"]

    weights = _ensemble_weights(all_candidates)
    if len(weights) >= 2:  # blending only means something with 2+ real contributors
        def ensemble_factory():
            return WeightedEnsembleRegressor(weights)
        ensemble_cv = evaluate_walk_forward(ensemble_factory, X, y, n_folds=CV_FOLDS)
        all_candidates["Ensemble"] = {
            "holdout": evaluate_split(ensemble_factory(), X_train, y_train, X_test, y_test, regime_labels),
            "cv": ensemble_cv,
            "weights": weights,
        }
        if best_cv_r2 > 0:
            relative_improvement = (ensemble_cv["r2_mean"] - best_cv_r2) / best_cv_r2
        else:
            relative_improvement = 1.0 if ensemble_cv["r2_mean"] > 0 else 0.0
        if ensemble_cv["r2_mean"] > best_cv_r2 and relative_improvement >= ENSEMBLE_MEANINGFUL_IMPROVEMENT:
            best_name = "Ensemble"

    best_metrics = all_candidates[best_name]["holdout"]
    best_cv = all_candidates[best_name]["cv"]

    if best_name == "Ensemble":
        final_model = WeightedEnsembleRegressor(weights)
    else:
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


def prepare_dataset_frame(dataset_key: str) -> tuple:
    """The dataset-level prep (load training CSV, build the derived-
    physics + lag/rolling feature frame) every variable/horizon slot of
    a dataset shares — factored out so a caller training many slots of
    the same dataset (train_dataset's bulk loop, or
    swdss.engine.retrain's automated cycle) only pays this cost once
    per dataset rather than once per slot.
    """
    config = DATASETS[dataset_key]
    base_df, feature_vars = _load_base_df(config)
    frame, feature_columns = build_feature_frame(base_df, feature_vars)
    return config, base_df, frame, feature_columns


def train_slot(dataset_key: str, variable: str, horizon: int, prepared: tuple | None = None) -> tuple:
    """Trains and evaluates ONE (dataset, variable, horizon) production
    slot — exactly the work train_dataset's loop body always did —
    without writing anything to disk. Returns (final_model, record) so
    a caller decides whether to persist it: train_dataset always does
    (unconditional bulk refresh, unchanged behavior); swdss.engine.
    retrain only does so if the new candidate actually beats the
    currently-deployed model's own CV score.

    `prepared`, if given, reuses an already-loaded
    (config, base_df, frame, feature_columns) tuple from
    prepare_dataset_frame — avoids re-reading the same training CSV for
    every horizon of the same variable.
    """
    config, base_df, frame, feature_columns = prepared or prepare_dataset_frame(dataset_key)

    target = base_df[variable].shift(-horizon)
    data = frame.copy()
    data["__target__"] = target
    data = data.dropna(subset=feature_columns + ["__target__"])

    X = data[feature_columns]
    y = data["__target__"]

    best_name, best_metrics, best_cv, final_model, all_candidates = _fit_best(X, y)

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
        "quiet_mae": best_metrics.get("quiet_mae"),
        "quiet_n": best_metrics.get("quiet_n"),
        "active_mae": best_metrics.get("active_mae"),
        "active_n": best_metrics.get("active_n"),
        "storm_mae": best_metrics.get("storm_mae"),
        "storm_n": best_metrics.get("storm_n"),
        "feature_columns": feature_columns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
    }
    return final_model, record


def train_dataset(dataset_key: str) -> list[dict]:
    prepared = prepare_dataset_frame(dataset_key)
    config = prepared[0]

    results = []

    for variable in config.variables:
        # Kp on both the production Analytics page and the experimental
        # cascade follows NOAA's real 3-hour cadence instead of an
        # arbitrary hourly horizon — trained separately by
        # train_kp_interval_model(), not as 5 fixed-horizon models here.
        if dataset_key in ("analytics", "experimental") and variable == "kp":
            continue

        for horizon in HORIZONS:
            final_model, record = train_slot(dataset_key, variable, horizon, prepared=prepared)

            path = model_path(dataset_key, variable, horizon)
            joblib.dump(final_model, path)
            record["model_path"] = str(path)
            results.append(record)
            print(
                f"[{dataset_key}] {variable} +{horizon}h -> {record['algorithm']} "
                f"(R2={record['r2']:.4f} MAE={record['mae']:.3f} RMSE={record['rmse']:.3f} | "
                f"CV R2={record['cv_r2_mean']:.4f}+/-{record['cv_r2_std']:.4f})"
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


def train_kp_interval_slot(dataset: str = "analytics") -> tuple:
    """The Kp-interval equivalent of train_slot — same "prepare, fit,
    build a record, don't write" contract, so swdss.engine.retrain can
    evaluate a fresh Kp-interval candidate before deciding whether it
    beats the currently-deployed one, exactly like every other
    production slot.
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
        "quiet_mae": best_metrics.get("quiet_mae"),
        "quiet_n": best_metrics.get("quiet_n"),
        "active_mae": best_metrics.get("active_mae"),
        "active_n": best_metrics.get("active_n"),
        "storm_mae": best_metrics.get("storm_mae"),
        "storm_n": best_metrics.get("storm_n"),
        "feature_columns": feature_columns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
    }
    return final_model, record


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
    final_model, record = train_kp_interval_slot(dataset)

    path = kp_interval_model_path(dataset)
    joblib.dump(final_model, path)
    record["model_path"] = str(path)

    metrics_doc_path = metrics_path(dataset)
    metrics_doc = {}
    if metrics_doc_path.exists():
        with open(metrics_doc_path) as f:
            metrics_doc = json.load(f)
    metrics_doc["kp_interval"] = record
    with open(metrics_doc_path, "w") as f:
        json.dump(metrics_doc, f, indent=2)

    print(
        f"[{dataset}] kp +next-interval -> {record['algorithm']} "
        f"(R2={record['r2']:.4f} MAE={record['mae']:.3f} RMSE={record['rmse']:.3f} | "
        f"CV R2={record['cv_r2_mean']:.4f}+/-{record['cv_r2_std']:.4f})"
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
