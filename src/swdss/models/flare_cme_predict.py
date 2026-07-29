"""Flare Outlook / CME Outlook classifiers, trained on SHARP magnetic-
complexity features (swdss.models.flare_cme_features). Fully isolated
from every other production model in this project — own directory
(models/flare_cme/), own metrics file, never touches jobs.py, train.py,
or any existing PRODUCTION_MATRIX entry.

This is a classification problem, not a regression one — "will an
M/X-class flare happen in the next 24h," not "what value will Bz have."
Scored accordingly:

- True Skill Statistic (TSS = sensitivity - (1 - specificity)) — the
  standard metric for rare-event flare forecasting, unlike plain
  accuracy, which a model that always predicts "no flare" would already
  win at, since flares are rare. TSS of 0 means no better than chance;
  1.0 is perfect.
- Brier score — mean squared error of the predicted probability against
  the actual 0/1 outcome, the standard way to score a probabilistic
  forecast rather than a hard yes/no call.

Given only ~2 months of history exists at the time this was built, a
plain chronological 80/20 split is used rather than walk-forward CV —
honestly labeled as a small-sample estimate, not a robust one. If
either label doesn't have enough positive examples on both sides of
that split, training is skipped entirely and that's reported plainly
rather than training on a degenerate single-class split.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, confusion_matrix

from swdss.models.flare_cme_features import SHARP_FEATURE_COLUMNS
from swdss.paths import MODELS_DIR

FLARE_CME_MODELS_DIR = MODELS_DIR / "flare_cme"
FLARE_MODEL_PATH = FLARE_CME_MODELS_DIR / "flare_model.joblib"
CME_MODEL_PATH = FLARE_CME_MODELS_DIR / "cme_model.joblib"
METRICS_PATH = FLARE_CME_MODELS_DIR / "metrics.json"

FEATURE_COLUMNS = (
    SHARP_FEATURE_COLUMNS
    + [f"{c}_24h_ago" for c in SHARP_FEATURE_COLUMNS]
    + [f"{c}_24h_delta" for c in SHARP_FEATURE_COLUMNS]
)

MIN_TEST_POSITIVES = 2  # below this, a TSS/Brier estimate on the test split is not worth reporting


def true_skill_statistic(y_true, y_pred_binary) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sensitivity - (1 - specificity)


def _train_one(matrix: pd.DataFrame, label_col: str) -> dict:
    data = matrix.dropna(subset=FEATURE_COLUMNS + [label_col]).sort_values("hour")
    n_positive_total = int(data[label_col].sum()) if not data.empty else 0

    if data.empty or n_positive_total < MIN_TEST_POSITIVES * 2:
        return {
            "status": "insufficient_positive_examples",
            "n_samples": len(data),
            "n_positive_total": n_positive_total,
        }

    split_idx = int(len(data) * 0.8)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    if train[label_col].nunique() < 2 or int(test[label_col].sum()) < MIN_TEST_POSITIVES:
        return {
            "status": "degenerate_split",
            "n_samples": len(data),
            "n_positive_total": n_positive_total,
            "n_positive_test": int(test[label_col].sum()),
        }

    X_train, y_train = train[FEATURE_COLUMNS], train[label_col]
    X_test, y_test = test[FEATURE_COLUMNS], test[label_col]

    model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred_binary = (proba >= 0.5).astype(int)
    tss = true_skill_statistic(y_test, pred_binary)

    # TSS <= 0 means no better than chance on this holdout (0 is exactly
    # chance-level; negative is worse). Same discipline as F10.7's
    # harmonic-model gate: the artifact is kept for research/iteration,
    # but never labeled "trained" (which the engine treats as ready to
    # serve live probabilities) until it's actually demonstrated real
    # skill — a small ~2-month, ~45-region dataset is exactly the kind
    # of sample size where this gate is expected to trip, not a sign
    # something is broken.
    status = "trained" if tss > 0 else "trained_no_measurable_skill"

    return {
        "status": status,
        "n_samples": len(data),
        "n_train": len(train),
        "n_test": len(test),
        "n_positive_total": n_positive_total,
        "n_positive_test": int(test[label_col].sum()),
        "tss": tss,
        "brier_score": float(brier_score_loss(y_test, proba)),
        "model": model,
    }


def train_flare_cme_models(matrix: pd.DataFrame) -> dict:
    flare_result = _train_one(matrix, "flare_label_24h")
    cme_result = _train_one(matrix, "cme_label_24h")

    FLARE_CME_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if flare_result.get("model") is not None:
        joblib.dump(flare_result.pop("model"), FLARE_MODEL_PATH)
    else:
        flare_result.pop("model", None)
    if cme_result.get("model") is not None:
        joblib.dump(cme_result.pop("model"), CME_MODEL_PATH)
    else:
        cme_result.pop("model", None)

    metrics = {"flare": flare_result, "cme": cme_result, "generated_at": pd.Timestamp.now(tz="UTC").isoformat()}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def load_flare_cme_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict_flare_probability(latest_features: pd.DataFrame) -> pd.Series | None:
    if not FLARE_MODEL_PATH.exists():
        return None
    model = joblib.load(FLARE_MODEL_PATH)
    return pd.Series(model.predict_proba(latest_features[FEATURE_COLUMNS])[:, 1], index=latest_features.index)


def predict_cme_probability(latest_features: pd.DataFrame) -> pd.Series | None:
    if not CME_MODEL_PATH.exists():
        return None
    model = joblib.load(CME_MODEL_PATH)
    return pd.Series(model.predict_proba(latest_features[FEATURE_COLUMNS])[:, 1], index=latest_features.index)
