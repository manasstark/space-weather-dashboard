"""Model explainability: local feature attribution for a live prediction.

Uses SHAP (TreeExplainer for XGBoost/RandomForest, LinearExplainer for
LinearRegression — covers every algorithm swdss.models.train ever
selects) when available, falling back to a model-agnostic zero-out
sensitivity check otherwise. Purely diagnostic and read-only: never
retrains, never modifies a model or a live prediction, and has no effect
on the production or experimental pipelines it inspects.
"""

import numpy as np

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from swdss.models.predict import _load_kp_interval_model, _load_metrics, _load_model, load_live_features

TOP_N = 8


def _select_shap_explainer(model, n_features: int):
    model_type = type(model).__name__
    if model_type in ("XGBRegressor", "RandomForestRegressor"):
        return shap.TreeExplainer(model)
    if model_type == "LinearRegression":
        masker = shap.maskers.Independent(np.zeros((1, n_features)))
        return shap.LinearExplainer(model, masker=masker)
    return None


def _permutation_fallback(model, X_row, feature_columns: list) -> list:
    """Model-agnostic local explanation for algorithms SHAP doesn't
    support here: how much the prediction changes if each feature is
    zeroed out, one at a time. Not the same statistical guarantee as
    SHAP's Shapley values, but a reasonable, always-available fallback.
    """
    base_pred = float(model.predict(X_row)[0])
    contributions = []
    for i, col in enumerate(feature_columns):
        perturbed = X_row.copy()
        perturbed.iloc[0, i] = 0.0
        perturbed_pred = float(model.predict(perturbed)[0])
        contributions.append((col, float(X_row.iloc[0, i]), base_pred - perturbed_pred))
    return contributions


def explain_prediction(dataset: str, variable: str, horizon) -> dict:
    """Explains the model's CURRENT prediction for (dataset, variable,
    horizon) using the most recently available live feature row (i.e.
    "why is the model saying this right now", not a reconstruction of
    exactly what it saw at some earlier tick).

    Returns {"method": "shap" | "permutation" | "unavailable",
    "model_name": str, "predicted_value": float | None,
    "contributions": [(feature, value, contribution), ...]} — the
    contributions list is sorted by |contribution| descending, capped at
    TOP_N.
    """
    metrics_doc = _load_metrics(dataset)
    if horizon == "interval":
        key = "kp_interval"
        model = _load_kp_interval_model(dataset)
    else:
        key = f"{variable}_{horizon}h"
        model = _load_model(dataset, variable, horizon)

    if key not in metrics_doc:
        return {"method": "unavailable", "model_name": None, "predicted_value": None, "contributions": []}

    meta = metrics_doc[key]
    feature_columns = meta["feature_columns"]

    frame = load_live_features(dataset)
    usable = frame.dropna(subset=feature_columns)
    if usable.empty:
        return {
            "method": "unavailable",
            "model_name": meta["algorithm"],
            "predicted_value": None,
            "contributions": [],
        }

    X_row = usable[feature_columns].iloc[[-1]]
    predicted_value = float(model.predict(X_row)[0])

    if _SHAP_AVAILABLE:
        explainer = _select_shap_explainer(model, len(feature_columns))
        if explainer is not None:
            sv = np.asarray(explainer.shap_values(X_row))
            sv_row = sv[0] if sv.ndim > 1 else sv
            contributions = list(zip(feature_columns, X_row.iloc[0].tolist(), sv_row.tolist()))
            contributions.sort(key=lambda t: abs(t[2]), reverse=True)
            return {
                "method": "shap",
                "model_name": meta["algorithm"],
                "predicted_value": predicted_value,
                "contributions": contributions[:TOP_N],
            }

    contributions = _permutation_fallback(model, X_row, feature_columns)
    contributions.sort(key=lambda t: abs(t[2]), reverse=True)
    return {
        "method": "permutation",
        "model_name": meta["algorithm"],
        "predicted_value": predicted_value,
        "contributions": contributions[:TOP_N],
    }
