"""Deterministic confidence scoring for a single forecast — NOT a machine
learning model. A documented, reproducible weighted heuristic over
signals the rest of the app already computes: the model's own held-out
R² and cross-validation stability (from its metrics.json entry, frozen
onto the job at creation), the forecast's horizon, this session's
prediction stability (jobs.stability_metric), and the variable's recent
evaluated-forecast error trend (jobs.compute_variable_metrics). Collapses
to a 5-category operational label — never fit against data, and never
used as a model input anywhere.
"""

HORIZON_PENALTY = {1: 1.0, 3: 0.9, 6: 0.8, 12: 0.65, 24: 0.5, "interval": 0.85}

STABILITY_SCORE = {"Stable": 1.0, "Moderately Stable": 0.6, "Unstable": 0.2}

# Ordered strictly descending — first threshold the score clears wins.
CONFIDENCE_THRESHOLDS = [
    (0.85, "Very High"),
    (0.70, "High"),
    (0.50, "Moderate"),
    (0.30, "Low"),
]

WEIGHTS = {
    "model_quality": 0.40,
    "cv_stability": 0.15,
    "horizon": 0.20,
    "session_stability": 0.15,
    "error_trend": 0.10,
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _trend_score(trend: list) -> float:
    """`trend` is compute_variable_metrics(dataset, variable)["trend"] —
    a chronologically sorted list of (completed_at, abs_error) tuples for
    ONE (dataset, variable), deliberately not get_prediction_statistics'
    dataset-wide trend (which would mix unrelated variables' error
    magnitudes, e.g. Speed in km/s against Bz in nT). Needs at least 4
    evaluated forecasts to say anything about direction; thinner history
    returns a neutral 0.7 rather than guessing off 1-2 points.
    """
    if not trend or len(trend) < 4:
        return 0.7
    errors = [e for _, e in trend]
    mid = len(errors) // 2
    earlier_mean = sum(errors[:mid]) / len(errors[:mid])
    later_mean = sum(errors[mid:]) / len(errors[mid:])
    if earlier_mean == 0:
        return 0.7
    change_pct = (later_mean - earlier_mean) / earlier_mean
    if change_pct <= -0.05:
        return 1.0  # improving — recent error meaningfully lower
    if change_pct >= 0.05:
        return 0.3  # degrading — recent error meaningfully higher
    return 0.7  # flat


def score_forecast_confidence(
    *,
    r2,
    cv_r2_std,
    horizon,
    stability_label,
    trend: list,
) -> tuple:
    """Returns (score in [0, 1], category)."""
    model_quality = _clip01(r2) if r2 is not None else 0.5
    cv_stability = 1.0 - min((cv_r2_std or 0.0) / 0.20, 1.0) if cv_r2_std is not None else 0.7
    horizon_score = HORIZON_PENALTY.get(horizon, 0.6)
    session_stability = STABILITY_SCORE.get(stability_label, 0.7)
    trend_score = _trend_score(trend)

    score = _clip01(
        WEIGHTS["model_quality"] * model_quality
        + WEIGHTS["cv_stability"] * cv_stability
        + WEIGHTS["horizon"] * horizon_score
        + WEIGHTS["session_stability"] * session_stability
        + WEIGHTS["error_trend"] * trend_score
    )

    category = "Very Low"
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if score >= threshold:
            category = label
            break

    return score, category
