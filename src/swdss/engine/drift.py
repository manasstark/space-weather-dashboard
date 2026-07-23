"""Lightweight operational drift monitoring — compares each model's
recent live evaluated-forecast error against its own training-time MAE
(the same number already shown in metrics.json / the job's frozen
metrics). Notifies only. Never retrains — retraining stays a deliberate,
human-triggered action in the Research Labs, never something this
engine does on its own.
"""

import pandas as pd

# A handful of evaluated forecasts is noise, not a trend — require at
# least this many before ever declaring drift, same "don't guess off 1-2
# points" spirit as swdss.engine.confidence._trend_score.
MIN_SAMPLES = 8
RECENT_WINDOW = 15
DRIFT_RATIO_THRESHOLD = 1.5


def detect_drift(evaluation_history: pd.DataFrame) -> list:
    """Returns a list of alert dicts (severity/source/message/since, plus
    the raw variable/dataset/horizon/training_mae/recent_mae for
    anything that wants the numbers directly) — one per (dataset,
    variable, horizon) whose recent mean absolute error has sustained at
    or above DRIFT_RATIO_THRESHOLD times its own training MAE across at
    least MIN_SAMPLES evaluated forecasts.
    """
    if evaluation_history.empty or "training_mae" not in evaluation_history.columns:
        return []

    alerts = []
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()
    grouped = evaluation_history.sort_values("completed_at").groupby(["dataset", "variable", "horizon"])

    for (dataset, variable, horizon), group in grouped:
        if len(group) < MIN_SAMPLES:
            continue

        recent = group.tail(RECENT_WINDOW)
        training_mae_series = recent["training_mae"].dropna()
        if training_mae_series.empty:
            continue

        # Most recently issued model's own training MAE — if the model
        # was ever promoted/retrained mid-window, compare against the
        # CURRENT model's baseline, not a stale earlier one.
        training_mae = float(training_mae_series.iloc[-1])
        if training_mae <= 0:
            continue

        recent_mae = float(recent["abs_error"].mean())
        ratio = recent_mae / training_mae
        if ratio >= DRIFT_RATIO_THRESHOLD:
            alerts.append({
                "severity": "warning",
                "source": "Drift",
                "message": (
                    f"MODEL DRIFT DETECTED — {variable.upper()} ({horizon}): training MAE {training_mae:.3f}, "
                    f"recent performance {recent_mae:.3f} ({ratio:.1f}x expected). Retraining recommended."
                ),
                "since": now_iso,
                "dataset": dataset,
                "variable": variable,
                "horizon": horizon,
                "training_mae": training_mae,
                "recent_mae": recent_mae,
                "ratio": ratio,
                "sample_size": len(recent),
            })

    return alerts
