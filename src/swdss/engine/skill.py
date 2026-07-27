"""Persistence-based forecast skill score — the standard operational
meteorology / space-weather verification question: does the model
actually beat the naive "assume no change" forecast, not merely
whether it's performing consistently with its own training benchmark
(which is all packages.build_verification_summary and drift.py check).

    skill_score = 1 - (MSE_model / MSE_persistence)

    > 0   the model has genuine forecast skill at this horizon
    <= 0  the model is no better than (or worse than) assuming the
          value known at issuance simply persists to the target hour —
          a real, actionable finding, not a bug in this module.

Requires evaluation_history rows to carry `persistence_squared_error`
(see orchestrator.refresh_dashboard_products, where the persistence
anchor — the last known hourly-mean observation at/before the job's own
creation time — is resolved once, at evaluation time, via the same
swdss.models.jobs.resolve_actual_value already used to resolve actuals).

AE never appears in this module's output: it has no live feed (see
swdss.models.registry's static_variables), so there is no "value known
at issuance" to persist from — persistence_squared_error is never
computed for it, and its rows are silently absent from every group
below rather than reported as a misleading skill score of 0.
"""

import pandas as pd

# Same "don't guess off a handful of points" floor used throughout this
# engine (swdss.engine.drift.MIN_SAMPLES, confidence._trend_score).
MIN_SAMPLES = 8


def compute_skill_scores(evaluation_history: pd.DataFrame) -> list:
    """Returns one row per (dataset, variable, horizon) with at least
    MIN_SAMPLES evaluated forecasts carrying a persistence comparison —
    everything else (too few samples, or AE) is simply absent, not
    reported with a fabricated score.
    """
    if evaluation_history.empty or "persistence_squared_error" not in evaluation_history.columns:
        return []

    df = evaluation_history.dropna(subset=["persistence_squared_error", "squared_error"])
    if df.empty:
        return []

    rows = []
    for (dataset, variable, horizon), group in df.groupby(["dataset", "variable", "horizon"]):
        n = len(group)
        if n < MIN_SAMPLES:
            continue
        model_mse = float(group["squared_error"].mean())
        persistence_mse = float(group["persistence_squared_error"].mean())
        if persistence_mse <= 0:
            continue
        skill = 1.0 - (model_mse / persistence_mse)
        rows.append({
            "dataset": dataset,
            "variable": variable,
            "horizon": horizon,
            "sample_n": n,
            "model_rmse": model_mse ** 0.5,
            "persistence_rmse": persistence_mse ** 0.5,
            "skill_score": skill,
            "verdict": "Beats Persistence" if skill > 0 else "No Skill Over Persistence",
        })

    rows.sort(key=lambda r: r["skill_score"])
    return rows
