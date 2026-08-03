"""Confidence calibration and activity-regime-conditioned error.

Turns two columns evaluation_history has been logging since the
Forecast Package system shipped — confidence_category_at_issuance and
activity_regime — into the first real check of whether either one
means anything, without touching how either is computed. This is pure
analysis over already-collected history: swdss.engine.confidence's
scoring weights and outlook.classify_activity_regime's bucketing are
both untouched by this module; it only reports on them.

Confidence calibration's success bar is regime-conditioned MAE where
there's enough history to compute one (see compute_regime_error_bands),
falling back to the model's own flat training MAE otherwise — the same
fallback pattern refresh_dashboard_products already uses for the
DISPLAYED forecast range. This module originally used the flat training
MAE unconditionally, which was the actual root cause of a real, measured
calibration inversion: analytics/dst's 1h model has a high training R²
(0.963, MAE=2.82 nT, both measured on a mostly-quiet multi-year corpus)
that earns it "Very High"/"High" confidence almost unconditionally, but
its LIVE regime-conditioned error runs 4.26 nT (Quiet) to 8.26 nT
(Active) — 1.5-3x its own flat training MAE — so a large share of its
"Very High"/"High"-issued forecasts were failing a success bar that
never reflected its real, day-to-day dispersion. Every other slot in
those buckets cleared 75%+; this one slot alone (21 of 121 "Very High"
rows) was enough to invert the whole aggregate ordering.
"""

import pandas as pd

# Same "don't guess off a handful of points" floor used throughout this
# engine (swdss.engine.drift.MIN_SAMPLES, swdss.engine.skill.MIN_SAMPLES).
MIN_SAMPLES = 8
SUCCESS_RATIO = 1.5

CONFIDENCE_ORDER = ["Very High", "High", "Moderate", "Low", "Very Low"]
REGIME_ORDER = ["Quiet", "Active", "Storm"]


def _regime_mae_lookup(evaluation_history: pd.DataFrame) -> dict:
    """(dataset, variable, horizon, activity_regime) -> mean_abs_error,
    for every slot with enough samples — same MIN_SAMPLES floor
    compute_regime_error_bands itself applies, reused directly rather
    than re-deriving it.
    """
    return {
        (row["dataset"], row["variable"], row["horizon"], row["activity_regime"]): row["mean_abs_error"]
        for row in compute_regime_error_bands(evaluation_history)
    }


def compute_confidence_calibration(evaluation_history: pd.DataFrame) -> list:
    """One row per confidence category actually issued: what fraction of
    forecasts made at that confidence level went on to verify as a
    success. A well-calibrated heuristic shows a monotonically
    decreasing success rate from Very High down to Low; a flat or
    inverted ordering means confidence.py's weights don't actually track
    real accuracy and should be revisited — this table is how that would
    be discovered, not assumed.

    Each forecast's success bar is 1.5x its OWN (dataset, variable,
    horizon, activity_regime) MAE where at least MIN_SAMPLES of history
    exists for that exact slot+regime; otherwise it falls back to 1.5x
    the model's flat training MAE, same as before. Judging a forecast
    issued during Active/Storm conditions against a bar computed mostly
    from Quiet-time history is exactly the mechanism that caused a real,
    measured inversion — see this module's own docstring.
    """
    if evaluation_history.empty or "confidence_category_at_issuance" not in evaluation_history.columns:
        return []

    df = evaluation_history.dropna(subset=["confidence_category_at_issuance", "abs_error", "training_mae"])
    df = df[df["training_mae"] > 0]
    if df.empty:
        return []

    regime_mae = _regime_mae_lookup(evaluation_history)
    has_regime = "activity_regime" in df.columns

    def _success_bar(row) -> float:
        if has_regime and pd.notna(row["activity_regime"]):
            key = (row["dataset"], row["variable"], row["horizon"], row["activity_regime"])
            if key in regime_mae:
                return SUCCESS_RATIO * regime_mae[key]
        return SUCCESS_RATIO * row["training_mae"]

    df = df.copy()
    df["_success_bar"] = df.apply(_success_bar, axis=1)
    df["_success"] = df["abs_error"] <= df["_success_bar"]

    rows = []
    for category, group in df.groupby("confidence_category_at_issuance"):
        n = len(group)
        if n < MIN_SAMPLES:
            rows.append({
                "confidence_category": category,
                "sample_n": n,
                "success_rate_pct": None,
                "status": "Insufficient Data",
            })
            continue
        success_rate = float(group["_success"].mean()) * 100
        rows.append({
            "confidence_category": category,
            "sample_n": n,
            "success_rate_pct": success_rate,
            "status": "OK",
        })

    rows.sort(key=lambda r: CONFIDENCE_ORDER.index(r["confidence_category"]) if r["confidence_category"] in CONFIDENCE_ORDER else len(CONFIDENCE_ORDER))
    return rows


def compute_regime_error_bands(evaluation_history: pd.DataFrame) -> list:
    """Mean absolute error per (dataset, variable, horizon, activity
    regime). The classic space-weather ML failure mode is a model that
    looks strong on an aggregate dominated by quiet-time samples and
    silently degrades exactly when a storm makes the forecast matter
    most. This never touches a model's own training or its dashboard-
    reported aggregate MAE; it only segments already-observed error by
    the regime each forecast was issued under.
    """
    if evaluation_history.empty or "activity_regime" not in evaluation_history.columns:
        return []

    df = evaluation_history.dropna(subset=["activity_regime", "abs_error"])
    if df.empty:
        return []

    rows = []
    for (dataset, variable, horizon, regime), group in df.groupby(["dataset", "variable", "horizon", "activity_regime"]):
        n = len(group)
        if n < MIN_SAMPLES:
            continue
        rows.append({
            "dataset": dataset,
            "variable": variable,
            "horizon": horizon,
            "activity_regime": regime,
            "sample_n": n,
            "mean_abs_error": float(group["abs_error"].mean()),
        })

    rows.sort(key=lambda r: (
        r["dataset"], r["variable"], r["horizon"],
        REGIME_ORDER.index(r["activity_regime"]) if r["activity_regime"] in REGIME_ORDER else len(REGIME_ORDER),
    ))
    return rows
