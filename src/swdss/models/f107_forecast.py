"""F10.7 solar radio flux forecasting — harmonic regression exploiting
the Sun's ~27-day synodic (Carrington) rotation period.

Why this approach, specifically: F10.7 is a much smoother, more quasi-
periodic signal than Kp/Dst/AE. Storm-time indices are hard because they
depend on rare, discontinuous solar-wind driving; F10.7 is dominated by
two well-understood, slowly-varying components instead — the ~11-year
solar cycle envelope, and a strong ~27-day periodicity from active
regions rotating in and out of Earth view as the Sun itself rotates.
That means classical harmonic/seasonal time-series methods, which would
be badly wrong for storm-time Kp, are genuinely strong here.

This module deliberately does NOT reach for a heavier dependency
(statsmodels' SARIMA) or a persisted/trained model artifact the way
production Kp/Dst/AE models work: a harmonic regression (a handful of
sin/cos Fourier terms at the rotation period, plus a linear trend term,
fit via ordinary least squares) is mathematically the same idea and
cheap enough to refit on every call — well under a second for the
data volumes this project has. No training script, no joblib artifact,
no metrics.json entry needed for something this inexpensive to compute
fresh every time.

Data-honesty note: a 27-day-period harmonic term is only meaningfully
constrained once the historical record actually spans multiple
rotations. MIN_DAYS_FOR_HARMONIC gates the harmonic model off (falling
back to the seasonal-naive baseline alone) when there isn't enough
history yet — this project's F10.7 archive starts from whenever
live_update.py was first run, so early on this will legitimately be in
fallback mode, and should silently upgrade itself as more days accumulate
rather than ever pretending precision the data doesn't support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROTATION_PERIOD_DAYS = 27.2753  # Carrington synodic solar rotation period
MIN_DAYS_FOR_HARMONIC = 40  # ~1.5 rotations — the minimum needed to constrain the 27-day term at all
MIN_HOLDOUT_SAMPLES_FOR_SKILL = 5
N_HARMONICS = 2


def daily_series(f107_df: pd.DataFrame) -> pd.Series:
    """Collapses the raw feed (which can carry several same-day readings
    as NOAA revises a day's value from preliminary to final) to one value
    per calendar day, taking the day's last (most-revised) reading.
    """
    if f107_df.empty or "f107_flux" not in f107_df.columns:
        return pd.Series(dtype=float)

    df = f107_df.dropna(subset=["f107_flux", "timestamp_utc"]).copy()
    if df.empty:
        return pd.Series(dtype=float)

    df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.date
    daily = df.groupby("date")["f107_flux"].last()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def _harmonic_design_matrix(t: np.ndarray, n_harmonics: int = N_HARMONICS) -> np.ndarray:
    columns = [t]
    for k in range(1, n_harmonics + 1):
        columns.append(np.sin(2 * np.pi * k * t / ROTATION_PERIOD_DAYS))
        columns.append(np.cos(2 * np.pi * k * t / ROTATION_PERIOD_DAYS))
    return np.column_stack(columns)


def _fit_harmonic_model(daily: pd.Series) -> LinearRegression:
    t = np.arange(len(daily), dtype=float)
    X = _harmonic_design_matrix(t)
    model = LinearRegression()
    model.fit(X, daily.values)
    return model


def harmonic_forecast(daily: pd.Series, horizon_days: int) -> float | None:
    """Fits the harmonic regression on all available history and predicts
    horizon_days beyond the last observed day. Returns None if there
    isn't enough history to constrain the rotation term at all.
    """
    if len(daily) < MIN_DAYS_FOR_HARMONIC:
        return None
    model = _fit_harmonic_model(daily)
    t_future = np.array([[len(daily) - 1 + horizon_days]], dtype=float)
    X_future = _harmonic_design_matrix(t_future.ravel())
    return float(model.predict(X_future)[0])


def seasonal_naive_forecast(daily: pd.Series, horizon_days: int) -> float | None:
    """The legitimate baseline given F10.7's known ~27-day periodicity:
    predict the value seen one rotation ago at the same phase, i.e. the
    reading from (horizon_days - ROTATION_PERIOD_DAYS) days from now,
    which is (round(ROTATION_PERIOD_DAYS) - horizon_days) days in the
    past. Falls back to ordinary persistence (last observed value) if
    there isn't even that much history yet.
    """
    if daily.empty:
        return None
    lookback_days = int(round(ROTATION_PERIOD_DAYS)) - horizon_days
    if lookback_days <= 0 or lookback_days >= len(daily):
        return float(daily.iloc[-1])
    return float(daily.iloc[-1 - lookback_days])


def compute_f107_skill(daily: pd.Series, horizon_days: int = 1) -> dict | None:
    """Walk-forward comparison of the harmonic model against the
    seasonal-naive baseline: for each of the last few days that has both
    a real outcome and enough preceding history to fit on, refit the
    harmonic model using only data strictly before that point (no
    leakage) and compare its forecast error against the naive baseline's.
    skill_score = 1 - (model MAE / naive MAE), the same definition
    swdss.engine.skill uses for the production forecasts — positive means
    genuine skill over the naive baseline, zero or negative means it
    currently doesn't have any. Returns None if there isn't enough
    history to run a real holdout (not just one lucky day).
    """
    n = len(daily)
    usable_holdout = n - MIN_DAYS_FOR_HARMONIC - horizon_days
    if usable_holdout < MIN_HOLDOUT_SAMPLES_FOR_SKILL:
        return None

    model_errors = []
    naive_errors = []
    holdout_points = range(n - usable_holdout, n - horizon_days)
    for i in holdout_points:
        train_slice = daily.iloc[: i + 1]
        actual = daily.iloc[i + horizon_days]

        model_pred = harmonic_forecast(train_slice, horizon_days)
        naive_pred = seasonal_naive_forecast(train_slice, horizon_days)
        if model_pred is None or naive_pred is None:
            continue
        model_errors.append(abs(model_pred - actual))
        naive_errors.append(abs(naive_pred - actual))

    if len(model_errors) < MIN_HOLDOUT_SAMPLES_FOR_SKILL:
        return None

    model_mae = float(np.mean(model_errors))
    naive_mae = float(np.mean(naive_errors))
    skill_score = None if naive_mae == 0 else 1 - (model_mae / naive_mae)
    return {
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "skill_score": skill_score,
        "n_samples": len(model_errors),
        "horizon_days": horizon_days,
    }


def forecast_f107(f107_df: pd.DataFrame) -> dict | None:
    """Top-level entry point: tomorrow's flux, a 7-day trend, the
    seasonal-naive baseline for both, and a skill score for tomorrow's
    horizon (the one with the most holdout samples available to check).
    Returns None if there's no F10.7 data at all yet.

    The harmonic model is only ever surfaced as the actual forecast once
    it has a positive, holdout-validated skill score against the
    seasonal-naive baseline on this exact dataset — not merely once
    MIN_DAYS_FOR_HARMONIC of history exists. With only slightly more than
    one rotation of history, a harmonic fit is technically computable but
    can swing wildly and has no track record yet; better to keep showing
    the honest, well-understood naive baseline than a number that merely
    looks more sophisticated. This mirrors the same discipline this
    project's Storm Learning tool applies elsewhere: never promote a
    result over its baseline without first measuring that it actually
    wins.
    """
    daily = daily_series(f107_df)
    if daily.empty:
        return None

    skill = compute_f107_skill(daily, horizon_days=1)
    harmonic_validated = skill is not None and skill["skill_score"] is not None and skill["skill_score"] > 0

    tomorrow_model = harmonic_forecast(daily, 1)
    tomorrow_naive = seasonal_naive_forecast(daily, 1)
    week_trend = [
        {
            "day": h,
            "model": harmonic_forecast(daily, h),
            "naive": seasonal_naive_forecast(daily, h),
        }
        for h in range(1, 8)
    ]

    if harmonic_validated:
        tomorrow_method = "harmonic"
    elif skill is not None:
        tomorrow_method = "seasonal_naive_fallback_underperformed"
    elif len(daily) < MIN_DAYS_FOR_HARMONIC:
        tomorrow_method = "seasonal_naive_fallback_insufficient_history"
    else:
        tomorrow_method = "seasonal_naive_fallback_unvalidated"

    return {
        "last_observed_date": daily.index[-1],
        "last_observed_value": float(daily.iloc[-1]),
        "n_days_history": len(daily),
        "harmonic_validated": harmonic_validated,
        "tomorrow_forecast": tomorrow_model if harmonic_validated else tomorrow_naive,
        "tomorrow_method": tomorrow_method,
        "tomorrow_naive_baseline": tomorrow_naive,
        "week_trend": week_trend,
        "skill": skill,
        "min_days_for_harmonic": MIN_DAYS_FOR_HARMONIC,
    }
