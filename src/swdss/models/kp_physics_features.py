"""Kp physics-informed feature engineering — Kp Research Laboratory only.

Unlike the IMF Research Lab's physics features (imf_physics_features.py),
which had to operate at raw MINUTE resolution because Bz's own physics is
minute-native, everything here operates on the SAME HOURLY analytics
frame (`analytics_features.csv`) production's own Kp model trains on —
Kp itself only exists at 3-hour resolution, so there is no finer-grained
"true" cadence being thrown away by working hourly. This keeps the Kp
Research Lab's physics features directly comparable to production's own
feature set, with no granularity duality to manage.

Every function is strictly causal — only rolling/expanding/shift-based
windows looking backward from each row, never forward — and takes a
DataFrame indexed by hourly timestamp with whichever of
bz_gsm/by_gsm/ey/vbz/dynamic_pressure/dst/kp/ae columns it needs,
returning the list of column names it created (same convention as
swdss.models.features and imf_physics_features.py).

Several of these (Storm Phase, Previous Storm Strength) are deliberately
simple, documented heuristics — not validated storm-detection algorithms
— in the same spirit as jobs.py's own `_STORM_CHECKS` threshold-based
storm/quiet segmentation.
"""

import numpy as np
import pandas as pd

DEFAULT_WINDOW_HOURS = 24
STORM_DST_THRESHOLD_NT = -20.0


def _consecutive_true_run_length(condition: pd.Series) -> pd.Series:
    reset_groups = (~condition).cumsum()
    return condition.groupby(reset_groups).cumsum()


def add_southward_duration(df: pd.DataFrame, bz_col: str = "bz_gsm") -> list[str]:
    """Consecutive HOURS with Bz < 0 (southward), reset to 0 the instant
    Bz turns northward — the hourly analogue of the IMF lab's
    minute-resolution southward-duration feature.
    """
    name = "southward_duration_hr"
    df[name] = _consecutive_true_run_length(df[bz_col] < 0)
    return [name]


def add_integrated_ey(df: pd.DataFrame, ey_col: str = "ey", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling sum of the geoeffective (positive) part of Ey over `window`
    hours — Ey is positive exactly when Bz is southward under this
    project's sign convention (ey = -speed * bz * 1e-3), so clipping to
    the positive part before summing integrates only the geoeffective
    driving, not cancelling contributions from northward stretches.
    """
    name = f"integrated_ey_{window}h"
    df[name] = df[ey_col].clip(lower=0).rolling(window).sum()
    return [name]


def add_integrated_vbz(df: pd.DataFrame, vbz_col: str = "vbz", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling sum of |VBz| over `window` hours. VBz = Speed * min(Bz, 0)
    is always <= 0 by construction (northward Bz clipped to 0), so its
    absolute value is a cumulative energy-input proxy over the window —
    the hourly analogue of the IMF lab's integrated-southward-Bz feature.
    """
    name = f"integrated_vbz_{window}h"
    df[name] = df[vbz_col].abs().rolling(window).sum()
    return [name]


def add_max_vbz(df: pd.DataFrame, vbz_col: str = "vbz", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Peak coupling strength over `window` hours. Since VBz <= 0 always
    (see add_integrated_vbz), the "maximum" (strongest) coupling
    corresponds to the most NEGATIVE raw value, i.e. a rolling minimum of
    the signed series — not a rolling maximum, which would just track
    how close VBz got to 0 (weakest coupling).
    """
    name = f"max_vbz_{window}h"
    df[name] = df[vbz_col].rolling(window).min()
    return [name]


def add_min_bz(df: pd.DataFrame, bz_col: str = "bz_gsm", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Most southward (most negative) Bz reading in the trailing `window`
    hours — how deep the strongest recent southward excursion was.
    """
    name = f"min_bz_{window}h"
    df[name] = df[bz_col].rolling(window).min()
    return [name]


def add_clock_angle(df: pd.DataFrame, by_col: str = "by_gsm", bz_col: str = "bz_gsm") -> list[str]:
    """IMF clock angle: atan2(By, Bz) in degrees, wrapped to [0, 360).
    0deg/360deg = purely northward, 180deg = purely southward (most
    geoeffective) — same definition as the IMF lab's minute-resolution
    version, computed here from the hourly-mean components instead.
    """
    name = "clock_angle_deg"
    angle = np.degrees(np.arctan2(df[by_col], df[bz_col]))
    df[name] = angle % 360
    return [name]


def add_clock_angle_change(df: pd.DataFrame, clock_angle_col: str = "clock_angle_deg") -> list[str]:
    """Hour-to-hour change in clock angle, as the shortest signed angular
    difference (a wrap from 359deg to 1deg reads as +2deg, not -358deg).
    """
    name = "clock_angle_change_deg"
    diff = df[clock_angle_col].diff()
    df[name] = (diff + 180) % 360 - 180
    return [name]


def add_storm_phase(df: pd.DataFrame, dst_col: str = "dst", threshold: float = STORM_DST_THRESHOLD_NT) -> list[str]:
    """Simplified, causal 3-state ordinal storm-phase classifier, derived
    purely from Dst's own trailing trajectory (no independent
    sudden-commencement flag is available in this dataset, so an
    "Initial Phase" state is deliberately not attempted):

      0 = Quiet     — dst > threshold (-20 nT default)
      1 = Main      — dst <= threshold and still falling or flat
                       (dst.diff() <= 0)
      2 = Recovery  — dst <= threshold and rising back toward baseline
                       (dst.diff() > 0)

    Uses only dst.diff() (backward-looking), so it is strictly causal —
    this is a plain rule-based heuristic in the same spirit as jobs.py's
    `_STORM_CHECKS`, not a validated storm-phase detection algorithm.
    """
    dst = df[dst_col]
    active = dst <= threshold
    change = dst.diff()
    phase = pd.Series(0, index=df.index, dtype="int64")
    phase[active & (change.fillna(-1.0) <= 0)] = 1
    phase[active & (change.fillna(-1.0) > 0)] = 2
    name = "storm_phase"
    df[name] = phase
    return [name]


def add_time_since_southward_turning(df: pd.DataFrame, bz_col: str = "bz_gsm") -> list[str]:
    """Hours elapsed since Bz most recently crossed from >= 0 to < 0 (a
    "southward turning" event), regardless of Bz's current sign — NaN
    before the first such event is observed in the frame.
    """
    bz = df[bz_col]
    turning = (bz < 0) & (bz.shift(1) >= 0)
    turning_ts = pd.Series(df.index, index=df.index).where(turning)
    last_turning_ts = turning_ts.ffill()
    name = "hours_since_southward_turn"
    df[name] = (pd.Series(df.index, index=df.index) - last_turning_ts).dt.total_seconds() / 3600.0
    return [name]


def add_previous_storm_strength(
    df: pd.DataFrame, dst_col: str = "dst", threshold: float = STORM_DST_THRESHOLD_NT
) -> list[str]:
    """The minimum Dst reached during the most recently COMPLETED storm
    episode (dst <= threshold), forward-filled until the next storm's own
    final minimum becomes known. Strictly causal: a storm's final minimum
    only becomes "known" the row immediately after that storm ends (once
    the following row confirms dst has risen back above threshold), never
    leaking a storm's own eventual depth into rows still inside it.
    NaN until the first storm in the frame has fully completed.
    """
    dst = df[dst_col]
    active = dst <= threshold
    storm_id = (active & ~active.shift(1, fill_value=False)).cumsum().where(active)
    running_min_in_storm = dst.groupby(storm_id).cummin()
    storm_final_min = running_min_in_storm.groupby(storm_id).transform("last")
    is_last_row_of_storm = active & ~active.shift(-1, fill_value=False)
    known_value = storm_final_min.where(is_last_row_of_storm)
    name = "previous_storm_strength"
    df[name] = known_value.shift(1).ffill()
    return [name]


def add_max_ae_6h(df: pd.DataFrame, ae_col: str = "ae", window: int = 6) -> list[str]:
    name = f"max_ae_{window}h"
    df[name] = df[ae_col].rolling(window).max()
    return [name]


def add_max_kp_24h(df: pd.DataFrame, kp_col: str = "kp", window: int = 24) -> list[str]:
    name = f"max_kp_{window}h"
    df[name] = df[kp_col].rolling(window).max()
    return [name]


def add_min_dst_24h(df: pd.DataFrame, dst_col: str = "dst", window: int = 24) -> list[str]:
    name = f"min_dst_{window}h"
    df[name] = df[dst_col].rolling(window).min()
    return [name]


def add_integrated_dynamic_pressure(
    df: pd.DataFrame, dp_col: str = "dynamic_pressure", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    name = f"integrated_dynamic_pressure_{window}h"
    df[name] = df[dp_col].rolling(window).sum()
    return [name]


# Display-name -> function registry, so the Research Lab UI can offer
# each physics feature as its own independent checkbox (per the "enable
# each physics feature individually" requirement) without the engine
# needing an if/elif per feature name.
PHYSICS_FEATURE_FUNCTIONS = {
    "Southward Duration": add_southward_duration,
    "Integrated Ey": add_integrated_ey,
    "Integrated VBz": add_integrated_vbz,
    "Maximum VBz": add_max_vbz,
    "Minimum Bz": add_min_bz,
    "Clock Angle": add_clock_angle,
    "Clock Angle Change": add_clock_angle_change,
    "Storm Phase": add_storm_phase,
    "Time Since Southward Turning": add_time_since_southward_turning,
    "Previous Storm Strength": add_previous_storm_strength,
    "Maximum AE (6h)": add_max_ae_6h,
    "Maximum Kp (24h)": add_max_kp_24h,
    "Minimum Dst (24h)": add_min_dst_24h,
    "Integrated Dynamic Pressure": add_integrated_dynamic_pressure,
}

# Clock Angle Change requires Clock Angle's own output column, so it must
# run after it whenever both are requested — PHYSICS_FEATURE_FUNCTIONS'
# insertion order already reflects this, and callers should preserve it
# rather than iterating in arbitrary/sorted order.
PHYSICS_FEATURE_DEPENDENCIES = {"Clock Angle Change": ["Clock Angle"]}
