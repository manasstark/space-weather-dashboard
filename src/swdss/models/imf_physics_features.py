"""IMF physics-informed feature engineering — Research Laboratory only.

Deliberately separate from swdss.models.features (the production feature
pipeline), and NOT interchangeable with it: production trains on HOURLY-
resampled data (matching Kp/Dst's own cadence for the combined model),
while every feature here operates on the RAW MINUTE-LEVEL IMF data. Bz's
physics genuinely lives at minute timescales — "how many consecutive
minutes has Bz been southward" or "60-minute LSTM sequences" only mean
something at that resolution; resampling to hourly first would destroy
exactly the information these features exist to capture. This divergence
is intentional and documented, not an oversight — see imf_research.py's
module docstring for the full research-vs-production data contract.

All functions take a DataFrame indexed by minute-level timestamp_utc with
at least a "bz_gsm" column (others as needed per function) and add new
columns in place, returning the list of column names created — same
convention as swdss.models.features, for a consistent calling style.

Southward Duration, Strong Southward Duration, Integrated Southward Bz,
Clock Angle, Clock Angle Rate, and Bt Persistence delegate to
swdss.physics — the Physics Engine and this project's single canonical
implementation of these formulas — parameterized to this lab's native
minute cadence. Magnetic Rotation (raw per-component deltas) stays
local: it isn't one of the canonical quantities migrated into the engine.
"""

import pandas as pd

from swdss.physics import core as physics_core
from swdss.physics import geometry as physics_geometry
from swdss.physics import persistence as physics_persistence

# Minute-cadence analogs of the production pipeline's hour-cadence lag
# set ([1,3,6,12,24]h) and rolling window (24h) — adapted to this
# pipeline's native minute resolution rather than reused verbatim, since
# a "1h lag" at hourly cadence and a "1-minute lag" at minute cadence
# serve the same purpose (let the model see a recent-past value) at
# timescales appropriate to each pipeline.
MINUTE_LAGS = [1, 5, 15, 30, 60]
MINUTE_ROLLING_WINDOW = 60
SOUTHWARD_INTEGRATION_WINDOWS = (15, 30, 60)
STRONG_SOUTHWARD_THRESHOLD_NT = -5.0


def add_minute_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int] = MINUTE_LAGS) -> list[str]:
    # Built as a dict and assigned once rather than one df[name] = ... per
    # lag — with len(columns) x len(lags) columns, per-column assignment
    # fragments the frame badly enough for pandas to warn about it.
    new_cols = {}
    for column in columns:
        for lag in lags:
            new_cols[f"{column}_lag{lag}m"] = df[column].shift(lag)
    df[list(new_cols)] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols)


def add_minute_rolling_features(
    df: pd.DataFrame, columns: list[str], window: int = MINUTE_ROLLING_WINDOW
) -> list[str]:
    new_cols = {}
    for column in columns:
        new_cols[f"{column}_{window}m"] = df[column].rolling(window).mean()
        new_cols[f"{column}_{window}m_std"] = df[column].rolling(window).std()
    df[list(new_cols)] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols)


def add_minute_change_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    new_cols = {f"{column}_change": df[column].diff() for column in columns}
    df[list(new_cols)] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols)


def add_southward_duration(df: pd.DataFrame, bz_col: str = "bz_gsm") -> list[str]:
    """1. Southward Duration — consecutive minutes with Bz < 0 (southward,
    geoeffective orientation), reset to 0 the instant Bz turns northward.
    A long streak means sustained reconnection-favorable driving, not
    just a momentary dip.
    """
    name = "southward_duration_min"
    df[name] = physics_core.southward_duration_series(df[bz_col])
    return [name]


def add_strong_southward_duration(df: pd.DataFrame, bz_col: str = "bz_gsm") -> list[str]:
    """2. Strong Southward Duration — consecutive minutes with
    Bz < STRONG_SOUTHWARD_THRESHOLD_NT (-5 nT), a stricter threshold
    associated with more intense reconnection than a bare Bz < 0 crossing.
    """
    name = "strong_southward_duration_min"
    df[name] = physics_core.strong_southward_duration_series(df[bz_col], STRONG_SOUTHWARD_THRESHOLD_NT)
    return [name]


def add_integrated_southward_bz(
    df: pd.DataFrame, bz_col: str = "bz_gsm", windows: tuple = SOUTHWARD_INTEGRATION_WINDOWS
) -> list[str]:
    """3. Integrated Southward Bz — rolling sum of |Bz| restricted to its
    southward (negative) part, over 15/30/60-minute windows. This is a
    cumulative energy-input proxy in the spirit of VBz/Burton et al.:
    a single deep southward spike and a longer, shallower southward
    stretch can integrate to the same total, which a simple instantaneous
    Bz reading can't distinguish but this can.
    """
    new_cols = {
        f"integrated_southward_bz_{window}m": physics_core.integrated_southward_bz_series(df[bz_col], window)
        for window in windows
    }
    df[list(new_cols)] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols)


def add_magnetic_rotation(
    df: pd.DataFrame, columns: tuple = ("bx_gsm", "by_gsm", "bz_gsm")
) -> list[str]:
    """4. Magnetic Rotation — ΔBx, ΔBy, ΔBz, the minute-to-minute change
    in each GSM component. A rapidly rotating field (large deltas across
    all three) indicates a structurally complex or turbulent IMF, as
    opposed to a steady, slowly-evolving field.
    """
    new_cols = {f"d{column}": df[column].diff() for column in columns}
    df[list(new_cols)] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols)


def add_clock_angle(df: pd.DataFrame, by_col: str = "by_gsm", bz_col: str = "bz_gsm") -> list[str]:
    """5. Magnetic Direction (IMF clock angle) — atan2(By, Bz) in degrees,
    wrapped to [0, 360). 0°/360° is purely northward (least geoeffective),
    180° is purely southward (most geoeffective); the angle tracks how
    the field's orientation in the GSM Y-Z plane evolves, independent of
    field magnitude.
    """
    name = "clock_angle_deg"
    df[name] = physics_geometry.clock_angle_series(df[by_col], df[bz_col])
    return [name]


def add_clock_angle_rate(df: pd.DataFrame, clock_angle_col: str = "clock_angle_deg") -> list[str]:
    """6. Clock Angle Rate of Change — first derivative of clock angle,
    computed as the shortest signed angular difference between
    consecutive minutes (so a wrap from 359° to 1° reads as +2°, not
    -358°). A fast-changing clock angle signals an evolving/rotating
    field structure (e.g. passing through a CME flux rope), even when
    Bz's raw value alone looks unremarkable.
    """
    name = "clock_angle_rate_deg"
    df[name] = physics_geometry.clock_angle_rate_series(df[clock_angle_col])
    return [name]


def add_bt_persistence(df: pd.DataFrame, bt_col: str = "bt", window: int = MINUTE_ROLLING_WINDOW) -> list[str]:
    """7. Magnetic Field Magnitude Persistence — rolling mean/std/max/min
    of Bt (total field strength) over `window` minutes. Distinguishes a
    persistently strong field (high rolling min, low std) from a brief
    spike riding on an otherwise weak background (high rolling max,
    high std) — two very different physical situations that share the
    same instantaneous Bt reading.
    """
    stats = physics_persistence.persistence_stats_series(df[bt_col], window)
    new_cols = {f"bt_persistence_{window}m_{stat}": series for stat, series in stats.items()}
    df[list(new_cols)] = pd.DataFrame(new_cols, index=df.index)
    return list(new_cols)


def add_all_imf_physics_features(df: pd.DataFrame) -> list[str]:
    """Adds every physics feature above onto `df` in place (minute-level,
    columns bt/bx_gsm/by_gsm/bz_gsm expected). Returns the full list of
    created column names, in the same order as the spec they implement.
    """
    created = []
    created += add_southward_duration(df)
    created += add_strong_southward_duration(df)
    created += add_integrated_southward_bz(df)
    created += add_magnetic_rotation(df)
    created += add_clock_angle(df)
    created += add_clock_angle_rate(df)
    created += add_bt_persistence(df)
    return created
