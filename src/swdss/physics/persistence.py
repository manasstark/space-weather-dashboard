"""Space Weather Physics Engine — rolling persistence/steadiness quantities.

Resolves the audit's Solar Wind Persistence partial-duplication:
kp_physics_features.py computed 4 rolling stats (mean/std/max/min);
ae_physics_features.py computed only 1 (std). One canonical, parameterized
function (persistence_stats_series) now computes all 4 for any column at
any window — every caller gets the full stat set. This is a strict
superset for AE (its "std"-only column is still available, just alongside
mean/max/min it didn't have before), so this migration only ADDS feature
options, it never removes or changes an existing one.

A single parameterized function replaces what were four near-identical
per-quantity implementations (Solar Wind, Bt, Bz, IMF Persistence) across
two labs — same formula shape, only the target column differs.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Persistence Stats (Solar Wind / Bt / Bz Persistence)
# ---------------------------------------------------------------------------
# Definition: rolling mean, std, max, and min of a given column over
#   `window` samples.
# Units: same units as the input column.
# Scientific interpretation: distinguishes a persistently strong/fast/
#   steady driver (high rolling min, low std) from a brief spike riding
#   on an otherwise weak background (high rolling max, high std) — two
#   physically different situations that share the same instantaneous
#   reading.
# Reference: standard rolling-window steadiness/persistence pattern used
#   throughout this project's research laboratories.
# Used by: IMF/Kp/AE Research Laboratories, for Solar Wind Speed, Bt, and
#   Bz persistence specifically.


def persistence_stats_series(column: pd.Series, window: int) -> dict[str, pd.Series]:
    rolling = column.rolling(window)
    return {
        "mean": rolling.mean(),
        "std": rolling.std(),
        "max": rolling.max(),
        "min": rolling.min(),
    }


# ---------------------------------------------------------------------------
# IMF Persistence (multi-component orientation steadiness)
# ---------------------------------------------------------------------------
# Definition: the mean of each GSM component's (Bx, By, Bz) own rolling
#   standard deviation over `window` samples, collapsed to one column.
# Units: nT.
# Scientific interpretation: a compact IMF-orientation steadiness index —
#   intentionally multi-component and collapsed (unlike the single-
#   column Bt/Bz Persistence above), measuring overall field-DIRECTION
#   steadiness rather than any one component's magnitude level.
# Reference: same "persistence via rolling std" pattern as Solar
#   Wind/Bt/Bz Persistence above, applied across all three IMF components.
# Used by: Kp Research Laboratory.


def imf_persistence_series(bx: pd.Series, by: pd.Series, bz: pd.Series, window: int) -> pd.Series:
    stds = pd.concat([bx.rolling(window).std(), by.rolling(window).std(), bz.rolling(window).std()], axis=1)
    return stds.mean(axis=1)
