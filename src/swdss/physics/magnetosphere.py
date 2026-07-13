"""Space Weather Physics Engine — magnetosphere boundary quantities."""

import numpy as np
import pandas as pd

# Nominal quiet-time subsolar standoff distance, in Earth radii — the
# Shue et al. (1998) formula's OWN output under canonical quiet
# conditions (Bz=0, Pdyn=2 nPa): (10.22 + 1.29*tanh(0.184*8.14)) *
# 2^(-1/6.6) ~= 10.25 Re. Used as the reference point for Estimated
# Compression below so that metric is self-consistent with the model
# it's built on, rather than an arbitrary external constant.
NOMINAL_STANDOFF_RE = (10.22 + 1.29 * np.tanh(0.184 * 8.14)) * 2 ** (-1 / 6.6)


# ---------------------------------------------------------------------------
# Magnetopause Stand-off Distance
# ---------------------------------------------------------------------------
# Definition: R0 = (10.22 + 1.29*tanh(0.184*(Bz+8.14))) * Pdyn^(-1/6.6)
#   (Bz in nT, Pdyn in nPa).
# Units: Earth radii (Re).
# Scientific interpretation: the empirical subsolar magnetopause
#   distance — how far from Earth the magnetopause boundary sits along
#   the Sun-Earth line. Compresses (R0 decreases) under strong dynamic
#   pressure or southward Bz.
# Expected range: roughly 8-11 Re in typical conditions, can compress to
#   6 Re or less during extreme events (occasionally inside geosynchronous
#   orbit at 6.6 Re).
# Reference: Shue, J.-H., et al. (1998), "Magnetopause location under
#   extreme solar wind conditions", J. Geophys. Res., 103(A8), 17691-17700.
# Used by: Kp/AE Research Laboratories.


def magnetopause_standoff_series(bz: pd.Series, dynamic_pressure: pd.Series) -> pd.Series:
    pdyn_safe = dynamic_pressure.where(dynamic_pressure > 0)
    return (10.22 + 1.29 * np.tanh(0.184 * (bz + 8.14))) * pdyn_safe ** (-1 / 6.6)


# ---------------------------------------------------------------------------
# Estimated Compression
# ---------------------------------------------------------------------------
# Definition: percentage deviation of the current magnetopause standoff
#   distance from its nominal quiet-time value (NOMINAL_STANDOFF_RE,
#   the Shue et al. formula's own output under Bz=0/Pdyn=2nPa).
# Formula: Compression[%] = (NOMINAL_STANDOFF_RE - R0) / NOMINAL_STANDOFF_RE * 100
# Units: percent.
# Scientific interpretation: a derived convenience metric (not itself a
#   named quantity from a single canonical paper) built on top of the
#   Shue et al. magnetopause model, expressing how compressed the
#   magnetosphere currently is relative to typical quiet conditions.
#   Positive = more compressed than typical (R0 smaller than nominal);
#   negative = more expanded/relaxed than typical.
# Expected range: roughly -10% to +10% in normal conditions, +30% or
#   more during extreme magnetopause compression events.
# Reference: derived from Shue et al. (1998) — see
#   magnetopause_standoff_series above.
# Used by: Kp/AE Research Laboratories.


def estimated_compression_series(magnetopause_standoff: pd.Series) -> pd.Series:
    return (NOMINAL_STANDOFF_RE - magnetopause_standoff) / NOMINAL_STANDOFF_RE * 100
