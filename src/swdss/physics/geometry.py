"""Space Weather Physics Engine — IMF field geometry.

Clock Angle and Clock Angle Rate were previously implemented identically
(same formula) in imf_physics_features.py, kp_physics_features.py,
ae_physics_features.py, imf_research.py, and dashboard/home.py — this
module is their single canonical home.

Magnetic Shear and IMF Rotation Rate resolve one of the audit's six
flagged inconsistencies: kp_physics_features.py's "Magnetic Shear" and
ae_physics_features.py's "Magnetic Shear" computed two genuinely
different quantities under the same name (Kp: rolling-accumulated clock
angle change; AE: single-step vector magnitude of B-component change).
Rather than picking a "winner" and discarding the other, both are kept
as two correctly-and-distinctly-named canonical quantities:

- Magnetic Shear = the AE lab's definition (vector magnitude of the
  field's hour-to-hour change) — a proxy for how much the field
  STRUCTURE changed.
- IMF Rotation Rate = the Kp lab's former "Magnetic Shear" definition,
  correctly renamed — a measure of how much the field DIRECTION rotated
  over a window, which is what it actually measures.

Documented consequence: the Kp Research Laboratory's "Magnetic Shear"
physics-feature toggle now computes what this module calls IMF Rotation
Rate (the underlying numbers are unchanged — only the name it should be
understood by is corrected). See kp_physics_features.py's migration note.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Clock Angle
# ---------------------------------------------------------------------------
# Definition: theta_c = atan2(By, Bz), wrapped to [0, 360)
# Units: degrees
# Scientific interpretation: the IMF's orientation in the GSM Y-Z plane,
#   independent of field magnitude. 0deg/360deg = purely northward (least
#   geoeffective, magnetosphere shielded); 180deg = purely southward (most
#   geoeffective, favors dayside reconnection).
# Expected range: [0, 360) degrees by construction.
# Reference: standard IMF clock angle definition used throughout solar
#   wind-magnetosphere coupling literature (e.g. Newell et al. 2007,
#   Boyle et al. 1997, and this project's own Akasofu epsilon/Boyle Index).
# Used by: IMF/Kp/AE Research Laboratories, dashboard Geospace scatter
#   plots, Physics Interpretation narrative panel.


def clock_angle_series(by: pd.Series, bz: pd.Series) -> pd.Series:
    return np.degrees(np.arctan2(by, bz)) % 360


def clock_angle_scalar(by: float, bz: float) -> float:
    return np.degrees(np.arctan2(by, bz)) % 360


# ---------------------------------------------------------------------------
# Clock Angle Rate (a.k.a. Clock Angle Change)
# ---------------------------------------------------------------------------
# Definition: shortest signed angular difference between consecutive
#   clock-angle readings — (delta + 180) % 360 - 180, so a wrap from
#   359deg to 1deg reads as +2deg, not -358deg.
# Units: degrees per sample interval (per minute or per hour, depending
#   on the caller's row cadence).
# Scientific interpretation: how fast the IMF orientation is rotating —
#   a large value signals an evolving/rotating field structure (e.g. a
#   CME flux rope passing), even when the instantaneous clock angle
#   alone looks unremarkable.
# Expected range: roughly -180 to +180 degrees per sample; typically
#   single digits in steady solar wind, larger during flux-rope passage.
# Reference: standard shortest-angular-difference convention; same
#   derivative-of-clock-angle concept used across this project's
#   research laboratories.
# Used by: IMF/Kp/AE Research Laboratories.


def clock_angle_rate_series(clock_angle: pd.Series) -> pd.Series:
    diff = clock_angle.diff()
    return (diff + 180) % 360 - 180


# ---------------------------------------------------------------------------
# Magnetic Shear
# ---------------------------------------------------------------------------
# Definition: |dB| = sqrt(dBx^2 + dBy^2 + dBz^2), the vector magnitude of
#   the sample-to-sample change in the full 3-component IMF vector.
# Units: nT per sample interval.
# Scientific interpretation: an explicit, disclosed SIMPLIFICATION —
#   "magnetic shear" properly refers to the angle between the IMF and
#   the magnetospheric field at the magnetopause, which requires a
#   magnetospheric field model this project doesn't have. This is a
#   proxy for "how rapidly is the field vector changing", not a true
#   shear-angle calculation.
# Expected range: near 0 nT in steady solar wind; several nT per sample
#   during turbulent/CME-sheath conditions.
# Reference: disclosed proxy, same self-aware-heuristic spirit as this
#   project's Storm Phase classifier — not a first-principles shear
#   calculation.
# Used by: AE Research Laboratory (previously its own local copy).


def magnetic_shear_series(bx: pd.Series, by: pd.Series, bz: pd.Series) -> pd.Series:
    return np.sqrt(bx.diff() ** 2 + by.diff() ** 2 + bz.diff() ** 2)


# ---------------------------------------------------------------------------
# IMF Rotation Rate
# ---------------------------------------------------------------------------
# Definition: rolling sum of |Clock Angle Rate| over `window` samples.
# Units: degrees, accumulated over the window.
# Scientific interpretation: how much the IMF orientation has rotated in
#   total over the trailing window — distinct from Clock Angle Rate
#   (the instantaneous per-sample rotation speed); this is the
#   accumulated rotation, capturing sustained tumbling that a single-step
#   rate could miss if it oscillates back and forth.
# Expected range: near 0 degrees for a steady field; can exceed 360
#   degrees (multiple full rotations) during a strongly rotating flux
#   rope passage over a long enough window.
# Reference: disclosed proxy built from the standard clock-angle-rate
#   definition above — this is the corrected name for what
#   kp_physics_features.py previously called "Magnetic Shear" (see this
#   module's docstring for why that name was wrong and this one is right).
# Used by: Kp Research Laboratory (previously mislabeled "Magnetic Shear").


def imf_rotation_rate_series(clock_angle_rate: pd.Series, window: int) -> pd.Series:
    return clock_angle_rate.abs().rolling(window).sum()
