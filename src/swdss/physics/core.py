"""Space Weather Physics Engine — core Sun-Earth coupling quantities.

Phase 0 of the Physics Engine migration extracted the three formulas
production already depended on — VBz, Ey, and Dynamic Pressure — byte-
for-byte from swdss.models.features.add_derived_physics_features. No
formula, constant, ordering, or intermediate computation was changed;
swdss.models.features delegates to this module rather than computing
these inline.

Phases 1-3 added the "Southward IMF" group below (Southward Duration,
Strong Southward Duration, Integrated Southward Bz, Integrated Ey,
Integrated VBz) — these were already byte-identical across the labs that
had them (IMF/Kp/AE), just at different cadences (minute vs hourly), so
each function here takes an explicit `window` parameter rather than
hard-coding a cadence, and is a pure consolidation with zero formula
changes anywhere.

Each core quantity is provided as TWO parallel functions where a scalar
caller exists in the codebase:

- a `*_series` function operating on pandas Series/DataFrame columns
  (used by training/research/dashboard code that already has a frame),
  written with the EXACT pandas call pattern production's own code used
  (e.g. `.clip(upper=0)`, not a numpy-based equivalent) — so migrating a
  caller onto this module changes zero arithmetic, only where the code
  lives.
- a `*_scalar` function operating on plain Python floats (used by
  single-reading callers like jobs.py's live snapshot), written with the
  EXACT Python builtin pattern that duplicated code already used (e.g.
  `min(bz, 0)`).
"""

import pandas as pd

# ---------------------------------------------------------------------------
# VBz — Burton et al. (1975) geoeffective coupling function
# ---------------------------------------------------------------------------
# Definition: VBz = Solar Wind Speed x min(Bz, 0)
# Units: nT.km/s
# Scientific interpretation: southward IMF (negative Bz) drives dayside
#   magnetic reconnection; energy injection into the ring current scales
#   with how fast the solar wind is moving the southward field past the
#   magnetopause. Northward (non-geoeffective) Bz is clipped to 0, so VBz
#   only ever spikes more negative — exactly when conditions are most
#   likely to drive a storm.
# Expected range: 0 (northward IMF, no coupling) down to roughly -5000 to
#   -10000+ nT.km/s during strong CME-driven southward IMF events.
# Reference: Burton, R. K., McPherron, R. L., & Russell, C. T. (1975),
#   "An empirical relationship between interplanetary conditions and Dst",
#   J. Geophys. Res., 80(31), 4204-4214.
# Used by: Analytics (Kp/Dst) production model, AE standalone production
#   model, Experimental cascade, IMF/Kp/AE Research Laboratories.


def vbz_series(speed: pd.Series, bz: pd.Series) -> pd.Series:
    return speed * bz.clip(upper=0)


def vbz_scalar(speed: float, bz: float) -> float:
    return speed * min(bz, 0)


# ---------------------------------------------------------------------------
# Ey — interplanetary dawn-dusk electric field
# ---------------------------------------------------------------------------
# Definition: Ey = -Solar Wind Speed x Bz x 1e-3
# Units: mV/m
# Scientific interpretation: the convective electric field the solar wind
#   carries, projected onto the dawn-dusk (Y) direction. Southward Bz
#   makes Ey positive — the sign convention used across space weather
#   literature for "geoeffective E-field" (positive Ey favors dayside
#   reconnection and ring current injection).
# Expected range: roughly -5 to +5 mV/m in quiet conditions, up to +15-20
#   mV/m or higher during strong southward IMF / CME sheath passage.
# Reference: standard interplanetary electric field definition used
#   throughout solar wind-magnetosphere coupling literature (e.g.
#   Kan & Lee, 1979, and the Burton et al. 1975 Dst-prediction framework
#   this project's VBz formula also derives from).
# Used by: Analytics (Kp/Dst) production model, AE standalone production
#   model, Experimental cascade, IMF/Kp/AE Research Laboratories,
#   the dashboard's Physics Interpretation narrative panel.


def ey_series(speed: pd.Series, bz: pd.Series) -> pd.Series:
    return -speed * bz * 1e-3


def ey_scalar(speed: float, bz: float) -> float:
    return -speed * bz * 1e-3


# ---------------------------------------------------------------------------
# Dynamic Pressure — solar wind ram pressure on the magnetopause
# ---------------------------------------------------------------------------
# Definition: Pdyn = 1.6726e-6 x Density x Speed^2
# Units: nPa
# Scientific interpretation: the ram (dynamic) pressure the solar wind
#   plasma exerts on the magnetopause, assuming a pure-proton plasma
#   (1.6726e-6 folds in the proton mass and the cm^-3/km/s -> nPa unit
#   conversion in one constant). A sudden jump in dynamic pressure (e.g.
#   at an interplanetary shock) compresses the magnetosphere and is a
#   classic sudden-commencement trigger.
# Expected range: roughly 1-3 nPa in typical solar wind, up to 10-20+ nPa
#   during a strong CME sheath or shock compression.
# Reference: standard solar wind ram pressure formula (proton-mass-only
#   approximation, ignoring alpha particle contribution to mass density),
#   used throughout the space weather literature and operationally by
#   NOAA SWPC.
# Used by: Analytics (Kp/Dst) production model, AE standalone production
#   model, Experimental cascade, IMF/Kp/AE Research Laboratories, the
#   dashboard's Heliosphere > Dynamic Pressure panel and Physics
#   Interpretation narrative panel.


def dynamic_pressure_series(density: pd.Series, speed: pd.Series) -> pd.Series:
    return 1.6726e-6 * density * speed**2


def dynamic_pressure_scalar(density: float, speed: float) -> float:
    return 1.6726e-6 * density * speed**2


def add_derived_physics_features(df: pd.DataFrame) -> list[str]:
    """Adds VBz, Ey, Dynamic Pressure onto `df` in place, in the exact
    same conditional-gating and column-order as
    swdss.models.features.add_derived_physics_features did before this
    extraction — each is only added when its required inputs (speed,
    bz_gsm, density) are present, a no-op for datasets missing any of
    them (e.g. standalone Solar Wind has no Bz; standalone IMF has no
    Speed/Density).

    This is the single production-facing entry point: swdss.models.
    features.add_derived_physics_features now delegates to this function
    — see that module's docstring for the production byte-identical
    contract this preserves.
    """
    created = []
    has_speed = "speed" in df.columns
    has_bz = "bz_gsm" in df.columns
    has_density = "density" in df.columns

    if has_speed and has_bz:
        df["vbz"] = vbz_series(df["speed"], df["bz_gsm"])
        created.append("vbz")
        df["ey"] = ey_series(df["speed"], df["bz_gsm"])
        created.append("ey")

    if has_speed and has_density:
        df["dynamic_pressure"] = dynamic_pressure_series(df["density"], df["speed"])
        created.append("dynamic_pressure")

    return created


def _consecutive_true_run_length(condition: pd.Series) -> pd.Series:
    """For a boolean series, returns at each row how many consecutive
    rows up to and including it have been True — resets to 0 the instant
    the condition is False. The "streak length" pattern behind both
    southward-duration functions below.
    """
    reset_groups = (~condition).cumsum()
    return condition.groupby(reset_groups).cumsum()


# ---------------------------------------------------------------------------
# Southward Duration
# ---------------------------------------------------------------------------
# Definition: consecutive samples with Bz < 0 (southward, geoeffective
#   orientation), reset to 0 the instant Bz turns northward.
# Units: samples (minutes or hours, depending on the caller's row cadence).
# Scientific interpretation: a long streak means sustained reconnection-
#   favorable driving, not just a momentary dip — distinguishes a brief
#   Bz excursion from prolonged southward IMF.
# Reference: standard consecutive-run-length pattern used throughout this
#   project's research laboratories.
# Used by: IMF/Kp/AE Research Laboratories.


def southward_duration_series(bz: pd.Series) -> pd.Series:
    return _consecutive_true_run_length(bz < 0)


# ---------------------------------------------------------------------------
# Strong Southward Duration
# ---------------------------------------------------------------------------
# Definition: consecutive samples with Bz below `threshold` (default -5 nT).
# Units: samples.
# Scientific interpretation: a stricter threshold than plain Southward
#   Duration, associated with more intense reconnection driving.
# Reference: same consecutive-run-length pattern, stricter threshold.
# Used by: IMF/Kp/AE Research Laboratories.

STRONG_SOUTHWARD_THRESHOLD_NT = -5.0


def strong_southward_duration_series(bz: pd.Series, threshold: float = STRONG_SOUTHWARD_THRESHOLD_NT) -> pd.Series:
    return _consecutive_true_run_length(bz < threshold)


# ---------------------------------------------------------------------------
# Integrated Southward Bz
# ---------------------------------------------------------------------------
# Definition: rolling sum of |Bz| restricted to its southward (negative)
#   part, over `window` samples.
# Units: nT, accumulated over the window.
# Scientific interpretation: a cumulative driving-energy proxy — a single
#   deep southward spike and a longer, shallower southward stretch can
#   integrate to the same total, which an instantaneous Bz reading alone
#   can't distinguish but this can.
# Reference: same rolling-sum-of-clipped-value pattern used throughout
#   this project's research laboratories.
# Used by: IMF/Kp/AE Research Laboratories.


def integrated_southward_bz_series(bz: pd.Series, window: int) -> pd.Series:
    return bz.clip(upper=0).abs().rolling(window).sum()


# ---------------------------------------------------------------------------
# Integrated Ey
# ---------------------------------------------------------------------------
# Definition: rolling sum of the geoeffective (positive) part of Ey over
#   `window` samples.
# Units: mV/m, accumulated over the window.
# Scientific interpretation: Ey is positive exactly when Bz is southward
#   (this project's sign convention), so clipping to the positive part
#   before summing integrates only the geoeffective driving.
# Reference: builds on Ey — see ey_series above.
# Used by: Kp/AE Research Laboratories.


def integrated_ey_series(ey: pd.Series, window: int) -> pd.Series:
    return ey.clip(lower=0).rolling(window).sum()


# ---------------------------------------------------------------------------
# Integrated VBz
# ---------------------------------------------------------------------------
# Definition: rolling sum of |VBz| over `window` samples.
# Units: nT.km/s, accumulated over the window.
# Scientific interpretation: VBz <= 0 always by construction (northward
#   Bz clipped to 0), so its absolute value integrated over time is a
#   cumulative energy-input proxy.
# Reference: builds on VBz — see vbz_series above.
# Used by: Kp/AE Research Laboratories.


def integrated_vbz_series(vbz: pd.Series, window: int) -> pd.Series:
    return vbz.abs().rolling(window).sum()
