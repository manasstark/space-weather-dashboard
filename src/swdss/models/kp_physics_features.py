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
storm/quiet segmentation. These, and other Kp-lab-specific composite/
threshold features (Maximum VBz, Minimum Bz, Time Since Southward
Turning, Previous Storm Strength, Maximum AE/Kp, Minimum Dst, Integrated
Dynamic Pressure), are NOT part of the Physics Engine's canonical
quantity list — they stay local to this lab.

Every other function here delegates to swdss.physics — the Physics
Engine and this project's single canonical implementation of these
formulas — instead of computing them locally. Two of the audit's six
flagged cross-lab inconsistencies are resolved by this migration, with a
documented, unavoidable consequence: rerunning these specific physics
experiments after this migration produces DIFFERENT feature values than
before, because the old Kp-lab formula was scientifically wrong (Newell
Coupling) or was one of two genuinely different quantities sharing one
name (Magnetic Shear):

- Newell Coupling Function: previously used the full 3D IMF magnitude
  (Bt); now uses only the transverse (Y-Z plane) component, matching
  Newell et al. (2007)'s actual published definition (see
  swdss.physics.coupling's module docstring). Feature values for any
  Kp Research Laboratory run using "Newell Coupling Function" after this
  migration differ from runs before it.

- Magnetic Shear: previously computed a rolling accumulation of clock
  angle change (a rotation-RATE-accumulation quantity); this was a
  mislabeling — that quantity is now correctly named "IMF Rotation Rate"
  (a new, additional entry in this lab's registry, so the old behavior
  remains available under its correct name). "Magnetic Shear" itself now
  computes what its name actually means here: the vector magnitude of
  the IMF's hour-to-hour change (matching the AE Research Laboratory's
  own Magnetic Shear, which was already using this definition — see
  swdss.physics.geometry's module docstring). Feature values for any Kp
  Research Laboratory run using "Magnetic Shear" after this migration
  differ from runs before it; the pre-migration quantity is still
  available under the new "IMF Rotation Rate" entry.

Akasofu epsilon, Boyle Index, and Plasma Beta were ALREADY the
scientifically canonical formulation in this lab (see swdss.physics.
coupling/plasma module docstrings for the verification) — no value
change for those three.
"""

import pandas as pd

from swdss.physics import core as physics_core
from swdss.physics import coupling as physics_coupling
from swdss.physics import geometry as physics_geometry
from swdss.physics import magnetosphere as physics_magnetosphere
from swdss.physics import persistence as physics_persistence
from swdss.physics import plasma as physics_plasma

DEFAULT_WINDOW_HOURS = 24
STORM_DST_THRESHOLD_NT = -20.0
STRONG_SOUTHWARD_THRESHOLD_NT = -5.0


def _consecutive_true_run_length(condition: pd.Series) -> pd.Series:
    reset_groups = (~condition).cumsum()
    return condition.groupby(reset_groups).cumsum()


# ---------------------------------------------------------------------------
# Physics Engine-backed features
# ---------------------------------------------------------------------------


def add_southward_duration(df: pd.DataFrame, bz_col: str = "bz_gsm") -> list[str]:
    """Consecutive HOURS with Bz < 0 (southward), reset to 0 the instant
    Bz turns northward — the hourly analogue of the IMF lab's
    minute-resolution southward-duration feature.
    """
    name = "southward_duration_hr"
    df[name] = physics_core.southward_duration_series(df[bz_col])
    return [name]


def add_strong_southward_duration(
    df: pd.DataFrame, bz_col: str = "bz_gsm", threshold: float = STRONG_SOUTHWARD_THRESHOLD_NT
) -> list[str]:
    """Consecutive HOURS with Bz below -5 nT — a stricter threshold than
    plain southward-duration, associated with more intense reconnection
    driving. Hourly analogue of the IMF lab's minute-resolution version.
    """
    name = "strong_southward_duration_hr"
    df[name] = physics_core.strong_southward_duration_series(df[bz_col], threshold)
    return [name]


def add_integrated_southward_bz(
    df: pd.DataFrame, bz_col: str = "bz_gsm", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """Rolling sum of |Bz| restricted to its southward part over `window`
    hours — a cumulative driving-energy proxy distinct from
    Integrated Ey/VBz (those weight by solar wind speed too; this is
    Bz's own magnitude alone). Hourly analogue of the IMF lab's version.
    """
    name = f"integrated_southward_bz_{window}h"
    df[name] = physics_core.integrated_southward_bz_series(df[bz_col], window)
    return [name]


def add_integrated_ey(df: pd.DataFrame, ey_col: str = "ey", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling sum of the geoeffective (positive) part of Ey over `window`
    hours — Ey is positive exactly when Bz is southward under this
    project's sign convention, so clipping to the positive part before
    summing integrates only the geoeffective driving.
    """
    name = f"integrated_ey_{window}h"
    df[name] = physics_core.integrated_ey_series(df[ey_col], window)
    return [name]


def add_integrated_vbz(df: pd.DataFrame, vbz_col: str = "vbz", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling sum of |VBz| over `window` hours — VBz <= 0 always by
    construction, so its absolute value is a cumulative energy-input
    proxy over the window.
    """
    name = f"integrated_vbz_{window}h"
    df[name] = physics_core.integrated_vbz_series(df[vbz_col], window)
    return [name]


def add_clock_angle(df: pd.DataFrame, by_col: str = "by_gsm", bz_col: str = "bz_gsm") -> list[str]:
    """IMF clock angle: atan2(By, Bz) in degrees, wrapped to [0, 360).
    0deg/360deg = purely northward, 180deg = purely southward (most
    geoeffective) — same definition as the IMF lab's minute-resolution
    version, computed here from the hourly-mean components instead.
    """
    name = "clock_angle_deg"
    df[name] = physics_geometry.clock_angle_series(df[by_col], df[bz_col])
    return [name]


def add_clock_angle_change(df: pd.DataFrame, clock_angle_col: str = "clock_angle_deg") -> list[str]:
    """Hour-to-hour change in clock angle, as the shortest signed angular
    difference (a wrap from 359deg to 1deg reads as +2deg, not -358deg).
    """
    name = "clock_angle_change_deg"
    df[name] = physics_geometry.clock_angle_rate_series(df[clock_angle_col])
    return [name]


def add_magnetic_shear(df: pd.DataFrame, columns: tuple = ("bx_gsm", "by_gsm", "bz_gsm")) -> list[str]:
    """Vector magnitude of the total IMF vector's hour-to-hour rotation:
    sqrt(dBx^2 + dBy^2 + dBz^2) — matches the AE Research Laboratory's
    own Magnetic Shear (this lab previously computed a different
    quantity under this name; see this module's docstring for the
    resolution — the pre-migration behavior is now available under
    "IMF Rotation Rate" below).
    """
    name = "magnetic_shear"
    df[name] = physics_geometry.magnetic_shear_series(df[columns[0]], df[columns[1]], df[columns[2]])
    return [name]


def add_imf_rotation_rate(
    df: pd.DataFrame, clock_angle_change_col: str = "clock_angle_change_deg", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """Rolling sum of |hour-to-hour clock angle change| over `window`
    hours — a proxy for how much the IMF orientation has rotated
    recently. This is what this lab called "Magnetic Shear" before this
    migration; see this module's docstring for why it was renamed.
    Requires Clock Angle Change first.
    """
    name = f"imf_rotation_rate_{window}h"
    df[name] = physics_geometry.imf_rotation_rate_series(df[clock_angle_change_col], window)
    return [name]


def add_magnetic_pressure(df: pd.DataFrame, bt_col: str = "bt") -> list[str]:
    """Magnetic pressure Pb = Bt^2 / (2*mu0), converted to nPa."""
    name = "magnetic_pressure_npa"
    df[name] = physics_plasma.magnetic_pressure_series(df[bt_col])
    return [name]


def add_thermal_pressure(df: pd.DataFrame, density_col: str = "density", temp_col: str = "temperature") -> list[str]:
    """Thermal pressure Pth = n*k*T, converted to nPa."""
    name = "thermal_pressure_npa"
    df[name] = physics_plasma.thermal_pressure_series(df[density_col], df[temp_col])
    return [name]


def add_total_pressure(
    df: pd.DataFrame,
    dp_col: str = "dynamic_pressure",
    magnetic_col: str = "magnetic_pressure_npa",
    thermal_col: str = "thermal_pressure_npa",
) -> list[str]:
    """Sum of dynamic (ram), magnetic, and thermal pressure. Requires
    Magnetic Pressure and Thermal Pressure first.
    """
    name = "total_pressure_npa"
    df[name] = physics_plasma.total_pressure_series(df[dp_col], df[magnetic_col], df[thermal_col])
    return [name]


def add_plasma_beta(
    df: pd.DataFrame, thermal_col: str = "thermal_pressure_npa", magnetic_col: str = "magnetic_pressure_npa"
) -> list[str]:
    """Plasma beta = thermal pressure / magnetic pressure. Requires
    Magnetic Pressure and Thermal Pressure first.
    """
    name = "plasma_beta"
    df[name] = physics_plasma.plasma_beta_series(df[thermal_col], df[magnetic_col])
    return [name]


def add_alfven_speed(df: pd.DataFrame, bt_col: str = "bt", density_col: str = "density") -> list[str]:
    """Alfvén speed VA[km/s] = 21.8 * Bt[nT] / sqrt(n[cm^-3])."""
    name = "alfven_speed_km_s"
    df[name] = physics_plasma.alfven_speed_series(df[bt_col], df[density_col])
    return [name]


def add_alfven_mach_number(df: pd.DataFrame, speed_col: str = "speed", alfven_col: str = "alfven_speed_km_s") -> list[str]:
    """Solar wind bulk speed divided by the local Alfvén speed. Requires
    Alfvén Speed first.
    """
    name = "alfven_mach_number"
    df[name] = physics_plasma.alfven_mach_number_series(df[speed_col], df[alfven_col])
    return [name]


def add_newell_coupling(
    df: pd.DataFrame, speed_col: str = "speed", by_col: str = "by_gsm", bz_col: str = "bz_gsm"
) -> list[str]:
    """Newell et al. (2007) universal coupling function, using the
    transverse (Y-Z plane) IMF component per the published definition
    — see this module's docstring for why this changed from the full 3D
    Bt this lab previously (incorrectly) used. No longer requires Clock
    Angle as a prerequisite — computed directly from By/Bz.
    """
    name = "newell_coupling"
    df[name] = physics_coupling.newell_coupling_series(df[speed_col], df[by_col], df[bz_col])
    return [name]


def add_akasofu_epsilon(
    df: pd.DataFrame, speed_col: str = "speed", bt_col: str = "bt", clock_angle_col: str = "clock_angle_deg"
) -> list[str]:
    """Akasofu (1981) epsilon parameter, in Watts. Requires Clock Angle first."""
    name = "akasofu_epsilon_w"
    df[name] = physics_coupling.akasofu_epsilon_series(df[speed_col], df[bt_col], df[clock_angle_col])
    return [name]


def add_boyle_index(
    df: pd.DataFrame, speed_col: str = "speed", bt_col: str = "bt", clock_angle_col: str = "clock_angle_deg"
) -> list[str]:
    """Boyle et al. (1997) empirical polar cap potential estimate, in kV.
    Requires Clock Angle first.
    """
    name = "boyle_polar_cap_potential_kv"
    df[name] = physics_coupling.boyle_index_series(df[speed_col], df[bt_col], df[clock_angle_col])
    return [name]


def add_integrated_energy_input(
    df: pd.DataFrame, epsilon_col: str = "akasofu_epsilon_w", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """Rolling sum of Akasofu epsilon over `window` hours — cumulative
    energy input (Watt-hours). Requires Akasofu Epsilon first.
    """
    name = f"integrated_energy_input_{window}h"
    df[name] = physics_coupling.integrated_energy_input_series(df[epsilon_col], window)
    return [name]


def add_magnetopause_standoff(
    df: pd.DataFrame, bz_col: str = "bz_gsm", dp_col: str = "dynamic_pressure"
) -> list[str]:
    """Shue et al. (1998) empirical subsolar magnetopause stand-off
    distance, in Earth radii. Pdyn <= 0 is undefined and produces NaN.
    """
    name = "magnetopause_standoff_re"
    df[name] = physics_magnetosphere.magnetopause_standoff_series(df[bz_col], df[dp_col])
    return [name]


def add_estimated_compression(df: pd.DataFrame, standoff_col: str = "magnetopause_standoff_re") -> list[str]:
    """Percentage deviation of the magnetopause standoff distance from
    its nominal quiet-time value. Requires Magnetopause Stand-off
    Distance first.
    """
    name = "estimated_compression_pct"
    df[name] = physics_magnetosphere.estimated_compression_series(df[standoff_col])
    return [name]


def add_solar_wind_persistence(df: pd.DataFrame, speed_col: str = "speed", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling mean/std/max/min of solar wind speed over `window` hours."""
    created = []
    stats = physics_persistence.persistence_stats_series(df[speed_col], window)
    for stat, series in stats.items():
        name = f"sw_persistence_{window}h_{stat}"
        df[name] = series
        created.append(name)
    return created


def add_imf_persistence(
    df: pd.DataFrame, columns: tuple = ("bx_gsm", "by_gsm", "bz_gsm"), window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """A compact IMF-orientation steadiness index over `window` hours."""
    name = f"imf_persistence_{window}h_variability"
    df[name] = physics_persistence.imf_persistence_series(df[columns[0]], df[columns[1]], df[columns[2]], window)
    return [name]


def add_bz_persistence(df: pd.DataFrame, bz_col: str = "bz_gsm", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling mean/std/max/min of Bz over `window` hours."""
    created = []
    stats = physics_persistence.persistence_stats_series(df[bz_col], window)
    for stat, series in stats.items():
        name = f"bz_persistence_{window}h_{stat}"
        df[name] = series
        created.append(name)
    return created


def add_bt_persistence(df: pd.DataFrame, bt_col: str = "bt", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling mean/std/max/min of Bt over `window` hours."""
    created = []
    stats = physics_persistence.persistence_stats_series(df[bt_col], window)
    for stat, series in stats.items():
        name = f"bt_persistence_{window}h_{stat}"
        df[name] = series
        created.append(name)
    return created


# ---------------------------------------------------------------------------
# Kp-lab-specific features — NOT part of the Physics Engine's canonical
# quantity list (composite/threshold heuristics specific to Kp's own
# Storm Phase framing, or simple rolling extrema not shared across labs).
# ---------------------------------------------------------------------------


def add_max_vbz(df: pd.DataFrame, vbz_col: str = "vbz", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Peak coupling strength over `window` hours. Since VBz <= 0 always,
    the "maximum" (strongest) coupling corresponds to the most NEGATIVE
    raw value, i.e. a rolling minimum of the signed series.
    """
    name = f"max_vbz_{window}h"
    df[name] = df[vbz_col].rolling(window).min()
    return [name]


def add_min_bz(df: pd.DataFrame, bz_col: str = "bz_gsm", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Most southward (most negative) Bz reading in the trailing `window` hours."""
    name = f"min_bz_{window}h"
    df[name] = df[bz_col].rolling(window).min()
    return [name]


def add_storm_phase(df: pd.DataFrame, dst_col: str = "dst", threshold: float = STORM_DST_THRESHOLD_NT) -> list[str]:
    """Simplified, causal 3-state ordinal storm-phase classifier, derived
    purely from Dst's own trailing trajectory:

      0 = Quiet     — dst > threshold (-20 nT default)
      1 = Main      — dst <= threshold and still falling or flat
      2 = Recovery  — dst <= threshold and rising back toward baseline

    Strictly causal — a plain rule-based heuristic, not a validated
    storm-phase detection algorithm.
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
    """Hours elapsed since Bz most recently crossed from >= 0 to < 0."""
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
    episode, forward-filled until the next storm's own final minimum
    becomes known. Strictly causal.
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
    "Strong Southward Duration": add_strong_southward_duration,
    "Integrated Southward Bz": add_integrated_southward_bz,
    "Magnetic Pressure": add_magnetic_pressure,
    "Thermal Pressure": add_thermal_pressure,
    "Total Pressure": add_total_pressure,
    "Plasma Beta": add_plasma_beta,
    "Alfvén Speed": add_alfven_speed,
    "Alfvén Mach Number": add_alfven_mach_number,
    "Newell Coupling Function": add_newell_coupling,
    "Akasofu ε": add_akasofu_epsilon,
    "Boyle Index": add_boyle_index,
    "Magnetopause Stand-off Distance": add_magnetopause_standoff,
    "Estimated Compression": add_estimated_compression,
    "Magnetic Shear": add_magnetic_shear,
    "IMF Rotation Rate": add_imf_rotation_rate,
    "Integrated Energy Input": add_integrated_energy_input,
    "Solar Wind Persistence": add_solar_wind_persistence,
    "IMF Persistence": add_imf_persistence,
    "Bz Persistence": add_bz_persistence,
    "Bt Persistence": add_bt_persistence,
}

# Every entry lists ALL transitively-required upstream features, in the
# exact order they must run — load_kp_research_frame's dependency
# resolver is one level deep (it does not itself recurse), so a feature
# whose prerequisite has its own prerequisite must flatten the full
# chain here itself. Keeping the full chain explicit here, rather than
# teaching the resolver to recurse, keeps that resolver simple and this
# the single source of truth for "what does X actually need computed
# first" for every feature in the registry.
PHYSICS_FEATURE_DEPENDENCIES = {
    "Clock Angle Change": ["Clock Angle"],
    "Akasofu ε": ["Clock Angle"],
    "Boyle Index": ["Clock Angle"],
    "Total Pressure": ["Magnetic Pressure", "Thermal Pressure"],
    "Plasma Beta": ["Magnetic Pressure", "Thermal Pressure"],
    "Alfvén Mach Number": ["Alfvén Speed"],
    "IMF Rotation Rate": ["Clock Angle", "Clock Angle Change"],
    "Integrated Energy Input": ["Clock Angle", "Akasofu ε"],
    "Estimated Compression": ["Magnetopause Stand-off Distance"],
}
