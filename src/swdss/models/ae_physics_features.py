"""AE physics-informed feature engineering — AE Research Laboratory only.

Like kp_physics_features.py, this operates on the SAME HOURLY cadence as
`ae_analytics_features.csv` (production's own training data for the
standalone AE model) — AE has no live NOAA/DONKI feed at all, and the
only historical archive that exists anywhere in this codebase
(`data/processed/ae/ae_processed.parquet`) is itself hourly, confirmed
directly against the file (26,304 rows / 3 years, one row per hour). So
unlike the IMF lab's minute-native physics features, there is no finer
cadence being thrown away here — hourly is genuinely the best resolution
available for AE anywhere in this project.

Every function is strictly causal (rolling/shift-based, backward-looking
only) and takes a DataFrame indexed by hourly timestamp with whichever of
speed/density/temperature/bt/bx_gsm/by_gsm/bz_gsm/ae columns it needs,
returning the list of column names it created.

Every quantity here delegates to swdss.physics — the Physics Engine and
this project's single canonical implementation of these formulas —
instead of computing locally. This migration resolves two of the
audit's six flagged cross-lab inconsistencies, with a documented,
unavoidable consequence: rerunning these specific physics experiments
after this migration produces DIFFERENT feature values than before,
because this lab's old formula was scientifically off:

- Plasma Beta: previously used a single-step approximate formula with
  constant 4.16e-5; verified against a first-principles derivation
  (2*mu0*n*k*T/B^2 in the appropriate units), the correct constant is
  3.4699e-5 — a ~20% difference. The Kp Research Laboratory's two-step
  (Thermal Pressure / Magnetic Pressure) formulation, which already used
  the correct constant, is now the shared canonical implementation (see
  swdss.physics.plasma's module docstring for the verification).

- Akasofu epsilon: previously a bare proportional form (v*B^2*sin^4
  (theta/2), no physical units); now the full SI-scaled form (Watts),
  matching the Kp Research Laboratory's prior implementation — see
  swdss.physics.coupling's module docstring for why the SI form was
  adopted as canonical. Integrated Energy Input (built on Akasofu
  epsilon here) changes as a direct consequence.

Newell Coupling Function, Boyle Index, and Magnetic Shear were ALREADY
the scientifically canonical formulation in this lab (see swdss.physics.
coupling/geometry module docstrings) — no value change for those three.
Magnetic Pressure, Thermal Pressure, Total Pressure, and IMF Rotation
Rate are NEW additions to this lab (available in the Kp Research
Laboratory before this migration; now available here too, for
cross-lab consistency).
"""

import pandas as pd

from swdss.physics import core as physics_core
from swdss.physics import coupling as physics_coupling
from swdss.physics import geometry as physics_geometry
from swdss.physics import magnetosphere as physics_magnetosphere
from swdss.physics import persistence as physics_persistence
from swdss.physics import plasma as physics_plasma

DEFAULT_WINDOW_HOURS = 24


# ---------------------------------------------------------------- core derived physics
# (Ey / VBz / Dynamic Pressure / Clock Angle / Southward Duration /
# Integrated Southward Bz) — the "Derived Physics" feature-group columns,
# always computed unconditionally by ae_research.py (like Kp's
# ey/vbz/dynamic_pressure), not part of the opt-in registry below.


def add_ey(df: pd.DataFrame, speed_col: str = "speed", bz_col: str = "bz_gsm") -> list[str]:
    name = "ey"
    df[name] = physics_core.ey_series(df[speed_col], df[bz_col])
    return [name]


def add_vbz(df: pd.DataFrame, speed_col: str = "speed", bz_col: str = "bz_gsm") -> list[str]:
    name = "vbz"
    df[name] = physics_core.vbz_series(df[speed_col], df[bz_col])
    return [name]


def add_dynamic_pressure(df: pd.DataFrame, density_col: str = "density", speed_col: str = "speed") -> list[str]:
    name = "dynamic_pressure"
    df[name] = physics_core.dynamic_pressure_series(df[density_col], df[speed_col])
    return [name]


def add_clock_angle(df: pd.DataFrame, by_col: str = "by_gsm", bz_col: str = "bz_gsm") -> list[str]:
    name = "clock_angle_deg"
    df[name] = physics_geometry.clock_angle_series(df[by_col], df[bz_col])
    return [name]


def add_southward_duration(df: pd.DataFrame, bz_col: str = "bz_gsm") -> list[str]:
    name = "southward_duration_hr"
    df[name] = physics_core.southward_duration_series(df[bz_col])
    return [name]


def add_integrated_southward_bz(
    df: pd.DataFrame, bz_col: str = "bz_gsm", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    name = f"integrated_southward_bz_{window}h"
    df[name] = physics_core.integrated_southward_bz_series(df[bz_col], window)
    return [name]


def add_all_core_derived_physics(df: pd.DataFrame) -> list[str]:
    created = []
    created += add_ey(df)
    created += add_vbz(df)
    created += add_dynamic_pressure(df)
    created += add_clock_angle(df)
    created += add_southward_duration(df)
    created += add_integrated_southward_bz(df)
    return created


# ---------------------------------------------------------------- physics feature experiments
# Individually toggleable, more exotic coupling functions — the "Physics
# Feature Experiments" section, distinct from the always-on core group
# above.


def add_newell_coupling(
    df: pd.DataFrame, speed_col: str = "speed", by_col: str = "by_gsm", bz_col: str = "bz_gsm"
) -> list[str]:
    """Newell et al. 2007 (JGR) universal coupling function — see
    swdss.physics.coupling for the full reference and unit convention.
    """
    name = "newell_coupling"
    df[name] = physics_coupling.newell_coupling_series(df[speed_col], df[by_col], df[bz_col])
    return [name]


def add_akasofu_epsilon(
    df: pd.DataFrame, speed_col: str = "speed", bt_col: str = "bt", clock_angle_col: str = "clock_angle_deg"
) -> list[str]:
    """Akasofu 1981 epsilon parameter, in Watts (full SI-scaled form —
    see this module's docstring for why this changed from a bare
    proportional index). Requires Clock Angle first.
    """
    name = "akasofu_epsilon"
    df[name] = physics_coupling.akasofu_epsilon_series(df[speed_col], df[bt_col], df[clock_angle_col])
    return [name]


def add_boyle_index(
    df: pd.DataFrame, speed_col: str = "speed", bt_col: str = "bt", clock_angle_col: str = "clock_angle_deg"
) -> list[str]:
    """Boyle et al. 1998 polar cap potential empirical formula. Requires
    Clock Angle first.
    """
    name = "boyle_index"
    df[name] = physics_coupling.boyle_index_series(df[speed_col], df[bt_col], df[clock_angle_col])
    return [name]


def add_magnetic_pressure(df: pd.DataFrame, bt_col: str = "bt") -> list[str]:
    """Magnetic pressure Pb = Bt^2/(2*mu0), converted to nPa. New to this
    lab — previously only available in the Kp Research Laboratory.
    """
    name = "magnetic_pressure_npa"
    df[name] = physics_plasma.magnetic_pressure_series(df[bt_col])
    return [name]


def add_thermal_pressure(df: pd.DataFrame, density_col: str = "density", temperature_col: str = "temperature") -> list[str]:
    """Thermal pressure Pth = n*k*T, converted to nPa. New to this lab."""
    name = "thermal_pressure_npa"
    df[name] = physics_plasma.thermal_pressure_series(df[density_col], df[temperature_col])
    return [name]


def add_total_pressure(
    df: pd.DataFrame,
    dp_col: str = "dynamic_pressure",
    magnetic_col: str = "magnetic_pressure_npa",
    thermal_col: str = "thermal_pressure_npa",
) -> list[str]:
    """Sum of dynamic, magnetic, and thermal pressure. New to this lab.
    Requires Magnetic Pressure and Thermal Pressure first.
    """
    name = "total_pressure_npa"
    df[name] = physics_plasma.total_pressure_series(df[dp_col], df[magnetic_col], df[thermal_col])
    return [name]


def add_plasma_beta(df: pd.DataFrame, thermal_col: str = "thermal_pressure_npa", magnetic_col: str = "magnetic_pressure_npa") -> list[str]:
    """Solar wind plasma beta — see this module's docstring for the
    constant-discrepancy resolution. Requires Magnetic Pressure and
    Thermal Pressure first (a dependency this lab's Plasma Beta didn't
    have before this migration, since it used to be a single-step
    formula).
    """
    name = "plasma_beta"
    df[name] = physics_plasma.plasma_beta_series(df[thermal_col], df[magnetic_col])
    return [name]


def add_alfven_speed(df: pd.DataFrame, bt_col: str = "bt", density_col: str = "density") -> list[str]:
    """Alfven speed VA[km/s] = 21.8 * Bt[nT] / sqrt(n[cm^-3]). New to this
    lab as a standalone toggle — previously only computed internally,
    unexposed, by Alfven Mach Number below.
    """
    name = "alfven_speed_km_s"
    df[name] = physics_plasma.alfven_speed_series(df[bt_col], df[density_col])
    return [name]


def add_alfven_mach_number(df: pd.DataFrame, speed_col: str = "speed", alfven_col: str = "alfven_speed_km_s") -> list[str]:
    """Alfven Mach number M_A = v / v_A. Requires Alfven Speed first (a
    dependency this feature didn't have before, since it used to compute
    Alfven speed internally).
    """
    name = "alfven_mach_number"
    df[name] = physics_plasma.alfven_mach_number_series(df[speed_col], df[alfven_col])
    return [name]


def add_magnetopause_standoff(df: pd.DataFrame, bz_col: str = "bz_gsm", dp_col: str = "dynamic_pressure") -> list[str]:
    """Shue et al. 1998 empirical magnetopause standoff distance model."""
    name = "magnetopause_standoff_re"
    df[name] = physics_magnetosphere.magnetopause_standoff_series(df[bz_col], df[dp_col])
    return [name]


def add_estimated_compression(df: pd.DataFrame, standoff_col: str = "magnetopause_standoff_re") -> list[str]:
    """Percentage deviation of the magnetopause standoff distance from
    its nominal quiet-time value. New to this lab. Requires Magnetopause
    Stand-off Distance first.
    """
    name = "estimated_compression_pct"
    df[name] = physics_magnetosphere.estimated_compression_series(df[standoff_col])
    return [name]


def add_integrated_ey(df: pd.DataFrame, ey_col: str = "ey", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    name = f"integrated_ey_{window}h"
    df[name] = physics_core.integrated_ey_series(df[ey_col], window)
    return [name]


def add_integrated_vbz(df: pd.DataFrame, vbz_col: str = "vbz", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    name = f"integrated_vbz_{window}h"
    df[name] = physics_core.integrated_vbz_series(df[vbz_col], window)
    return [name]


def add_integrated_energy_input(
    df: pd.DataFrame, epsilon_col: str = "akasofu_epsilon", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """Rolling sum of Akasofu epsilon over `window` hours — a genuine
    cumulative-ENERGY quantity (Watt-hours), which is why Akasofu epsilon
    (an actual power/energy-rate quantity) is integrated here rather than
    Newell coupling (a flux-opening rate, not an energy quantity) — this
    lab previously integrated Newell coupling instead; see this module's
    docstring. Requires Akasofu Epsilon first (a dependency this feature
    didn't have before, since it used to recompute Newell internally).
    """
    name = f"integrated_energy_input_{window}h"
    df[name] = physics_coupling.integrated_energy_input_series(df[epsilon_col], window)
    return [name]


def add_clock_angle_change(df: pd.DataFrame, clock_angle_col: str = "clock_angle_deg") -> list[str]:
    """Hour-to-hour change in clock angle. New to this lab as a standalone
    toggle (previously only computed internally, unexposed, by this lab's
    now-removed Clock Angle Persistence). Requires Clock Angle first.
    """
    name = "clock_angle_change_deg"
    df[name] = physics_geometry.clock_angle_rate_series(df[clock_angle_col])
    return [name]


def add_imf_rotation_rate(
    df: pd.DataFrame, clock_angle_change_col: str = "clock_angle_change_deg", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """Rolling sum of |hour-to-hour clock angle change| over `window`
    hours. New to this lab — previously only available (mislabeled as
    "Magnetic Shear") in the Kp Research Laboratory; see
    kp_physics_features.py's docstring for that resolution. Requires
    Clock Angle Change first.
    """
    name = f"imf_rotation_rate_{window}h"
    df[name] = physics_geometry.imf_rotation_rate_series(df[clock_angle_change_col], window)
    return [name]


def add_magnetic_shear(df: pd.DataFrame, columns: tuple = ("bx_gsm", "by_gsm", "bz_gsm")) -> list[str]:
    """Magnitude of the total IMF vector's hour-to-hour rotation:
    sqrt(dBx^2 + dBy^2 + dBz^2) — an explicit, disclosed SIMPLIFICATION;
    see swdss.physics.geometry's module docstring.
    """
    name = "magnetic_shear"
    df[name] = physics_geometry.magnetic_shear_series(df[columns[0]], df[columns[1]], df[columns[2]])
    return [name]


def add_clock_angle_persistence(
    df: pd.DataFrame, clock_angle_col: str = "clock_angle_deg", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """Rolling standard deviation of clock angle over `window` hours — a
    LOW value means the IMF orientation has been stable/persistent.
    """
    name = f"clock_angle_persistence_{window}h_std"
    df[name] = df[clock_angle_col].rolling(window).std()
    return [name]


def add_strong_southward_duration(
    df: pd.DataFrame, bz_col: str = "bz_gsm", threshold: float = physics_core.STRONG_SOUTHWARD_THRESHOLD_NT
) -> list[str]:
    """Consecutive HOURS with Bz below -5 nT — a stricter threshold than
    the always-on Southward Duration core column, associated with more
    intense reconnection driving. New to this lab (previously only
    available in the Kp Research Laboratory); added specifically so the
    AE Optimization Study's Coupling Physics experiment can test it in
    isolation alongside the other 13 coupling variables.
    """
    name = "strong_southward_duration_hr"
    df[name] = physics_core.strong_southward_duration_series(df[bz_col], threshold)
    return [name]


def add_solar_wind_persistence(df: pd.DataFrame, speed_col: str = "speed", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling mean/std/max/min of Solar Wind Speed over `window` hours.
    Previously std-only in this lab; now the full stat set (a strict
    superset — the previously-available std column is unchanged, mean/
    max/min are new additions), matching the Kp Research Laboratory's
    own Solar Wind Persistence.
    """
    created = []
    stats = physics_persistence.persistence_stats_series(df[speed_col], window)
    for stat, series in stats.items():
        name = f"solar_wind_persistence_{window}h_{stat}"
        df[name] = series
        created.append(name)
    return created


# Display-name -> function registry for the opt-in "Physics Feature
# Experiments" section — each independently toggleable, mirroring
# kp_physics_features.py's PHYSICS_FEATURE_FUNCTIONS convention.
PHYSICS_FEATURE_FUNCTIONS = {
    "Newell Coupling Function": add_newell_coupling,
    "Akasofu Epsilon Parameter": add_akasofu_epsilon,
    "Boyle Index": add_boyle_index,
    "Magnetic Pressure": add_magnetic_pressure,
    "Thermal Pressure": add_thermal_pressure,
    "Total Pressure": add_total_pressure,
    "Plasma Beta": add_plasma_beta,
    "Alfven Speed": add_alfven_speed,
    "Alfven Mach Number": add_alfven_mach_number,
    "Magnetopause Stand-off Distance": add_magnetopause_standoff,
    "Estimated Compression": add_estimated_compression,
    "Integrated Ey": add_integrated_ey,
    "Integrated VBz": add_integrated_vbz,
    "Integrated Energy Input": add_integrated_energy_input,
    "Clock Angle Change": add_clock_angle_change,
    "Clock Angle Persistence": add_clock_angle_persistence,
    "IMF Rotation Rate": add_imf_rotation_rate,
    "Magnetic Shear": add_magnetic_shear,
    "Solar Wind Persistence": add_solar_wind_persistence,
    "Strong Southward Duration": add_strong_southward_duration,
}

# Every entry lists ALL transitively-required upstream features, in the
# exact order they must run — see kp_physics_features.py's identical
# note for why this resolver is one level deep by design.
PHYSICS_FEATURE_DEPENDENCIES = {
    "Akasofu Epsilon Parameter": [],  # uses clock_angle_deg, always present (core group)
    "Boyle Index": [],  # uses clock_angle_deg, always present (core group)
    "Total Pressure": ["Magnetic Pressure", "Thermal Pressure"],
    "Plasma Beta": ["Magnetic Pressure", "Thermal Pressure"],
    "Alfven Mach Number": ["Alfven Speed"],
    "Integrated Energy Input": ["Akasofu Epsilon Parameter"],
    "Clock Angle Persistence": [],  # uses clock_angle_deg, always present (core group)
    "IMF Rotation Rate": ["Clock Angle Change"],
    "Estimated Compression": ["Magnetopause Stand-off Distance"],
    "Strong Southward Duration": [],  # uses bz_gsm directly, no prerequisites
}
