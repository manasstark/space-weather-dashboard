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

Several formulas here (Akasofu epsilon, Boyle Index, Plasma Beta, Alfven
Mach Number) are standard SOLAR WIND practical/empirical formulas using
commonly-cited approximate constants (not first-principles derivations),
and Magnetic Shear/Clock Angle Persistence/Solar Wind Persistence are
explicit, disclosed proxies (no magnetopause field model exists in this
project) — same self-disclosed-heuristic spirit as kp_physics_features.py's
Storm Phase classifier. Citations are given per function.
"""

import numpy as np
import pandas as pd

DEFAULT_WINDOW_HOURS = 24
EARTH_RADIUS_KM = 6371.0


def _clock_angle_rad(df: pd.DataFrame, clock_angle_col: str = "clock_angle_deg") -> pd.Series:
    return np.radians(df[clock_angle_col])


# ---------------------------------------------------------------- core derived physics
# (Ey / VBz / Dynamic Pressure / Clock Angle / Southward Duration /
# Integrated Southward Bz) — the "Derived Physics" feature-group columns,
# always computed unconditionally by ae_research.py (like Kp's
# ey/vbz/dynamic_pressure), not part of the opt-in registry below.


def add_ey(df: pd.DataFrame, speed_col: str = "speed", bz_col: str = "bz_gsm") -> list[str]:
    name = "ey"
    df[name] = -df[speed_col] * df[bz_col] * 1e-3
    return [name]


def add_vbz(df: pd.DataFrame, speed_col: str = "speed", bz_col: str = "bz_gsm") -> list[str]:
    name = "vbz"
    df[name] = df[speed_col] * df[bz_col].clip(upper=0)
    return [name]


def add_dynamic_pressure(df: pd.DataFrame, density_col: str = "density", speed_col: str = "speed") -> list[str]:
    name = "dynamic_pressure"
    df[name] = 1.6726e-6 * df[density_col] * df[speed_col] ** 2
    return [name]


def add_clock_angle(df: pd.DataFrame, by_col: str = "by_gsm", bz_col: str = "bz_gsm") -> list[str]:
    name = "clock_angle_deg"
    angle = np.degrees(np.arctan2(df[by_col], df[bz_col]))
    df[name] = angle % 360
    return [name]


def add_southward_duration(df: pd.DataFrame, bz_col: str = "bz_gsm") -> list[str]:
    name = "southward_duration_hr"
    condition = df[bz_col] < 0
    reset_groups = (~condition).cumsum()
    df[name] = condition.groupby(reset_groups).cumsum()
    return [name]


def add_integrated_southward_bz(
    df: pd.DataFrame, bz_col: str = "bz_gsm", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    name = f"integrated_southward_bz_{window}h"
    df[name] = df[bz_col].clip(upper=0).abs().rolling(window).sum()
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
    """Newell et al. 2007 (JGR) universal coupling function:
    dPhi/dt ~ v^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2), the most widely
    used empirical solar-wind/magnetosphere coupling function in the
    literature (originally calibrated against AE, Dst, and other
    geomagnetic indices simultaneously) — used here in its standard
    proportional form (v in km/s, B_T in nT), not scaled to a physical
    unit system, matching this project's existing VBz/Ey convention of
    using coupling functions as ML features rather than SI-exact outputs.
    """
    name = "newell_coupling"
    bt_yz = np.sqrt(df[by_col] ** 2 + df[bz_col] ** 2)
    clock_angle = np.arctan2(df[by_col], df[bz_col])
    df[name] = (df[speed_col] ** (4 / 3)) * (bt_yz ** (2 / 3)) * (np.abs(np.sin(clock_angle / 2)) ** (8 / 3))
    return [name]


def add_akasofu_epsilon(
    df: pd.DataFrame, speed_col: str = "speed", bt_col: str = "bt", clock_angle_col: str = "clock_angle_deg"
) -> list[str]:
    """Akasofu 1981 epsilon parameter, in its standard proportional form
    (v * B^2 * sin^4(theta_c/2)) — the l0^2 and 4*pi/mu_0 physical
    constants are dropped, same convention as Newell coupling above,
    since an ML model absorbs constant scale factors anyway.
    """
    name = "akasofu_epsilon"
    theta = _clock_angle_rad(df, clock_angle_col)
    df[name] = df[speed_col] * df[bt_col] ** 2 * np.sin(theta / 2) ** 4
    return [name]


def add_boyle_index(
    df: pd.DataFrame, speed_col: str = "speed", bt_col: str = "bt", clock_angle_col: str = "clock_angle_deg"
) -> list[str]:
    """Boyle et al. 1998 polar cap potential empirical formula:
    Phi_PC [kV] = 1e-4 * v^2 + 11.7 * Bt * sin^3(theta_c/2)
    (v in km/s, Bt in nT) — a standard, widely-cited empirical proxy for
    the cross-polar-cap potential driven by dayside reconnection.
    """
    name = "boyle_index"
    theta = _clock_angle_rad(df, clock_angle_col)
    df[name] = 1e-4 * df[speed_col] ** 2 + 11.7 * df[bt_col] * np.sin(theta / 2) ** 3
    return [name]


def add_plasma_beta(df: pd.DataFrame, density_col: str = "density", temperature_col: str = "temperature", bt_col: str = "bt") -> list[str]:
    """Solar wind plasma beta (ratio of thermal to magnetic pressure),
    using the standard practical approximate formula
    beta = 4.16e-5 * n * T / Bt^2 (n in cm^-3, T in K, Bt in nT) commonly
    used in solar wind physics — an approximation, not an exact
    first-principles calculation (ignores electron pressure and any
    non-Maxwellian corrections).
    """
    name = "plasma_beta"
    df[name] = 4.16e-5 * df[density_col] * df[temperature_col] / (df[bt_col] ** 2)
    return [name]


def add_alfven_mach_number(df: pd.DataFrame, speed_col: str = "speed", density_col: str = "density", bt_col: str = "bt") -> list[str]:
    """Alfven Mach number M_A = v / v_A, using the standard practical
    formula for solar-wind Alfven speed v_A[km/s] = 21.8 * Bt[nT] /
    sqrt(n[cm^-3]) (proton-mass-only approximation, ignoring alpha
    particle contribution to mass density).
    """
    name = "alfven_mach_number"
    alfven_speed = 21.8 * df[bt_col] / np.sqrt(df[density_col].clip(lower=1e-6))
    df[name] = df[speed_col] / alfven_speed
    return [name]


def add_magnetopause_standoff(df: pd.DataFrame, bz_col: str = "bz_gsm", dp_col: str = "dynamic_pressure") -> list[str]:
    """Shue et al. 1998 empirical magnetopause standoff distance model:
    R0 [R_E] = (10.22 + 1.29*tanh(0.184*(Bz+8.14))) * Dp^(-1/6.6)
    (Bz in nT, Dp in nPa) — the standard, widely-used empirical
    magnetopause location model (this is the full model, not a
    simplified proportional form, since it's already well-calibrated
    for direct use as-is).
    """
    name = "magnetopause_standoff_re"
    dp_safe = df[dp_col].clip(lower=1e-6)
    df[name] = (10.22 + 1.29 * np.tanh(0.184 * (df[bz_col] + 8.14))) * dp_safe ** (-1 / 6.6)
    return [name]


def add_integrated_ey(df: pd.DataFrame, ey_col: str = "ey", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    name = f"integrated_ey_{window}h"
    df[name] = df[ey_col].clip(lower=0).rolling(window).sum()
    return [name]


def add_integrated_vbz(df: pd.DataFrame, vbz_col: str = "vbz", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    name = f"integrated_vbz_{window}h"
    df[name] = df[vbz_col].abs().rolling(window).sum()
    return [name]


def add_integrated_energy_input(
    df: pd.DataFrame,
    speed_col: str = "speed",
    by_col: str = "by_gsm",
    bz_col: str = "bz_gsm",
    window: int = DEFAULT_WINDOW_HOURS,
) -> list[str]:
    """Rolling sum of the Newell coupling function over `window` hours —
    a cumulative energy-input proxy. Computes Newell coupling internally
    (rather than requiring add_newell_coupling to have already run) so
    this is usable regardless of whether "Newell Coupling Function" is
    separately enabled as its own feature.
    """
    name = f"integrated_energy_input_{window}h"
    bt_yz = np.sqrt(df[by_col] ** 2 + df[bz_col] ** 2)
    clock_angle = np.arctan2(df[by_col], df[bz_col])
    newell = (df[speed_col] ** (4 / 3)) * (bt_yz ** (2 / 3)) * (np.abs(np.sin(clock_angle / 2)) ** (8 / 3))
    df[name] = newell.rolling(window).sum()
    return [name]


def add_clock_angle_persistence(
    df: pd.DataFrame, clock_angle_col: str = "clock_angle_deg", window: int = DEFAULT_WINDOW_HOURS
) -> list[str]:
    """Rolling standard deviation of clock angle over `window` hours — a
    LOW value means the IMF orientation has been stable/persistent; a
    HIGH value means the field has been rotating/tumbling. Same
    "persistence via rolling std" pattern as the IMF lab's Bt Persistence.
    """
    name = f"clock_angle_persistence_{window}h_std"
    df[name] = df[clock_angle_col].rolling(window).std()
    return [name]


def add_magnetic_shear(df: pd.DataFrame, columns: tuple = ("bx_gsm", "by_gsm", "bz_gsm")) -> list[str]:
    """Magnitude of the total IMF vector's hour-to-hour rotation:
    sqrt(dBx^2 + dBy^2 + dBz^2). An explicit, disclosed SIMPLIFICATION —
    "magnetic shear" properly refers to the angle between the IMF and
    the magnetospheric field at the magnetopause, which requires a
    magnetospheric field model this project doesn't have. This is a
    proxy for "how rapidly is the field direction/structure changing",
    not a true shear-angle calculation.
    """
    name = "magnetic_shear"
    squared_sum = sum(df[c].diff() ** 2 for c in columns)
    df[name] = np.sqrt(squared_sum)
    return [name]


def add_solar_wind_persistence(df: pd.DataFrame, speed_col: str = "speed", window: int = DEFAULT_WINDOW_HOURS) -> list[str]:
    """Rolling standard deviation of Solar Wind Speed over `window` hours
    — a LOW value means steady driving conditions; a HIGH value means
    turbulent/variable solar wind (e.g. during a CME passage).
    """
    name = f"solar_wind_persistence_{window}h_std"
    df[name] = df[speed_col].rolling(window).std()
    return [name]


# Display-name -> function registry for the opt-in "Physics Feature
# Experiments" section — each independently toggleable, mirroring
# kp_physics_features.py's PHYSICS_FEATURE_FUNCTIONS convention.
PHYSICS_FEATURE_FUNCTIONS = {
    "Newell Coupling Function": add_newell_coupling,
    "Akasofu Epsilon Parameter": add_akasofu_epsilon,
    "Boyle Index": add_boyle_index,
    "Plasma Beta": add_plasma_beta,
    "Alfven Mach Number": add_alfven_mach_number,
    "Magnetopause Stand-off Distance": add_magnetopause_standoff,
    "Integrated Ey": add_integrated_ey,
    "Integrated VBz": add_integrated_vbz,
    "Integrated Energy Input": add_integrated_energy_input,
    "Clock Angle Persistence": add_clock_angle_persistence,
    "Magnetic Shear": add_magnetic_shear,
    "Solar Wind Persistence": add_solar_wind_persistence,
}
