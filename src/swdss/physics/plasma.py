"""Space Weather Physics Engine — solar wind plasma pressure/regime quantities.

Resolves the audit's Plasma Beta inconsistency: kp_physics_features.py
computed beta as (thermal pressure / magnetic pressure) using two
separately-derived pressure constants (net effective constant
~3.470e-5); ae_physics_features.py used a single-step approximate
formula with constant 4.16e-5 — a ~20% discrepancy for identical inputs.

Verified by first-principles derivation: beta = 2*mu0*n*k*T / B^2, with
n in cm^-3 -> m^-3 (x1e6), B in nT -> T (x1e-9), gives a constant of
3.46995e-5 — matching the Kp lab's constant (3.47005e-5, from its own
two-step Pth/Pb formula) to 4 significant figures. Kp's formula was
scientifically correct; AE's was not. Kp's two-step form (via
Magnetic/Thermal Pressure, exposed here as their own reusable functions)
is adopted as canonical. This changes the AE Research Laboratory's
Plasma Beta feature values for any run after this migration.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Magnetic Pressure
# ---------------------------------------------------------------------------
# Definition: P_B = Bt^2 / (2*mu0), converted to nPa.
# Formula: P_B[nPa] = 3.9789e-4 * Bt[nT]^2
# Units: nPa.
# Scientific interpretation: the pressure the IMF's own magnetic field
#   exerts, part of the total pressure balance (with thermal and ram
#   pressure) the magnetopause responds to.
# Expected range: roughly 0.001-0.05 nPa for typical Bt of 2-10 nT —
#   an order of magnitude smaller than typical dynamic pressure, which
#   dominates the solar wind's pressure balance (consistent with plasma
#   beta typically being of order 1, not << 1).
# Reference: standard magnetic pressure formula (B^2/2*mu0) with the
#   nT-to-nPa unit conversion folded into the constant.
# Used by: Kp Research Laboratory (previously AE lacked this quantity
#   as a standalone feature — now available to it too).


def magnetic_pressure_series(bt: pd.Series) -> pd.Series:
    return 3.9789e-4 * bt**2


# ---------------------------------------------------------------------------
# Thermal Pressure
# ---------------------------------------------------------------------------
# Definition: P_th = n*k*T, converted to nPa.
# Formula: P_th[nPa] = 1.3807e-8 * n[cm^-3] * T[K]
# Units: nPa.
# Scientific interpretation: the solar wind plasma's own thermal
#   pressure — part of the total pressure balance.
# Expected range: roughly 0.001-0.02 nPa for typical solar wind density
#   (~5 cm^-3) and temperature (~1e5 K).
# Reference: standard ideal-gas thermal pressure formula (n*k*T) with
#   the cm^-3/K-to-nPa unit conversion folded into the constant.
# Used by: Kp Research Laboratory.


def thermal_pressure_series(density: pd.Series, temperature: pd.Series) -> pd.Series:
    return 1.3807e-8 * density * temperature


# ---------------------------------------------------------------------------
# Total Pressure
# ---------------------------------------------------------------------------
# Definition: sum of dynamic (ram), magnetic, and thermal pressure.
# Units: nPa.
# Scientific interpretation: the full pressure balance the magnetopause
#   responds to — dynamic pressure dominates under typical conditions,
#   but magnetic/thermal pressure become more significant in low-speed,
#   high-density, or strong-field solar wind.
# Reference: sum of the three component pressures defined above and in
#   swdss.physics.core.dynamic_pressure_series.
# Used by: Kp Research Laboratory.


def total_pressure_series(dynamic_pressure: pd.Series, magnetic_pressure: pd.Series, thermal_pressure: pd.Series) -> pd.Series:
    return dynamic_pressure + magnetic_pressure + thermal_pressure


# ---------------------------------------------------------------------------
# Plasma Beta
# ---------------------------------------------------------------------------
# Definition: beta = P_thermal / P_magnetic.
# Units: dimensionless.
# Scientific interpretation: beta >> 1 means the solar wind's dynamics
#   are thermally (not magnetically) dominated; beta ~ 1 is typical for
#   ambient solar wind; beta << 1 (magnetically dominated) is associated
#   with structures like magnetic clouds/flux ropes within CMEs.
# Expected range: roughly 0.1-10 in typical solar wind, often < 1 inside
#   a CME's magnetic cloud.
# Reference: standard plasma beta definition; see this module's
#   docstring for the first-principles verification of the constant used.
# Used by: Kp/AE Research Laboratories.


def plasma_beta_series(thermal_pressure: pd.Series, magnetic_pressure: pd.Series) -> pd.Series:
    denom = magnetic_pressure.replace(0.0, np.nan)
    return thermal_pressure / denom


# ---------------------------------------------------------------------------
# Alfven Speed
# ---------------------------------------------------------------------------
# Definition: V_A = Bt / sqrt(mu0 * rho), reduced to convenient solar
#   wind units for a proton-mass-only plasma.
# Formula: V_A[km/s] = 21.8 * Bt[nT] / sqrt(n[cm^-3])
# Units: km/s.
# Scientific interpretation: the characteristic speed at which magnetic
#   disturbances propagate through the plasma — the reference speed for
#   Alfven Mach Number below.
# Expected range: roughly 30-80 km/s in typical solar wind.
# Reference: standard practical Alfven speed formula for a pure-proton
#   plasma (ignoring alpha particle contribution to mass density).
# Used by: Kp/AE Research Laboratories.


def alfven_speed_series(bt: pd.Series, density: pd.Series) -> pd.Series:
    denom = np.sqrt(density).replace(0.0, np.nan)
    return 21.8 * bt / denom


# ---------------------------------------------------------------------------
# Alfven Mach Number
# ---------------------------------------------------------------------------
# Definition: M_A = v / V_A.
# Units: dimensionless.
# Scientific interpretation: how super-Alfvenic the driving solar wind
#   flow is — relevant to bow shock formation and the efficiency of
#   energy transfer at the magnetopause.
# Expected range: roughly 4-10 in typical solar wind.
# Reference: standard Alfven Mach number definition.
# Used by: Kp/AE Research Laboratories.


def alfven_mach_number_series(speed: pd.Series, alfven_speed: pd.Series) -> pd.Series:
    denom = alfven_speed.replace(0.0, np.nan)
    return speed / denom
