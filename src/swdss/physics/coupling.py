"""Space Weather Physics Engine — solar wind/magnetosphere coupling functions.

Resolves three of the audit's six flagged inconsistencies:

- Newell Coupling Function: kp_physics_features.py used the full 3D IMF
  magnitude (Bt); ae_physics_features.py used only the Y-Z transverse
  component (sqrt(By^2+Bz^2)). Newell et al. (2007) explicitly define
  the coupling function in terms of the TRANSVERSE field component, not
  the total field magnitude — Bx (the radial, largely non-geoeffective
  component) is excluded by design in the original paper. AE's
  implementation was scientifically correct; Kp's has been migrated to
  match it. This changes the Kp Research Laboratory's Newell Coupling
  Function feature values for any run after this migration.

- Akasofu epsilon: kp_physics_features.py used the full SI-scaled form
  (l0=7 Earth radii folded into a Watts-scale constant); ae_physics_
  features.py used a bare proportional form (v*B^2*sin^4(theta/2), no
  constant, no physical units). The full SI form is adopted as canonical
  — it produces genuinely interpretable Watt-scale values matching
  published Akasofu epsilon figures (~10^10-10^12 W), which serves this
  project's documentation goals (expected range, scientific
  interpretation) far better than an arbitrary-scale index. This changes
  the AE Research Laboratory's Akasofu Epsilon Parameter feature values
  (and its Integrated Energy Input, which is built on it) for any run
  after this migration.

- Boyle Index: kp_physics_features.py clipped Bt >= 0 and took abs() of
  the sin^3 term; ae_physics_features.py did neither. Since Bt is a field
  magnitude (always >= 0 by construction) and sin(theta_c/2) is always
  >= 0 for theta_c in [0, 360) (theta_c/2 in [0, 180)), these guards are
  provably no-ops for any valid input — the two formulas are numerically
  identical for realistic data. Kp's more defensive version (protects
  against corrupted/out-of-range input rows) is adopted as canonical;
  this has no numerical effect on either lab's historical results for
  any row with valid data.
"""

import numpy as np
import pandas as pd

# Akasofu (1981) empirical scale length: l0 = 7 Earth radii, folded into
# the constant below so callers work in convenient solar-wind units
# (speed in km/s, Bt in nT) rather than raw SI.
AKASOFU_L0_RE = 7.0
AKASOFU_WATTS_CONSTANT = 1.5825e6  # derived from l0=7Re — see akasofu_epsilon_series


# ---------------------------------------------------------------------------
# Newell Coupling Function
# ---------------------------------------------------------------------------
# Definition: dPhi_MP/dt = v^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2),
#   where B_T = sqrt(By^2 + Bz^2) is the TRANSVERSE (Y-Z plane) IMF
#   component — not the total 3D field magnitude Bt.
# Units: used in its standard proportional form (v in km/s, B_T in nT) —
#   not scaled to a physical flux unit, matching this project's existing
#   VBz/Ey convention of using coupling functions as ML features.
# Scientific interpretation: the most widely used empirical solar-wind/
#   magnetosphere coupling function in the literature, originally
#   calibrated against AE, Dst, and other geomagnetic indices
#   simultaneously — a proxy for the rate of magnetic flux opened at the
#   dayside magnetopause via reconnection.
# Expected range: roughly 0-3000 (dimensionless proportional units) in
#   typical solar wind, higher during fast/strongly-southward driving.
# Reference: Newell, P. T., Sotirelis, T., Liou, K., Meng, C.-I., &
#   Rich, F. J. (2007), "A nearly universal solar wind-magnetosphere
#   coupling function inferred from 10 magnetospheric state variables",
#   J. Geophys. Res., 112, A01206.
# Used by: Kp/AE Research Laboratories.


def newell_coupling_series(speed: pd.Series, by: pd.Series, bz: pd.Series) -> pd.Series:
    bt_transverse = np.sqrt(by**2 + bz**2)
    clock_angle = np.arctan2(by, bz)
    return (speed ** (4 / 3)) * (bt_transverse ** (2 / 3)) * (np.abs(np.sin(clock_angle / 2)) ** (8 / 3))


# ---------------------------------------------------------------------------
# Akasofu epsilon parameter
# ---------------------------------------------------------------------------
# Definition: epsilon = v * B^2 * sin^4(theta_c/2) * l0^2 / mu0, with
#   l0 = 7 Earth radii (the standard empirical scale length). Collapsing
#   the SI constants for v[km/s] and B[nT] into one factor gives
#   epsilon[W] = 1.5825e6 * v * Bt^2 * sin^4(theta_c/2).
# Units: Watts.
# Scientific interpretation: an estimate of the total power the solar
#   wind couples into the magnetosphere-ionosphere system — the
#   quantity Akasofu's original work correlated against auroral/ring-
#   current dissipation.
# Expected range: roughly 10^9-10^10 W in quiet conditions, 10^11-10^12 W
#   or higher during geomagnetic storms.
# Reference: Akasofu, S.-I. (1981), "Energy coupling between the solar
#   wind and the magnetosphere", Space Sci. Rev., 28(2), 121-190.
# Used by: Kp/AE Research Laboratories.


def akasofu_epsilon_series(speed: pd.Series, bt: pd.Series, clock_angle_deg: pd.Series) -> pd.Series:
    theta_rad = np.radians(clock_angle_deg)
    return AKASOFU_WATTS_CONSTANT * speed.clip(lower=0) * bt.clip(lower=0) ** 2 * (np.sin(theta_rad / 2) ** 4)


# ---------------------------------------------------------------------------
# Boyle Index (polar cap potential)
# ---------------------------------------------------------------------------
# Definition: Phi_PC = 1e-4 * v^2 + 11.7 * Bt * sin^3(theta_c/2)
#   (v in km/s, Bt in nT).
# Units: kV.
# Scientific interpretation: a standard, widely-cited empirical proxy for
#   the cross-polar-cap potential driven by dayside reconnection.
# Expected range: roughly 20-60 kV in quiet conditions, 100+ kV during
#   strong driving.
# Reference: Boyle, C. B., Reiff, P. H., & Hairston, M. R. (1997),
#   "Empirical polar cap potentials", J. Geophys. Res., 102(A1), 111-125.
# Used by: Kp/AE Research Laboratories.


def boyle_index_series(speed: pd.Series, bt: pd.Series, clock_angle_deg: pd.Series) -> pd.Series:
    theta_rad = np.radians(clock_angle_deg)
    return 1e-4 * speed**2 + 11.7 * bt.clip(lower=0) * (np.abs(np.sin(theta_rad / 2)) ** 3)


# ---------------------------------------------------------------------------
# Integrated Energy Input
# ---------------------------------------------------------------------------
# Definition: rolling sum of Akasofu epsilon over `window` samples.
# Units: Watt-hours (or Watt-minutes, depending on the caller's row
#   cadence) — a genuine cumulative-energy quantity, since integrating a
#   Watts-rate over time gives energy. This is why Akasofu epsilon (an
#   actual power/energy-rate quantity) is the canonical thing to
#   integrate here, rather than Newell coupling (a flux-opening RATE,
#   not an energy quantity) — ae_physics_features.py previously
#   integrated Newell coupling instead; this migration changes its
#   Integrated Energy Input feature values.
# Scientific interpretation: cumulative energy input over the trailing
#   window — distinguishes a single deep driving spike from a longer,
#   shallower period that integrates to the same total.
# Reference: builds directly on Akasofu (1981) — see akasofu_epsilon_series.
# Used by: Kp/AE Research Laboratories.


def integrated_energy_input_series(akasofu_epsilon: pd.Series, window: int) -> pd.Series:
    return akasofu_epsilon.rolling(window).sum()
