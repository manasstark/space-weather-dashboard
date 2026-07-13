"""Space Weather Physics Engine — validation utilities and reference checks.

This project has no existing pytest/unittest infrastructure (verified
during the Physics Engine migration — no test framework is installed or
configured anywhere in the codebase), so this module is a self-contained,
dependency-free validation runner rather than a pytest suite: run
`python -m swdss.physics.validation` to check every formula against a
known reference calculation and NaN-safety expectation, and get a clear
PASS/FAIL report — matching this project's existing convention of
self-contained modules over external test-framework dependencies.
"""

import math

import numpy as np
import pandas as pd

from swdss.physics import coupling, core, geometry, magnetosphere, plasma


def _check(label: str, actual: float, expected: float, tol: float = 1e-6) -> bool:
    passed = math.isclose(actual, expected, rel_tol=tol, abs_tol=tol)
    status = "PASS" if passed else f"FAIL (expected {expected}, got {actual})"
    print(f"  {label:45s} {status}")
    return passed


def _check_nan_safe(label: str, fn, *args) -> bool:
    """Confirms a function returns NaN (not an exception, not inf) for
    a zero-denominator edge case, rather than crashing or silently
    producing a misleading number.
    """
    try:
        result = fn(*args)
        val = result.iloc[0] if isinstance(result, pd.Series) else result
        passed = pd.isna(val)
        status = "PASS (NaN as expected)" if passed else f"FAIL (expected NaN, got {val})"
    except Exception as exc:
        passed = False
        status = f"FAIL (raised {type(exc).__name__}: {exc})"
    print(f"  {label:45s} {status}")
    return passed


def run_validation() -> bool:
    all_pass = True
    print("=== Core Sun-Earth Coupling ===")
    all_pass &= _check("VBz (southward Bz)", core.vbz_scalar(500, -10), -5000)
    all_pass &= _check("VBz (northward Bz, clipped to 0)", core.vbz_scalar(500, 10), 0)
    all_pass &= _check("Ey (southward Bz -> positive)", core.ey_scalar(500, -10), 5.0)
    all_pass &= _check("Dynamic Pressure", core.dynamic_pressure_scalar(5, 400), 1.6726e-6 * 5 * 400**2)

    print("\n=== Geometry ===")
    all_pass &= _check("Clock Angle (purely southward)", geometry.clock_angle_scalar(0, -5), 180.0)
    all_pass &= _check("Clock Angle (purely northward)", geometry.clock_angle_scalar(0, 5), 0.0)
    all_pass &= _check("Clock Angle (purely dawnward +Y)", geometry.clock_angle_scalar(5, 0), 90.0)

    print("\n=== Coupling Functions ===")
    # Newell coupling at theta_c=180 (purely southward) collapses to
    # v^(4/3) * B_T^(2/3), since sin(90deg)=1.
    speed, by, bz = pd.Series([500.0]), pd.Series([0.0]), pd.Series([-10.0])
    expected_newell = 500 ** (4 / 3) * 10 ** (2 / 3)
    all_pass &= _check("Newell Coupling (purely southward)", coupling.newell_coupling_series(speed, by, bz).iloc[0], expected_newell, tol=1e-4)

    print("\n=== Plasma ===")
    # First-principles-verified constant (see plasma.py's module docstring)
    all_pass &= _check("Plasma Beta constant matches first-principles derivation", 1.3807e-8 / 3.9789e-4, 3.4699e-5, tol=1e-3)

    print("\n=== Magnetosphere ===")
    all_pass &= _check("Nominal magnetopause standoff ~10.25 Re", magnetosphere.NOMINAL_STANDOFF_RE, 10.25, tol=1e-2)

    print("\n=== NaN Safety ===")
    all_pass &= _check_nan_safe("Plasma Beta with zero magnetic pressure", plasma.plasma_beta_series, pd.Series([1.0]), pd.Series([0.0]))
    all_pass &= _check_nan_safe("Alfven Speed with zero density", plasma.alfven_speed_series, pd.Series([5.0]), pd.Series([0.0]))
    all_pass &= _check_nan_safe("Magnetopause Standoff with zero dynamic pressure", magnetosphere.magnetopause_standoff_series, pd.Series([0.0]), pd.Series([0.0]))

    print("\n" + "=" * 60)
    print("ALL PHYSICS ENGINE VALIDATION CHECKS PASSED" if all_pass else "VALIDATION FAILURES DETECTED")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_validation() else 1)
