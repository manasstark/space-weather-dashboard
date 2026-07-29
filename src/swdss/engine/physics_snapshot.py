"""Builds the live "physics summary" attached to every forecast snapshot
— descriptive only, computed from the existing Physics Engine
(swdss.physics.*) against the same live Analytics feature frame
predict.py already builds for the Kp/Dst production models. Never
generates a new ML feature; every quantity here already has a canonical
formula documented in swdss.physics.

vbz/ey/dynamic_pressure are read straight off the frame (load_live_
features already computes them via add_derived_physics_features) rather
than recomputed here, so there is exactly one place that formula runs.
Everything else (clock angle, coupling functions, plasma/pressure
quantities, magnetopause geometry) isn't part of the base Analytics
feature set, so it's computed here directly from the same tail window.
"""

from swdss.models.predict import load_live_features
from swdss.physics.core import (
    integrated_ey_series,
    integrated_southward_bz_series,
    integrated_vbz_series,
    southward_duration_series,
    strong_southward_duration_series,
)
from swdss.physics.coupling import akasofu_epsilon_series, boyle_index_series, newell_coupling_series
from swdss.physics.geometry import (
    clock_angle_rate_series,
    clock_angle_series,
    imf_rotation_rate_series,
    magnetic_shear_series,
)
from swdss.physics.magnetosphere import estimated_compression_series, magnetopause_standoff_series
from swdss.physics.plasma import (
    alfven_mach_number_series,
    alfven_speed_series,
    magnetic_pressure_series,
    plasma_beta_series,
    thermal_pressure_series,
)

WINDOW_HOURS = 48
ROLLING_WINDOW = 24

# The computed numeric physics quantities build_physics_snapshot produces
# — deliberately excludes the derived *_label / overall_coupling string
# fields below, since those are descriptive categorizations of the
# numbers here, not independent inputs whose absence would represent a
# genuine data gap. Kept as an explicit list (rather than introspecting
# the returned dict's keys) so this module's own schema is the single
# source of truth physics_completeness() checks against.
PHYSICS_QUANTITY_KEYS = (
    "dynamic_pressure", "clock_angle_deg", "clock_angle_rate", "vbz", "ey",
    "newell_coupling", "akasofu_epsilon_watts", "boyle_index_kv",
    "magnetic_pressure", "thermal_pressure", "plasma_beta", "alfven_speed",
    "alfven_mach_number", "magnetic_shear", "imf_rotation_rate",
    "southward_duration_hr", "strong_southward_duration_hr",
    "integrated_southward_bz", "integrated_ey", "integrated_vbz",
    "magnetopause_standoff_re", "estimated_compression_pct",
)


def _last(series):
    if series is None:
        return None
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else None


def _clock_angle_label(value) -> str:
    if value is None:
        return "Unknown"
    if value <= 45 or value >= 315:
        return "Northward"
    if 135 <= value <= 225:
        return "Southward"
    return "Southward-leaning" if 90 < value < 270 else "Northward-leaning"


def _plasma_beta_label(value) -> str:
    if value is None:
        return "Unknown"
    if value < 0.5:
        return "Low (magnetically dominated)"
    if value <= 2:
        return "Typical"
    return "High (thermally dominated)"


def _coupling_label(newell) -> str:
    # Rough operational bands over Newell coupling's typical 0-3000 range
    # (see swdss.physics.coupling's docstring) — a descriptive label only,
    # not a threshold used anywhere else in the app.
    if newell is None:
        return "Unknown"
    if newell < 300:
        return "Weak"
    if newell < 1200:
        return "Moderate"
    return "Strong"


def _dynamic_pressure_label(value) -> str:
    if value is None:
        return "Unknown"
    if value < 3:
        return "Typical"
    if value < 10:
        return "Elevated"
    return "High (shock/CME sheath likely)"


def build_physics_snapshot() -> dict:
    """Returns a flat dict — this becomes snapshot["physics_summary"]."""
    frame = load_live_features("analytics")
    tail = frame.tail(WINDOW_HOURS)

    speed, density, temperature = tail["speed"], tail["density"], tail["temperature"]
    bt, bx, by, bz = tail["bt"], tail["bx_gsm"], tail["by_gsm"], tail["bz_gsm"]
    vbz = tail["vbz"] if "vbz" in tail.columns else None
    ey = tail["ey"] if "ey" in tail.columns else None
    dynamic_pressure = tail["dynamic_pressure"] if "dynamic_pressure" in tail.columns else None

    clock_angle = clock_angle_series(by, bz)
    clock_angle_rate = clock_angle_rate_series(clock_angle)

    newell = newell_coupling_series(speed, by, bz)
    akasofu = akasofu_epsilon_series(speed, bt, clock_angle)
    boyle = boyle_index_series(speed, bt, clock_angle)

    magnetic_pressure = magnetic_pressure_series(bt)
    thermal_pressure = thermal_pressure_series(density, temperature)
    alfven_speed = alfven_speed_series(bt, density)

    magnetopause_standoff = (
        magnetopause_standoff_series(bz, dynamic_pressure) if dynamic_pressure is not None else None
    )

    physics = {
        "dynamic_pressure": _last(dynamic_pressure),
        "clock_angle_deg": _last(clock_angle),
        "clock_angle_rate": _last(clock_angle_rate),
        "vbz": _last(vbz),
        "ey": _last(ey),
        "newell_coupling": _last(newell),
        "akasofu_epsilon_watts": _last(akasofu),
        "boyle_index_kv": _last(boyle),
        "magnetic_pressure": _last(magnetic_pressure),
        "thermal_pressure": _last(thermal_pressure),
        "plasma_beta": _last(plasma_beta_series(thermal_pressure, magnetic_pressure)),
        "alfven_speed": _last(alfven_speed),
        "alfven_mach_number": _last(alfven_mach_number_series(speed, alfven_speed)),
        "magnetic_shear": _last(magnetic_shear_series(bx, by, bz)),
        "imf_rotation_rate": _last(imf_rotation_rate_series(clock_angle_rate, ROLLING_WINDOW)),
        "southward_duration_hr": _last(southward_duration_series(bz)),
        "strong_southward_duration_hr": _last(strong_southward_duration_series(bz)),
        "integrated_southward_bz": _last(integrated_southward_bz_series(bz, ROLLING_WINDOW)),
        "integrated_ey": _last(integrated_ey_series(ey, ROLLING_WINDOW)) if ey is not None else None,
        "integrated_vbz": _last(integrated_vbz_series(vbz, ROLLING_WINDOW)) if vbz is not None else None,
        "magnetopause_standoff_re": _last(magnetopause_standoff),
    }
    physics["estimated_compression_pct"] = (
        _last(estimated_compression_series(magnetopause_standoff))
        if magnetopause_standoff is not None
        else None
    )

    physics["clock_angle_label"] = _clock_angle_label(physics["clock_angle_deg"])
    physics["plasma_beta_label"] = _plasma_beta_label(physics["plasma_beta"])
    physics["coupling_label"] = _coupling_label(physics["newell_coupling"])
    physics["dynamic_pressure_label"] = _dynamic_pressure_label(physics["dynamic_pressure"])

    strong = physics["coupling_label"] == "Strong" and physics["clock_angle_label"] == "Southward"
    moderate = physics["coupling_label"] == "Moderate"
    physics["overall_coupling"] = "Strong" if strong else ("Moderate" if moderate else "Weak")

    return physics


def physics_completeness(physics: dict) -> dict:
    """Reports the health of the COMPUTED physics, not just whether the
    Physics Engine process ran — a quantity can be silently None (a
    genuinely missing/NaN input propagated through, e.g. no dynamic
    pressure because density was unavailable) even while build_physics_
    snapshot() itself completes without raising. Returns
    {available, total, missing: [names]} against PHYSICS_QUANTITY_KEYS.
    """
    missing = [key for key in PHYSICS_QUANTITY_KEYS if physics.get(key) is None]
    total = len(PHYSICS_QUANTITY_KEYS)
    return {"available": total - len(missing), "total": total, "missing": missing}
