"""Rule-based (no LLM, no black box) physics interpretation of current
Sun-Earth coupling conditions, synthesized from live Solar Wind + IMF +
geomagnetic readings using established space-weather physics: the
Burton et al. (1975) VBz coupling function, the interplanetary
dawn-dusk electric field Ey, and the IMF clock angle. Every statement
below is a direct, reproducible function of the input numbers — nothing
here is generated or inferred by a language model.
"""

import math

VBZ_QUIET_THRESHOLD = 0
VBZ_WEAK_THRESHOLD = -1000
VBZ_MODERATE_THRESHOLD = -3000

AE_QUIET_THRESHOLD = 100
AE_UNSETTLED_THRESHOLD = 300
AE_ACTIVE_THRESHOLD = 500


def _solar_wind_state(speed, density) -> str:
    if speed is None or density is None:
        return "Insufficient live Solar Wind data to assess current state."

    if speed < 400:
        speed_label = "Slow solar wind"
    elif speed < 500:
        speed_label = "Moderate-speed solar wind"
    elif speed < 700:
        speed_label = "Fast solar wind"
    else:
        speed_label = "Very fast solar wind"

    if density < 5:
        density_label = "low density"
    elif density < 10:
        density_label = "moderate density"
    elif density < 30:
        density_label = "high density"
    else:
        density_label = "very high density (possible shock/CME sheath)"

    dynamic_pressure = 1.6726e-6 * density * speed**2
    return (
        f"{speed_label} ({speed:.0f} km/s) with {density_label} ({density:.1f} p/cm3). "
        f"Dynamic pressure on the magnetopause is approximately {dynamic_pressure:.2f} nPa."
    )


def _imf_orientation(bt, bx, by, bz) -> str:
    if bz is None or bt is None:
        return "Insufficient live IMF data to assess current orientation."

    orientation = "southward" if bz < 0 else "northward"
    clock_angle = math.degrees(math.atan2(by, bz)) if by is not None else None
    clock_text = f", clock angle {clock_angle:.0f} degrees from north" if clock_angle is not None else ""
    consequence = (
        "Southward orientation favors dayside magnetic reconnection."
        if bz < 0
        else "Northward orientation suppresses dayside reconnection — a shielding configuration."
    )
    return f"IMF is {orientation} (Bz={bz:.2f} nT, Bt={bt:.2f} nT{clock_text}). {consequence}"


def _magnetic_coupling(speed, bz) -> str:
    if speed is None or bz is None:
        return "Insufficient data to assess magnetic coupling."

    vbz = speed * min(bz, 0)
    ey = -speed * bz * 1e-3

    if vbz >= VBZ_QUIET_THRESHOLD:
        label = "Minimal coupling — northward IMF blocks efficient reconnection."
    elif vbz > VBZ_WEAK_THRESHOLD:
        label = "Weak coupling — limited solar wind energy transfer into the magnetosphere."
    elif vbz > VBZ_MODERATE_THRESHOLD:
        label = "Moderate coupling — noticeable energy injection into the magnetosphere is likely."
    else:
        label = "Strong coupling — significant solar wind energy transfer into the magnetosphere."

    return f"{label} (VBz={vbz:.0f} nT-km/s, Ey={ey:.2f} mV/m)"


def _auroral_activity(ae) -> str:
    if ae is None:
        return "No recent AE reading available to assess auroral electrojet activity."
    if ae < AE_QUIET_THRESHOLD:
        label = "Quiet"
    elif ae < AE_UNSETTLED_THRESHOLD:
        label = "Unsettled"
    elif ae < AE_ACTIVE_THRESHOLD:
        label = "Active"
    else:
        label = "Storm-level"
    return f"{label} auroral electrojet activity (latest known AE ~{ae:.0f} nT)."


def _ring_current_response(dst) -> str:
    if dst is None:
        return "No recent Dst reading available to assess ring current state."
    if dst > -30:
        label, risk = "Quiet", "low ring current activity"
    elif dst > -50:
        label, risk = "Weak disturbance", "minor ring current buildup"
    elif dst > -100:
        label, risk = "Moderate storm", "significant ring current intensification"
    elif dst > -200:
        label, risk = "Intense storm", "strong ring current, geomagnetic storm conditions"
    else:
        label, risk = "Severe storm", "extreme ring current intensification"
    return f"{label} ring current state (Dst={dst:.0f} nT) — {risk}."


def _geomagnetic_activity(kp) -> str:
    if kp is None:
        return "No recent Kp reading available to assess geomagnetic activity."
    if kp <= 3:
        label, risk = "Quiet", "normal geomagnetic conditions"
    elif kp < 5:
        label, risk = "Active", "unsettled field, no storm"
    elif kp < 6:
        label, risk = "G1 storm", "minor storm conditions"
    elif kp < 7:
        label, risk = "G2 storm", "moderate storm conditions"
    elif kp < 8:
        label, risk = "G3 storm", "strong storm conditions"
    else:
        label, risk = "G4-G5 storm", "severe to extreme storm conditions"
    return f"{label} geomagnetic conditions (Kp={kp:.1f}) — {risk}."


def physics_interpretation(speed, density, temperature, bt, bx, by, bz, kp=None, dst=None, ae=None) -> dict:
    """Returns a dict of six rule-based narrative statements — Current
    Solar Wind State, Current IMF Orientation, Expected Magnetic
    Coupling, Expected Auroral Activity, Expected Ring Current Response,
    Expected Geomagnetic Activity — each a pure, reproducible function of
    the given readings. `temperature`, `bx` are accepted for API
    completeness/future use but only feed sections that need them today.
    """
    return {
        "solar_wind_state": _solar_wind_state(speed, density),
        "imf_orientation": _imf_orientation(bt, bx, by, bz),
        "magnetic_coupling": _magnetic_coupling(speed, bz),
        "auroral_activity": _auroral_activity(ae),
        "ring_current": _ring_current_response(dst),
        "geomagnetic_activity": _geomagnetic_activity(kp),
    }
