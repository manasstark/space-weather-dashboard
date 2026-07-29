"""Drag-Based Model (DBM) for CME transit time — Vrsnak et al. (2013),
"Propagation of Interplanetary Coronal Mass Ejections: The Drag-Based
Model", Solar Physics.

Why this exists: the arrival estimate this project used everywhere
(dashboard/lib/event_explorer.py's estimate_cme_arrival, and the CME
alert message in swdss.engine.alerts) was pure kinematics — 1 AU divided
by the CME's launch speed, held constant for the entire Sun-Earth
transit. Real CMEs don't travel at constant speed: a CME faster than the
ambient solar wind decelerates toward it via aerodynamic-style drag; one
slower than the ambient wind accelerates toward it. For a fast CME this
can mean several hundred km/s of speed change by 1 AU, which the old
formula had no way to represent.

DBM treats the CME as a single point mass subject to:

    dv/dt = -Gamma * (v - w) * |v - w|

    v      = CME speed
    w      = ambient solar wind speed
    Gamma  = drag parameter (km^-1), lumping CME mass/cross-section and
             ambient density into one empirical constant

This has a closed-form solution for v(t) and r(t), unlike a full MHD
simulation (WSA-ENLIL-class), which makes it the standard fast, real-
time-capable operational tool the field actually reaches for first, and
tractable to run here without a new heavy dependency.

Gamma has no single true value — Vrsnak et al. (2013) report fitted
values roughly spanning 1e-8 to 2e-7 km^-1 depending on CME mass/width
and ambient density, and there is no way to know a specific event's true
Gamma in advance. Rather than guess one number, dbm_arrival_ensemble
sweeps Gamma (and the ambient speed, itself only ever an estimate at
forecast time) across this literature range and reports the resulting
spread of transit times as a min/median/max window — the same spirit as
the confidence bands already used elsewhere in this project's forecast
engine, applied here to a time-of-arrival distribution instead of a
value distribution.

Known, deliberately-not-solved limitation: DBM (like every model in this
family, including WSA-ENLIL) has no way to predict the sign of the
arriving CME's Bz — that is set by the flux rope's internal magnetic
orientation, not by transit dynamics, and remains unobservable with any
confidence before in-situ arrival. This module estimates *when*, never
*how geoeffective*.
"""

from __future__ import annotations

import math
from datetime import timedelta

AU_KM = 1.496e8
R0_KM = 20 * 6.957e5  # ~20 solar radii — roughly where LASCO height-time speed fits are referenced and where the drag regime is assumed to take over
GAMMA_RANGE_KM_INV = (0.2e-7, 2.0e-7)  # Vrsnak et al. (2013) typical fitted range
DEFAULT_GAMMA_KM_INV = 0.5e-7
DEFAULT_AMBIENT_SPEED_KM_S = 400.0  # typical slow solar wind, used only when no live reading is available


def _dbm_distance_km(v0: float, w: float, gamma: float, t_seconds: float) -> float:
    """DBM's closed-form r(t) (distance travelled beyond R0), t in seconds."""
    if abs(v0 - w) < 1e-9:
        return w * t_seconds
    epsilon = 1.0 if v0 > w else -1.0
    return w * t_seconds + (epsilon / gamma) * math.log(1 + epsilon * gamma * (v0 - w) * t_seconds)


def _dbm_transit_seconds(v0: float, w: float, gamma: float, target_km: float = AU_KM) -> float | None:
    """Solves r(t) = target_km - R0_KM for t via bisection — r(t) is
    transcendental in t, but monotonically increasing in t for any
    physically valid v0/w/gamma > 0, so bisection is safe and simple
    (no scipy dependency needed for a single root-find like this).
    """
    if v0 <= 0 or w <= 0 or gamma <= 0:
        return None

    distance = target_km - R0_KM
    if distance <= 0:
        return 0.0

    lo, hi = 0.0, distance / min(v0, w)
    tries = 0
    while _dbm_distance_km(v0, w, gamma, hi) < distance and tries < 40:
        hi *= 1.5
        tries += 1

    for _ in range(80):
        mid = (lo + hi) / 2
        if _dbm_distance_km(v0, w, gamma, mid) < distance:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def dbm_transit_hours(
    launch_speed_km_s: float | None,
    ambient_speed_km_s: float | None = None,
    gamma_km_inv: float = DEFAULT_GAMMA_KM_INV,
) -> float | None:
    """Single-point DBM transit time estimate, in hours. Falls back to
    DEFAULT_AMBIENT_SPEED_KM_S if no live ambient solar wind reading is
    available (e.g. a data gap right at CME launch time).
    """
    if launch_speed_km_s is None or launch_speed_km_s <= 0:
        return None
    ambient = ambient_speed_km_s if ambient_speed_km_s and ambient_speed_km_s > 0 else DEFAULT_AMBIENT_SPEED_KM_S
    seconds = _dbm_transit_seconds(float(launch_speed_km_s), float(ambient), gamma_km_inv)
    return None if seconds is None else seconds / 3600.0


def dbm_arrival_ensemble(
    launch_speed_km_s: float | None,
    ambient_speed_km_s: float | None = None,
    n_gamma: int = 5,
    ambient_spread_frac: float = 0.15,
) -> dict | None:
    """Sweeps Gamma across its literature range and the ambient speed
    +/-ambient_spread_frac around the best current estimate (the DBM
    ensemble idea — cf. Moestl et al.'s DBEM), returning a transit-time
    window instead of one number. Neither Gamma nor the ambient wind
    speed the CME will actually encounter along its path is truly known
    at forecast time, so reporting a spread is more honest than a false-
    precision point estimate.
    """
    if launch_speed_km_s is None or launch_speed_km_s <= 0:
        return None

    ambient_center = ambient_speed_km_s if ambient_speed_km_s and ambient_speed_km_s > 0 else DEFAULT_AMBIENT_SPEED_KM_S
    lo_gamma, hi_gamma = GAMMA_RANGE_KM_INV
    gammas = [lo_gamma + i * (hi_gamma - lo_gamma) / (n_gamma - 1) for i in range(n_gamma)]
    ambients = [
        ambient_center * (1 - ambient_spread_frac),
        ambient_center,
        ambient_center * (1 + ambient_spread_frac),
    ]

    hours = []
    for gamma in gammas:
        for ambient in ambients:
            h = dbm_transit_hours(launch_speed_km_s, ambient, gamma)
            if h is not None:
                hours.append(h)

    if not hours:
        return None

    hours.sort()
    n = len(hours)
    return {
        "min_hours": hours[0],
        "median_hours": hours[n // 2],
        "max_hours": hours[-1],
        "n_samples": n,
        "gamma_range_km_inv": GAMMA_RANGE_KM_INV,
        "ambient_speed_used_km_s": ambient_center,
        "launch_speed_km_s": launch_speed_km_s,
    }


def estimate_cme_arrival_dbm(launch_time, launch_speed_km_s, ambient_speed_km_s=None):
    """Top-level convenience: given a CME's launch time and speed, return
    a dict with the DBM ensemble's transit-time window converted into
    actual arrival timestamps. Returns None if launch_speed_km_s is
    missing/invalid, the same contract the old kinematic
    estimate_cme_arrival used.
    """
    ensemble = dbm_arrival_ensemble(launch_speed_km_s, ambient_speed_km_s)
    if ensemble is None or launch_time is None:
        return None

    return {
        "arrival_min": launch_time + timedelta(hours=ensemble["min_hours"]),
        "arrival_median": launch_time + timedelta(hours=ensemble["median_hours"]),
        "arrival_max": launch_time + timedelta(hours=ensemble["max_hours"]),
        "travel_hours_min": ensemble["min_hours"],
        "travel_hours_median": ensemble["median_hours"],
        "travel_hours_max": ensemble["max_hours"],
        "ambient_speed_used_km_s": ensemble["ambient_speed_used_km_s"],
        "gamma_range_km_inv": ensemble["gamma_range_km_inv"],
        "launch_speed_km_s": ensemble["launch_speed_km_s"],
    }
