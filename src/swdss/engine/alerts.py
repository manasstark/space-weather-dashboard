"""Rule-based operational alerts — fixed templates triggered by
documented conditions over the outlook level, physics summary, data
freshness, and recent CME detections. No ML, no free-text generation.
"""

import pandas as pd

from swdss.paths import PROCESSED_DIR
from swdss.physics.cme_dbm import estimate_cme_arrival_dbm

SOUTHWARD_DURATION_ALERT_HOURS = 6
MAGNETOPAUSE_COMPRESSION_ALERT_PCT = 15
CME_RECENT_WINDOW_HOURS = 24


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def ambient_solar_wind_speed(target_time, tolerance_hours: int = 3):
    """Reads the Solar Wind processed parquet directly (same reasoning as
    the CME read below — this module must never import dashboard/home.py)
    for the speed reading nearest target_time, used as the DBM ambient
    wind speed. Returns None if no reading falls within tolerance.
    """
    path = PROCESSED_DIR / "solar_wind" / "solar_wind_processed.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "timestamp_utc" not in df.columns or "solar_wind_speed" not in df.columns:
        return None

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "solar_wind_speed"])
    if df.empty:
        return None

    diffs = (df["timestamp_utc"] - target_time).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(hours=tolerance_hours):
        return None
    return float(df.loc[idx, "solar_wind_speed"])


def _recent_cme_alerts() -> list:
    """Reads the CME processed parquet directly (not through dashboard/
    home.py — that module runs Streamlit script code at import time and
    must never be imported from engine/live_update code) for any CME
    detected within CME_RECENT_WINDOW_HOURS, with the same Drag-Based
    Model (DBM) arrival estimate dashboard.lib.event_explorer's
    estimate_cme_arrival uses (swdss.physics.cme_dbm) — replacing the
    constant-speed heuristic both previously shared.
    """
    path = PROCESSED_DIR / "cme" / "cme_processed.parquet"
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path)
    except Exception:
        return []
    if df.empty or "timestamp_utc" not in df.columns:
        return []

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    cutoff = _now() - pd.Timedelta(hours=CME_RECENT_WINDOW_HOURS)
    recent = df[df["timestamp_utc"] >= cutoff]

    alerts = []
    for _, row in recent.iterrows():
        speed = row.get("speed")
        if speed is None or pd.isna(speed) or float(speed) <= 0:
            continue
        launch_time = row["timestamp_utc"]
        ambient_speed = ambient_solar_wind_speed(launch_time)
        dbm = estimate_cme_arrival_dbm(launch_time, float(speed), ambient_speed)
        if dbm is None:
            continue
        arrival = dbm["arrival_median"]
        window_hours = dbm["travel_hours_max"] - dbm["travel_hours_min"]
        alerts.append({
            "severity": "warning",
            "source": "CME",
            "message": (
                f"CME Arrival — detected {launch_time.strftime('%d %b %H:%M UTC')} at "
                f"{speed:.0f} km/s, estimated Earth arrival ~{arrival.strftime('%d %b %H:%M UTC')} "
                f"(DBM window ±{window_hours / 2:.0f}h)."
            ),
            "since": launch_time.isoformat(),
        })
    return alerts


def build_alerts(*, outlook_level: str, physics: dict, freshness: dict) -> list:
    alerts = []
    now_iso = _now().isoformat()

    if outlook_level in ("Moderate Storm", "Strong Storm"):
        severity = "critical" if outlook_level == "Strong Storm" else "warning"
        alerts.append({
            "severity": severity,
            "source": "Outlook",
            "message": f"Elevated Geomagnetic Activity — Overall Outlook: {outlook_level}.",
            "since": now_iso,
        })

    if physics.get("clock_angle_label") == "Southward" and (physics.get("southward_duration_hr") or 0) >= SOUTHWARD_DURATION_ALERT_HOURS:
        alerts.append({
            "severity": "warning",
            "source": "Physics",
            "message": f"Strong Southward IMF — sustained for {physics['southward_duration_hr']:.0f}+ hours.",
            "since": now_iso,
        })

    if physics.get("dynamic_pressure_label") in ("Elevated", "High (shock/CME sheath likely)"):
        alerts.append({
            "severity": "warning",
            "source": "Physics",
            "message": f"Elevated Dynamic Pressure — {physics.get('dynamic_pressure'):.2f} nPa.",
            "since": now_iso,
        })

    if physics.get("coupling_label") == "Strong":
        alerts.append({
            "severity": "info",
            "source": "Physics",
            "message": f"High Newell Coupling — {physics.get('newell_coupling'):.0f} (strong solar wind-magnetosphere coupling).",
            "since": now_iso,
        })

    compression = physics.get("estimated_compression_pct")
    if compression is not None and compression >= MAGNETOPAUSE_COMPRESSION_ALERT_PCT:
        alerts.append({
            "severity": "warning",
            "source": "Physics",
            "message": f"Strong Magnetopause Compression — {compression:.0f}% above nominal standoff distance.",
            "since": now_iso,
        })

    for name, status in (freshness or {}).items():
        if status.get("status") == "Stale":
            alerts.append({
                "severity": "warning",
                "source": "Data Feed",
                "message": f"{name.replace('_', ' ').title()} data feed is stale ({status.get('age')}).",
                "since": now_iso,
            })

    alerts.extend(_recent_cme_alerts())

    return alerts
