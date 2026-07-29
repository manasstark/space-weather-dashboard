"""Shared, dependency-free data-loading, lookup, and formatting helpers
used across dashboard/home.py and every dashboard/lib/*.py module below
it (event_explorer, forecast_dialogs, solar_activity, library).

Extracted verbatim from dashboard/home.py (no formula, styling, or
behavior changes) as part of splitting home.py's remaining ~6,000 lines
into topic-scoped modules, the same pattern already used for the three
Research Laboratories (see shared_ui.py's own docstring). This module
exists specifically because several small utilities here — format_value,
load_processed_data, latest_label_time, time_caption, etc. — are needed
by more than one of the newly-split modules, and none of them may import
from home.py itself (that would be circular, since home.py imports them).

One deliberate behavior-preserving refactor happened during this
extraction: `nearest_master_row` previously read a bare module-level
global (`master_df`) that only existed in home.py's own namespace,
which would have been a real circular dependency once this function
moved to a separate module. Since `load_master_data` is already
`@st.cache_data`-decorated, `nearest_master_row` now simply calls it
directly instead — identical data, identical caching behavior, zero
signature change for any caller, but no more reaching into another
module's global namespace.
"""

import base64
from html import escape

import pandas as pd
import streamlit as st

from dashboard.lib.shared_ui import REFRESH_SECONDS
from swdss.paths import MASTER_V1_PATH, PROCESSED_DIR


def get_base64_image(path) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()


DATASET_LABELS = {
    "solar_wind": "Solar Wind",
    "imf": "IMF",
    "kp": "Kp",
    "dst": "Dst",
    "analytics": "Analytics (Combined Earth)",
}

STATUS_BADGE_STYLES = {
    "in_progress": ("Predicting", "#7a5a1f", "#fff3cd"),
    "evaluating": ("Evaluating", "#1f5a7a", "#d0eaf7"),
    "completed": ("Completed", "#1f5a2e", "#d4f4dd"),
    "stopped": ("Stopped", "#4a4a4a", "#e2e2e2"),
    "failed": ("Failed", "#7a1f1f", "#fddede"),
    "pending": ("Pending Official Kyoto Data", "#7a5a1f", "#fff3cd"),
    "verified": ("Verified", "#1f5a2e", "#d4f4dd"),
    "quicklook_pending": ("Awaiting Quicklook Estimate", "#5a1f8a", "#f2e8ff"),
    "quicklook_verified": ("Quicklook Estimate Available", "#5a1f8a", "#f2e8ff"),
    "kp_waiting_official": ("Waiting for Official NOAA Kp", "#1f5a7a", "#d0eaf7"),
}

QUICKLOOK_CONFIDENCE_COLORS = {
    # Bright terminal-palette values (dashboard.lib.design_tokens'
    # ACCENT/AMBER/RED) — the previous dark, desaturated values were
    # chosen for the retired metric_card's light-grey Win-95 background
    # and would be near-invisible on terminal_metric's dark panel.
    "high": "#39d98a",
    "moderate": "#e3b341",
    "low": "#f85149",
}


def status_badge_html(status: str) -> str:
    text, border_color, bg_color = STATUS_BADGE_STYLES.get(status, (status.title(), "#808080", "#dcdcdc"))
    return (
        f'<span style="display:inline-block; padding:2px 10px; border-radius:10px; '
        f'background:{bg_color}; border:1px solid {border_color}; color:{border_color}; '
        f'font-size:0.75rem; font-weight:700;">{escape(text)}</span>'
    )


def format_value(value, suffix: str = "", decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    if decimals == 0:
        return f"{value:.0f}{suffix}"
    return f"{value:.{decimals}f}{suffix}"


def format_event_time(value) -> str:
    if pd.isna(value):
        return "N/A"

    timestamp = pd.to_datetime(value, utc=True)
    return timestamp.strftime("%d %b %H:%M UTC")


def latest_label_time(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.to_datetime(value, utc=True).strftime("%d %b %H:%M UTC")


def time_caption(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return f"Recorded at {row['timestamp_utc']}"


def reference_tooltip(column: str, value) -> str:
    if pd.isna(value):
        return "No reference available"

    value = float(value)

    if column == "bz":
        if value > 0:
            return "Bz > 0 nT: Northward IMF. Low geomagnetic coupling."
        if value >= -5:
            return "Bz 0 to -5 nT: Weak southward IMF. Minor activity possible."
        if value >= -10:
            return "Bz -5 to -10 nT: Moderate southward IMF. Storm possible."
        if value >= -20:
            return "Bz -10 to -20 nT: Strong southward IMF. Strong storm coupling."
        return "Bz < -20 nT: Extreme southward IMF. Severe storm potential."

    if column == "kp":
        if value <= 3:
            return "Kp 0-3: Quiet geomagnetic conditions."
        if value < 5:
            return "Kp 4: Active geomagnetic field."
        if value < 6:
            return "Kp 5: G1 minor geomagnetic storm."
        if value < 7:
            return "Kp 6: G2 moderate geomagnetic storm."
        if value < 8:
            return "Kp 7: G3 strong geomagnetic storm."
        return "Kp 8-9: G4-G5 severe to extreme geomagnetic storm."

    if column == "dst":
        if value > -30:
            return "Dst > -30 nT: Quiet or weak storm activity."
        if value > -50:
            return "Dst -30 to -50 nT: Weak storm."
        if value > -100:
            return "Dst -50 to -100 nT: Moderate storm."
        if value > -200:
            return "Dst -100 to -200 nT: Intense storm."
        return "Dst < -200 nT: Superstorm."

    return ""


def variable_meaning_and_risk(column: str, value) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "No data", "N/A"

    value = float(value)

    if column == "speed":
        if value < 400:
            return "Slow solar wind", "Usually quiet"
        if value < 500:
            return "Moderate speed", "Normal/active"
        if value < 700:
            return "Fast solar wind", "Storm possible with southward Bz"
        return "Very fast wind", "Enhanced storm potential"

    if column == "density":
        if value < 5:
            return "Low density", "Weak pressure"
        if value < 10:
            return "Moderate density", "Normal solar wind"
        if value < 30:
            return "High density", "Compression possible"
        return "Very high density", "Shock/CME sheath possible"

    if column == "temperature":
        if value < 50000:
            return "Cool wind", "Usually quiet"
        if value < 150000:
            return "Typical wind", "Normal"
        if value < 500000:
            return "Hot wind", "Disturbed flow possible"
        return "Very hot plasma", "Shock/CME heating possible"

    if column == "bz":
        if value > 0:
            return "Northward IMF", "Low coupling"
        if value >= -5:
            return "Weak southward IMF", "Minor activity possible"
        if value >= -10:
            return "Moderate southward IMF", "Storm possible"
        if value >= -20:
            return "Strong southward IMF", "Strong storm coupling"
        return "Extreme southward IMF", "Severe storm potential"

    if column == "kp":
        if value <= 3:
            return "Quiet", "Normal"
        if value < 5:
            return "Active", "Unsettled field"
        if value < 6:
            return "G1 storm", "Minor storm"
        if value < 7:
            return "G2 storm", "Moderate storm"
        if value < 8:
            return "G3 storm", "Strong storm"
        return "G4-G5 storm", "Severe/extreme storm"

    if column == "dst":
        if value > -30:
            return "Quiet", "Low storm activity"
        if value > -50:
            return "Weak storm", "Minor ring current"
        if value > -100:
            return "Moderate storm", "Storm underway"
        if value > -200:
            return "Intense storm", "Strong disturbance"
        return "Superstorm", "Extreme disturbance"

    return "", ""


def render_simple_retro_table(df: pd.DataFrame, display_names: dict | None = None) -> None:
    """Generic retro-styled HTML table for arbitrary dataframes (e.g. the
    'Latest Events' / 'Latest CMEs' tables), matching the same vintage
    look used by the reference pagers.
    """
    display_names = display_names or {}

    def fmt_cell(value) -> str:
        if isinstance(value, pd.Timestamp):
            return format_event_time(value)
        if pd.isna(value):
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    html = """
    <style>
    .retro-table-simple {
        border-collapse: collapse;
        background: #ffffff;
        color: #000000;
        font-family: "MS Sans Serif", Tahoma, sans-serif;
        font-size: 14px;
        border-top: 2px solid #808080;
        border-left: 2px solid #808080;
        border-right: 2px solid #ffffff;
        border-bottom: 2px solid #ffffff;
    }
    .retro-table-simple th, .retro-table-simple td {
        border: 1px solid #d0d0d0;
        padding: 6px 10px;
        white-space: nowrap;
    }
    .retro-table-simple th {
        background: #efefef;
        color: #606060;
        font-weight: 400;
        text-align: left;
    }
    </style>
    <table class="retro-table-simple">
    """

    html += (
        "<thead><tr>"
        + "".join(f"<th>{escape(display_names.get(col, col))}</th>" for col in df.columns)
        + "</tr></thead><tbody>"
    )
    for _, row in df.iterrows():
        html += "<tr>" + "".join(f"<td>{escape(fmt_cell(row[col]))}</td>" for col in df.columns) + "</tr>"
    html += "</tbody></table>"

    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(ttl=REFRESH_SECONDS)
def load_master_data(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


@st.cache_data(ttl=REFRESH_SECONDS)
def load_processed_data(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name / f"{name}_processed.parquet"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


def seven_day_window(df: pd.DataFrame) -> pd.DataFrame:
    latest_time = df["timestamp_utc"].max()
    start_time = latest_time - pd.Timedelta(days=7)
    return df[df["timestamp_utc"] >= start_time].copy()


def recent_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty or "timestamp_utc" not in df.columns:
        return df
    latest_time = df["timestamp_utc"].max()
    start_time = latest_time - pd.Timedelta(days=days)
    return df[df["timestamp_utc"] >= start_time].copy()


def latest_non_null(df: pd.DataFrame, column: str):
    clean = df.dropna(subset=[column])
    if clean.empty:
        return float("nan"), None

    row = clean.iloc[-1]
    return row[column], row["timestamp_utc"]


def latest_value(df: pd.DataFrame, column: str, dataset: str | None = None):
    source_df = load_processed_data(dataset) if dataset else df

    if source_df.empty:
        source_df = df

    return latest_non_null(source_df, column)


def row_at_extreme(df: pd.DataFrame, column: str, mode: str = "max") -> pd.Series | None:
    clean = df.dropna(subset=[column])
    if clean.empty:
        return None
    idx = clean[column].idxmax() if mode == "max" else clean[column].idxmin()
    return clean.loc[idx]


def row_at_extreme_from_source(dataset_name: str, column: str, mode: str = "max") -> pd.Series | None:
    """Like row_at_extreme, but reads the true minute-level extreme from
    the dataset's own processed file instead of master_df_v1 (which
    resamples Solar Wind/IMF to hourly means — averaging away real
    spikes, e.g. a true 681 km/s peak showing up as a smoothed 650).
    The other fields needed for the hover card (Kp, Dst, etc.) are then
    pulled from whichever hourly master_df row is closest to that exact
    extreme's timestamp, since those datasets aren't minute-resolution.
    """
    source_df = recent_window(load_processed_data(dataset_name), 7)
    row = row_at_extreme(source_df, column, mode)
    if row is None:
        return None

    context_row, _ = nearest_master_row(row["timestamp_utc"])
    if context_row is None:
        return row

    enriched = context_row.copy()
    enriched[column] = row[column]
    enriched["timestamp_utc"] = row["timestamp_utc"]
    return enriched


def nearest_row_in(df: pd.DataFrame, target_time, tolerance_hours: float = 6) -> pd.Series | None:
    if df.empty or target_time is None or pd.isna(target_time) or "timestamp_utc" not in df.columns:
        return None
    diffs = (df["timestamp_utc"] - target_time).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(hours=tolerance_hours):
        return None
    return df.loc[idx]


def nearest_master_row(target_time, tolerance_hours: int = 3):
    master_df = load_master_data(MASTER_V1_PATH)
    if target_time is None or pd.isna(target_time) or master_df.empty or "timestamp_utc" not in master_df.columns:
        return None, False

    data_available = target_time <= master_df["timestamp_utc"].max()

    diffs = (master_df["timestamp_utc"] - target_time).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(hours=tolerance_hours):
        return None, data_available

    return master_df.loc[idx], data_available
