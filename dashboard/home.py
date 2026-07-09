import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from html import escape

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "features" / "master_df_v1.parquet"
REFRESH_SECONDS = 15

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swdss.ingest.kyoto_ae_quicklook import annotate_quicklook_image, fetch_quicklook_image, quicklook_image_url
from swdss.models.explainability import explain_prediction
from swdss.models.hypothesis import (
    archive_hypothesis,
    create_hypothesis,
    delete_hypothesis,
    duplicate_hypothesis,
    evaluate_hypothesis,
    get_hypothesis,
    list_hypotheses,
    reactivate_hypothesis,
    update_hypothesis,
    update_manual_conclusion,
    update_notes,
)
from swdss.models.physics_interpretation import physics_interpretation
from swdss.models.imf_research import (
    ALL_TRAINABLE_MODELS,
    DEFAULT_GRANULARITY,
    DEFAULT_HORIZON,
    DEFAULT_SEQUENCE_LENGTH,
    FUTURE_MODELS,
    GRANULARITY_OPTIONS,
    HOURLY_HORIZONS,
    HYPERPARAM_SCHEMA,
    MINUTE_HORIZONS,
    SEQUENCE_LENGTH_OPTIONS,
    SEQUENCE_MODELS,
    TABULAR_MODELS,
    TARGET_OPTIONS,
    compare_runs,
    delete_run,
    get_run,
    list_runs,
    load_research_frame,
    promote_run,
    train_horizon_sweep,
    train_research_model,
)
from swdss.models import ae_research, kp_research
from swdss.models.jobs import (
    average_prediction,
    classify_quicklook_error,
    delete_job,
    final_percentage_error,
    find_matching_job,
    forecast_drift,
    forecast_evaluation_label,
    get_job,
    get_job_stats,
    get_prediction_statistics,
    get_running_jobs,
    get_saved_jobs,
    job_mae,
    model_quality_label,
    poll_jobs,
    production_bias,
    production_error,
    quicklook_error,
    quicklook_label,
    quicklook_relative_error,
    refresh_quicklook_estimate,
    rolling_final_error,
    save_job,
    stability_metric,
    start_job,
    stop_job,
)
from swdss.models.predict import latest_minute_observation
from swdss.models.registry import (
    AE_VARIABLES,
    ANALYTICS_VARIABLES,
    EXPERIMENTAL_VARIABLES,
    HORIZONS,
    IMF_VARIABLES,
    SOLAR_WIND_VARIABLES,
    VARIABLE_LABELS,
    VARIABLE_UNITS,
)


st.set_page_config(
    page_title="Space Weather DSS",
    layout="wide",
)


def get_base64_image(path: Path) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()


def apply_retro_windows_style() -> None:
    bg_path = PROJECT_ROOT / "dashboard" / "assets" / "magnetosphere_bg.jpeg"
    bg_css = ""

    if bg_path.exists():
        encoded = get_base64_image(bg_path)
        bg_css = f"""
        [class*="stApp"] {{
            background-image:
                linear-gradient(rgba(5, 5, 15, 0.78), rgba(5, 5, 15, 0.78)),
                url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """

    st.markdown(
        f"""
        <style>
        {bg_css}

        html, body, [class*="stApp"] {{
            color: #ffffff;
            font-family: "MS Sans Serif", "Tahoma", sans-serif;
        }}

        section[data-testid="stSidebar"] {{
            background: rgba(20, 20, 30, 0.85);
            border-right: 2px solid #808080;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: #ffffff !important;
        }}

        div[data-testid="stMetric"] {{
            background: #dcdcdc;
            color: #000000;
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            padding: 12px;
        }}

        div[data-testid="stMetric"] *,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        label[data-testid="stMetricLabel"],
        label[data-testid="stMetricLabel"] *,
        div[data-testid="stWidgetLabel"],
        div[data-testid="stWidgetLabel"] *,
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {{
            color: #000000 !important;
        }}

        .stRadio,
        .stRadio > div,
        .stRadio > div > div,
        [data-testid="stRadio"],
        [data-testid="stRadio"] > div,
        [role="radiogroup"] {{
            width: 100% !important;
        }}

        .stRadio > div {{
            background: #dcdcdc;
            box-sizing: border-box;
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            padding: 8px;
        }}

        div[data-testid="stDataFrame"] {{
            border-top: 2px solid #808080;
            border-left: 2px solid #808080;
            border-right: 2px solid #ffffff;
            border-bottom: 2px solid #ffffff;
        }}

        h1, h2, h3 {{
            color: #ffffff;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {{
            color: #ffffff;
        }}

        div.stButton > button,
        div.stButton > button *,
        button[kind="secondary"],
        button[kind="secondary"] *,
        button[kind="primary"],
        button[kind="primary"] *,
        [data-testid^="stBaseButton"],
        [data-testid^="stBaseButton"] *,
        [data-testid="stButton"] button,
        [data-testid="stButton"] button * {{
            color: #000000 !important;
        }}

        div[data-testid="stAlert"] {{
            background-color: #eef3f8 !important;
            border: 1px solid #808080;
        }}

        div[data-testid="stAlert"],
        div[data-testid="stAlert"] * {{
            color: #000000 !important;
        }}

        div[data-testid="stToastContainer"] {{
            background: transparent !important;
            border: none !important;
        }}

        div[data-testid="stToastContainer"] > div {{
            background-color: #aaaaaa !important;
            border: 1px solid #808080 !important;
        }}

        div[data-testid="stToastContainer"] * {{
            color: #000000 !important;
        }}

        .stRadio label,
        .stRadio label *,
        .stRadio span,
        .stRadio p {{
            color: #000000 !important;
        }}

        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label *,
        [data-testid="stCheckbox"] span,
        [data-testid="stCheckbox"] p {{
            color: #ffffff !important;
        }}

        div[data-testid="stDataFrame"] * {{
            color: #000000;
        }}

        div.stButton > button {{
            display: flex;
            align-items: center;
            justify-content: center;
            background: #dcdcdc;
            color: #000000;
            border-radius: 0px;
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            min-width: 34px;
            min-height: 32px;
            height: auto;
            padding: 6px 12px;
            font-weight: 700;
            font-family: "MS Sans Serif", Tahoma, sans-serif;
            box-shadow: none;
        }}

        div.stButton > button p {{
            margin: 0;
            color: #000000 !important;
        }}

        .hover-card {{
            position: relative;
            display: block;
        }}

        .hover-card-tooltip {{
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            margin-top: 6px;
            z-index: 999;
            background: #050505;
            color: #f2f2f2 !important;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 10px 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            line-height: 1.45;
            white-space: pre;
        }}

        .hover-card-tooltip * {{
            color: #f2f2f2 !important;
        }}

        .hover-card:hover .hover-card-tooltip {{
            display: block;
        }}

        div.stButton > button:active {{
            border-top: 2px solid #808080;
            border-left: 2px solid #808080;
            border-right: 2px solid #ffffff;
            border-bottom: 2px solid #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


RETRO_CHART_COLORWAY = ["#0000FF", "#008000", "#FF0000", "#00BFBF", "#BF00BF", "#BFBF00", "#404040"]
RETRO_CHART_FONT = "Courier New, Consolas, monospace"


def apply_retro_chart_style(fig) -> None:
    """Classic engineering-software look: white paper, boxed mirrored
    axes, light gridlines, monospace font, MATLAB-style line colors.
    """
    fig.update_layout(
        font=dict(family=RETRO_CHART_FONT, size=12, color="#1a1a1a"),
        title_font=dict(family=RETRO_CHART_FONT, size=15, color="#000000"),
        paper_bgcolor="#f4f1ea",
        plot_bgcolor="#ffffff",
        colorway=RETRO_CHART_COLORWAY,
        legend=dict(
            bgcolor="#ffffff",
            bordercolor="#000000",
            borderwidth=1,
            font=dict(family=RETRO_CHART_FONT, size=11),
        ),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    axis_style = dict(
        showline=True,
        linecolor="#000000",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="#000000",
        tickfont=dict(family=RETRO_CHART_FONT, size=11),
        gridcolor="#cccccc",
        gridwidth=0.6,
        zeroline=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)


def plot_retro(fig, **kwargs) -> None:
    apply_retro_chart_style(fig)
    kwargs.setdefault("use_container_width", True)
    st.plotly_chart(fig, **kwargs)


def auto_refresh(seconds: int = REFRESH_SECONDS) -> None:
    if st_autorefresh is not None:
        st_autorefresh(interval=seconds * 1000, key="auto_refresh")
    else:
        st.warning("Install streamlit-autorefresh for automatic dashboard refresh.")


@st.cache_data(ttl=REFRESH_SECONDS)
def load_master_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


def seven_day_window(df: pd.DataFrame) -> pd.DataFrame:
    latest_time = df["timestamp_utc"].max()
    start_time = latest_time - pd.Timedelta(days=7)
    return df[df["timestamp_utc"] >= start_time].copy()


def freshness_status(df: pd.DataFrame, column: str, max_age_minutes: int) -> tuple[str, str]:
    if df.empty or column not in df.columns:
        return "No data", "N/A"

    clean = df.dropna(subset=[column])
    if clean.empty:
        return "No data", "N/A"

    latest_time = clean["timestamp_utc"].max()
    now_utc = pd.Timestamp.now(tz="UTC")
    age_minutes = (now_utc - latest_time).total_seconds() / 60

    status = "Fresh" if age_minutes <= max_age_minutes else "Stale"

    if age_minutes < 60:
        age_text = f"{age_minutes:.1f} min old"
    else:
        age_text = f"{age_minutes / 60:.1f} hr old"

    return status, age_text



def format_value(value, suffix: str = "", decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    if decimals == 0:
        return f"{value:.0f}{suffix}"
    return f"{value:.{decimals}f}{suffix}"


def metric_card(label: str, value: str, caption: str = "", tooltip: str = "", value_color: str = "#000000") -> None:
    """Fixed-height card so a row of cards always lines up evenly, even when
    one value wraps to two lines (e.g. a date + time) or one caption is
    longer than its neighbors. The value area reserves space for two lines
    and vertically centers its content, so short and long values still
    start their caption at the same point; nothing is clamped or hidden,
    so a card with unusually long text simply grows past the fixed height
    rather than losing content.
    """
    title_attr = f' title="{escape(tooltip)}"' if tooltip else ""
    info_icon = " ⓘ" if tooltip else ""
    st.markdown(
        f"""
        <div{title_attr} style="
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            padding: 14px 16px;
            height: 158px;
            box-sizing: border-box;
            background: #dcdcdc;
            color: #000000;
            font-family: 'MS Sans Serif', Tahoma, sans-serif;
            display: flex;
            flex-direction: column;
        ">
            <div style="font-size: 0.85rem; color: #000080; line-height: 1.2;">{label}{info_icon}</div>
            <div style="
                font-size: 1.6rem;
                font-weight: 700;
                color: {value_color};
                line-height: 1.15;
                min-height: 2.3em;
                display: flex;
                align-items: center;
                overflow-wrap: break-word;
                word-break: break-word;
            ">{value}</div>
            <div style="font-size: 0.72rem; color: #404040; line-height: 1.25; flex-grow: 1;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    "high": "#1f5a2e",
    "moderate": "#7a5a1f",
    "low": "#7a1f1f",
}


def status_badge_html(status: str) -> str:
    text, border_color, bg_color = STATUS_BADGE_STYLES.get(status, (status.title(), "#808080", "#dcdcdc"))
    return (
        f'<span style="display:inline-block; padding:2px 10px; border-radius:10px; '
        f'background:{bg_color}; border:1px solid {border_color}; color:{border_color}; '
        f'font-size:0.75rem; font-weight:700;">{escape(text)}</span>'
    )


def extreme_card_with_hover(
    label: str,
    value: str,
    caption: str,
    row: pd.Series | None,
    skip_field: str = "",
) -> None:
    field_order = [
        ("speed", "SPD ", "solar_wind_speed", " km/s", 1),
        ("density", "DENS", "proton_density", " p/cm3", 2),
        ("temp", "TEMP", "temperature", " K", 0),
        ("bz", "Bz  ", "bz", " nT", 2),
        ("kp", "Kp  ", "kp", "", 1),
        ("dst", "Dst ", "dst", " nT", 1),
    ]

    lines = []
    for key, tag, column, suffix, decimals in field_order:
        if key == skip_field:
            continue
        cell_value = None if row is None else row.get(column)
        lines.append(f"{tag}: {format_value(cell_value, suffix, decimals)}")

    tooltip_text = "\n".join(lines) if lines else "No data"

    st.markdown(
        f"""
        <div class="hover-card">
            <div style="
                border-top: 2px solid #ffffff;
                border-left: 2px solid #ffffff;
                border-right: 2px solid #808080;
                border-bottom: 2px solid #808080;
                padding: 16px;
                min-height: 118px;
                background: #dcdcdc;
                color: #000000;
                font-family: 'MS Sans Serif', Tahoma, sans-serif;
            ">
                <div style="font-size: 0.85rem; color: #000080;">{label}</div>
                <div style="font-size: 1.8rem; font-weight: 700; margin-top: 6px; color: #000000;">{value}</div>
                <div style="font-size: 0.8rem; color: #404040; margin-top: 8px;">{caption}</div>
            </div>
            <div class="hover-card-tooltip">{escape(tooltip_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_note(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            min-height: 28px;
            margin-top: 6px;
            margin-bottom: 20px;
            padding: 4px 10px;
            background: rgba(0, 0, 0, 0.45);
            border-radius: 2px;
            display: inline-block;
            color: #f2f2f2;
            font-size: 0.85rem;
            white-space: nowrap;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def latest_non_null(df: pd.DataFrame, column: str):
    clean = df.dropna(subset=[column])
    if clean.empty:
        return np.nan, None

    row = clean.iloc[-1]
    return row[column], row["timestamp_utc"]


@st.cache_data(ttl=REFRESH_SECONDS)
def load_processed_data(name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / name / f"{name}_processed.parquet"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


def latest_value(df: pd.DataFrame, column: str, dataset: str | None = None):
    source_df = load_processed_data(dataset) if dataset else df

    if source_df.empty:
        source_df = df

    return latest_non_null(source_df, column)


def format_event_time(value) -> str:
    if pd.isna(value):
        return "N/A"

    timestamp = pd.to_datetime(value, utc=True)
    return timestamp.strftime("%d %b %H:%M UTC")


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


def status_terminal(df: pd.DataFrame) -> None:
    speed, speed_time = latest_value(df, "solar_wind_speed", "solar_wind")
    density, density_time = latest_value(df, "proton_density", "solar_wind")
    temperature, temperature_time = latest_value(df, "temperature", "solar_wind")
    bz, bz_time = latest_value(df, "bz", "imf")
    kp, kp_time = latest_value(df, "kp", "kp")
    dst, dst_time = latest_value(df, "dst", "dst")

    rows = [
        ("Speed", format_value(speed, " km/s", 1), speed_time, "speed", speed),
        ("Density", format_value(density, " p/cm3", 2), density_time, "density", density),
        ("Temp", format_value(temperature, " K", 0), temperature_time, "temperature", temperature),
        ("Bz", format_value(bz, " nT", 2), bz_time, "bz", bz),
        ("Kp", format_value(kp, "", 1), kp_time, "kp", kp),
        ("Dst", format_value(dst, " nT", 1), dst_time, "dst", dst),
    ]

    rows_html = ""
    for name, value_text, time_value, column, raw_value in rows:
        meaning, risk = variable_meaning_and_risk(column, raw_value)
        rows_html += (
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(value_text)}</td>"
            f"<td>{escape(latest_label_time(time_value))}</td>"
            f"<td>{escape(meaning)}</td>"
            f"<td>{escape(risk)}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <style>
        .terminal-wrap {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            box-sizing: border-box;
            width: 100%;
            overflow-x: auto;
        }}
        .terminal-wrap .terminal-title {{
            color: #f2f2f2;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .terminal-wrap .terminal-window {{
            color: #9adfff;
            font-size: 0.78rem;
            margin-bottom: 10px;
        }}
        table.terminal-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.76rem;
        }}
        table.terminal-table th,
        table.terminal-table td {{
            border: 1px solid #333333;
            padding: 4px 8px;
            text-align: left;
            white-space: normal;
            word-wrap: break-word;
            overflow-wrap: break-word;
            color: #f2f2f2 !important;
        }}
        table.terminal-table th:nth-child(1), table.terminal-table td:nth-child(1) {{ width: 9%; }}
        table.terminal-table th:nth-child(2), table.terminal-table td:nth-child(2) {{ width: 13%; }}
        table.terminal-table th:nth-child(3), table.terminal-table td:nth-child(3) {{ width: 18%; }}
        table.terminal-table th:nth-child(4), table.terminal-table td:nth-child(4) {{ width: 28%; }}
        table.terminal-table th:nth-child(5), table.terminal-table td:nth-child(5) {{ width: 32%; }}
        table.terminal-table th {{
            color: #00ff88 !important;
            font-weight: 700;
        }}
        </style>
        <div class="terminal-wrap">
            <div class="terminal-title">SW-DSS STATUS TERMINAL</div>
            <div class="terminal-window">DATASET WINDOW: {escape(date_window_label(df))}</div>
            <table class="terminal-table">
                <thead>
                    <tr>
                        <th>VAR</th>
                        <th>VALUE</th>
                        <th>TIME (UTC)</th>
                        <th>MEANING</th>
                        <th>RISK</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    anchor_time = pick_anchor_time(dst_time, kp_time, bz_time)
    if st.button("🔍 Solar Event", key="terminal_solar_event"):
        open_dialog("reverse_explorer", (anchor_time, "Current Conditions"))


def latest_label_time(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.to_datetime(value, utc=True).strftime("%d %b %H:%M UTC")


def date_window_label(df: pd.DataFrame) -> str:
    if df.empty or "timestamp_utc" not in df.columns:
        return "No date window"

    start = df["timestamp_utc"].min()
    end = df["timestamp_utc"].max()

    return f"{start.strftime('%d %b')} to {end.strftime('%d %b')}"


def render_simple_retro_table(df: pd.DataFrame, display_names: dict | None = None) -> None:
    """Generic retro-styled HTML table for arbitrary dataframes (e.g. the
    'Latest Events' / 'Latest CMEs' tables), matching the same vintage
    look used by the reference pagers and top_event_table().
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


def top_event_table(df: pd.DataFrame, key_column: str, mode: str, title: str) -> None:
    if key_column not in df.columns:
        st.info(f"{key_column} column not available.")
        return

    if "timestamp_utc" not in df.columns:
        st.info("timestamp_utc column not available.")
        return

    clean = df.dropna(subset=[key_column]).copy()

    if clean.empty:
        st.info(f"No data available for {title}.")
        return

    ascending = mode == "lowest"
    top_df = clean.sort_values(key_column, ascending=ascending).head(5).copy()

    top_df["event_time"] = top_df["timestamp_utc"].apply(format_event_time)

    all_columns = [
        "solar_wind_speed",
        "proton_density",
        "temperature",
        "bz",
        "kp",
        "dst",
    ]

    remaining_columns = [
        col for col in all_columns
        if col in top_df.columns and col != key_column
    ]

    final_columns = ["event_time", key_column, *remaining_columns]
    top_df = top_df[final_columns]

    display_names = {
        "event_time": "Time (UTC)",
        "solar_wind_speed": "Speed",
        "proton_density": "Density",
        "temperature": "Temperature",
        "bz": "Bz",
        "kp": "Kp",
        "dst": "Dst",
    }

    def fmt(col, val):
        if pd.isna(val):
            return "N/A"
        if col == "event_time":
            return str(val)
        if col == "temperature":
            return f"{float(val):.0f}"
        if col in ["bz", "kp"]:
            return f"{float(val):.2f}"
        if col == "dst":
            return f"{float(val):.0f}"
        return f"{float(val):.1f}"

    html = """
    <style>
    .retro-table {
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
    .retro-table th, .retro-table td {
        border: 1px solid #d0d0d0;
        padding: 8px 10px;
        white-space: nowrap;
    }
    .retro-table th {
        background: #efefef;
        color: #606060;
        font-weight: 400;
        text-align: left;
    }
    .hover-cell {
        cursor: help;
        text-decoration: underline dotted #000080;
    }
    </style>
    <table class="retro-table">
    """

    html += "<thead><tr>"
    for col in final_columns:
        html += f"<th>{escape(display_names.get(col, col))}</th>"
    html += "</tr></thead><tbody>"

    for _, row in top_df.iterrows():
        html += "<tr>"
        for col in final_columns:
            value = fmt(col, row[col])
            tooltip = reference_tooltip(col, row[col]) if col in ["bz", "kp", "dst"] else ""
            class_name = "hover-cell" if tooltip else ""
            html += f'<td class="{class_name}" title="{escape(tooltip)}">{escape(value)}</td>'
        html += "</tr>"

    html += "</tbody></table>"

    st.markdown(f"### {title}")
    st.markdown(html, unsafe_allow_html=True)


def render_paged_reference_table(reference_tables: list[dict], session_key: str, key_prefix: str) -> None:
    """Shared renderer for the Range/Meaning/Risk-style reference pagers
    used across Home, Heliosphere, Geospace, and Photosphere. Pads every
    table to the same row count so switching tables never resizes the
    panel (no scrolling, no page-shift), and keeps the prev/next buttons
    below the table instead of overlapping it.
    """
    if session_key not in st.session_state:
        st.session_state[session_key] = 0

    idx = st.session_state[session_key]
    current = reference_tables[idx]
    columns = current.get("columns", ["Range", "Meaning", "Risk"])
    max_rows = max(len(t["data"]) for t in reference_tables)

    rows = current["data"]
    if rows and isinstance(rows[0], dict):
        rows = [[entry.get(col, "") for col in columns] for entry in rows]

    padded_rows = list(rows)
    blank_row = ["" for _ in columns]
    while len(padded_rows) < max_rows:
        padded_rows.append(blank_row)

    table_html = """
    <style>
    .retro-table-photo {
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
    .retro-table-photo th, .retro-table-photo td {
        border: 1px solid #d0d0d0;
        padding: 5px 10px;
        white-space: nowrap;
    }
    .retro-table-photo td.blank-row {
        border-color: transparent;
        background: #ffffff;
    }
    .retro-table-photo th {
        background: #efefef;
        color: #606060;
        font-weight: 400;
        text-align: left;
    }
    </style>
    <table class="retro-table-photo">
    """

    table_html += "<thead><tr>" + "".join(f"<th>{escape(col)}</th>" for col in columns) + "</tr></thead><tbody>"
    for row in padded_rows:
        is_blank = row is blank_row
        cell_class = ' class="blank-row"' if is_blank else ""
        table_html += (
            "<tr>" + "".join(f"<td{cell_class}>{escape(str(cell)) or '&nbsp;'}</td>" for cell in row) + "</tr>"
        )
    table_html += "</tbody></table>"

    st.markdown(
        f"""
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff; white-space:nowrap; margin-bottom:2px;">
            {escape(current['title'])}
        </div>
        {table_html}
        """,
        unsafe_allow_html=True,
    )

    prev_col, next_col, _spacer_col = st.columns([0.12, 0.12, 0.76])

    with prev_col:
        if st.button("‹", key=f"{key_prefix}_prev"):
            st.session_state[session_key] = (idx - 1) % len(reference_tables)
            st.rerun()

    with next_col:
        if st.button("›", key=f"{key_prefix}_next"):
            st.session_state[session_key] = (idx + 1) % len(reference_tables)
            st.rerun()


def heliosphere_reference_window() -> None:
    render_paged_reference_table(
        [
            {
                "title": "Speed Reference",
                "data": [
                    {"Range": "< 400 km/s", "Meaning": "Slow solar wind", "Risk": "Usually quiet"},
                    {"Range": "400-500 km/s", "Meaning": "Moderate speed", "Risk": "Normal/active"},
                    {
                        "Range": "500-700 km/s",
                        "Meaning": "Fast solar wind",
                        "Risk": "Storm possible with southward Bz",
                    },
                    {"Range": "> 700 km/s", "Meaning": "Very fast wind", "Risk": "Enhanced storm potential"},
                ],
            },
            {
                "title": "Density Reference",
                "data": [
                    {"Range": "< 5 p/cm3", "Meaning": "Low density", "Risk": "Weak pressure"},
                    {"Range": "5-10 p/cm3", "Meaning": "Moderate density", "Risk": "Normal solar wind"},
                    {"Range": "10-30 p/cm3", "Meaning": "High density", "Risk": "Compression possible"},
                    {"Range": "> 30 p/cm3", "Meaning": "Very high density", "Risk": "Shock/CME sheath possible"},
                ],
            },
            {
                "title": "Temperature Reference",
                "data": [
                    {"Range": "< 50,000 K", "Meaning": "Cool wind", "Risk": "Usually quiet"},
                    {"Range": "50,000-150,000 K", "Meaning": "Typical wind", "Risk": "Normal"},
                    {"Range": "150,000-500,000 K", "Meaning": "Hot wind", "Risk": "Disturbed flow possible"},
                    {"Range": "> 500,000 K", "Meaning": "Very hot plasma", "Risk": "Shock/CME heating possible"},
                ],
            },
            {
                "title": "Bz Reference",
                "data": [
                    {"Range": "Bz > 0 nT", "Meaning": "Northward IMF", "Risk": "Low coupling"},
                    {"Range": "0 to -5 nT", "Meaning": "Weak southward IMF", "Risk": "Minor activity possible"},
                    {"Range": "-5 to -10 nT", "Meaning": "Moderate southward IMF", "Risk": "Storm possible"},
                    {
                        "Range": "-10 to -20 nT",
                        "Meaning": "Strong southward IMF",
                        "Risk": "Strong storm coupling",
                    },
                    {"Range": "< -20 nT", "Meaning": "Extreme southward IMF", "Risk": "Severe storm potential"},
                ],
            },
        ],
        session_key="heliosphere_reference_idx",
        key_prefix="heliosphere_reference",
    )


def geospace_reference_window() -> None:
    render_paged_reference_table(
        [
            {
                "title": "Kp Reference",
                "data": [
                    {"Range": "0-3", "Meaning": "Quiet", "Risk": "Normal"},
                    {"Range": "4", "Meaning": "Active", "Risk": "Unsettled field"},
                    {"Range": "5", "Meaning": "G1 storm", "Risk": "Minor storm"},
                    {"Range": "6", "Meaning": "G2 storm", "Risk": "Moderate storm"},
                    {"Range": "7", "Meaning": "G3 storm", "Risk": "Strong storm"},
                    {"Range": "8-9", "Meaning": "G4-G5 storm", "Risk": "Severe/extreme storm"},
                ],
            },
            {
                "title": "Dst Reference",
                "data": [
                    {"Range": "Dst > -30 nT", "Meaning": "Quiet", "Risk": "Low storm activity"},
                    {"Range": "-30 to -50 nT", "Meaning": "Weak storm", "Risk": "Minor ring current"},
                    {"Range": "-50 to -100 nT", "Meaning": "Moderate storm", "Risk": "Storm underway"},
                    {"Range": "-100 to -200 nT", "Meaning": "Intense storm", "Risk": "Strong disturbance"},
                    {"Range": "< -200 nT", "Meaning": "Superstorm", "Risk": "Extreme disturbance"},
                ],
            },
        ],
        session_key="geospace_reference_idx",
        key_prefix="geospace_reference",
    )


def photosphere_reference_window() -> None:
    reference_tables = [
        {
            "title": "CME Speed Reference",
            "columns": ["Speed (km/s)", "Category", "Interpretation"],
            "data": [
                ["< 300", "Very Slow", "Usually weak, low impact"],
                ["300-500", "Slow", "Typical solar wind speed"],
                ["500-800", "Moderate", "Can produce moderate disturbances"],
                ["800-1200", "Fast", "Higher chance of geomagnetic effects"],
                ["1200-1800", "Very Fast", "Potentially geoeffective CME"],
                ["> 1800", "Extreme", "Major space weather event possible"],
            ],
        },
        {
            "title": "CME Longitude Reference",
            "columns": ["Longitude", "Interpretation"],
            "data": [
                ["-30° to +30°", "Near Earth-directed (highest concern)"],
                ["±30° to ±60°", "Possible Earth impact"],
                ["±60° to ±120°", "Unlikely Earth impact"],
                ["> ±120°", "Usually away from Earth"],
            ],
        },
        {
            "title": "CME Half Angle (Width)",
            "columns": ["Half Angle", "Category", "Interpretation"],
            "data": [
                ["< 20°", "Narrow", "Usually localized"],
                ["20°-40°", "Moderate", "Medium-sized CME"],
                ["40°-60°", "Wide", "Greater chance of Earth impact"],
                ["> 60°", "Halo / Very Wide", "Potentially Earth-directed"],
            ],
        },
        {
            "title": "Solar Flare Classification",
            "columns": ["Flare Class", "Peak X-ray Flux (W/m²)", "Interpretation"],
            "data": [
                ["A", "< 10⁻⁷", "Very weak"],
                ["B", "10⁻⁷ - 10⁻⁶", "Weak"],
                ["C", "10⁻⁶ - 10⁻⁵", "Minor"],
                ["M", "10⁻⁵ - 10⁻⁴", "Strong"],
                ["X", "> 10⁻⁴", "Extreme"],
            ],
        },
        {
            "title": "Radio Burst Reference",
            "columns": ["Type", "Meaning", "Importance"],
            "data": [
                ["Type II", "Shock wave", "Strong CME indicator"],
                ["Type III", "Fast electron beams", "Flare indicator"],
                ["Type IV", "Large magnetic structure", "Major eruption"],
            ],
        },
        {
            "title": "F10.7 Solar Flux",
            "columns": ["Flux (SFU)", "Solar Activity"],
            "data": [
                ["< 70", "Very Low"],
                ["70-100", "Quiet"],
                ["100-150", "Moderate"],
                ["150-200", "Active"],
                ["200-300", "High"],
                ["> 300", "Very High"],
            ],
        },
    ]

    render_paged_reference_table(
        reference_tables,
        session_key="photosphere_reference_idx",
        key_prefix="photosphere_reference",
    )


def time_caption(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return f"Recorded at {row['timestamp_utc']}"


def nearest_row_in(df: pd.DataFrame, target_time, tolerance_hours: float = 6) -> pd.Series | None:
    if df.empty or target_time is None or pd.isna(target_time) or "timestamp_utc" not in df.columns:
        return None
    diffs = (df["timestamp_utc"] - target_time).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(hours=tolerance_hours):
        return None
    return df.loc[idx]


def render_overview_chart(days: int = 4) -> None:
    """A stacked multi-panel time series, mirroring the layout of NOAA's
    real-time solar wind plot: IMF Bt/Bz, Dst, Density, Temperature,
    Speed, and Kp. Solar Wind/IMF panels use the real minute-level
    processed data (not the hourly-averaged master_df) so genuine spikes
    and noise show up instead of being smoothed away. Kp/Dst stay at
    their native 3-hour/1-hour cadence.

    All traces are placed on one literal shared x-axis (not separate
    per-row axes like make_subplots would give you) and stacked purely
    via y-axis "domain" slices. That's what makes hovering over any one
    panel trigger every panel's own tooltip at that same x position
    (synced by one vertical crosshair line), instead of only the panel
    directly under the cursor.
    """
    sw = recent_window(load_processed_data("solar_wind"), days)
    imf = recent_window(load_processed_data("imf"), days)
    kp_df = recent_window(load_processed_data("kp"), days)
    dst_df = recent_window(load_processed_data("dst"), days)

    panel_labels = [
        "IMF Bt / Bz (nT)",
        "Dst (nT)",
        "Proton Density (p/cm³)",
        "Temperature (K)",
        "Solar Wind Speed (km/s)",
        "Kp Index",
    ]
    panel_count = len(panel_labels)
    gap = 0.025
    panel_height = (1 - gap * (panel_count - 1)) / panel_count

    def domain_for(panel_index: int) -> list[float]:
        top = 1 - panel_index * (panel_height + gap)
        bottom = top - panel_height
        return [max(bottom, 0.0), top]

    fig = go.Figure()

    point_marker = dict(size=3, opacity=0.6)

    if not imf.empty:
        if "bt" in imf.columns:
            fig.add_trace(
                go.Scatter(
                    x=imf["timestamp_utc"],
                    y=imf["bt"],
                    name="Bt",
                    yaxis="y1",
                    mode="lines+markers",
                    line=dict(color="#404040", width=1),
                    marker=dict(**point_marker, color="#404040"),
                )
            )
        if "bz" in imf.columns:
            fig.add_trace(
                go.Scatter(
                    x=imf["timestamp_utc"],
                    y=imf["bz"],
                    name="Bz",
                    yaxis="y1",
                    mode="lines+markers",
                    line=dict(color="#FF0000", width=1),
                    marker=dict(**point_marker, color="#FF0000"),
                )
            )

    if not dst_df.empty and "dst" in dst_df.columns:
        fig.add_trace(
            go.Scatter(
                x=dst_df["timestamp_utc"],
                y=dst_df["dst"],
                name="Dst",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#BF00BF", width=1.5),
                marker=dict(size=5, opacity=0.6, color="#BF00BF"),
                fill="tozeroy",
            )
        )

    if not sw.empty:
        if "proton_density" in sw.columns:
            fig.add_trace(
                go.Scatter(
                    x=sw["timestamp_utc"],
                    y=sw["proton_density"],
                    name="Density",
                    yaxis="y3",
                    mode="lines+markers",
                    line=dict(color="#FF8C00", width=1),
                    marker=dict(**point_marker, color="#FF8C00"),
                )
            )
        if "temperature" in sw.columns:
            fig.add_trace(
                go.Scatter(
                    x=sw["timestamp_utc"],
                    y=sw["temperature"],
                    name="Temperature",
                    yaxis="y4",
                    mode="lines+markers",
                    line=dict(color="#008000", width=1),
                    marker=dict(**point_marker, color="#008000"),
                )
            )
        if "solar_wind_speed" in sw.columns:
            fig.add_trace(
                go.Scatter(
                    x=sw["timestamp_utc"],
                    y=sw["solar_wind_speed"],
                    name="Speed",
                    yaxis="y5",
                    mode="lines+markers",
                    line=dict(color="#BFBF00", width=1.5),
                    marker=dict(**point_marker, color="#BFBF00"),
                )
            )

    if not kp_df.empty and "kp" in kp_df.columns:
        fig.add_trace(
            go.Bar(x=kp_df["timestamp_utc"], y=kp_df["kp"], name="Kp", yaxis="y6", marker_color="#2e7d32")
        )

    fig.update_layout(
        xaxis=dict(
            domain=[0, 1],
            anchor="y6",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#888888",
            spikethickness=1,
        ),
        yaxis=dict(domain=domain_for(0), anchor="x"),
        yaxis2=dict(domain=domain_for(1), anchor="x"),
        yaxis3=dict(domain=domain_for(2), anchor="x", type="log"),
        yaxis4=dict(domain=domain_for(3), anchor="x", type="log"),
        yaxis5=dict(domain=domain_for(4), anchor="x"),
        yaxis6=dict(domain=domain_for(5), anchor="x", range=[0, 9]),
        hovermode="x",
        height=620,
        showlegend=True,
        title=f"Sun-to-Earth Overview (Last {days} Days)",
        annotations=[
            dict(
                text=label,
                xref="paper",
                yref="paper",
                x=0,
                y=domain_for(i)[1] + 0.012,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=12, color="#000000"),
            )
            for i, label in enumerate(panel_labels)
        ],
    )

    apply_retro_chart_style(fig)

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="overview_chart",
        on_select="rerun",
    )

    points = event.selection.points if event and event.selection else []

    if not points:
        st.caption("Click anywhere on the chart to see every panel's value at that exact time.")
        return

    clicked_time = pd.to_datetime(points[0].get("x"), utc=True)

    sw_row = nearest_row_in(sw, clicked_time, tolerance_hours=1)
    imf_row = nearest_row_in(imf, clicked_time, tolerance_hours=1)
    kp_row = nearest_row_in(kp_df, clicked_time, tolerance_hours=4)
    dst_row = nearest_row_in(dst_df, clicked_time, tolerance_hours=2)

    lines = [
        f"Bt: {format_value(None if imf_row is None else imf_row.get('bt'), ' nT', 2)}",
        f"Bz: {format_value(None if imf_row is None else imf_row.get('bz'), ' nT', 2)}",
        f"Dst: {format_value(None if dst_row is None else dst_row.get('dst'), ' nT', 1)}",
        f"Density: {format_value(None if sw_row is None else sw_row.get('proton_density'), ' p/cm3', 2)}",
        f"Temperature: {format_value(None if sw_row is None else sw_row.get('temperature'), ' K', 0)}",
        f"Speed: {format_value(None if sw_row is None else sw_row.get('solar_wind_speed'), ' km/s', 1)}",
        f"Kp: {format_value(None if kp_row is None else kp_row.get('kp'), '', 1)}",
    ]

    _render_chain_box(
        f"All Panels at {clicked_time.strftime('%d %b %Y %H:%M UTC')}",
        lines,
        height=26 + (len(lines) + 1) * 19,
    )


def line_chart(df: pd.DataFrame, columns: list[str], title: str) -> None:
    available = [col for col in columns if col in df.columns]
    if not available:
        st.info("No data available for this chart yet.")
        return
    chart_df = df[["timestamp_utc", *available]].dropna(how="all", subset=available)
    fig = px.line(chart_df, x="timestamp_utc", y=available, title=title)
    fig.update_layout(height=390, legend_title_text="")
    plot_retro(fig)


def correlation_explorer(df: pd.DataFrame, columns: list[str], title: str) -> None:
    available = [col for col in columns if col in df.columns and df[col].notna().any()]

    st.subheader(title)

    if len(available) < 2:
        st.info("Not enough data for correlation analysis.")
        return

    corr = df[available].corr(numeric_only=True, min_periods=10)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix",
    )
    fig.update_layout(height=430)
    plot_retro(fig)

    col1, col2 = st.columns(2)
    x_col = col1.selectbox("X variable", available, key=f"{title}_x")
    y_options = [col for col in available if col != x_col]
    y_col = col2.selectbox("Y variable", y_options, key=f"{title}_y")

    scatter_df = df[["timestamp_utc", x_col, y_col]].dropna(subset=[x_col, y_col])
    if scatter_df.empty:
        st.info("No overlapping data for selected variables.")
        return

    corr_value = scatter_df[[x_col, y_col]].corr().iloc[0, 1]
    st.metric("Selected Correlation", format_value(corr_value, decimals=3))

    scatter = px.scatter(
        scatter_df,
        x=x_col,
        y=y_col,
        trendline="ols",
        hover_data=["timestamp_utc"],
        title=f"{x_col} vs {y_col}",
    )
    scatter.update_layout(height=430)
    plot_retro(scatter)


def render_prediction_queue_stats(dataset: str) -> None:
    stats = get_job_stats(dataset)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Running", str(stats["running"]), "Active prediction jobs")
    with c2:
        metric_card("Completed Today", str(stats["completed_today"]), "Jobs finished today")
    with c3:
        avg_mae_text = "N/A" if stats["avg_mae"] is None else f"{stats['avg_mae']:.2f}"
        metric_card("Average MAE", avg_mae_text, "Across all completed jobs")


def render_prediction_job_tiles(jobs: list[dict], empty_message: str) -> None:
    if not jobs:
        st.info(empty_message)
        return

    cols_per_row = 4
    for row_start in range(0, len(jobs), cols_per_row):
        chunk = jobs[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, job in zip(cols, chunk):
            with col:
                label = VARIABLE_LABELS.get(job["variable"], job["variable"])
                tile_icon = {"in_progress": "🟡", "evaluating": "🔵", "stopped": "⏹️"}.get(job["status"], "✅")
                tile_color = {"in_progress": "#7a5a1f", "evaluating": "#1f4a7a", "stopped": "#4a4a4a"}.get(
                    job["status"], "#3a3a3a"
                )
                start_hour = pd.Timestamp(job["start_hour"])
                is_kp_interval = job["dataset"] in ("analytics", "experimental") and job["variable"] == "kp"
                horizon_label = "Next Interval" if is_kp_interval else f"{job['horizon']}h"

                st.markdown(
                    f"""
                    <div style="
                        background:{tile_color};
                        border:2px solid #808080;
                        border-radius:4px 4px 0 0;
                        height:64px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:1.6rem;
                    ">{tile_icon}</div>
                    <div style="
                        background:#1c1c24;
                        border:2px solid #808080;
                        border-top:none;
                        border-radius:0 0 4px 4px;
                        padding:8px 10px 10px 10px;
                        margin-bottom:6px;
                    ">
                        <div style="font-weight:700; font-size:0.95rem; color:#f2f2f2; line-height:1.3;">
                            {escape(label)} &mdash; {horizon_label}
                        </div>
                        <div style="font-size:0.74rem; color:#b8b8c0; margin-top:4px; line-height:1.3;">
                            Started {start_hour.strftime('%d %b %H:%M UTC')}
                        </div>
                        <div style="margin-top:6px;">{status_badge_html(job['status'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Open", key=f"job_tile_{job['job_id']}", use_container_width=True):
                    open_dialog("prediction_job", job["job_id"])


@st.dialog("Saved Predictions", width="large", dismissible=False)
def show_saved_predictions(dataset: str) -> None:
    render_dialog_close_button("close_saved_predictions")

    label = DATASET_LABELS.get(dataset, dataset.title())
    st.subheader(f"{label} — Saved Predictions")

    jobs = get_saved_jobs(dataset)
    render_prediction_job_tiles(jobs, "No saved predictions yet. Save a completed job to keep it here permanently.")


def format_analytics_inputs(inputs: dict) -> str:
    """Renders the Solar Wind + IMF readings captured for one Analytics
    tick as two terminal lines — the live upstream conditions feeding the
    combined Kp/Dst model, distinct from the single self-referential
    reading the standalone Solar Wind/IMF/Kp/Dst tabs show.
    """
    if not inputs:
        return ""
    sw_parts = []
    if inputs.get("speed") is not None:
        sw_parts.append(f"Speed {inputs['speed']:.1f} km/s")
    if inputs.get("density") is not None:
        sw_parts.append(f"Density {inputs['density']:.2f} p/cm3")
    if inputs.get("temperature") is not None:
        sw_parts.append(f"Temp {inputs['temperature']:.0f} K")

    imf_parts = []
    for key, lbl in [("bt", "Bt"), ("bx_gsm", "Bx"), ("by_gsm", "By"), ("bz_gsm", "Bz")]:
        if inputs.get(key) is not None:
            imf_parts.append(f"{lbl} {inputs[key]:.2f} nT")

    lines = []
    if sw_parts:
        lines.append(f"<div>Solar Wind: {escape(' | '.join(sw_parts))}</div>")
    if imf_parts:
        lines.append(f"<div>IMF: {escape(' | '.join(imf_parts))}</div>")
    return "".join(lines)


def render_explainability_section(dataset: str, variable: str, horizon) -> None:
    """"Why did the model predict this value?" — shared across every job
    dialog (production and Research Lab alike). Purely diagnostic: reads
    the model and the most recently available live feature row, never
    retrains anything or affects the live prediction itself.
    """
    with st.expander("🔍 Why did the model predict this? (Feature Importance)"):
        try:
            result = explain_prediction(dataset, variable, horizon)
        except Exception as exc:
            st.warning(f"Could not compute explainability right now: {exc}")
            return

        if result["method"] == "unavailable" or not result["contributions"]:
            st.info("Not enough live feature history yet to explain this model's current prediction.")
            return

        method_label = "SHAP (exact Shapley values)" if result["method"] == "shap" else "Permutation sensitivity (SHAP unavailable for this model type)"
        st.caption(f"Method: {method_label} — Model: {result['model_name']}. Based on the most recently available live feature row.")

        contrib_df = pd.DataFrame(
            [
                {"Feature": feat, "Current Value": round(val, 3), "Contribution": round(contrib, 4)}
                for feat, val, contrib in result["contributions"]
            ]
        )
        fig = go.Figure()
        colors = ["#1f7a3a" if c >= 0 else "#7a1f1f" for c in contrib_df["Contribution"]]
        fig.add_trace(
            go.Bar(
                x=contrib_df["Contribution"],
                y=contrib_df["Feature"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title="Top Contributing Variables",
            height=320,
            xaxis_title="Contribution to prediction",
            yaxis=dict(autorange="reversed"),
        )
        plot_retro(fig)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        st.caption(
            "Positive contribution pushes the prediction up; negative pushes it down. "
            f"Predicted value: {result['predicted_value']:.2f}." if result["predicted_value"] is not None else ""
        )


def render_kp_forecast_dialog(job: dict) -> None:
    """Kp on the Analytics page runs TWO independent products from one job
    record — see predict.predict_kp_interval / predict_kp_rolling for the
    full reasoning:

    - Mode 1, Production Forecast: `production_prediction`, frozen at job
      creation from data strictly BEFORE the target interval (matching
      how the model was trained — features from block B predict block
      B+1, never block B's own value). This is the number used for
      official model evaluation and never changes after creation.
    - Mode 2, Operational Rolling Estimate: every tick logged after
      creation (job["ticks"]), computed from whatever data is freshest
      right now — deliberately off the training distribution, an
      "operational situational awareness" read, always labeled
      Experimental, never used for official evaluation.

    They're rendered in clearly separate sections below and never allowed
    to overwrite each other, matching how they're stored.
    """
    ticks = job["ticks"]
    metrics = job.get("metrics", {})

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])
    target_interval_end = target_hour + pd.Timedelta(hours=3)
    production_prediction = job.get("production_prediction")
    production_observed_at = job.get("production_observed_at")

    st.subheader("Kp Forecast")
    st.caption(
        "Two independent products: a frozen Production Forecast (used for official model "
        "evaluation) and an Experimental Rolling Estimate (situational awareness only). "
        "Neither ever overwrites the other."
    )

    # ==================== Production Forecast (Frozen) ====================
    st.markdown("##### Production Forecast (Frozen)")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        metric_card("Prediction Generated", created_at.strftime("%H:%M:%S UTC"), "")
    with pc2:
        metric_card(
            "Prediction Time",
            "N/A" if production_observed_at is None else pd.Timestamp(production_observed_at).strftime("%H:%M UTC"),
            "Input data timestamp — strictly before the interval, matching training",
        )
    with pc3:
        metric_card(
            "Forecast Interval", f"{target_hour.strftime('%H:%M')}–{target_interval_end.strftime('%H:%M UTC')}", ""
        )
    with pc4:
        metric_card(
            "Frozen Forecast Value",
            "N/A" if production_prediction is None else f"{production_prediction:.2f}",
            "Never recomputed — used for official model evaluation",
        )

    badge_key = "kp_waiting_official" if job["status"] in ("in_progress", "evaluating") else job["status"]
    st.markdown(status_badge_html(badge_key), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f}
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================== Operational Rolling Estimate (Experimental) ==================
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Operational Rolling Estimate (Experimental)")
    st.warning(
        "⚠️ Uses whatever upstream data is freshest right now, including data from inside the "
        "forecast interval itself — a situational-awareness read, not a calibrated forecast. "
        "Never used for official model evaluation; see Production Forecast above for that."
    )

    latest_tick = ticks[-1] if ticks else None
    latest_inputs = (latest_tick.get("inputs") or {}) if latest_tick else {}
    latest_minute_at = (
        pd.Timestamp(latest_tick["minute_at"]) if latest_tick and latest_tick.get("minute_at") else None
    )
    rolling_value = latest_tick["predicted_value"] if latest_tick else None
    drift = forecast_drift(job)

    if len(ticks) >= 2:
        rolling_diff = ticks[-1]["predicted_value"] - ticks[-2]["predicted_value"]
        if abs(rolling_diff) < 1e-9:
            rolling_trend = "Stable"
        elif rolling_diff > 0:
            rolling_trend = "Increasing"
        else:
            rolling_trend = "Decreasing"
    else:
        rolling_trend = "N/A"

    def _fmt(value, dp, suffix=""):
        return "N/A" if value is None else f"{value:.{dp}f}{suffix}"

    r1, r2c, r3, r4, r5 = st.columns(5)
    with r1:
        metric_card("Current Time", pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S UTC"), "")
    with r2c:
        metric_card(
            "Latest Solar Wind Update",
            _fmt(latest_inputs.get("speed"), 1, " km/s"),
            "" if latest_minute_at is None else latest_minute_at.strftime("%H:%M:%S UTC"),
        )
    with r3:
        metric_card(
            "Latest IMF Update",
            "N/A" if latest_inputs.get("bz_gsm") is None else f"Bz {latest_inputs['bz_gsm']:.2f} nT",
            "" if latest_minute_at is None else latest_minute_at.strftime("%H:%M:%S UTC"),
        )
    with r4:
        metric_card("Latest Dst", _fmt(latest_inputs.get("dst"), 1, " nT"), "")
    with r5:
        metric_card("Latest AE", _fmt(latest_inputs.get("ae"), 1, " nT"), "")

    r6, r7, r8, r9, r10 = st.columns(5)
    with r6:
        metric_card(
            "Current Rolling Estimate",
            "N/A" if rolling_value is None else f"{rolling_value:.2f}",
            "Experimental — not the production forecast",
        )
    with r7:
        metric_card(
            "Last Updated", "N/A" if latest_minute_at is None else latest_minute_at.strftime("%H:%M:%S UTC"), ""
        )
    with r8:
        metric_card("Trend", rolling_trend, "")
    with r9:
        if drift is None:
            metric_card("Prediction Drift", "N/A", "")
        else:
            sign = "+" if drift >= 0 else ""
            metric_card(
                "Prediction Drift",
                f"{sign}{drift:.2f}",
                "First rolling estimate to latest",
                tooltip="How much the rolling estimate has moved as new data arrived.",
            )
    with r10:
        r2_val = metrics.get("r2")
        metric_card(
            "Confidence",
            model_quality_label(r2_val),
            "Reflects the underlying model only — not calibrated for off-distribution input",
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Prediction History — Rolling Estimate Timeline")

    if ticks:
        estimate_col = f"Rolling Estimate ({target_hour.strftime('%H:%M')}–{target_interval_end.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Latest Dst (nT)",
            "Latest Official Kp",
            "Latest AE (nT)",
            estimate_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("dst"), 1),
                _fmt(inputs.get("kp"), 2),
                _fmt(inputs.get("ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .kp-console {{
                background: #050505;
                border: 2px solid #ffffff;
                box-shadow: 3px 3px 0px #808080;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.kp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.kp-console-table th, table.kp-console-table td {{
                border: 1px solid #1a3a2a;
                padding: 3px 8px;
                text-align: right;
                color: #d8ffe8;
            }}
            table.kp-console-table th {{
                color: #00ff88;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.kp-console-table td:first-child, table.kp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="kp-console">
                <table class="kp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name="Rolling Estimate (Experimental)",
            )
        )
        fig.update_layout(
            title="Rolling Estimate Timeline (Experimental)",
            height=360,
            legend_title_text="",
            yaxis_title="Rolling Kp Estimate",
        )
        plot_retro(fig)

    # ==================== Verification ====================
    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Verification")

        actual = job["actual_value"]
        prod_err = production_error(job)
        prod_bias = production_bias(job)
        roll_err = rolling_final_error(job)
        final_rolling_value = ticks[-1]["predicted_value"] if ticks else None

        st.markdown("**Production Forecast → Official Kp** (official model evaluation)")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            metric_card(
                "Frozen Forecast", "N/A" if production_prediction is None else f"{production_prediction:.2f}", ""
            )
        with v2:
            metric_card(
                "Official Kp",
                "Pending" if actual is None else f"{actual:.2f}",
                "" if job["status"] == "completed" else "Stopped before the interval closed",
            )
        with v3:
            metric_card(
                "Production Error", "N/A" if prod_err is None else f"{prod_err:.2f}", "Primary official accuracy metric"
            )
        with v4:
            metric_card(
                "Production Bias",
                "N/A" if prod_bias is None else f"{prod_bias:+.2f}",
                "Signed — positive means the model runs high",
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("**Final Rolling Estimate → Official Kp** (research comparison only, never official)")
        v5, v6, v7 = st.columns(3)
        with v5:
            metric_card(
                "Final Rolling Estimate",
                "N/A" if final_rolling_value is None else f"{final_rolling_value:.2f}",
                "Experimental",
            )
        with v6:
            metric_card("Official Kp", "Pending" if actual is None else f"{actual:.2f}", "")
        with v7:
            metric_card("Rolling Error", "N/A" if roll_err is None else f"{roll_err:.2f}", "Research only")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            r2 = metrics.get("r2")
            metric_card("Model Quality", model_quality_label(r2), "N/A" if r2 is None else f"R² = {r2:.4f}")
        with m2:
            metric_card("MAE", f"{metrics.get('mae', float('nan')):.3f}", "Model's typical training error")
        with m3:
            metric_card("RMSE", f"{metrics.get('rmse', float('nan')):.3f}", "")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section(job["dataset"], "kp", "interval")
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


def render_dst_forecast_dialog(job: dict) -> None:
    """Dst on the Analytics page is still horizon-based (1h/3h/6h/12h/24h,
    not NOAA's fixed publishing cadence like Kp), but otherwise gets the
    exact same live console + completion summary architecture as Kp's
    dedicated dialog, since both are driven by the same combined Solar
    Wind + IMF + geomagnetic feature set.
    """
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    horizon = job["horizon"]

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])

    st.subheader("Dst Forecast")
    st.markdown(
        f"**Started:** {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
        f"**Target:** {target_hour.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Horizon:** {horizon}h"
    )
    st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} nT
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} nT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Live Forecast Console")

    def _fmt(value, dp):
        return "N/A" if value is None else f"{value:.{dp}f}"

    if ticks:
        forecast_col = f"Forecast ({target_hour.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Latest Dst (nT)",
            "Latest Official Kp",
            "Latest AE (nT)",
            forecast_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("dst"), 1),
                _fmt(inputs.get("kp"), 2),
                _fmt(inputs.get("ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .kp-console {{
                background: #050505;
                border: 2px solid #ffffff;
                box-shadow: 3px 3px 0px #808080;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.kp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.kp-console-table th, table.kp-console-table td {{
                border: 1px solid #1a3a2a;
                padding: 3px 8px;
                text-align: right;
                color: #d8ffe8;
            }}
            table.kp-console-table th {{
                color: #00ff88;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.kp-console-table td:first-child, table.kp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="kp-console">
                <table class="kp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name="Predicted Dst",
            )
        )
        fig.update_layout(
            title="Dst Forecast Drift Toward Target",
            height=360,
            legend_title_text="",
            yaxis_title="Predicted Dst (nT)",
        )
        plot_retro(fig)

    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Forecast Summary")

        final_pred = ticks[-1]["predicted_value"] if ticks else None
        avg_pred = average_prediction(job)
        actual = job["actual_value"]
        final_error = None if (final_pred is None or actual is None) else abs(final_pred - actual)
        avg_error = None if (avg_pred is None or actual is None) else abs(avg_pred - actual)
        drift = forecast_drift(job)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("Started", created_at.strftime("%H:%M:%S UTC"), "")
        with s2:
            metric_card("Target", target_hour.strftime("%H:%M UTC"), "")
        with s3:
            metric_card("Horizon", f"{horizon}h", "")
        with s4:
            metric_card(
                "Final Prediction",
                "N/A" if final_pred is None else f"{final_pred:.2f} nT",
                "Last forecast before the target arrived — the operational forecast",
            )

        s5, s6, s7, s8 = st.columns(4)
        with s5:
            metric_card(
                "Average Prediction",
                "N/A" if avg_pred is None else f"{avg_pred:.2f} nT",
                "Stability indicator, not the operational forecast",
                tooltip="Mean of every prediction generated during the session.",
            )
        with s6:
            metric_card(
                "Actual Dst",
                "Pending" if actual is None else f"{actual:.2f} nT",
                "" if job["status"] == "completed" else "Stopped before the target arrived",
            )
        with s7:
            metric_card(
                "Final Prediction Error",
                "N/A" if final_error is None else f"{final_error:.2f} nT",
                "Primary operational accuracy metric",
            )
        with s8:
            metric_card(
                "Average Prediction Error",
                "N/A" if avg_error is None else f"{avg_error:.2f} nT",
                "Secondary stability metric",
            )

        s9, s10, s11, s12 = st.columns(4)
        with s9:
            if drift is None:
                metric_card("Forecast Drift", "N/A", "")
            else:
                sign = "+" if drift >= 0 else ""
                metric_card(
                    "Forecast Drift",
                    f"{sign}{drift:.2f} nT",
                    "Final minus initial prediction",
                    tooltip="How much the forecast moved from the first tick to the last.",
                )
        with s10:
            r2 = metrics.get("r2")
            metric_card("Model Quality", model_quality_label(r2), "N/A" if r2 is None else f"R² = {r2:.4f}")
        with s11:
            metric_card("MAE", f"{metrics.get('mae', float('nan')):.3f} nT", "Model's typical training error")
        with s12:
            metric_card("RMSE", f"{metrics.get('rmse', float('nan')):.3f} nT", "")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section(job["dataset"], "dst", job["horizon"])
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


def render_ae_forecast_dialog(job: dict) -> None:
    """AE V1: independent target driven by Solar Wind + IMF + derived
    physics only (no Kp/Dst inputs, no chained predictions — see README's
    staged AE plan). Otherwise the same live console + completion summary
    architecture as Dst's dedicated dialog.

    AE has no live NOAA/DONKI feed (only a historical file), so "Latest
    AE" below is the last known value, not a per-minute live reading.

    Prediction and verification are two separate, independent phases (see
    swdss.models.jobs._advance_predicting / _verify_static_jobs):
    completion (status='completed') happens purely because the target
    hour arrived — it never waits on an AE observation that may never
    come. Verification (verification_status) is a separate, ongoing check
    against Kyoto WDC's published real-time AE digital data (never NOAA,
    which has no AE product) of whether that source has since been
    published to cover
    that hour; until then it stays 'pending', shown as "Awaiting Official
    AE Verification" rather than blocking the job from completing.
    """
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    horizon = job["horizon"]

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])
    r2 = metrics.get("r2")
    verification_status = job.get("verification_status")

    st.subheader("AE Forecast")
    st.markdown(
        f"**Started:** {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
        f"**Target:** {target_hour.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Horizon:** {horizon}h  \n"
        f"**Model Quality:** {model_quality_label(r2)}"
    )
    badge_col, verify_col = st.columns([0.3, 0.7])
    with badge_col:
        st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    with verify_col:
        if job["status"] == "completed" and verification_status:
            st.markdown(status_badge_html(verification_status), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} nT
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} nT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Live Forecast Console")

    def _fmt(value, dp):
        return "N/A" if value is None else f"{value:.{dp}f}"

    if ticks:
        forecast_col = f"Forecast ({target_hour.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Ey (mV/m)",
            "VBz",
            "Dynamic Pressure (nPa)",
            "Latest AE (nT)",
            forecast_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("ey"), 3),
                _fmt(inputs.get("vbz"), 1),
                _fmt(inputs.get("dynamic_pressure"), 3),
                _fmt(inputs.get("ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .kp-console {{
                background: #050505;
                border: 2px solid #ffffff;
                box-shadow: 3px 3px 0px #808080;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.kp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.kp-console-table th, table.kp-console-table td {{
                border: 1px solid #1a3a2a;
                padding: 3px 8px;
                text-align: right;
                color: #d8ffe8;
            }}
            table.kp-console-table th {{
                color: #00ff88;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.kp-console-table td:first-child, table.kp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="kp-console">
                <table class="kp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name="Predicted AE",
            )
        )
        fig.update_layout(
            title="AE Forecast Drift Toward Target",
            height=360,
            legend_title_text="",
            yaxis_title="Predicted AE (nT)",
        )
        plot_retro(fig)

    if job["status"] == "completed" and verification_status == "pending":
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        checked_at = job.get("verification_checked_at")
        checked_caption = (
            f" Last checked {pd.Timestamp(checked_at).strftime('%Y-%m-%d %H:%M:%S UTC')}." if checked_at else ""
        )
        st.markdown("**Prediction Complete — Verification Pending**")
        st.info(
            "Official Kyoto AE data has not yet been published. "
            "Estimated publication delay: approximately 10-20 days.\n\n"
            "The target hour has arrived and the forecast is frozen below. NOAA/DONKI publish no AE "
            "product, so **verification** (confirming the official AE value) is a completely separate, "
            "ongoing check against **Kyoto World Data Center**'s published digital AE data — once per "
            "day is sufficient, since Kyoto WDC only publishes in batches." + checked_caption
        )

    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Forecast Summary")

        final_pred = ticks[-1]["predicted_value"] if ticks else None
        avg_pred = average_prediction(job)
        actual = job["actual_value"]
        final_error = None if (final_pred is None or actual is None) else abs(final_pred - actual)
        avg_error = None if (avg_pred is None or actual is None) else abs(avg_pred - actual)
        drift = forecast_drift(job)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("Started", created_at.strftime("%H:%M:%S UTC"), "")
        with s2:
            metric_card("Target", target_hour.strftime("%H:%M UTC"), "")
        with s3:
            metric_card("Horizon", f"{horizon}h", "")
        with s4:
            metric_card(
                "Final Prediction",
                "N/A" if final_pred is None else f"{final_pred:.2f} nT",
                "Last forecast before the target arrived — the operational forecast",
            )

        s5, s6, s7, s8 = st.columns(4)
        with s5:
            metric_card(
                "Average Prediction",
                "N/A" if avg_pred is None else f"{avg_pred:.2f} nT",
                "Stability indicator, not the operational forecast",
                tooltip="Mean of every prediction generated during the session.",
            )
        with s6:
            if job["status"] == "stopped":
                actual_caption = "Stopped before the target arrived"
            elif verification_status == "verified":
                actual_caption = "Verified against Kyoto WDC's published AE digital data"
            else:
                actual_caption = "Awaiting Official AE (Kyoto WDC checked automatically)"
            metric_card(
                "Actual AE",
                "Pending" if actual is None else f"{actual:.2f} nT",
                actual_caption,
            )
        with s7:
            metric_card(
                "Final Prediction Error",
                "N/A" if final_error is None else f"{final_error:.2f} nT",
                "Primary operational accuracy metric",
            )
        with s8:
            metric_card(
                "Average Prediction Error",
                "N/A" if avg_error is None else f"{avg_error:.2f} nT",
                "Secondary stability metric",
            )

        s9, s10, s11, s12 = st.columns(4)
        with s9:
            if drift is None:
                metric_card("Forecast Drift", "N/A", "")
            else:
                sign = "+" if drift >= 0 else ""
                metric_card(
                    "Forecast Drift",
                    f"{sign}{drift:.2f} nT",
                    "Final minus initial prediction",
                    tooltip="How much the forecast moved from the first tick to the last.",
                )
        with s10:
            metric_card("Model Quality", model_quality_label(r2), "N/A" if r2 is None else f"R² = {r2:.4f}")
        with s11:
            metric_card("MAE", f"{metrics.get('mae', float('nan')):.3f} nT", "Model's typical training error")
        with s12:
            metric_card("RMSE", f"{metrics.get('rmse', float('nan')):.3f} nT", "")

        s13, s14, s15, s16 = st.columns(4)
        with s13:
            pct_error = final_percentage_error(job)
            metric_card(
                "Percentage Error",
                "N/A" if pct_error is None else f"{pct_error:.1f}%",
                "Final prediction error as % of the official AE value",
            )
        with s14:
            verified_at = job.get("verified_at")
            metric_card(
                "Verification Date",
                "N/A" if verified_at is None else pd.Timestamp(verified_at).strftime("%Y-%m-%d %H:%M UTC"),
                "When Kyoto WDC's data first covered this target hour",
            )
        with s15:
            metric_card(
                "Verification Status",
                "Verified" if verification_status == "verified" else "Pending Official Kyoto Data",
                "Kyoto WDC digital AE" if verification_status == "verified" else "Checked ~daily",
            )
        with s16:
            st.empty()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section("ae", "ae", job["horizon"])
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


def _experimental_badge_html() -> str:
    return (
        "<div style=\"display:inline-block;background:#5a1f8a;color:#f2e8ff;"
        "border:2px solid #d8b8ff;border-radius:4px;padding:2px 10px;"
        "font-weight:700;font-size:0.78rem;letter-spacing:0.03em;\">"
        "🧪 EXPERIMENTAL — RESEARCH ONLY, NOT PRODUCTION</div>"
    )


def render_experimental_forecast_dialog(job: dict) -> None:
    """AE V3 (research/experimental — see README's staged AE plan): the
    cascaded pipeline that feeds Predicted AE (never observed AE) into
    Kp/Dst as an extra feature. Completely separate models/training data
    from the production "analytics" pipeline — this dialog only ever
    reads production jobs (via find_matching_job) for side-by-side
    comparison, it never influences them.
    """
    variable = job["variable"]
    horizon = job["horizon"]
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    label = VARIABLE_LABELS.get(variable, variable)
    unit = VARIABLE_UNITS.get(variable, "")
    decimals = 2 if variable == "kp" else 1

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])
    is_kp_interval = variable == "kp"

    st.markdown(_experimental_badge_html(), unsafe_allow_html=True)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.subheader(f"Experimental {label} Forecast — Cascaded via Predicted AE")
    st.markdown(
        f"**Started:** {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
        f"**Target:** {target_hour.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Horizon:** {'Next official NOAA interval' if is_kp_interval else f'{horizon}h'}  \n"
        f"**Model Quality:** {model_quality_label(metrics.get('r2'))}"
    )
    st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #d8b8ff;
            box-shadow: 3px 3px 0px #5a1f8a;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #e8caff;
        }}
        </style>
        <div class="job-terminal">
            <div>EXPERIMENTAL MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} {escape(unit)}
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} {escape(unit)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Live Forecast Console (Experimental — Cascaded)")

    def _fmt(value, dp):
        return "N/A" if value is None else f"{value:.{dp}f}"

    if ticks:
        forecast_col = f"Experimental Forecast ({target_hour.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Previous Kp",
            "Previous Dst (nT)",
            "Predicted AE (nT)",
            forecast_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("kp"), 2),
                _fmt(inputs.get("dst"), 1),
                _fmt(inputs.get("predicted_ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .exp-console {{
                background: #050505;
                border: 2px solid #d8b8ff;
                box-shadow: 3px 3px 0px #5a1f8a;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.exp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.exp-console-table th, table.exp-console-table td {{
                border: 1px solid #3a1a4a;
                padding: 3px 8px;
                text-align: right;
                color: #f0e0ff;
            }}
            table.exp-console-table th {{
                color: #e8caff;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.exp-console-table td:first-child, table.exp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="exp-console">
                <table class="exp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    # Look up a comparable production job — same variable, same exact
    # target hour — "whenever possible", per the research spec. Only ever
    # reads production's job table; never writes to or influences it.
    production_job = find_matching_job("analytics", variable, target_hour)

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name=f"Experimental {label}",
            )
        )
        if production_job and production_job["ticks"]:
            prod_plotted = [t for t in production_job["ticks"] if t["minute_at"] is not None]
            if prod_plotted:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.Timestamp(t["minute_at"]) for t in prod_plotted],
                        y=[t["predicted_value"] for t in prod_plotted],
                        mode="lines+markers",
                        name=f"Production {label}",
                    )
                )
        fig.update_layout(
            title=f"{label} Forecast Drift — Production vs. Experimental",
            height=360,
            legend_title_text="",
        )
        plot_retro(fig)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Production vs. Experimental — Side by Side")
    if production_job is None:
        st.info(
            f"No Production {label} job found targeting {target_hour.strftime('%Y-%m-%d %H:%M UTC')}. "
            "Start one from the Prediction tab (same horizon, similar start time) to compare live."
        )
    else:
        prod_final = production_job["ticks"][-1]["predicted_value"] if production_job["ticks"] else None
        exp_final = ticks[-1]["predicted_value"] if ticks else None
        actual = job["actual_value"] if job["actual_value"] is not None else production_job["actual_value"]

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card(
                "Production (latest)",
                "N/A" if prod_final is None else f"{prod_final:.2f} {unit}",
                f"Model: {production_job['model_name']}",
            )
        with c2:
            metric_card(
                "Experimental (latest)",
                "N/A" if exp_final is None else f"{exp_final:.2f} {unit}",
                f"Model: {job['model_name']}",
            )
        with c3:
            metric_card(
                "Actual",
                "Pending" if actual is None else f"{actual:.2f} {unit}",
                "Shared ground truth for both pipelines",
            )

    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Evaluation")

        exp_final = ticks[-1]["predicted_value"] if ticks else None
        actual = job["actual_value"]
        if production_job is not None and actual is None:
            actual = production_job["actual_value"]
        exp_error = None if (exp_final is None or actual is None) else abs(exp_final - actual)

        prod_final = None
        prod_error = None
        prod_mae = None
        prod_r2 = None
        if production_job is not None:
            prod_final = production_job["ticks"][-1]["predicted_value"] if production_job["ticks"] else None
            prod_error = None if (prod_final is None or actual is None) else abs(prod_final - actual)
            prod_mae = production_job.get("metrics", {}).get("mae")
            prod_r2 = production_job.get("metrics", {}).get("r2")

        comparison_df = pd.DataFrame(
            [
                {
                    "Pipeline": "Production",
                    f"Final Prediction ({unit})": "N/A" if prod_final is None else round(prod_final, decimals),
                    f"Absolute Error ({unit})": "N/A" if prod_error is None else round(prod_error, decimals),
                    "Model MAE (offline, training)": "N/A" if prod_mae is None else round(prod_mae, 3),
                    "Model R² (offline, training)": "N/A" if prod_r2 is None else round(prod_r2, 4),
                },
                {
                    "Pipeline": "Experimental",
                    f"Final Prediction ({unit})": "N/A" if exp_final is None else round(exp_final, decimals),
                    f"Absolute Error ({unit})": "N/A" if exp_error is None else round(exp_error, decimals),
                    "Model MAE (offline, training)": round(metrics.get("mae", float("nan")), 3),
                    "Model R² (offline, training)": round(metrics.get("r2", float("nan")), 4),
                },
            ]
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        if actual is None:
            st.caption("Actual observation still Pending — errors above will populate once it's available.")
        elif prod_error is not None and exp_error is not None:
            if exp_error < prod_error:
                st.success(
                    f"For this target, the experimental cascade performed better "
                    f"({exp_error:.{decimals}f} {unit} vs {prod_error:.{decimals}f} {unit} absolute error)."
                )
            elif exp_error > prod_error:
                st.warning(
                    f"For this target, the production pipeline performed better "
                    f"({prod_error:.{decimals}f} {unit} vs {exp_error:.{decimals}f} {unit} absolute error)."
                )
            else:
                st.info("Both pipelines produced the same absolute error for this target.")
        else:
            st.caption("No matching Production job to compare error against for this target.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section("experimental", job["variable"], "interval" if job["variable"] == "kp" else job["horizon"])
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


@st.dialog("Prediction Job", width="large", dismissible=False)
def show_prediction_job(job_id: str) -> None:
    render_dialog_close_button("close_prediction_job")

    job = get_job(job_id)
    if job is None:
        st.error("This prediction job could not be found.")
        return

    poll_jobs(job["dataset"])
    job = get_job(job_id)

    dataset = job["dataset"]
    variable = job["variable"]

    if dataset == "analytics" and variable == "kp":
        render_kp_forecast_dialog(job)
        return
    if dataset == "analytics" and variable == "dst":
        render_dst_forecast_dialog(job)
        return
    if dataset == "ae":
        render_ae_forecast_dialog(job)
        return
    if dataset == "experimental":
        render_experimental_forecast_dialog(job)
        return

    horizon = job["horizon"]
    label = VARIABLE_LABELS.get(variable, variable)
    unit = VARIABLE_UNITS.get(variable, "")
    decimals = 0 if unit == "K" else 2

    start_hour = pd.Timestamp(job["start_hour"])
    target_hour = pd.Timestamp(job["target_hour"])
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    is_kp_interval = dataset == "analytics" and variable == "kp"
    horizon_display = "Next NOAA Interval" if is_kp_interval else f"{horizon}h"

    st.subheader(f"{label} — {horizon_display} Forecast")
    badge_col, caption_col = st.columns([0.15, 0.85])
    with badge_col:
        st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    with caption_col:
        st.caption(
            f"Started {start_hour.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Target: {target_hour.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    live_ts, live_val = latest_minute_observation(dataset, variable)
    latest_predicted = ticks[-1]["predicted_value"] if ticks else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Current NOAA",
            format_value(live_val, f" {unit}", decimals),
            "N/A" if live_ts is None else f"As of {live_ts.strftime('%H:%M:%S UTC')}",
        )
    with c2:
        r2 = metrics.get("r2")
        quality = model_quality_label(r2)
        metric_card(
            "Model Quality",
            quality,
            "N/A" if r2 is None else f"R² = {r2:.4f}",
            tooltip="How well this model performed on held-out training data, categorized from its R² score.",
        )
    with c3:
        if latest_predicted is not None and live_val is not None:
            change = latest_predicted - live_val
            if abs(change) < 1e-9:
                trend, color = "No Change", "#404040"
            elif change > 0:
                trend, color = "Increase", "#1f7a3a"
            else:
                trend, color = "Decrease", "#a31f1f"
            sign = "+" if change >= 0 else ""
            metric_card(
                "Expected Change",
                f"{sign}{format_value(change, f' {unit}', decimals)}",
                f"Trend: {trend}",
                tooltip="Difference between the latest forecast and the current live NOAA reading.",
                value_color=color,
            )
        else:
            metric_card("Expected Change", "N/A", "", tooltip="Difference between the latest forecast and the current live NOAA reading.")
    with c4:
        stability_label, stability_delta = stability_metric(job)
        if stability_label is None:
            metric_card(
                "Stability",
                "N/A",
                "Not enough updates to evaluate stability",
                tooltip="How much the prediction has varied across the most recent updates in this session.",
            )
        else:
            metric_card(
                "Stability",
                stability_label,
                f"Δ Prediction = {stability_delta:.2f} {unit}",
                tooltip="How much the prediction has varied across the most recent updates in this session.",
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if job["status"] == "completed" and job["actual_value"] is not None:
        final_pred = ticks[-1]["predicted_value"] if ticks else None
        error = None if final_pred is None else abs(final_pred - job["actual_value"])
        model_mae = metrics.get("mae")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            metric_card(
                "Final Prediction",
                format_value(final_pred, f" {unit}", decimals),
                f"Target {target_hour.strftime('%H:%M UTC')}",
            )
        with a2:
            metric_card(
                "Actual NOAA",
                format_value(job["actual_value"], f" {unit}", decimals),
                f"At {target_hour.strftime('%H:%M UTC')}",
            )
        with a3:
            metric_card(
                "Absolute Error",
                "N/A" if error is None else format_value(error, f" {unit}", decimals),
                "",
                tooltip="Difference between the final prediction and the actual NOAA observation, in native units.",
            )
        with a4:
            eval_label = forecast_evaluation_label(error, model_mae)
            metric_card(
                "Forecast Evaluation",
                eval_label,
                "" if model_mae is None else f"Model's typical error: {model_mae:.{decimals}f} {unit}",
                tooltip="How this forecast's error compares to the model's typical error (MAE) from training.",
            )
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        .job-terminal .pipeline-step {{ color: #9adfff; margin-left: 12px; }}
        .job-terminal div {{ margin-bottom: 3px; }}
        .job-terminal.scroll {{ max-height: 320px; overflow-y: auto; }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} {escape(unit)}
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} {escape(unit)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if not ticks:
        st.info("No ticks logged yet.")
        return

    blocks = []
    for i, tick in enumerate(reversed(ticks)):
        minute_at = tick["minute_at"]
        minute_label = "N/A" if minute_at is None else pd.Timestamp(minute_at).strftime("%H:%M:%S UTC")
        pred_text = f"{tick['predicted_value']:.{decimals}f} {unit}"
        used_horizon = tick.get("used_horizon", horizon)
        used_horizon_text = "Next Interval" if used_horizon == "interval" else f"{used_horizon}h"
        next_step = "Waiting For Next NOAA Update..." if i == 0 and job["status"] == "in_progress" else "Superseded"

        if dataset == "analytics" and tick.get("inputs"):
            header = f"<div>[{escape(minute_label)}] Multi-Source Update</div>" + format_analytics_inputs(
                tick["inputs"]
            )
            features_step = "Features Generated (Solar Wind + IMF + Geomagnetic Lags)"
        else:
            noaa_text = "N/A" if tick["noaa_value"] is None else f"{tick['noaa_value']:.{decimals}f} {unit}"
            header = f"<div>[{escape(minute_label)}] NOAA {escape(label)}: {escape(noaa_text)}</div>"
            features_step = "Features Generated"

        blocks.append(
            header
            + f"<div class='pipeline-step'>&rarr; {features_step}</div>"
            + f"<div class='pipeline-step'>&rarr; Model Loaded (Horizon: {used_horizon_text})</div>"
            + f"<div class='pipeline-step'>&rarr; Prediction (Target {target_hour.strftime('%H:%M UTC')}): {escape(pred_text)}</div>"
            + f"<div class='pipeline-step'>&rarr; {next_step}</div>"
        )

    st.markdown(
        f"""
        <div class="job-terminal scroll">
            {''.join(blocks)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name=f"Predicted ({target_hour.strftime('%H:%M UTC')})",
            )
        )
        fig.update_layout(
            title=f"{label} — Prediction Drift Toward Target",
            height=360,
            legend_title_text="",
        )
        plot_retro(fig)

    if job["status"] == "evaluating" and dataset == "analytics":
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.info(f"Waiting for NOAA to publish the official {label} value for this target. The job will complete automatically once it's available.")
    elif job["status"] == "evaluating":
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Evaluation In Progress — Target Hour Filling")
        st.caption(
            f"Collecting NOAA minute readings for {target_hour.strftime('%H:%M')}–"
            f"{(target_hour + pd.Timedelta(hours=1)).strftime('%H:%M UTC')}. Once the hour closes, "
            "this collapses into the final Job Summary below."
        )
        eval_ticks = job.get("eval_ticks", [])
        if eval_ticks:
            eval_df = pd.DataFrame(
                [
                    {
                        "Minute": pd.Timestamp(t["minute_at"]).strftime("%H:%M:%S UTC") if t["minute_at"] else "N/A",
                        f"NOAA Value ({unit})": "N/A" if t["noaa_value"] is None else round(t["noaa_value"], decimals),
                        f"Running Average ({unit})": (
                            "N/A" if t["running_avg"] is None else round(t["running_avg"], decimals)
                        ),
                    }
                    for t in reversed(eval_ticks)
                ]
            )
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
        else:
            st.info("Waiting for the first NOAA reading in this hour...")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Job Summary")

    mae = job_mae(job)
    summary_df = pd.DataFrame(
        [
            {
                "Variable": label,
                "Horizon": horizon_display,
                "Started": start_hour.strftime("%Y-%m-%d %H:%M UTC"),
                "Target": target_hour.strftime("%Y-%m-%d %H:%M UTC"),
                f"Initial Prediction ({unit})": round(ticks[0]["predicted_value"], decimals),
                f"Final Prediction ({unit})": round(ticks[-1]["predicted_value"], decimals),
                f"Actual NOAA ({unit})": "Pending" if job["actual_value"] is None else round(job["actual_value"], decimals),
                f"Mean Error - All Ticks ({unit})": "N/A" if mae is None else round(mae, decimals),
                "Model": job["model_name"],
                "R²": round(metrics.get("r2", float("nan")), 4),
            }
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


def prediction_panel(dataset: str, variables: list[str]) -> None:
    poll_jobs(dataset)

    if dataset == "ae":
        st.caption(
            "This forecast estimates the auroral electrojet activity expected during the next hour "
            "based on the upstream solar wind and interplanetary magnetic field conditions available "
            "at the time the forecast is issued. Differences from subsequent observations may reflect "
            "evolving solar wind conditions, the nonlinear response of Earth's magnetosphere, or model "
            "limitations."
        )

    render_prediction_queue_stats(dataset)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    variable = col1.selectbox(
        "Target Variable",
        variables,
        format_func=lambda v: VARIABLE_LABELS[v],
        key=f"{dataset}_pred_var",
    )

    is_kp_interval = dataset in ("analytics", "experimental") and variable == "kp"
    if is_kp_interval:
        with col2:
            st.markdown("**Forecast Horizon**")
            st.caption("Next official NOAA Kp interval (published every 3h: 00, 03, 06... UTC)")
        horizon = 1  # placeholder only — predict_kp_interval ignores it and always targets the next interval
    else:
        horizon = col2.selectbox(
            "Forecast Horizon",
            HORIZONS,
            format_func=lambda h: f"{h} Hour" + ("s" if h != 1 else ""),
            key=f"{dataset}_pred_horizon",
        )

    btn_col, saved_col = st.columns(2)
    button_label = "Start Experimental Prediction" if dataset == "experimental" else "Start Prediction"
    with btn_col:
        if st.button(button_label, key=f"{dataset}_pred_btn", use_container_width=True):
            try:
                job, created = start_job(dataset, variable, horizon)
            except Exception as exc:
                st.error(f"Could not start prediction: {exc}")
                return
            label = VARIABLE_LABELS.get(variable, variable)
            horizon_text = "next NOAA interval" if is_kp_interval else f"{horizon}h"
            prefix = "experimental " if dataset == "experimental" else ""
            if created:
                st.toast(f"Started {prefix}{label} {horizon_text} prediction.")
            else:
                start_hour = pd.Timestamp(job["start_hour"])
                st.warning(
                    f"A {prefix}{label} {horizon_text} prediction is already in progress "
                    f"(started {start_hour.strftime('%H:%M UTC')}). Open its card below to view live drift."
                )
    with saved_col:
        if st.button("📁 Saved Predictions", key=f"{dataset}_saved_btn", use_container_width=True):
            open_dialog("saved_predictions", dataset)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Running Predictions")
    render_prediction_job_tiles(
        get_running_jobs(dataset),
        "No predictions running. Pick a variable and horizon, then click Start Prediction.",
    )


def prediction_statistics_panel(dataset: str) -> None:
    """Analyzes the forecasting SYSTEM itself — aggregated across every
    completed, evaluated job for this dataset — rather than any single
    forecast. That's covered by the Predictions tab.
    """
    stats = get_prediction_statistics(dataset)

    if stats["count"] == 0:
        st.info("No completed, evaluated forecasts yet. Statistics will appear here once predictions finish.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Forecast Count", str(stats["count"]), "Completed & evaluated forecasts")
    with c2:
        rate = stats["success_rate"]
        metric_card(
            "Success Rate",
            "N/A" if rate is None else f"{rate:.0f}%",
            "Within 1.5x the model's typical error",
            tooltip="Percentage of forecasts whose final error came in at or below 1.5x the model's own typical training error (MAE).",
        )
    with c3:
        best_model = stats["best_model"]
        best_mae = stats["mae_by_model"].get(best_model) if best_model else None
        metric_card(
            "Best-Performing Model",
            best_model or "N/A",
            "" if best_mae is None else f"Avg error: {best_mae:.3f}",
            tooltip="The algorithm with the lowest average absolute error across all completed forecasts that used it.",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    col_var, col_horizon = st.columns(2)
    with col_var:
        st.markdown("##### Mean Absolute Error by Variable")
        var_df = pd.DataFrame(
            [
                {"Variable": VARIABLE_LABELS.get(v, v), "MAE": round(mae, 3)}
                for v, mae in sorted(stats["mae_by_variable"].items(), key=lambda kv: kv[1])
            ]
        )
        st.dataframe(var_df, use_container_width=True, hide_index=True)
    with col_horizon:
        st.markdown("##### Mean Absolute Error by Forecast Horizon")
        horizon_df = pd.DataFrame(
            [
                {"Horizon": f"{h}h", "MAE": round(mae, 3)}
                for h, mae in sorted(stats["mae_by_horizon"].items())
            ]
        )
        st.dataframe(horizon_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Historical Forecast Error Trend")
    trend = stats["trend"]
    if len(trend) >= 2:
        trend_df = pd.DataFrame(trend, columns=["completed_at", "abs_error"])
        trend_df["completed_at"] = pd.to_datetime(trend_df["completed_at"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend_df["completed_at"],
                y=trend_df["abs_error"],
                mode="lines+markers",
                name="Absolute Error",
            )
        )
        fig.update_layout(
            title="Forecast Error Over Time (Lower Is Better)",
            height=360,
            legend_title_text="",
        )
        plot_retro(fig)
    else:
        st.info("Need at least 2 completed forecasts to plot a trend.")


def current_analysis_solar_wind(df: pd.DataFrame) -> None:
    st.subheader("Solar Wind Current Analysis")

    speed_row = row_at_extreme_from_source("solar_wind", "solar_wind_speed", "max")
    density_row = row_at_extreme_from_source("solar_wind", "proton_density", "max")
    temp_row = row_at_extreme_from_source("solar_wind", "temperature", "max")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(
            "Highest Speed",
            format_value(None if speed_row is None else speed_row["solar_wind_speed"], " km/s", 1),
            time_caption(speed_row),
        )
        if speed_row is not None:
            card_note(
                f"Density: {format_value(speed_row.get('proton_density'), ' p/cm3', 2)} | "
                f"Temp: {format_value(speed_row.get('temperature'), ' K', 0)}"
            )
    with c2:
        metric_card(
            "Highest Density",
            format_value(None if density_row is None else density_row["proton_density"], " p/cm3", 2),
            time_caption(density_row),
        )
        if density_row is not None:
            card_note(
                f"Speed: {format_value(density_row.get('solar_wind_speed'), ' km/s', 1)} | "
                f"Temp: {format_value(density_row.get('temperature'), ' K', 0)}"
            )
    with c3:
        metric_card(
            "Highest Temperature",
            format_value(None if temp_row is None else temp_row["temperature"], " K", 0),
            time_caption(temp_row),
        )
        if temp_row is not None:
            card_note(
                f"Density: {format_value(temp_row.get('proton_density'), ' p/cm3', 2)} | "
                f"Speed: {format_value(temp_row.get('solar_wind_speed'), ' km/s', 1)}"
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    line_chart(df, ["solar_wind_speed", "proton_density", "temperature"], "Solar Wind Speed, Density, Temperature")
    correlation_explorer(df, ["solar_wind_speed", "proton_density", "temperature"], "Solar Wind Correlations")


def current_analysis_imf(df: pd.DataFrame) -> None:
    st.subheader("IMF Current Analysis")

    bz_row = row_at_extreme_from_source("imf", "bz", "min")
    bt_row = row_at_extreme_from_source("imf", "bt", "max")
    bx_row = row_at_extreme_from_source("imf", "bx", "max")
    by_row = row_at_extreme_from_source("imf", "by", "max")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Lowest Bz", format_value(None if bz_row is None else bz_row["bz"], " nT", 2), time_caption(bz_row))
        if bz_row is not None:
            card_note(
                f"Bt: {format_value(bz_row.get('bt'), ' nT', 2)} | "
                f"Bx: {format_value(bz_row.get('bx'), ' nT', 2)} | "
                f"By: {format_value(bz_row.get('by'), ' nT', 2)}"
            )
        else:
            card_note("No associated IMF values")

    with c2:
        metric_card("Highest Bt", format_value(None if bt_row is None else bt_row["bt"], " nT", 2), time_caption(bt_row))
        if bt_row is not None:
            card_note(
                f"Bz: {format_value(bt_row.get('bz'), ' nT', 2)} | "
                f"Bx: {format_value(bt_row.get('bx'), ' nT', 2)} | "
                f"By: {format_value(bt_row.get('by'), ' nT', 2)}"
            )
        else:
            card_note("No associated IMF values")

    with c3:
        metric_card("Highest Bx", format_value(None if bx_row is None else bx_row["bx"], " nT", 2), time_caption(bx_row))
        if bx_row is not None:
            card_note(
                f"Bt: {format_value(bx_row.get('bt'), ' nT', 2)} | "
                f"Bz: {format_value(bx_row.get('bz'), ' nT', 2)} | "
                f"By: {format_value(bx_row.get('by'), ' nT', 2)}"
            )
        else:
            card_note("No associated IMF values")

    with c4:
        metric_card("Highest By", format_value(None if by_row is None else by_row["by"], " nT", 2), time_caption(by_row))
        if by_row is not None:
            card_note(
                f"Bt: {format_value(by_row.get('bt'), ' nT', 2)} | "
                f"Bx: {format_value(by_row.get('bx'), ' nT', 2)} | "
                f"Bz: {format_value(by_row.get('bz'), ' nT', 2)}"
            )
        else:
            card_note("No associated IMF values")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    line_chart(df, ["bt", "bx", "by", "bz"], "IMF Components")
    correlation_explorer(df, ["bt", "bx", "by", "bz"], "IMF Correlations")


def _add_geospace_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived physics columns used by the Geospace scatter explorer."""
    d = df.copy().sort_values("timestamp_utc").reset_index(drop=True)

    if "solar_wind_speed" in d.columns and "bz" in d.columns:
        d["ey"] = -d["solar_wind_speed"] * d["bz"] * 1e-3

    if "solar_wind_speed" in d.columns and "proton_density" in d.columns:
        d["dynamic_pressure"] = 1.6726e-6 * d["proton_density"] * d["solar_wind_speed"] ** 2

    if "by" in d.columns and "bz" in d.columns:
        d["clock_angle"] = np.degrees(np.arctan2(d["by"], d["bz"]))

    if "kp" in d.columns:
        d["prev_kp"] = d["kp"].shift(1)

    if "dst" in d.columns:
        d["prev_dst"] = d["dst"].shift(1)

    try:
        from swdss.paths import PROCESSED_DIR
        ae_path = PROCESSED_DIR / "ae" / "ae_processed.parquet"
        ae_df = pd.read_parquet(ae_path)[["timestamp_utc", "ae"]].copy()
        ae_df["prev_ae"] = ae_df["ae"].shift(1)
        d = d.merge(ae_df[["timestamp_utc", "prev_ae"]], on="timestamp_utc", how="left")
    except Exception:
        d["prev_ae"] = np.nan

    if "bz" in d.columns:
        d["southward_duration"] = (d["bz"] < 0).astype(float).rolling(24, min_periods=1).sum()

    return d


def current_analysis_kp(df: pd.DataFrame) -> None:
    st.subheader("Kp Current Analysis")

    kp_row = row_at_extreme(df, "kp", "max")
    latest = df.dropna(subset=["kp"]).tail(1)
    latest_row = None if latest.empty else latest.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Highest Kp", format_value(None if kp_row is None else kp_row["kp"], "", 1), time_caption(kp_row))
        if kp_row is not None:
            card_note(
                f"Bz: {format_value(kp_row.get('bz'), ' nT', 2)} | "
                f"Dst: {format_value(kp_row.get('dst'), ' nT', 1)}"
            )
        else:
            card_note("No associated Bz/Dst")

    with c2:
        metric_card("Update Cadence", "3 hours", "NOAA Kp product cadence")
        card_note("Kp is a 3-hour planetary index")

    with c3:
        latest_value = np.nan if latest_row is None else latest_row["kp"]
        metric_card("Latest Kp", format_value(latest_value, "", 1), time_caption(latest_row))
        if latest_row is not None:
            card_note(
                f"Bz: {format_value(latest_row.get('bz'), ' nT', 2)} | "
                f"Dst: {format_value(latest_row.get('dst'), ' nT', 1)}"
            )
        else:
            card_note("No associated Bz/Dst")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    line_chart(df, ["kp"], "Kp Index")
    df_geo = _add_geospace_derived_cols(df)
    correlation_explorer(
        df_geo,
        [
            "kp", "solar_wind_speed", "proton_density", "temperature",
            "bt", "bx", "by", "bz", "dst",
            "ey", "dynamic_pressure", "clock_angle",
            "prev_kp", "prev_ae", "prev_dst", "southward_duration",
        ],
        "Kp With Solar Wind, IMF & Derived Physics",
    )


def current_analysis_dst(df: pd.DataFrame) -> None:
    st.subheader("Dst Current Analysis")

    dst_row = row_at_extreme(df, "dst", "min")
    latest = df.dropna(subset=["dst"]).tail(1)
    latest_row = None if latest.empty else latest.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Lowest Dst", format_value(None if dst_row is None else dst_row["dst"], " nT", 1), time_caption(dst_row))
        if dst_row is not None:
            card_note(
                f"Bz: {format_value(dst_row.get('bz'), ' nT', 2)} | "
                f"Kp: {format_value(dst_row.get('kp'), '', 1)}"
            )
        else:
            card_note("No associated Bz/Kp")

    with c2:
        metric_card("Update Cadence", "1 hour", "NOAA/Kyoto Dst product cadence")
        card_note("Dst is an hourly ring-current index")

    with c3:
        latest_value = np.nan if latest_row is None else latest_row["dst"]
        metric_card("Latest Dst", format_value(latest_value, " nT", 1), time_caption(latest_row))
        if latest_row is not None:
            card_note(
                f"Bz: {format_value(latest_row.get('bz'), ' nT', 2)} | "
                f"Kp: {format_value(latest_row.get('kp'), '', 1)}"
            )
        else:
            card_note("No associated Bz/Kp")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    line_chart(df, ["dst"], "Dst Index")
    df_geo = _add_geospace_derived_cols(df)
    correlation_explorer(
        df_geo,
        [
            "dst", "kp", "solar_wind_speed", "proton_density", "temperature",
            "bt", "bx", "by", "bz",
            "ey", "dynamic_pressure", "clock_angle",
            "prev_kp", "prev_ae", "prev_dst", "southward_duration",
        ],
        "Dst With Solar Wind, IMF & Derived Physics",
    )


def earth_analysis(df: pd.DataFrame) -> None:
    bz_row = row_at_extreme_from_source("imf", "bz", "min")
    kp_row = row_at_extreme(df, "kp", "max")
    dst_row = row_at_extreme(df, "dst", "min")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Lowest Bz", format_value(None if bz_row is None else bz_row["bz"], " nT", 2), time_caption(bz_row))
        if bz_row is not None:
            card_note(
                f"Speed: {format_value(bz_row.get('solar_wind_speed'), ' km/s', 1)} | "
                f"Density: {format_value(bz_row.get('proton_density'), ' p/cm3', 2)} | "
                f"Temp: {format_value(bz_row.get('temperature'), ' K', 0)}"
            )
    with c2:
        metric_card("Highest Kp", format_value(None if kp_row is None else kp_row["kp"], "", 1), time_caption(kp_row))
        if kp_row is not None:
            card_note(
                f"Speed: {format_value(kp_row.get('solar_wind_speed'), ' km/s', 1)} | "
                f"Density: {format_value(kp_row.get('proton_density'), ' p/cm3', 2)} | "
                f"Bz: {format_value(kp_row.get('bz'), ' nT', 2)}"
            )
    with c3:
        metric_card("Lowest Dst", format_value(None if dst_row is None else dst_row["dst"], " nT", 1), time_caption(dst_row))
        if dst_row is not None:
            card_note(
                f"Speed: {format_value(dst_row.get('solar_wind_speed'), ' km/s', 1)} | "
                f"Density: {format_value(dst_row.get('proton_density'), ' p/cm3', 2)} | "
                f"Bz: {format_value(dst_row.get('bz'), ' nT', 2)}"
            )

    variables = ["solar_wind_speed", "proton_density", "temperature", "bt", "bx", "by", "bz", "kp", "dst"]
    line_chart(df, variables, "Combined 7-Day Space Weather Variables")
    correlation_explorer(df, variables, "Combined Dataset Correlations")


def home_page(df: pd.DataFrame) -> None:
    st.title("Space Weather Decision Support System")
    st.caption("7-day NOAA-based summary. Page refreshes every minute.")

    speed_row = row_at_extreme_from_source("solar_wind", "solar_wind_speed", "max")
    density_row = row_at_extreme_from_source("solar_wind", "proton_density", "max")
    temp_row = row_at_extreme_from_source("solar_wind", "temperature", "max")
    bz_row = row_at_extreme_from_source("imf", "bz", "min")
    kp_row = row_at_extreme(df, "kp", "max")
    dst_row = row_at_extreme(df, "dst", "min")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        extreme_card_with_hover(
            "Highest Speed",
            format_value(None if speed_row is None else speed_row["solar_wind_speed"], " km/s", 1),
            time_caption(speed_row),
            speed_row,
            skip_field="speed",
        )
        if speed_row is not None and st.button("🔍 Solar Event", key="extreme_event_speed", use_container_width=True):
            open_dialog("reverse_explorer", (speed_row["timestamp_utc"], "Highest Speed"))
    with c2:
        extreme_card_with_hover(
            "Highest Density",
            format_value(None if density_row is None else density_row["proton_density"], " p/cm3", 2),
            time_caption(density_row),
            density_row,
            skip_field="density",
        )
        if density_row is not None and st.button("🔍 Solar Event", key="extreme_event_density", use_container_width=True):
            open_dialog("reverse_explorer", (density_row["timestamp_utc"], "Highest Density"))
    with c3:
        extreme_card_with_hover(
            "Highest Temperature",
            format_value(None if temp_row is None else temp_row["temperature"], " K", 0),
            time_caption(temp_row),
            temp_row,
            skip_field="temp",
        )
        if temp_row is not None and st.button("🔍 Solar Event", key="extreme_event_temp", use_container_width=True):
            open_dialog("reverse_explorer", (temp_row["timestamp_utc"], "Highest Temperature"))
    with c4:
        extreme_card_with_hover(
            "Lowest Bz",
            format_value(None if bz_row is None else bz_row["bz"], " nT", 2),
            time_caption(bz_row),
            bz_row,
            skip_field="bz",
        )
        if bz_row is not None and st.button("🔍 Solar Event", key="extreme_event_bz", use_container_width=True):
            open_dialog("reverse_explorer", (bz_row["timestamp_utc"], "Lowest Bz"))
    with c5:
        extreme_card_with_hover(
            "Highest Kp",
            format_value(None if kp_row is None else kp_row["kp"], "", 1),
            time_caption(kp_row),
            kp_row,
            skip_field="kp",
        )
        if kp_row is not None and st.button("🔍 Solar Event", key="extreme_event_kp", use_container_width=True):
            open_dialog("reverse_explorer", (kp_row["timestamp_utc"], "Highest Kp"))
    with c6:
        extreme_card_with_hover(
            "Lowest Dst",
            format_value(None if dst_row is None else dst_row["dst"], " nT", 1),
            time_caption(dst_row),
            dst_row,
            skip_field="dst",
        )
        if dst_row is not None and st.button("🔍 Solar Event", key="extreme_event_dst", use_container_width=True):
            open_dialog("reverse_explorer", (dst_row["timestamp_utc"], "Lowest Dst"))

    st.divider()

    render_overview_chart()

    st.divider()

    solar_event_news_feed()

    st.divider()

    top5_col, heliomap_col = st.columns(2)

    with top5_col:
        with st.container(height=620, border=False):
            st.subheader("Top 5 Recorded Conditions")

            tab1, tab2, tab3 = st.tabs(["Lowest Bz", "Highest Kp", "Lowest Dst"])

            with tab1:
                top_event_table(df, "bz", "lowest", "Top 5 Most Negative Bz Events")

            with tab2:
                top_event_table(df, "kp", "highest", "Top 5 Highest Kp Events")

            with tab3:
                top_event_table(df, "dst", "lowest", "Top 5 Lowest Dst Events")

    with heliomap_col:
        with st.container(height=620, border=False):
            heliomap_panel()


def heliosphere_page(df: pd.DataFrame) -> None:
    st.title("Heliosphere")
    tabs = st.tabs(["Solar Wind", "IMF", "Derived Parameters", "Dynamic Pressure", "Travel Time"])

    with tabs[0]:
        inner = st.tabs(["Current Analysis", "Predictions", "Prediction Statistics"])
        with inner[0]:
            current_analysis_solar_wind(df)
        with inner[1]:
            prediction_panel("solar_wind", SOLAR_WIND_VARIABLES)
        with inner[2]:
            prediction_statistics_panel("solar_wind")

    with tabs[1]:
        inner = st.tabs(["Current Analysis", "Predictions", "Prediction Statistics", "Research Laboratory"])
        with inner[0]:
            current_analysis_imf(df)
        with inner[1]:
            prediction_panel("imf", IMF_VARIABLES)
        with inner[2]:
            prediction_statistics_panel("imf")
        with inner[3]:
            render_imf_research_laboratory()

    with tabs[2]:
        st.subheader("Derived Parameters")
        st.info("Derived parameters will use solar wind + IMF features in the next version.")

    with tabs[3]:
        st.subheader("Dynamic Pressure")
        if {"proton_density", "solar_wind_speed"}.issubset(df.columns):
            pressure = df.copy()
            pressure["dynamic_pressure"] = 1.6726e-6 * pressure["proton_density"] * pressure["solar_wind_speed"] ** 2
            line_chart(pressure, ["dynamic_pressure"], "Estimated Solar Wind Dynamic Pressure")
        else:
            st.info("Need proton_density and solar_wind_speed columns.")

    with tabs[4]:
        st.subheader("Travel Time")
        st.info("Travel time model will be added after CME and L1 propagation logic.")


def geospace_page(df: pd.DataFrame) -> None:
    st.title("Geospace")
    tabs = st.tabs(["Kp", "Dst"])

    with tabs[0]:
        current_analysis_kp(df)

    with tabs[1]:
        current_analysis_dst(df)


def recent_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty or "timestamp_utc" not in df.columns:
        return df
    latest_time = df["timestamp_utc"].max()
    start_time = latest_time - pd.Timedelta(days=days)
    return df[df["timestamp_utc"] >= start_time].copy()


EVENT_TYPE_CATEGORY = {
    "FLA": "Flare",
    "XRA": "X-ray Event",
    "RSP": "Radio Burst",
    "RBR": "Radio Burst",
    "RNS": "Radio Burst",
    "DSF": "Filament Eruption",
    "EPL": "Filament Eruption",
    "BSL": "Other",
}

RADIO_BURST_TYPES = ["RSP", "RBR", "RNS"]


def count_associated_cmes(events: pd.DataFrame, cme_df: pd.DataFrame, hours: int = 4) -> int:
    if events.empty or cme_df.empty:
        return 0
    if "timestamp_utc" not in events.columns or "timestamp_utc" not in cme_df.columns:
        return 0

    events_sorted = events.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    cme_sorted = cme_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    if events_sorted.empty or cme_sorted.empty:
        return 0

    merged = pd.merge_asof(
        events_sorted[["timestamp_utc"]],
        cme_sorted[["timestamp_utc"]].rename(columns={"timestamp_utc": "cme_time"}),
        left_on="timestamp_utc",
        right_on="cme_time",
        direction="forward",
        tolerance=pd.Timedelta(hours=hours),
    )
    return int(merged["cme_time"].notna().sum())


def render_event_timeline(events: pd.DataFrame, max_events: int = 10) -> None:
    if events.empty:
        st.info("No events available for the timeline.")
        return

    timeline_events = events.sort_values("timestamp_utc", ascending=False).head(max_events)

    rows_html = ""
    last_date = None

    for _, row in timeline_events.iterrows():
        ts = row["timestamp_utc"]
        date_str = ts.strftime("%d %b")
        time_str = ts.strftime("%H:%M UTC")

        flare_class = row.get("flare_class")
        event_type = row.get("event_type")
        region = row.get("active_region")

        label_parts = []
        if pd.notna(flare_class):
            label_parts.append(f"{flare_class} Flare")
        elif pd.notna(event_type):
            label_parts.append(EVENT_TYPE_CATEGORY.get(str(event_type), str(event_type)))
        if pd.notna(region):
            label_parts.append(f"AR{region}")

        label = " — ".join(label_parts) if label_parts else "Event"

        if date_str != last_date:
            rows_html += f'<div style="color:#9adfff; font-weight:700; margin-top:10px;">{escape(date_str)}</div>'
            last_date = date_str

        rows_html += (
            f'<div style="color:#f2f2f2; padding-left:8px;">{escape(time_str)}&nbsp;&nbsp;{escape(label)}</div>'
            '<div style="color:#808080; padding-left:8px;">&darr;</div>'
        )

    st.markdown(
        f"""
        <div style="
            background:#050505;
            border:2px solid #ffffff;
            box-shadow:3px 3px 0px #808080;
            padding:14px;
            font-family:'Courier New', monospace;
            font-size:0.82rem;
            line-height:1.5;
        ">{rows_html}</div>
        """,
        unsafe_allow_html=True,
    )


def dedupe_near_duplicate_events(df: pd.DataFrame, window_minutes: int = 15) -> pd.DataFrame:
    """Collapse repeated reports of the same eruption (e.g. several Type III
    bursts a few minutes apart) into a single feed entry, keeping the
    earliest occurrence of each cluster.
    """
    if df.empty:
        return df

    ordered = df.sort_values("timestamp_utc")
    window = pd.Timedelta(minutes=window_minutes)
    last_seen: dict = {}
    keep_indices = []

    for idx, row in ordered.iterrows():
        key = (row.get("event_type"), row.get("radio_burst_type"))
        ts = row["timestamp_utc"]
        previous_ts = last_seen.get(key)

        if previous_ts is None or (ts - previous_ts) > window:
            keep_indices.append(idx)
            last_seen[key] = ts

    return df.loc[keep_indices]


def _event_severity(row: pd.Series) -> int:
    flare_class = str(row.get("flare_class") or "")
    burst_type = str(row.get("radio_burst_type") or "").upper()

    if flare_class.upper().startswith("X"):
        return 3
    if burst_type == "II":
        return 3
    if flare_class.upper().startswith("M"):
        return 2
    if burst_type == "III":
        return 1
    return 0


def find_associated_cme(event_time, cme_df: pd.DataFrame, hours: int = 6) -> pd.Series | None:
    if cme_df.empty or "timestamp_utc" not in cme_df.columns:
        return None

    window_df = cme_df[
        (cme_df["timestamp_utc"] >= event_time) & (cme_df["timestamp_utc"] <= event_time + pd.Timedelta(hours=hours))
    ]
    if window_df.empty:
        return None

    return window_df.sort_values("timestamp_utc").iloc[0]


def estimate_cme_arrival(cme_row: pd.Series):
    speed = cme_row.get("speed")
    if speed is None or pd.isna(speed) or float(speed) <= 0:
        return None, None

    au_km = 1.496e8
    travel_hours = (au_km / float(speed)) / 3600
    arrival_time = cme_row["timestamp_utc"] + pd.Timedelta(hours=travel_hours)
    return arrival_time, travel_hours


def nearest_master_row(target_time, tolerance_hours: int = 3):
    if target_time is None or pd.isna(target_time) or master_df.empty or "timestamp_utc" not in master_df.columns:
        return None, False

    data_available = target_time <= master_df["timestamp_utc"].max()

    diffs = (master_df["timestamp_utc"] - target_time).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(hours=tolerance_hours):
        return None, data_available

    return master_df.loc[idx], data_available


def render_chain_step(title: str, lines: list[str], last: bool = False) -> None:
    lines_html = "".join(f"<div>{escape(line)}</div>" for line in lines)
    st.markdown(
        f"""
        <div style="
            background:#dcdcdc;
            color:#000000;
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            padding:12px 16px;
            font-family:'MS Sans Serif', Tahoma, sans-serif;
        ">
            <div style="font-weight:700; color:#000080; margin-bottom:4px;">{escape(title)}</div>
            <div style="font-size:0.85rem; line-height:1.5;">{lines_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not last:
        st.markdown(
            "<div style='text-align:center; font-size:1.3rem; color:#9adfff; margin:2px 0;'>&darr;</div>",
            unsafe_allow_html=True,
        )


def _render_chain_box(title: str, lines: list[str], height: int) -> None:
    lines_html = "".join(f"<div>{escape(line)}</div>" for line in lines)
    st.markdown(
        f"""
        <div style="
            background:#dcdcdc;
            color:#000000;
            border: 2px solid #808080;
            border-top-color:#ffffff;
            border-left-color:#ffffff;
            padding:8px 12px;
            height:{height}px;
            box-sizing:border-box;
            overflow-y:auto;
            margin-bottom:10px;
            font-family:'MS Sans Serif', Tahoma, sans-serif;
        ">
            <div style="font-weight:700; color:#000080; margin-bottom:4px;">{escape(title)}</div>
            <div style="font-size:0.82rem; line-height:1.35;">{lines_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chain_grid(steps: list[tuple[str, list[str]]], columns: int = 2) -> None:
    """Lay chain steps out in a plain ordered grid, reading order left to
    right then top to bottom. Cards within the same row share a height
    sized to that row's tallest card, so rows align cleanly without a
    single oversized height forced onto every card in the dialog.
    """
    title_px, padding_px, line_px = 26, 18, 19

    for row_start in range(0, len(steps), columns):
        row_steps = steps[row_start : row_start + columns]
        max_lines = max(len(lines) for _, lines in row_steps)
        row_height = title_px + padding_px + max_lines * line_px

        cols = st.columns(columns)
        for col, step in zip(cols, row_steps):
            with col:
                _render_chain_box(*step, height=row_height)


def style_retro_dialog() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] [role="dialog"] {
            background-color: #1a1a1a !important;
            border: 2px solid #ffffff !important;
            box-shadow: 4px 4px 0px #808080 !important;
            padding: 0 !important;
            max-height: 85vh !important;
            overflow-y: auto !important;
        }
        div[data-testid="stDialog"] [role="dialog"] > div {
            padding: 8px 14px !important;
        }
        div[data-testid="stDialog"] [data-testid="stVerticalBlock"] {
            gap: 0.3rem !important;
        }
        div[data-testid="stDialog"] h1,
        div[data-testid="stDialog"] h2,
        div[data-testid="stDialog"] h3,
        div[data-testid="stDialog"] p,
        div[data-testid="stDialog"] span,
        div[data-testid="stDialog"] label {
            color: #f2f2f2 !important;
        }

        /* st.info/warning/success/error render on a light background (see
        the div[data-testid="stAlert"] rule in apply_retro_windows_style).
        That rule alone loses here: div[data-testid="stDialog"] p above has
        higher CSS specificity than div[data-testid="stAlert"] *, so inside
        a dialog it was winning and forcing near-white text onto the
        alert's light background — unreadable. This selector matches both
        attributes, giving it higher specificity than either rule alone,
        so alert text stays dark inside dialogs too. */
        div[data-testid="stDialog"] div[data-testid="stAlert"],
        div[data-testid="stDialog"] div[data-testid="stAlert"] * {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def open_dialog(kind: str, payload) -> None:
    """Streamlit's own "dialog stays open across reruns" tracking can be
    unreliable across full-page reruns (e.g. the global auto_refresh timer),
    not just reruns triggered from inside the dialog itself. So instead of
    relying on that, we track which dialog should be open ourselves and
    re-assert it from one central dispatcher on every single script run.
    """
    st.session_state.active_dialog = (kind, payload)
    st.rerun()


def close_active_dialog() -> None:
    st.session_state.active_dialog = None
    st.rerun()


def render_dialog_close_button(key: str) -> None:
    style_retro_dialog()
    _, close_col = st.columns([10, 1])
    with close_col:
        if st.button("✕", key=key, use_container_width=True):
            close_active_dialog()


SAVED_EVENTS_PATH = PROJECT_ROOT / "data" / "saved_events.json"


def load_saved_events() -> list[dict]:
    if not SAVED_EVENTS_PATH.exists():
        return []
    try:
        with open(SAVED_EVENTS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def _write_saved_events(records: list[dict]) -> None:
    SAVED_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVED_EVENTS_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, default=str)


def save_event_record(row: pd.Series) -> bool:
    record = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            record[key] = value.isoformat()
        elif pd.isna(value):
            record[key] = None
        else:
            record[key] = value
    record["saved_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    records = load_saved_events()
    new_key = (record.get("timestamp_utc"), record.get("event_type"))
    if any((r.get("timestamp_utc"), r.get("event_type")) == new_key for r in records):
        return False

    records.append(record)
    _write_saved_events(records)
    return True


def remove_saved_event(index: int) -> None:
    records = load_saved_events()
    if 0 <= index < len(records):
        records.pop(index)
        _write_saved_events(records)


def record_to_series(record: dict) -> pd.Series:
    data = dict(record)
    if data.get("timestamp_utc"):
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    return pd.Series(data)


LIBRARY_DIR = PROJECT_ROOT / "data" / "library"
LIBRARY_INDEX_PATH = PROJECT_ROOT / "data" / "library_index.json"
LIBRARY_CATEGORIES = ["Concepts", "Articles", "Research Papers"]
LIBRARY_CATEGORY_COLORS = {
    "Concepts": "#1f4a7a",
    "Articles": "#1f7a3a",
    "Research Papers": "#5a1f7a",
}


def load_library_index() -> list[dict]:
    if not LIBRARY_INDEX_PATH.exists():
        return []
    try:
        with open(LIBRARY_INDEX_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def _write_library_index(records: list[dict]) -> None:
    LIBRARY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIBRARY_INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, default=str)


def add_library_document(category: str, title: str, uploaded_file) -> None:
    category_dir = LIBRARY_DIR / category.lower().replace(" ", "_")
    category_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
    file_path = category_dir / stored_name
    with open(file_path, "wb") as out_file:
        out_file.write(uploaded_file.getbuffer())

    records = load_library_index()
    records.append(
        {
            "title": title or uploaded_file.name,
            "category": category,
            "filename": uploaded_file.name,
            "stored_path": str(file_path.relative_to(PROJECT_ROOT)),
            "added_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    )
    _write_library_index(records)


def remove_library_document(index: int) -> None:
    records = load_library_index()
    if 0 <= index < len(records):
        record = records.pop(index)
        stored_path = PROJECT_ROOT / record.get("stored_path", "")
        if stored_path.exists():
            try:
                stored_path.unlink()
            except OSError:
                pass
        _write_library_index(records)


@st.dialog("Space Weather Concepts", width="large", dismissible=False)
def show_space_weather_library() -> None:
    render_dialog_close_button("close_library")

    records = load_library_index()
    tabs = st.tabs(LIBRARY_CATEGORIES)

    for tab, category in zip(tabs, LIBRARY_CATEGORIES):
        with tab:
            with st.expander("➕ Add Document"):
                title_input = st.text_input("Title", key=f"library_title_{category}")
                uploaded_file = st.file_uploader("Choose a file", key=f"library_upload_{category}")
                if st.button("Save", key=f"library_save_{category}"):
                    if uploaded_file is not None:
                        add_library_document(category, title_input, uploaded_file)
                        st.toast("Document saved.")
                        st.rerun()
                    else:
                        st.warning("Choose a file first.")

            category_records = [(i, r) for i, r in enumerate(records) if r.get("category") == category]

            if not category_records:
                st.info(f"No {category.lower()} saved yet.")
                continue

            color = LIBRARY_CATEGORY_COLORS.get(category, "#3a3a3a")
            cols_per_row = 4

            for row_start in range(0, len(category_records), cols_per_row):
                chunk = category_records[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, (idx, record) in zip(cols, chunk):
                    with col:
                        st.markdown(
                            f"""
                            <div style="
                                background:{color};
                                border:2px solid #808080;
                                border-radius:4px;
                                height:90px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                margin-bottom:6px;
                                font-size:2rem;
                            ">📄</div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.caption(record.get("title", "Untitled"))

                        stored_path = PROJECT_ROOT / record.get("stored_path", "")
                        if stored_path.exists():
                            with open(stored_path, "rb") as file:
                                st.download_button(
                                    "Open",
                                    data=file.read(),
                                    file_name=record.get("filename", "document"),
                                    key=f"library_open_{idx}",
                                    use_container_width=True,
                                )

                        if st.button("🗑", key=f"library_remove_{idx}", use_container_width=True):
                            remove_library_document(idx)
                            st.rerun()


def event_title(row: pd.Series) -> str:
    flare_class = row.get("flare_class")
    burst_type = row.get("radio_burst_type")
    event_type = str(row.get("event_type") or "")

    if pd.notna(flare_class):
        return f"{flare_class} Flare"
    if pd.notna(burst_type):
        return f"Type {burst_type} Radio Burst"
    return EVENT_TYPE_CATEGORY.get(event_type, event_type or "Solar Event")


@st.dialog("Event Explorer", width="large", dismissible=False)
def show_event_explorer(event: pd.Series) -> None:
    render_dialog_close_button("close_event_explorer")

    ts = event["timestamp_utc"]
    cme_df = load_processed_data("cme")

    title = event_title(event)
    event_type = str(event.get("event_type") or "")
    region = event.get("active_region")
    severity = _event_severity(event)
    risk_label = {3: "High Impact Potential", 2: "Moderate Impact Potential"}.get(severity, "Low Risk")
    region_text = f"AR{region}" if pd.notna(region) else "Unknown"

    steps: list[tuple[str, list[str]]] = [
        (
            "Solar Event",
            [
                f"Title: {title}",
                f"Time: {latest_label_time(ts)}",
                f"Type: {EVENT_TYPE_CATEGORY.get(event_type, event_type)}",
                f"Region: {region_text}",
                f"Risk: {risk_label}",
            ],
        )
    ]

    cme_match = find_associated_cme(ts, cme_df)

    if cme_match is None:
        steps.append(("Associated CME", ["No CME detected within 6 hours of this event."]))
        steps.append(("Arrival at Earth", ["Not applicable — no associated CME."]))
        steps.append(("Solar Wind Changes", ["Not applicable."]))
        steps.append(("IMF Changes (Bz, Bt)", ["Not applicable."]))
        steps.append(("Kp Response", ["Not applicable."]))
        steps.append(("Dst Response", ["Not applicable."]))
        render_chain_grid(steps)
        return

    steps.append(
        (
            "Associated CME",
            [
                f"Speed: {format_value(cme_match.get('speed'), ' km/s', 1)}",
                f"Latitude: {format_value(cme_match.get('latitude'), '°', 1)}",
                f"Longitude: {format_value(cme_match.get('longitude'), '°', 1)}",
                f"Half Angle: {format_value(cme_match.get('half_angle'), '°', 1)}",
                f"Start Time: {latest_label_time(cme_match['timestamp_utc'])}",
            ],
        )
    )

    arrival_time, travel_hours = estimate_cme_arrival(cme_match)

    if arrival_time is None:
        steps.append(("Arrival at Earth", ["Could not estimate — missing CME speed."]))
        steps.append(("Solar Wind Changes", ["Not applicable."]))
        steps.append(("IMF Changes (Bz, Bt)", ["Not applicable."]))
        steps.append(("Kp Response", ["Not applicable."]))
        steps.append(("Dst Response", ["Not applicable."]))
        render_chain_grid(steps)
        return

    steps.append(
        (
            "Arrival at Earth (Estimated)",
            [
                f"Estimated Arrival: {latest_label_time(arrival_time)}",
                f"Travel Time: {travel_hours:.1f} hours",
                "Heuristic constant-speed transit model.",
            ],
        )
    )

    response_row, data_available = nearest_master_row(arrival_time)
    not_available_note = (
        "Not yet recorded (arrival is in the future)."
        if not data_available
        else "No recorded data within ±3 hours of arrival."
    )

    if response_row is not None:
        steps.append(
            (
                "Solar Wind Changes",
                [
                    f"Speed: {format_value(response_row.get('solar_wind_speed'), ' km/s', 1)}",
                    f"Density: {format_value(response_row.get('proton_density'), ' p/cm3', 2)}",
                    f"Temperature: {format_value(response_row.get('temperature'), ' K', 0)}",
                    f"At: {latest_label_time(response_row['timestamp_utc'])}",
                ],
            )
        )
        steps.append(
            (
                "IMF Changes (Bz, Bt)",
                [
                    f"Bz: {format_value(response_row.get('bz'), ' nT', 2)}",
                    f"Bt: {format_value(response_row.get('bt'), ' nT', 2)}",
                    f"At: {latest_label_time(response_row['timestamp_utc'])}",
                ],
            )
        )
        steps.append(
            (
                "Kp Response",
                [
                    f"Kp: {format_value(response_row.get('kp'), '', 1)}",
                    f"At: {latest_label_time(response_row['timestamp_utc'])}",
                ],
            )
        )
        steps.append(
            (
                "Dst Response",
                [
                    f"Dst: {format_value(response_row.get('dst'), ' nT', 1)}",
                    f"At: {latest_label_time(response_row['timestamp_utc'])}",
                ],
            )
        )
    else:
        steps.append(("Solar Wind Changes", [not_available_note]))
        steps.append(("IMF Changes (Bz, Bt)", [not_available_note]))
        steps.append(("Kp Response", [not_available_note]))
        steps.append(("Dst Response", [not_available_note]))

    render_chain_grid(steps)


def pick_anchor_time(*times):
    valid = [t for t in times if t is not None and not pd.isna(t)]
    if not valid:
        return None
    return max(valid)


def find_cme_for_arrival(target_time, cme_df: pd.DataFrame, tolerance_hours: int = 12):
    if target_time is None or pd.isna(target_time) or cme_df.empty or "speed" not in cme_df.columns:
        return None, None, None

    candidates = cme_df.dropna(subset=["speed", "timestamp_utc"])
    if candidates.empty:
        return None, None, None

    best_row, best_arrival, best_travel, best_diff = None, None, None, None

    for _, row in candidates.iterrows():
        arrival_time, travel_hours = estimate_cme_arrival(row)
        if arrival_time is None:
            continue

        diff = abs((arrival_time - target_time).total_seconds())
        if best_diff is None or diff < best_diff:
            best_row, best_arrival, best_travel, best_diff = row, arrival_time, travel_hours, diff

    if best_row is None or best_diff > tolerance_hours * 3600:
        return None, None, None

    return best_row, best_arrival, best_travel


def find_source_event_for_cme(cme_row: pd.Series, events_df: pd.DataFrame, hours_before: int = 6):
    if events_df.empty or "timestamp_utc" not in events_df.columns:
        return None

    cme_time = cme_row["timestamp_utc"]
    window_df = events_df[
        (events_df["timestamp_utc"] >= cme_time - pd.Timedelta(hours=hours_before))
        & (events_df["timestamp_utc"] <= cme_time)
    ]
    if window_df.empty:
        return None

    window_df = window_df.copy()
    window_df["severity"] = window_df.apply(_event_severity, axis=1)
    window_df = window_df.sort_values(["severity", "timestamp_utc"], ascending=[False, False])
    return window_df.iloc[0]


@st.dialog("Event Explorer", width="large", dismissible=False)
def show_reverse_event_explorer(target_time, effect_label: str) -> None:
    render_dialog_close_button("close_reverse_explorer")

    if target_time is None or pd.isna(target_time):
        st.info("No timestamp available for this value yet.")
        return

    cme_df = load_processed_data("cme")
    events_df = load_processed_data("solar_events")

    cme_row, arrival_time, travel_hours = find_cme_for_arrival(target_time, cme_df)

    if cme_row is None:
        render_chain_grid(
            [
                ("Solar Event", ["No matching CME found within ±12 hours of this reading."]),
                (f"Observed: {effect_label}", [f"Time: {latest_label_time(target_time)}"]),
            ]
        )
        return

    source_event = find_source_event_for_cme(cme_row, events_df)
    if source_event is not None:
        title = event_title(source_event)
        region = source_event.get("active_region")
        region_text = f"AR{region}" if pd.notna(region) else "Unknown"
        severity = _event_severity(source_event)
        risk_label = {3: "High Impact Potential", 2: "Moderate Impact Potential"}.get(severity, "Low Risk")
        solar_event_step = (
            "Solar Event",
            [
                f"Title: {title}",
                f"Time: {latest_label_time(source_event['timestamp_utc'])}",
                f"Region: {region_text}",
                f"Risk: {risk_label}",
            ],
        )
    else:
        solar_event_step = ("Solar Event", ["No clear source flare/burst found before this CME."])

    render_chain_grid(
        [
            solar_event_step,
            (
                "Associated CME",
                [
                    f"Speed: {format_value(cme_row.get('speed'), ' km/s', 1)}",
                    f"Latitude: {format_value(cme_row.get('latitude'), '°', 1)}",
                    f"Longitude: {format_value(cme_row.get('longitude'), '°', 1)}",
                    f"Half Angle: {format_value(cme_row.get('half_angle'), '°', 1)}",
                    f"Start Time: {latest_label_time(cme_row['timestamp_utc'])}",
                ],
            ),
            (
                "Arrival at Earth (Estimated)",
                [
                    f"Estimated Arrival: {latest_label_time(arrival_time)}",
                    f"Travel Time: {travel_hours:.1f} hours",
                    "Heuristic constant-speed transit model.",
                ],
            ),
            (f"Observed: {effect_label}", [f"Time: {latest_label_time(target_time)}"]),
        ]
    )


def render_storyboard_scene(frame: int, has_cme: bool) -> None:
    if not has_cme:
        position_class, moving_class = "pos-sun", ""
    elif frame == 0:
        position_class, moving_class = "pos-sun", ""
    elif frame == 1:
        position_class, moving_class = "pos-sun", "anim-to-sat"
    elif frame == 2:
        position_class, moving_class = "pos-sat", ""
    elif frame == 3:
        position_class, moving_class = "pos-sat", "anim-to-earth"
    else:
        position_class, moving_class = "pos-earth", ("pulse" if frame == 4 else "")

    st.markdown(
        f"""
        <style>
        @keyframes moveToSat {{ from {{ left: 4%; }} to {{ left: 46%; }} }}
        @keyframes moveToEarth {{ from {{ left: 46%; }} to {{ left: 90%; }} }}
        @keyframes pulseGlow {{
            0%, 100% {{ transform: translateY(-50%) scale(1); }}
            50% {{ transform: translateY(-50%) scale(1.5); }}
        }}
        .storyboard-track {{
            position: relative;
            height: 70px;
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            margin-bottom: 14px;
            overflow: hidden;
        }}
        .storyboard-fixed-icon {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.6rem;
        }}
        .storyboard-moving-icon {{
            position: absolute;
            top: 50%;
            font-size: 1.6rem;
            transform: translateY(-50%);
        }}
        .pos-sun {{ left: 4%; }}
        .pos-sat {{ left: 46%; }}
        .pos-earth {{ left: 90%; }}
        .anim-to-sat {{ animation: moveToSat 2.6s linear forwards; }}
        .anim-to-earth {{ animation: moveToEarth 2.6s linear forwards; }}
        .pulse {{ animation: pulseGlow 1s ease-in-out infinite; }}
        </style>
        <div class="storyboard-track">
            <div class="storyboard-fixed-icon" style="left:2%;">☀️</div>
            <div class="storyboard-fixed-icon" style="left:45%;">🛰️</div>
            <div class="storyboard-fixed-icon" style="left:88%;">🌍</div>
            <div class="storyboard-moving-icon {position_class} {moving_class}">💨</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Event Storyboard", width="large", dismissible=False)
def play_event_animation(event: pd.Series) -> None:
    render_dialog_close_button("close_storyboard")

    cme_df = load_processed_data("cme")

    ts = event["timestamp_utc"]
    title = event_title(event)
    region = event.get("active_region")
    region_text = f"AR{region}" if pd.notna(region) else "Unknown"
    severity = _event_severity(event)
    risk_label = {3: "High Impact Potential", 2: "Moderate Impact Potential"}.get(severity, "Low Risk")

    cme_match = find_associated_cme(ts, cme_df)
    has_cme = cme_match is not None

    arrival_time, travel_hours, response_row, data_available = None, None, None, False
    if has_cme:
        arrival_time, travel_hours = estimate_cme_arrival(cme_match)
        if arrival_time is not None:
            response_row, data_available = nearest_master_row(arrival_time)

    max_frame = 5 if has_cme else 1

    event_key = f"{ts.isoformat()}_{title}"
    if st.session_state.get("storyboard_event_key") != event_key:
        st.session_state.storyboard_event_key = event_key
        st.session_state.storyboard_frame = 0

    frame = st.session_state.storyboard_frame

    render_storyboard_scene(frame, has_cme)

    if frame == 0:
        render_chain_step(
            "☀️ Solar Event Detected",
            [
                f"{title}",
                f"Time: {latest_label_time(ts)}",
                f"Region: {region_text}",
                f"Risk: {risk_label}",
            ],
            last=True,
        )
    elif not has_cme:
        render_chain_step(
            "No CME Detected",
            ["No associated CME found within 6 hours of this event. Sequence ends here."],
            last=True,
        )
    elif frame == 1:
        render_chain_step(
            "💨 CME Traveling Toward Earth",
            [
                f"Speed: {format_value(cme_match.get('speed'), ' km/s', 1)}",
                f"Estimated Travel Time: {f'{travel_hours:.1f} hours' if travel_hours else 'N/A'}",
            ],
            last=True,
        )
    elif frame == 2:
        if response_row is not None:
            speed_meaning, speed_risk = variable_meaning_and_risk("speed", response_row.get("solar_wind_speed"))
            bz_meaning, bz_risk = variable_meaning_and_risk("bz", response_row.get("bz"))
            render_chain_step(
                "🛰️ Satellite Checkpoint (L1)",
                [
                    f"Solar Wind Speed: {format_value(response_row.get('solar_wind_speed'), ' km/s', 1)} — {speed_meaning} ({speed_risk})",
                    f"Bz: {format_value(response_row.get('bz'), ' nT', 2)} — {bz_meaning} ({bz_risk})",
                    f"Recorded: {latest_label_time(response_row['timestamp_utc'])}",
                ],
                last=True,
            )
        else:
            note = "Not yet recorded (arrival is in the future)." if not data_available else "No recorded data near estimated arrival."
            render_chain_step("🛰️ Satellite Checkpoint (L1)", [note], last=True)
    elif frame == 3:
        render_chain_step(
            "💨 Continuing Toward Earth",
            ["Disturbance moving from L1 toward the magnetosphere."],
            last=True,
        )
    elif frame == 4:
        if response_row is not None:
            kp_meaning, kp_risk = variable_meaning_and_risk("kp", response_row.get("kp"))
            dst_meaning, dst_risk = variable_meaning_and_risk("dst", response_row.get("dst"))
            render_chain_step(
                "🌍 Earth Impact",
                [
                    f"Kp: {format_value(response_row.get('kp'), '', 1)} — {kp_meaning} ({kp_risk})",
                    f"Dst: {format_value(response_row.get('dst'), ' nT', 1)} — {dst_meaning} ({dst_risk})",
                    f"Recorded: {latest_label_time(response_row['timestamp_utc'])}",
                ],
                last=True,
            )
        else:
            note = "Not yet recorded (arrival is in the future)." if not data_available else "No recorded data near estimated arrival."
            render_chain_step("🌍 Earth Impact", [note], last=True)
    else:
        render_chain_step("✅ Sequence Complete", ["Press Restart to play again."], last=True)

    if st.button("⟲ Restart"):
        st.session_state.storyboard_frame = 0
        st.rerun()

    if frame < max_frame:
        if st_autorefresh is not None:
            st_autorefresh(interval=2800, key="storyboard_autoplay")
        else:
            st.warning("Install streamlit-autorefresh for auto-play. Use the Restart button to step manually.")
        st.session_state.storyboard_frame = frame + 1


@st.dialog("Event Animations", width="large", dismissible=False)
def show_animations_grid() -> None:
    render_dialog_close_button("close_animations_grid")

    full_chain = get_full_chain_events()
    if full_chain.empty:
        st.info("No events with a complete data chain yet. Check back as more data accumulates.")
        return

    ordered = full_chain.sort_values(["severity", "timestamp_utc"], ascending=[False, False])

    severity_colors = {3: "#7a1f1f", 2: "#7a5a1f", 1: "#1f4a7a", 0: "#3a3a3a"}

    cols_per_row = 4
    entries = list(ordered.iterrows())

    for row_start in range(0, len(entries), cols_per_row):
        chunk = entries[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, (idx, row) in zip(cols, chunk):
            with col:
                ts = row["timestamp_utc"]
                title = event_title(row)
                severity = int(row.get("severity", 0))
                color = severity_colors.get(severity, "#3a3a3a")

                st.markdown(
                    f"""
                    <div style="
                        background:{color};
                        border:2px solid #808080;
                        border-radius:4px;
                        height:90px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        margin-bottom:6px;
                        font-size:2rem;
                    ">🎬</div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(title, key=f"anim_grid_{idx}", use_container_width=True):
                    open_dialog("storyboard", row)

                st.caption(ts.strftime("%d %b %Y, %H:%M UTC"))


def _render_event_buttons(events: pd.DataFrame, key_prefix: str) -> None:
    for idx, row in events.iterrows():
        ts = row["timestamp_utc"]
        label = f"{event_title(row)} - {ts.strftime('%d %b %Y')}"

        btn_col, save_col = st.columns([0.85, 0.15])
        with btn_col:
            if st.button(label, key=f"{key_prefix}_{idx}", use_container_width=True):
                open_dialog("event_explorer", row)
        with save_col:
            if st.button("💾", key=f"{key_prefix}_save_{idx}", use_container_width=True):
                if save_event_record(row):
                    st.toast("Event saved.")
                else:
                    st.toast("Already saved.")


def render_event_button_list(
    events: pd.DataFrame,
    key_prefix: str,
    scrollable: bool = False,
    height: int = 260,
) -> None:
    if events.empty:
        st.info("No events to show.")
        return

    if scrollable:
        with st.container(height=height):
            _render_event_buttons(events, key_prefix)
    else:
        _render_event_buttons(events, key_prefix)


@st.dialog("Saved Solar Events", width="large", dismissible=False)
def show_saved_events() -> None:
    render_dialog_close_button("close_saved_events")

    records = load_saved_events()

    if not records:
        st.info("No saved events yet. Use the 💾 button next to an event in the News Feed to save it here.")
        return

    indexed = [(i, record_to_series(record)) for i, record in enumerate(records)]

    groups: dict = {}
    for i, row in indexed:
        ts = row.get("timestamp_utc")
        date_key = ts.date() if pd.notna(ts) else None
        groups.setdefault(date_key, []).append((i, row, ts))

    dated_keys = sorted((d for d in groups if d is not None), reverse=True)
    ordered_keys = dated_keys + ([None] if None in groups else [])

    for date_key in ordered_keys:
        entries = groups[date_key]
        date_label = date_key.strftime("%d %B %Y") if date_key is not None else "Unknown date"

        with st.expander(f"{date_label} ({len(entries)})", expanded=False):
            entries_sorted = sorted(entries, key=lambda e: e[2] if pd.notna(e[2]) else pd.Timestamp.min, reverse=True)

            for i, row, ts in entries_sorted:
                time_text = ts.strftime("%d %b %Y %H:%M UTC") if pd.notna(ts) else "Unknown time"
                label = f"{event_title(row)} - {time_text}"

                view_col, remove_col = st.columns([0.85, 0.15])
                with view_col:
                    if st.button(label, key=f"saved_event_view_{i}", use_container_width=True):
                        open_dialog("event_explorer", row)
                with remove_col:
                    if st.button("🗑", key=f"saved_event_remove_{i}", use_container_width=True):
                        remove_saved_event(i)
                        st.rerun()


def get_notable_solar_events() -> pd.DataFrame:
    events_df = load_processed_data("solar_events")
    if events_df.empty:
        return pd.DataFrame()

    candidates = recent_window(events_df, 2)
    if candidates.empty:
        candidates = events_df.sort_values("timestamp_utc").tail(40)

    candidates = candidates.copy()
    candidates["severity"] = candidates.apply(_event_severity, axis=1)

    if "event_type" in candidates.columns:
        notable = candidates[candidates["event_type"].astype(str).isin(["FLA", "XRA", "RSP", "RBR", "RNS"])]
    else:
        notable = candidates

    return dedupe_near_duplicate_events(notable)


def get_full_chain_events(lookback_days: int = 30, limit: int = 30) -> pd.DataFrame:
    """Events with a complete Sun-to-Earth chain: an associated CME whose
    estimated arrival lands on a real recorded master_df row (i.e. data
    exists all the way through Kp/Dst), not just a CME still in transit.
    """
    events_df = load_processed_data("solar_events")
    cme_df = load_processed_data("cme")
    if events_df.empty or cme_df.empty:
        return pd.DataFrame()

    candidates = recent_window(events_df, lookback_days)
    if candidates.empty:
        candidates = events_df.copy()

    if "event_type" in candidates.columns:
        candidates = candidates[candidates["event_type"].astype(str).isin(["FLA", "XRA", "RSP", "RBR", "RNS"])]

    candidates = dedupe_near_duplicate_events(candidates)
    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.copy()
    candidates["severity"] = candidates.apply(_event_severity, axis=1)
    candidates = candidates.sort_values(["severity", "timestamp_utc"], ascending=[False, False])

    seen_cme_keys = set()
    full_chain_rows = []
    for _, row in candidates.iterrows():
        cme_match = find_associated_cme(row["timestamp_utc"], cme_df)
        if cme_match is None:
            continue

        cme_key = cme_match.get("activity_id") or cme_match["timestamp_utc"]
        if cme_key in seen_cme_keys:
            continue

        arrival_time, _ = estimate_cme_arrival(cme_match)
        if arrival_time is None:
            continue

        response_row, _ = nearest_master_row(arrival_time)
        if response_row is None:
            continue

        seen_cme_keys.add(cme_key)
        full_chain_rows.append(row)
        if len(full_chain_rows) >= limit:
            break

    if not full_chain_rows:
        return pd.DataFrame()

    return pd.DataFrame(full_chain_rows)


def solar_event_news_feed() -> None:
    st.subheader("Solar Activity News Feed")

    notable = get_notable_solar_events()

    if notable.empty:
        st.info("No notable solar events in the recent window.")
        return

    by_severity = notable.sort_values(["severity", "timestamp_utc"], ascending=[False, False])
    by_latest = notable.sort_values("timestamp_utc", ascending=False)

    if "news_feed_severity_expanded" not in st.session_state:
        st.session_state.news_feed_severity_expanded = False
    if "news_feed_latest_expanded" not in st.session_state:
        st.session_state.news_feed_latest_expanded = False

    severity_col, latest_col = st.columns(2)

    with severity_col:
        st.markdown("#### By Severity")
        expanded = st.session_state.news_feed_severity_expanded
        limit = len(by_severity) if expanded else 5
        render_event_button_list(by_severity.head(limit), "news_feed_sev", scrollable=expanded)

        if len(by_severity) > 5:
            button_label = "Show fewer" if st.session_state.news_feed_severity_expanded else "More"
            if st.button(button_label, key="news_feed_severity_more"):
                st.session_state.news_feed_severity_expanded = not st.session_state.news_feed_severity_expanded
                st.rerun()

    with latest_col:
        st.markdown("#### Latest Recorded")
        expanded = st.session_state.news_feed_latest_expanded
        limit = len(by_latest) if expanded else 5
        render_event_button_list(by_latest.head(limit), "news_feed_latest", scrollable=expanded)

        if len(by_latest) > 5:
            button_label = "Show fewer" if st.session_state.news_feed_latest_expanded else "More"
            if st.button(button_label, key="news_feed_latest_more"):
                st.session_state.news_feed_latest_expanded = not st.session_state.news_feed_latest_expanded
                st.rerun()


def solar_events_analysis() -> None:
    st.subheader("Solar Events — Current Analysis")

    events_df = load_processed_data("solar_events")
    if events_df.empty:
        st.info("No solar events data available yet. Run the live updater to populate this.")
        return

    cme_df = load_processed_data("cme")

    recent = dedupe_near_duplicate_events(recent_window(events_df, 7))
    today = dedupe_near_duplicate_events(recent_window(events_df, 1))

    total_events_today = len(today)

    total_flares = 0
    total_radio_bursts = 0
    if "event_type" in recent.columns:
        event_type_str = recent["event_type"].astype(str)
        total_flares = int(event_type_str.eq("FLA").sum())
        total_radio_bursts = int(event_type_str.isin(RADIO_BURST_TYPES).sum())

    x_class_count = 0
    m_class_count = 0
    if "flare_class" in recent.columns:
        classes = recent["flare_class"].dropna().astype(str).str.upper()
        x_class_count = int(classes.str.startswith("X").sum())
        m_class_count = int(classes.str.startswith("M").sum())

    most_active_region = "N/A"
    if "active_region" in recent.columns:
        region_values = recent["active_region"].dropna().astype(str)
        if not region_values.empty:
            most_active_region = region_values.value_counts().idxmax()

    associated_cme_count = count_associated_cmes(recent, cme_df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Today's Solar Events", str(total_events_today))
    with c2:
        metric_card("Total Solar Flares (7d)", str(total_flares))
    with c3:
        metric_card("Total Radio Bursts (7d)", str(total_radio_bursts))
    with c4:
        metric_card("X-Class Flares (7d)", str(x_class_count))
    with c5:
        metric_card("Associated CMEs (7d)", str(associated_cme_count))
    with c6:
        metric_card("Most Active Region", str(most_active_region))

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.markdown("### Events Timeline")
    render_event_timeline(recent)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    event_charts = []

    if "flare_class" in recent.columns:
        class_letters = recent["flare_class"].dropna().astype(str).str[0].str.upper()
        class_letters = class_letters[class_letters.isin(["X", "M", "C", "B", "A"])]
        if not class_letters.empty:
            class_label_map = {"X": "X-Class", "M": "M-Class", "C": "C-Class", "B": "B-Class", "A": "A-Class"}
            order = ["X-Class", "M-Class", "C-Class", "B-Class", "A-Class"]
            class_counts = class_letters.map(class_label_map).value_counts()
            class_counts = class_counts.reindex(order).dropna().reset_index()
            class_counts.columns = ["class", "count"]
            fig = px.bar(class_counts, x="class", y="count", title="Flare Class Distribution (Last 7 Days)")
            fig.update_layout(height=340)
            event_charts.append(fig)

    if not recent.empty:
        daily_counts = recent.copy()
        daily_counts["date"] = daily_counts["timestamp_utc"].dt.date.astype(str)
        counts_by_day = daily_counts.groupby("date").size().reset_index(name="events")
        fig = px.bar(counts_by_day, x="date", y="events", title="Events Per Day (Last 7 Days)")
        fig.update_layout(height=340)
        event_charts.append(fig)

    if "event_type" in recent.columns and not recent.empty:
        category_series = recent["event_type"].astype(str).map(lambda t: EVENT_TYPE_CATEGORY.get(t, "Other"))
        category_counts = category_series.value_counts().reset_index()
        category_counts.columns = ["category", "count"]
        fig = px.pie(category_counts, names="category", values="count", title="Event Type Distribution (Last 7 Days)")
        fig.update_layout(height=340)
        event_charts.append(fig)

    if "active_region" in recent.columns:
        region_data = recent.dropna(subset=["active_region"]).copy()
        if not region_data.empty:
            region_data["active_region"] = region_data["active_region"].astype(str)
            region_counts = region_data["active_region"].value_counts().reset_index()
            region_counts.columns = ["active_region", "event_count"]
            region_counts = region_counts.head(10)
            fig = px.bar(region_counts, x="active_region", y="event_count", title="Active Region Frequency (Last 7 Days)")
            fig.update_layout(height=340)
            event_charts.append(fig)

    if "duration_minutes" in recent.columns:
        duration_data = recent.dropna(subset=["duration_minutes"])
        duration_data = duration_data[duration_data["duration_minutes"] > 0]
        if not duration_data.empty:
            fig = px.histogram(duration_data, x="duration_minutes", nbins=20, title="Event Duration (Minutes, Last 7 Days)")
            fig.update_layout(height=340)
            event_charts.append(fig)

    for row_start in range(0, len(event_charts), 2):
        chart_cols = st.columns(2)
        for col, chart_fig in zip(chart_cols, event_charts[row_start : row_start + 2]):
            with col:
                plot_retro(chart_fig)

    st.markdown("### Statistics (7 Days)")
    average_events_per_day = len(recent) / 7 if not recent.empty else 0

    st1, st2, st3, st4, st5, st6, st7 = st.columns(7)
    st1.metric("Total Events", str(len(recent)))
    st2.metric("Total Flares", str(total_flares))
    st3.metric("Total Radio Bursts", str(total_radio_bursts))
    st4.metric("X-Class Count", str(x_class_count))
    st5.metric("M-Class Count", str(m_class_count))
    st6.metric("Most Active Region", str(most_active_region))
    st7.metric("Avg Events/Day", format_value(average_events_per_day, "", 1))

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.markdown("### Latest Events")
    display_cols = [
        col for col in ["timestamp_utc", "event_type", "flare_class", "active_region", "duration_minutes", "note"]
        if col in recent.columns
    ]
    if display_cols:
        latest_events = recent.sort_values("timestamp_utc", ascending=False).head(15)[display_cols]
        render_simple_retro_table(
            latest_events,
            display_names={
                "timestamp_utc": "Time (UTC)",
                "event_type": "Type",
                "flare_class": "Flare Class",
                "active_region": "Region",
                "duration_minutes": "Duration (min)",
                "note": "Note",
            },
        )
    else:
        st.info("No displayable event columns found.")


EARTH_DIRECTED_LONGITUDE_DEG = 30


def cme_analysis() -> None:
    st.subheader("CME — Current Analysis")

    cme_df = load_processed_data("cme")
    if cme_df.empty:
        st.info("No CME data available yet. Run the live updater to populate this.")
        return

    recent = recent_window(cme_df, 7)
    speeds = recent["speed"].dropna() if "speed" in recent.columns else pd.Series(dtype=float)

    latest_speed, latest_speed_time = latest_non_null(cme_df, "speed")
    fastest_row = row_at_extreme(recent, "speed", "max") if not speeds.empty else None
    avg_speed = speeds.mean() if not speeds.empty else None
    total_cmes = len(recent)

    earth_directed_count = 0
    if "longitude" in recent.columns:
        earth_directed_count = int(
            recent["longitude"].dropna().abs().le(EARTH_DIRECTED_LONGITUDE_DEG).sum()
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Latest CME Speed", format_value(latest_speed, " km/s", 1), latest_label_time(latest_speed_time))
    with c2:
        metric_card(
            "Fastest CME (7d)",
            format_value(None if fastest_row is None else fastest_row["speed"], " km/s", 1),
            time_caption(fastest_row),
        )
    with c3:
        metric_card("Average Speed (7d)", format_value(avg_speed, " km/s", 1))
    with c4:
        metric_card("Total CMEs This Week", str(total_cmes))
    with c5:
        metric_card(
            "Earth-Directed CMEs (7d)",
            str(earth_directed_count),
            f"|longitude| ≤ {EARTH_DIRECTED_LONGITUDE_DEG}°",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.markdown("### CME Statistics (7 Days)")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    if not speeds.empty:
        s1.metric("Max Speed", format_value(speeds.max(), " km/s", 1))
        s2.metric("Min Speed", format_value(speeds.min(), " km/s", 1))
        s3.metric("Average Speed", format_value(speeds.mean(), " km/s", 1))
        s4.metric("Median Speed", format_value(speeds.median(), " km/s", 1))
        s5.metric("Std Deviation", format_value(speeds.std(), " km/s", 1))
        s6.metric("Total CMEs", str(total_cmes))
    else:
        st.info("No CME speed data available for statistics.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    cme_charts = []

    if "speed" in recent.columns and not recent.empty:
        speed_data = recent.dropna(subset=["speed"]).sort_values("timestamp_utc")
        fig = px.line(speed_data, x="timestamp_utc", y="speed", markers=True, title="CME Speed vs Time (Last 7 Days)")
        fig.update_layout(height=340)
        cme_charts.append(fig)

        fig_hist = px.histogram(speed_data, x="speed", nbins=15, title="CME Speed Distribution (Last 7 Days)")
        fig_hist.update_layout(height=340)
        cme_charts.append(fig_hist)

    if "longitude" in recent.columns and not recent.empty:
        longitude_data = recent.dropna(subset=["longitude"])
        if not longitude_data.empty:
            fig = px.scatter(
                longitude_data,
                x="longitude",
                y="speed" if "speed" in longitude_data.columns else None,
                color="active_region" if "active_region" in longitude_data.columns else None,
                title="CME Longitude Distribution (near 0° is Earth-facing)",
            )
            fig.update_layout(height=340)
            cme_charts.append(fig)

    if "half_angle" in recent.columns and not recent.empty:
        width_data = recent.dropna(subset=["half_angle"])
        if not width_data.empty:
            fig = px.scatter(
                width_data,
                x="timestamp_utc",
                y="half_angle",
                size="speed" if "speed" in width_data.columns else None,
                color="speed" if "speed" in width_data.columns else None,
                title="CME Width (Half Angle) Over Time",
            )
            fig.update_layout(height=340)
            cme_charts.append(fig)

    for row_start in range(0, len(cme_charts), 2):
        chart_cols = st.columns(2)
        for col, chart_fig in zip(chart_cols, cme_charts[row_start : row_start + 2]):
            with col:
                plot_retro(chart_fig)

    st.markdown("### Latest CMEs")
    display_cols = [
        col for col in ["timestamp_utc", "active_region", "speed", "latitude", "longitude", "half_angle", "cme_type"]
        if col in recent.columns
    ]
    if display_cols:
        latest_cmes = recent.sort_values("timestamp_utc", ascending=False).head(15)[display_cols]
        render_simple_retro_table(
            latest_cmes,
            display_names={
                "timestamp_utc": "Time (UTC)",
                "active_region": "Region",
                "speed": "Speed",
                "latitude": "Latitude",
                "longitude": "Longitude",
                "half_angle": "Half Angle",
                "cme_type": "Type",
            },
        )
    else:
        st.info("No displayable CME columns found.")


SUN_DISK_RADIUS_DEG = 90


def add_sun_disk_background(fig, radius: int = SUN_DISK_RADIUS_DEG) -> None:
    sun_path = PROJECT_ROOT / "dashboard" / "assets" / "sun_disk.jpg"
    if not sun_path.exists():
        fig.add_shape(
            type="circle",
            x0=-radius,
            y0=-radius,
            x1=radius,
            y1=radius,
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.08)",
        )
        return

    encoded = get_base64_image(sun_path)
    fig.add_layout_image(
        dict(
            source=f"data:image/jpeg;base64,{encoded}",
            xref="x",
            yref="y",
            x=-radius,
            y=radius,
            sizex=radius * 2,
            sizey=radius * 2,
            sizing="stretch",
            layer="below",
        )
    )
    fig.update_layout(plot_bgcolor="#050505", paper_bgcolor="#050505")


def heliomap_solar_events_tab() -> None:
    events_df = load_processed_data("solar_events")
    if events_df.empty or "heliographic_lat" not in events_df.columns:
        st.info("No positional solar event data available yet.")
        return

    recent = recent_window(events_df, 7)
    located = recent.dropna(subset=["heliographic_lat", "heliographic_lon"]).copy()

    inner = st.tabs(["Region Map", "Highest Activity Region"])

    with inner[0]:
        if located.empty:
            st.info("No located solar events in the last 7 days.")
        else:
            located["category"] = located["event_type"].astype(str).map(
                lambda t: EVENT_TYPE_CATEGORY.get(t, "Other")
            )
            located["severity"] = located.apply(_event_severity, axis=1)
            located["marker_size"] = 8 + located["severity"] * 6

            fig = px.scatter(
                located,
                x="heliographic_lon",
                y="heliographic_lat",
                color="category",
                size="marker_size",
                hover_data=["timestamp_utc", "active_region", "flare_class"],
                title="Solar Event Positions (Last 7 Days)",
            )
            fig.update_xaxes(range=[-100, 100], title="Longitude (°)")
            fig.update_yaxes(range=[-100, 100], title="Latitude (°)", scaleanchor="x", scaleratio=1)
            add_sun_disk_background(fig)
            fig.update_layout(height=420)
            plot_retro(fig)

    with inner[1]:
        if "active_region" in recent.columns:
            region_data = recent.dropna(subset=["active_region"]).copy()
            if not region_data.empty:
                region_data["active_region"] = region_data["active_region"].astype(str)
                region_counts = region_data["active_region"].value_counts().reset_index()
                region_counts.columns = ["active_region", "event_count"]
                region_counts = region_counts.head(10)
                fig = px.bar(
                    region_counts, x="active_region", y="event_count", title="Most Active Regions (Last 7 Days)"
                )
                fig.update_layout(height=360)
                plot_retro(fig)
            else:
                st.info("No active region data available.")
        else:
            st.info("No active region column available.")


def heliomap_cme_tab() -> None:
    cme_df = load_processed_data("cme")
    if cme_df.empty or "latitude" not in cme_df.columns:
        st.info("No positional CME data available yet.")
        return

    recent = recent_window(cme_df, 7)
    located = recent.dropna(subset=["latitude", "longitude"]).copy()

    inner = st.tabs(["Region Map", "Highest Activity Region"])

    with inner[0]:
        if located.empty:
            st.info("No located CMEs in the last 7 days.")
        else:
            fig = px.scatter(
                located,
                x="longitude",
                y="latitude",
                color="speed",
                color_continuous_scale="Inferno",
                size="half_angle" if "half_angle" in located.columns else None,
                hover_data=["timestamp_utc", "active_region"],
                title="CME Source Positions (Last 7 Days)",
            )
            fig.add_vrect(
                x0=-EARTH_DIRECTED_LONGITUDE_DEG,
                x1=EARTH_DIRECTED_LONGITUDE_DEG,
                fillcolor="#39ff6a",
                opacity=0.28,
                line_width=2,
                line_color="#39ff6a",
                line_dash="dash",
                layer="above",
                annotation_text="Earth-directed zone",
                annotation_position="top",
                annotation_font_color="#39ff6a",
                annotation_font_size=13,
            )
            fig.update_xaxes(title="Source Longitude (°)", range=[-180, 180])
            fig.update_yaxes(title="Source Latitude (°)", range=[-90, 90])
            add_sun_disk_background(fig)
            fig.update_layout(height=420)
            plot_retro(fig)
            st.caption("Sun image covers the Earth-facing disk (±90° longitude); points beyond it are far-side CMEs.")

    with inner[1]:
        if "active_region" in recent.columns:
            region_data = recent.dropna(subset=["active_region"]).copy()
            if not region_data.empty:
                region_data["active_region"] = region_data["active_region"].astype(str)
                region_counts = region_data["active_region"].value_counts().reset_index()
                region_counts.columns = ["active_region", "cme_count"]
                region_counts = region_counts.head(10)
                fig = px.bar(
                    region_counts,
                    x="active_region",
                    y="cme_count",
                    title="Most Active CME-Producing Regions (Last 7 Days)",
                )
                fig.update_layout(height=360)
                plot_retro(fig)
            else:
                st.info("No active region data available for CMEs.")
        else:
            st.info("No active region column available.")


def heliomap_panel() -> None:
    st.markdown("#### Heliomap")
    st.caption("Where on the Sun recent activity originated.")

    tabs = st.tabs(["Solar Events", "CME"])
    with tabs[0]:
        heliomap_solar_events_tab()
    with tabs[1]:
        heliomap_cme_tab()


def cme_predictions() -> None:
    st.subheader("CME — Predictions")

    tabs = st.tabs(["Predict Arrival Time", "Estimated Travel Time", "Estimated Storm Risk"])
    for tab in tabs:
        with tab:
            st.info("Prediction module will be added later.")


def f107_classification(value) -> str:
    if value is None or pd.isna(value):
        return "No data"
    value = float(value)
    if value < 100:
        return "Low"
    if value < 150:
        return "Moderate"
    if value < 200:
        return "High"
    return "Very High"


F107_BANDS = [
    ("Low", 0, 100),
    ("Moderate", 100, 150),
    ("High", 150, 200),
    ("Very High", 200, None),
]


def f107_activity_classification(value) -> None:
    current_label = f107_classification(value)

    cells = ""
    for label, low, high in F107_BANDS:
        is_current = label == current_label
        range_text = f"{low}+" if high is None else f"{low}-{high}"
        background = "#1a3d1a" if is_current else "#1a1a1a"
        border = "2px solid #00ff88" if is_current else "1px solid #333333"
        marker = "● " if is_current else ""
        cells += (
            f'<div style="flex:1; padding:8px; text-align:center; background:{background}; '
            f'border:{border}; color:#f2f2f2; font-family:\'Courier New\', monospace; font-size:0.8rem;">'
            f"{marker}{escape(label)}<br><span style='color:#9adfff;'>{escape(range_text)}</span></div>"
        )

    st.markdown(
        f"""
        <div style="display:flex; gap:6px; margin-bottom:14px;">{cells}</div>
        """,
        unsafe_allow_html=True,
    )


def f107_analysis() -> None:
    st.subheader("F10.7 — Current Analysis")

    f107_df = load_processed_data("f107")
    if f107_df.empty:
        st.info("No F10.7 data available yet. Run the live updater to populate this.")
        return

    monthly = recent_window(f107_df, 30)
    week = recent_window(f107_df, 7)

    latest_flux, latest_flux_time = latest_non_null(f107_df, "f107_flux")
    highest_row = row_at_extreme(week, "f107_flux", "max") if "f107_flux" in week.columns and not week.empty else None
    lowest_row = row_at_extreme(week, "f107_flux", "min") if "f107_flux" in week.columns and not week.empty else None
    avg_flux = week["f107_flux"].dropna().mean() if "f107_flux" in week.columns and not week.empty else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Today's Flux", format_value(latest_flux, "", 1), latest_label_time(latest_flux_time))
    with c2:
        metric_card(
            "Highest Flux (7d)",
            format_value(None if highest_row is None else highest_row["f107_flux"], "", 1),
            time_caption(highest_row),
        )
    with c3:
        metric_card(
            "Lowest Flux (7d)",
            format_value(None if lowest_row is None else lowest_row["f107_flux"], "", 1),
            time_caption(lowest_row),
        )
    with c4:
        metric_card("Average Flux (7d)", format_value(avg_flux, "", 1))

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.markdown("### Activity Classification")
    f107_activity_classification(latest_flux)

    if "f107_flux" in monthly.columns and not monthly.empty:
        plot_df = monthly[["timestamp_utc", "f107_flux"]].dropna().sort_values("timestamp_utc").copy()
        plot_df["7-day average"] = plot_df["f107_flux"].rolling(7, min_periods=1).mean()

        f107_charts = []

        fig = px.line(plot_df, x="timestamp_utc", y="f107_flux", title="F10.7 Flux vs Time (Last 30 Days)")
        fig.update_layout(height=340)
        f107_charts.append(fig)

        fig_avg = px.line(
            plot_df,
            x="timestamp_utc",
            y="7-day average",
            title="F10.7 7-Day Moving Average (Trend)",
        )
        fig_avg.update_traces(line_color="#00ff88")
        fig_avg.update_layout(height=340)
        f107_charts.append(fig_avg)

        fig_hist = px.histogram(plot_df, x="f107_flux", nbins=20, title="F10.7 Flux Distribution (Last 30 Days)")
        fig_hist.update_layout(height=340)
        f107_charts.append(fig_hist)

        for row_start in range(0, len(f107_charts), 2):
            chart_cols = st.columns(2)
            for col, chart_fig in zip(chart_cols, f107_charts[row_start : row_start + 2]):
                with col:
                    plot_retro(chart_fig)


def f107_predictions() -> None:
    st.subheader("F10.7 — Predictions")

    tabs = st.tabs(["Tomorrow's Flux", "Next Week Trend"])
    for tab in tabs:
        with tab:
            st.info("Prediction module will be added later.")


def photosphere_page(df: pd.DataFrame) -> None:
    st.title("Photosphere")
    tabs = st.tabs(["Solar Events", "CME", "F10.7"])

    with tabs[0]:
        inner = st.tabs(["Current Analysis", "Predictions"])
        with inner[0]:
            solar_events_analysis()
        with inner[1]:
            st.info("Prediction module will be added later.")

    with tabs[1]:
        inner = st.tabs(["Current Analysis", "Predictions"])
        with inner[0]:
            cme_analysis()
        with inner[1]:
            cme_predictions()

    with tabs[2]:
        inner = st.tabs(["Current Analysis", "Predictions"])
        with inner[0]:
            f107_analysis()
        with inner[1]:
            f107_predictions()


def render_quicklook_verification_tab() -> None:
    """Immediate, approximate visual comparison against Kyoto WDC's
    continuously-updating real-time (quicklook) AE graph — NOT the
    official verification system. That remains the separate, delayed
    workflow driven by Kyoto's official digital AE data (Production
    Prediction tab), which stays the sole authoritative source; this tab
    exists only so a user doesn't have to wait the ~10-20 days that takes.
    """
    st.warning(
        "Quicklook values are estimated from the Kyoto real-time graph and may differ from the "
        "official digital AE values published later. This is **NOT** the official verification "
        "system — the official Kyoto digital AE data (Production Prediction tab) remains the "
        "authoritative source."
    )

    poll_jobs("ae")
    jobs = get_running_jobs("ae") + get_saved_jobs("ae")
    completed_jobs = [j for j in jobs if j["status"] == "completed" and j["ticks"]]

    if not completed_jobs:
        st.info("No completed AE predictions yet. Start one from the Production Prediction tab.")
        return

    completed_jobs.sort(key=lambda j: j["created_at"], reverse=True)
    labels = [
        f"Target {pd.Timestamp(j['target_hour']).strftime('%Y-%m-%d %H:%M UTC')} "
        f"(started {pd.Timestamp(j['created_at']).strftime('%H:%M UTC')})"
        for j in completed_jobs
    ]
    selected = st.selectbox("Prediction to compare", labels, key="quicklook_job_select")
    job = completed_jobs[labels.index(selected)]

    final_pred = job["ticks"][-1]["predicted_value"]
    quicklook_ae = job.get("quicklook_ae")
    approx_error = quicklook_error(job)
    relative_error = quicklook_relative_error(job)
    error_label = classify_quicklook_error(approx_error)
    confidence = job.get("quicklook_confidence")
    range_low = job.get("quicklook_range_low")
    range_high = job.get("quicklook_range_high")
    hour_coverage = job.get("quicklook_hour_coverage")
    estimate_label = quicklook_label(job)

    range_caption = "Estimated from Kyoto Quicklook graph. Not Official."
    if range_low is not None and range_high is not None:
        range_caption = f"Range {range_low:.0f}–{range_high:.0f} nT · " + range_caption

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Prediction Time", pd.Timestamp(job["created_at"]).strftime("%H:%M:%S UTC"), "")
    with c2:
        metric_card("Forecast Target", pd.Timestamp(job["target_hour"]).strftime("%Y-%m-%d %H:%M UTC"), "")
    with c3:
        metric_card("Predicted AE", f"{final_pred:.2f} nT", "")
    with c4:
        metric_card(
            estimate_label,
            "N/A" if quicklook_ae is None else f"{quicklook_ae:.2f} nT",
            range_caption,
        )
    with c5:
        metric_card(
            "Absolute Error",
            "N/A" if approx_error is None else f"{approx_error:.2f} nT",
            "Predicted vs. Quicklook estimate — not the official error",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        metric_card(
            "Graph Coverage",
            "N/A" if hour_coverage is None else f"{hour_coverage * 100:.0f}%",
            "% of the forecast hour with a curve drawn in Kyoto's graph",
            value_color=QUICKLOOK_CONFIDENCE_COLORS.get(confidence, "#000000"),
        )
    with d2:
        metric_card(
            "Quicklook Confidence",
            "N/A" if confidence is None else confidence.title(),
            "<40% Low · 40–80% Moderate · >80% High coverage",
            value_color=QUICKLOOK_CONFIDENCE_COLORS.get(confidence, "#000000"),
        )
    with d3:
        metric_card(
            "Estimated Range",
            "N/A" if range_low is None or range_high is None else f"{range_low:.0f}–{range_high:.0f} nT",
            "Uncertainty band from graph extraction",
        )
    with d4:
        metric_card(
            "Error Classification",
            error_label or "N/A",
            "Predicted vs. Quicklook estimate, coarse band",
        )
    with d5:
        metric_card(
            "Relative Error",
            "N/A" if relative_error is None else f"{relative_error:.1f}%",
            "Absolute error as a % of the Quicklook estimate",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    if estimate_label == "Partial Quicklook Estimate":
        coverage_pct = 0 if hour_coverage is None else round(hour_coverage * 100)
        st.warning(
            f"⚠️ Only the first {coverage_pct}% of the forecast hour is currently available in the "
            "Kyoto Quicklook graph. This estimate should not be interpreted as the final hourly AE. "
            "It will keep updating automatically (and can be refreshed manually) as Kyoto draws more "
            "of the hour."
        )
    elif estimate_label == "Complete Quicklook Estimate":
        st.success(
            "✅ The forecast hour is now fully drawn in Kyoto's Quicklook graph — this estimate "
            "reflects the complete hour, though it remains an approximate visual read, not the "
            "official Kyoto digital AE value."
        )

    status_col, refresh_col = st.columns([0.7, 0.3])
    with status_col:
        badge_key = "quicklook_verified" if quicklook_ae is not None else "quicklook_pending"
        st.markdown(status_badge_html(badge_key), unsafe_allow_html=True)
    with refresh_col:
        if st.button("🔄 Refresh Quicklook Estimate", key=f"refresh_quicklook_{job['job_id']}"):
            refresh_quicklook_estimate(job["job_id"])
            st.toast("Quicklook estimate refreshed.")
            st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Graph Metadata")
    checked_at = job.get("quicklook_checked_at")
    graph_created = job.get("quicklook_graph_created")
    age_source_ts = graph_created or checked_at
    if age_source_ts is not None:
        age_minutes = int((pd.Timestamp.now(tz="UTC") - pd.Timestamp(age_source_ts)).total_seconds() / 60)
        image_age = f"{age_minutes} min"
        if graph_created is None:
            image_age += " (since last check)"
    else:
        image_age = "N/A"

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        metric_card(
            "Graph Created",
            "N/A" if graph_created is None else pd.Timestamp(graph_created).strftime("%H:%M UTC"),
            "When Kyoto last regenerated this day's graph image",
        )
    with g2:
        metric_card(
            "Quicklook Timestamp",
            "N/A" if checked_at is None else pd.Timestamp(checked_at).strftime("%H:%M:%S UTC"),
            "When this estimate was last read from the graph",
        )
    with g3:
        metric_card("Forecast Target", pd.Timestamp(job["target_hour"]).strftime("%H:%M UTC"), "")
    with g4:
        metric_card("Image Age", image_age, "Time since the graph image was generated")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Kyoto Quicklook Graph")
    day = pd.Timestamp(job["target_hour"]).normalize()
    try:
        image = fetch_quicklook_image(day)
        annotated = annotate_quicklook_image(image, job["target_hour"], final_pred, quicklook_ae)
        st.image(annotated)
        st.caption(
            f"Source: {job.get('quicklook_image_url') or quicklook_image_url(day)} — blue dashed line: "
            "**Target Time**. Blue horizontal line: **Prediction**. Red circle: **Estimated AE** read "
            "from the curve at that time. **Estimated from Kyoto Quicklook graph. Not Official.**"
        )
    except Exception as exc:
        st.warning(f"Could not load the Kyoto Quicklook graph right now: {exc}")

    with st.expander("How was the Quicklook Estimate obtained?"):
        st.markdown(
            "```\n"
            "Kyoto Quicklook image\n"
            "        ↓\n"
            "  Image Processing\n"
            "        ↓\n"
            "   Axis Detection\n"
            "        ↓\n"
            "   Curve Detection\n"
            "        ↓\n"
            "   Interpolation\n"
            "        ↓\n"
            "   Estimated AE\n"
            "```"
        )
        st.markdown(
            "Kyoto WDC's real-time graph is fetched as an image, and the AE/AO curve's pixel height is "
            "located within the target hour's column range using fixed axis calibration, then converted "
            "back to nT. **This is an approximation of a chart reading, not an official Kyoto value** — "
            "it exists only to give an immediate, rough sense of comparison. Confidence and Estimated "
            "Range above reflect how much of the hour had a clearly detectable curve and how consistent "
            "the detected height was, not a statistical error bound."
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.info(
        "This comparison uses the Kyoto Quicklook graph, an approximate read of Kyoto's real-time "
        "image. The official digital AE values are published later by Kyoto WDC (typically ~10-20 "
        "days). When available, this prediction will automatically receive an official verification "
        "below, independent of the Quicklook estimate above."
    )
    st.markdown("##### Official Verification (separate, delayed workflow)")
    verification_status = job.get("verification_status")
    actual = job["actual_value"]
    o1, o2, o3 = st.columns(3)
    with o1:
        metric_card(
            "Official Verification Status",
            "Verified" if verification_status == "verified" else "Pending Official Kyoto Data",
            "Kyoto WDC digital AE — the authoritative source",
        )
    with o2:
        metric_card("Official AE", "Pending" if actual is None else f"{actual:.2f} nT", "")
    with o3:
        official_error = None if actual is None else abs(final_pred - actual)
        metric_card("Official Error", "N/A" if official_error is None else f"{official_error:.2f} nT", "")


def analytics_page(df: pd.DataFrame) -> None:
    st.title("Analytics")
    st.subheader("Combined Earth Analysis")
    inner = st.tabs(["Current Analysis", "Prediction", "AE Predictions", "Research & Experiments"])
    with inner[0]:
        earth_analysis(df)
    with inner[1]:
        prediction_panel("analytics", ANALYTICS_VARIABLES)
    with inner[2]:
        ae_inner = st.tabs(["Production Prediction", "Quicklook Verification", "Research Laboratory"])
        with ae_inner[0]:
            prediction_panel("ae", AE_VARIABLES)
        with ae_inner[1]:
            render_quicklook_verification_tab()
        with ae_inner[2]:
            render_ae_research_laboratory()
    with inner[3]:
        research_inner = st.tabs(["Experimental Predictions", "Kp Research Laboratory"])
        with research_inner[0]:
            st.markdown(_experimental_badge_html(), unsafe_allow_html=True)
            st.caption(
                "Research feature: cascades Predicted AE (from the frozen AE model) into the Kp/Dst "
                "models as an extra feature, instead of the production pipeline's observed AE. "
                "Completely separate models and training data from the Prediction tab — for "
                "comparison purposes only, not a replacement for it."
            )
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            prediction_panel("experimental", EXPERIMENTAL_VARIABLES)
        with research_inner[1]:
            render_kp_research_laboratory()


def render_independent_models_tab() -> None:
    """Reproduces the production architecture: each variable predicted
    independently from observed history, never from another model's
    prediction. Reuses the exact same panels as the Analytics page — same
    jobs, same models, same data — just surfaced here for architecture
    comparison purposes.
    """
    st.markdown(
        "**Architecture:** Live NOAA &rarr; Feature Engineering &rarr; AE Model &rarr; Kp Model &rarr; "
        "Dst Model. None of the models use another model's prediction — only observed historical "
        "values are used.",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### AE (independent)")
    prediction_panel("ae", AE_VARIABLES)
    st.divider()
    st.markdown("##### Kp / Dst (independent — combined Sun-Earth model, observed AE)")
    prediction_panel("analytics", ANALYTICS_VARIABLES)


def render_physics_cascaded_tab() -> None:
    """The experimental cascade: Predicted AE (never observed AE) feeds
    forward into the experimental Kp/Dst models as an extra feature.
    Reuses the same "experimental" dataset already built for the
    Analytics page's Experimental Predictions tab.
    """
    st.markdown(_experimental_badge_html(), unsafe_allow_html=True)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "**Architecture:** Live NOAA &rarr; Solar Wind &rarr; IMF &rarr; Derived Physics &rarr; AE "
        "Prediction Model &rarr; Predicted AE &rarr; Experimental Kp Model &rarr; Experimental Dst "
        "Model. Predicted AE is the only intermediate predicted variable — observed AE is never used, "
        "and predicted Kp is never fed into the Dst model.",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    prediction_panel("experimental", EXPERIMENTAL_VARIABLES)


def render_model_comparison_tab() -> None:
    """Aggregated, historical comparison across every completed &
    evaluated job for each architecture — not a single-forecast
    comparison (that's already on each dataset's own Prediction tab), but
    the system-level question: which architecture performs better overall.
    """
    st.caption(
        "Aggregated across every completed forecast for each architecture — a system-level "
        "comparison, not a single prediction. AE has one shared model (Predicted AE feeds both "
        "architectures identically), so it has no Production/Experimental split of its own."
    )

    ae_stats = get_prediction_statistics("ae")
    prod_stats = get_prediction_statistics("analytics")
    exp_stats = get_prediction_statistics("experimental")

    st.markdown("##### AE (shared model)")
    a1, a2, a3 = st.columns(3)
    with a1:
        metric_card("Verified Forecasts", str(ae_stats["count"]), "")
    with a2:
        rate = ae_stats["success_rate"]
        metric_card("Success Rate", "N/A" if rate is None else f"{rate:.0f}%", "Within 1.5x model's typical error")
    with a3:
        metric_card("Best-Performing Model", ae_stats.get("best_model") or "N/A", "")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Kp & Dst — Production vs. Experimental")

    for variable in ["kp", "dst"]:
        label = VARIABLE_LABELS[variable]
        prod_mae = prod_stats["mae_by_variable"].get(variable)
        exp_mae = exp_stats["mae_by_variable"].get(variable)

        st.markdown(f"**{label}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(
                "Production MAE",
                "N/A" if prod_mae is None else f"{prod_mae:.3f}",
                f"n={prod_stats['count']} verified forecasts (all variables)" if prod_stats["count"] else "",
            )
        with c2:
            metric_card("Experimental MAE", "N/A" if exp_mae is None else f"{exp_mae:.3f}", "")
        with c3:
            if prod_mae is not None and exp_mae is not None and prod_mae:
                diff_pct = (prod_mae - exp_mae) / prod_mae * 100
                metric_card(
                    "Prediction Difference",
                    f"{diff_pct:+.1f}%",
                    "Positive = experimental has lower MAE",
                )
            else:
                metric_card("Prediction Difference", "N/A", "Need completed & verified jobs from both")
        with c4:
            if prod_mae is not None and exp_mae is not None:
                if abs(prod_mae - exp_mae) < 1e-9:
                    verdict, color = "Tie", "#404040"
                elif exp_mae < prod_mae:
                    verdict, color = "Experimental Better", "#1f7a3a"
                else:
                    verdict, color = "Production Better", "#7a1f1f"
                metric_card("Verdict", verdict, "Lower MAE wins", value_color=color)
            else:
                metric_card("Verdict", "Not enough data", "")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("###### Production — Forecast Error Trend")
        prediction_statistics_panel("analytics")
    with comp_col2:
        st.markdown("###### Experimental — Forecast Error Trend")
        prediction_statistics_panel("experimental")


def _pipeline_node_html(label: str, active: bool, sublabel: str = "") -> str:
    bg = "#1f5a2e" if active else "#2a2a32"
    border = "#4ade80" if active else "#808080"
    text_color = "#d4f4dd" if active else "#e8e8e8"
    sub_html = (
        f"<div style='font-size:0.66rem;color:#b8b8c0;margin-top:2px;'>{escape(sublabel)}</div>"
        if sublabel
        else ""
    )
    return (
        f"<div style='display:inline-block;background:{bg};border:2px solid {border};"
        f"border-radius:4px;padding:10px 16px;text-align:center;color:{text_color};"
        f"font-family:&quot;Courier New&quot;,monospace;font-weight:700;min-width:110px;'>"
        f"{escape(label)}{sub_html}</div>"
    )


def _pipeline_arrow_html(vertical: bool = True) -> str:
    symbol = "&darr;" if vertical else "&rarr;"
    return f"<span style='font-size:1.3rem;color:#808080;padding:0 8px;'>{symbol}</span>"


def render_prediction_pipeline_tab() -> None:
    """A visual side-by-side of the two architectures. Nodes highlight
    green while a matching job is actively running/evaluating for that
    variable+architecture — a lightweight, always-correct way to satisfy
    "highlight each node as predictions progress" without a separate
    animation/state system.
    """
    st.caption(
        "Visual comparison of the two forecasting architectures. A node highlights green while a "
        "matching prediction job is actively running."
    )

    poll_jobs("ae")
    poll_jobs("analytics")
    poll_jobs("experimental")

    def _is_running(dataset: str, variable: str) -> bool:
        return any(
            j["variable"] == variable and j["status"] in ("in_progress", "evaluating")
            for j in get_running_jobs(dataset) + get_saved_jobs(dataset)
        )

    running_ae = _is_running("ae", "ae")
    running_kp_prod = _is_running("analytics", "kp")
    running_dst_prod = _is_running("analytics", "dst")
    running_kp_exp = _is_running("experimental", "kp")
    running_dst_exp = _is_running("experimental", "dst")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###### Independent")
        top_row = (
            _pipeline_node_html("Live NOAA", True)
            + _pipeline_arrow_html(False)
            + _pipeline_node_html("Feature Engineering", True)
        )
        st.markdown(f"<div style='text-align:center'>{top_row}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;font-size:1.3rem;color:#808080;'>&darr;</div>",
            unsafe_allow_html=True,
        )
        bottom_row = (
            _pipeline_node_html("AE", running_ae)
            + "&nbsp;&nbsp;"
            + _pipeline_node_html("Kp", running_kp_prod)
            + "&nbsp;&nbsp;"
            + _pipeline_node_html("Dst", running_dst_prod)
        )
        st.markdown(f"<div style='text-align:center'>{bottom_row}</div>", unsafe_allow_html=True)
        st.caption("AE, Kp, and Dst predict independently, in parallel — none feed into another.")

    with col2:
        st.markdown("###### Physics Cascaded")
        chain = (
            _pipeline_node_html("Live NOAA", True)
            + _pipeline_arrow_html(False)
            + _pipeline_node_html("Feature Engineering", True)
            + _pipeline_arrow_html(False)
            + _pipeline_node_html("AE", running_ae, "Predicted AE")
            + _pipeline_arrow_html(False)
            + _pipeline_node_html("Kp", running_kp_exp)
            + _pipeline_arrow_html(False)
            + _pipeline_node_html("Dst", running_dst_exp)
        )
        st.markdown(f"<div style='text-align:center'>{chain}</div>", unsafe_allow_html=True)
        st.caption("Predicted AE feeds forward into both Kp and Dst as an extra feature — never observed AE, never predicted Kp into Dst.")


def render_physics_interpretation_panel(df: pd.DataFrame) -> None:
    """Rule-based (no LLM) physics narrative of current Sun-Earth
    coupling conditions — see swdss.models.physics_interpretation for the
    underlying, fully reproducible threshold logic.
    """
    st.caption(
        "Rule-based physics interpretation of current conditions — no LLM, no black box. Every "
        "statement below is a direct, reproducible function of the live Solar Wind, IMF, and "
        "geomagnetic readings using established space-weather physics (Burton et al. 1975 coupling, "
        "IMF clock angle, standard Kp/Dst storm thresholds)."
    )

    def _latest_scalar(column: str):
        value, _ = latest_value(df, column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)

    speed = _latest_scalar("solar_wind_speed")
    density = _latest_scalar("proton_density")
    temperature = _latest_scalar("temperature")
    bt = _latest_scalar("bt")
    bx = _latest_scalar("bx")
    by = _latest_scalar("by")
    bz = _latest_scalar("bz")
    kp = _latest_scalar("kp")
    dst = _latest_scalar("dst")
    _, ae_raw = latest_minute_observation("ae", "ae")
    ae = float(ae_raw) if ae_raw is not None else None

    sections = physics_interpretation(speed, density, temperature, bt, bx, by, bz, kp, dst, ae)

    st.markdown("##### Current Solar Wind State")
    st.info(sections["solar_wind_state"])
    st.markdown("##### Current IMF Orientation")
    st.info(sections["imf_orientation"])
    st.markdown("##### Expected Magnetic Coupling")
    st.info(sections["magnetic_coupling"])
    st.markdown("##### Expected Auroral Activity")
    st.info(sections["auroral_activity"])
    st.markdown("##### Expected Ring Current Response")
    st.info(sections["ring_current"])
    st.markdown("##### Expected Geomagnetic Activity")
    st.info(sections["geomagnetic_activity"])


CONCLUSION_COLORS = {"Supported": "#1f7a3a", "Not Supported": "#7a1f1f", "Inconclusive": "#7a5a1f"}


# ==================== IMF Research Laboratory (Bz experimentation) ====================
# Fully isolated from the Production Prediction tab — see
# swdss.models.imf_research module docstring for the production-safety
# contract. Nothing below ever writes to the production model files.


def _imf_research_notes() -> None:
    st.info(
        "**Why Bz is hard to forecast.** Bz is fundamentally different from Solar Wind Speed or "
        "Density: it's the *orientation* of the interplanetary magnetic field, not a scalar plasma "
        "quantity. Its evolution depends on evolving magnetic structures — CME flux-rope rotation, "
        "turbulence, current-sheet crossings — not just how fast or dense the plasma is. Sequence "
        "models (LSTM/GRU) may capture this temporal magnetic evolution better than single-row "
        "regression models, since they see how the field has been *rotating* over time, not just its "
        "instantaneous value."
    )


def _imf_research_hyperparam_inputs(model_type: str, key_prefix: str) -> dict:
    schema = HYPERPARAM_SCHEMA.get(model_type, {})
    values = {}
    if not schema:
        st.caption("No tunable hyperparameters for this model.")
        return values
    cols = st.columns(min(len(schema), 4))
    for i, (name, spec) in enumerate(schema.items()):
        with cols[i % len(cols)]:
            label = name.replace("_", " ").title()
            if spec["type"] == "int":
                values[name] = st.number_input(
                    label,
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=spec["default"],
                    step=1,
                    key=f"{key_prefix}_{model_type}_{name}",
                )
            else:
                values[name] = st.number_input(
                    label,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    step=0.01,
                    key=f"{key_prefix}_{model_type}_{name}",
                )
    return values


def _imf_research_model_selector(key_prefix: str) -> str:
    options = ALL_TRAINABLE_MODELS + FUTURE_MODELS

    def _fmt(name):
        return f"{name} (coming soon)" if name in FUTURE_MODELS else name

    choice = st.selectbox("Model Architecture", options, format_func=_fmt, key=f"{key_prefix}_model_select")
    if choice in FUTURE_MODELS:
        st.warning(f"{choice} is a registered placeholder for future work — not trainable yet.")
    return choice


def _imf_research_granularity_horizon(key_prefix: str) -> tuple:
    """Shared Forecast Granularity + Forecast Horizon control pair — the
    axis that was missing before this redesign (see imf_research.py
    module docstring): "Minute" targets shift(-horizon) minutes on live
    minute data; "Hourly" targets shift(-horizon) hours on the SAME
    historical CSVs production trains on, via the identical
    swdss.models.features functions. Returns (granularity, horizon).
    """
    granularity = st.radio(
        "Forecast Granularity", GRANULARITY_OPTIONS, horizontal=True, key=f"{key_prefix}_granularity"
    )
    horizon_options = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    unit = "min" if granularity == "Minute" else "hour"
    horizon = st.radio(
        "Forecast Horizon",
        horizon_options,
        format_func=lambda h: f"{h} {unit}" + ("s" if h != 1 else ""),
        horizontal=True,
        key=f"{key_prefix}_horizon",
    )
    if granularity == "Hourly":
        st.caption("Sourced from the same 3-year historical CSVs production trains on — genuinely comparable R².")
    else:
        st.caption("Sourced from the live ~7-day minute-level buffer — includes the new physics features.")
    return granularity, horizon


def _imf_research_run_row(run: dict, best_run_id: str = None, key_prefix: str = "runs") -> None:
    """key_prefix disambiguates widget keys when the same run is rendered
    from more than one tab in the same script run (e.g. a trained LSTM/GRU
    run appears in both Training Runs' full log and Sequence Models' own
    list) — Streamlit renders every tab's content on every rerun (inactive
    tabs are just CSS-hidden), so two calls with the same run_id would
    otherwise register duplicate widget keys and crash.
    """
    m = run["metrics"]
    is_best = run["run_id"] == best_run_id
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns([0.24, 0.13, 0.13, 0.13, 0.13, 0.24])
        with c1:
            star = "⭐ " if is_best else ""
            promoted_tag = " 🚀" if run.get("promoted") else ""
            st.markdown(f"**{star}{run['model_type']}**{promoted_tag}")
            granularity = run.get("granularity", "Minute")
            horizon_label = run.get("horizon_label") or (f"{run.get('horizon', 1)}m" if granularity == "Minute" else f"{run.get('horizon', 1)}h")
            seq_note = f" · seq={run['sequence_length']}{'m' if granularity == 'Minute' else 'h'}" if run.get("sequence_length") else ""
            st.caption(
                f"{run['target']} · {granularity} · +{horizon_label}{seq_note} · "
                f"{pd.Timestamp(run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        with c2:
            metric_card("R²", f"{m['r2']:.4f}", "")
        with c3:
            metric_card("MAE", f"{m['mae']:.3f}", "")
        with c4:
            metric_card("RMSE", f"{m['rmse']:.3f}", "")
        with c5:
            metric_card("Bias", f"{m['bias']:+.3f}", "")
        with c6:
            st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button(
                    "Promoted" if run.get("promoted") else "Promote",
                    key=f"{key_prefix}_promote_{run['run_id']}",
                    disabled=run.get("promoted", False),
                    use_container_width=True,
                ):
                    promote_run(run["run_id"])
                    st.toast("Marked as promoted (label only — production untouched).")
                    st.rerun()
            with b2:
                if st.button("Delete", key=f"{key_prefix}_delete_{run['run_id']}", use_container_width=True):
                    delete_run(run["run_id"])
                    st.toast("Run deleted.")
                    st.rerun()


def render_imf_training_runs_tab() -> None:
    st.caption(
        "Train a new experimental model — any target, any granularity/horizon, any architecture "
        "including LSTM/GRU. Every run is stored separately from the production predictor and NEVER "
        "overwrites it — Promote only labels a run for your own tracking; wiring a model into "
        "production is always a manual, deliberate step."
    )
    target_label = st.selectbox("Target Variable", list(TARGET_OPTIONS), index=0, key="imf_research_target")
    granularity, horizon = _imf_research_granularity_horizon("train_runs")
    model_type = _imf_research_model_selector("train_runs")

    sequence_length = None
    if model_type in SEQUENCE_MODELS:
        unit = "minutes" if granularity == "Minute" else "hours"
        sequence_length = st.selectbox(
            f"Sequence Length ({unit}, look-back window — independent of the forecast horizon above)",
            SEQUENCE_LENGTH_OPTIONS,
            index=SEQUENCE_LENGTH_OPTIONS.index(DEFAULT_SEQUENCE_LENGTH),
            key="train_runs_seqlen",
        )

    st.markdown("**Hyperparameters**")
    hyperparams = _imf_research_hyperparam_inputs(model_type, "train_runs") if model_type in ALL_TRAINABLE_MODELS else {}

    if st.button(
        "🧪 Train Model", key="train_runs_train_btn", type="primary", disabled=model_type not in ALL_TRAINABLE_MODELS
    ):
        with st.spinner(f"Training {model_type} on {target_label} ({granularity}, +{horizon})..."):
            try:
                run = train_research_model(
                    target_label,
                    model_type,
                    granularity=granularity,
                    horizon=horizon,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Training Run Log")
    runs = list_runs()
    if not runs:
        st.info("No training runs yet. Train a model above.")
        return

    filt1, filt2 = st.columns(2)
    with filt1:
        target_filter = st.selectbox("Filter by target", ["All"] + list(TARGET_OPTIONS), key="train_runs_filter")
    with filt2:
        granularity_filter = st.selectbox(
            "Filter by granularity", ["All"] + GRANULARITY_OPTIONS, key="train_runs_granularity_filter"
        )
    filtered = runs
    if target_filter != "All":
        filtered = [r for r in filtered if r["target"] == target_filter]
    if granularity_filter != "All":
        filtered = [r for r in filtered if r.get("granularity", "Minute") == granularity_filter]
    best_id = max(filtered, key=lambda r: r["metrics"]["r2"])["run_id"] if filtered else None
    for run in filtered:
        _imf_research_run_row(run, best_run_id=best_id, key_prefix="training_runs")


def render_imf_model_comparison_tab() -> None:
    st.caption(
        "Compare trained models for a single target side by side — R²/MAE/RMSE/MAPE/Bias, best "
        "model highlighted."
    )
    target_label = st.selectbox("Target", list(TARGET_OPTIONS), key="imf_compare_target")
    granularity_filter = st.selectbox("Granularity", ["All"] + GRANULARITY_OPTIONS, key="imf_compare_granularity")
    runs = list_runs(target_label)
    if granularity_filter != "All":
        runs = [r for r in runs if r.get("granularity", "Minute") == granularity_filter]
    if not runs:
        st.info(f"No trained models for {target_label} yet — train one in Training Runs.")
        return

    def _run_label(r):
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        return f"{r['model_type']} · {gran} +{hz} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    default_n = min(4, len(run_labels))
    selected = st.multiselect("Models to compare", run_labels, default=run_labels[:default_n], key="imf_compare_select")
    chosen = [runs[run_labels.index(s)] for s in selected]
    if not chosen:
        st.info("Select at least one model above.")
        return

    best_id = max(chosen, key=lambda r: r["metrics"]["r2"])["run_id"]

    rows = []
    for r in chosen:
        m = r["metrics"]
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        rows.append(
            {
                "Model": r["model_type"] + (" ⭐" if r["run_id"] == best_id else ""),
                "Granularity": gran,
                "Horizon": hz,
                "R²": round(m["r2"], 4),
                "MAE": round(m["mae"], 4),
                "RMSE": round(m["rmse"], 4),
                "MAPE (%)": round(m["mape"], 2) if m["mape"] is not None else None,
                "Bias": round(m["bias"], 4),
                "Trained": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Predicted vs. Actual / Residuals")
    detail_label = st.selectbox("Inspect model", selected, key="imf_compare_detail")
    detail_run = chosen[selected.index(detail_label)]
    sample = detail_run["prediction_sample"]
    y_true = sample["y_true"]
    y_pred = sample["y_pred"]
    residuals = [p - t for p, t in zip(y_pred, y_true)]

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=4), name="Predicted vs Actual"))
        lo, hi = min(y_true + y_pred), max(y_true + y_pred)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
        fig.update_layout(
            title="Predicted vs. Actual (held-out test sample)", height=340, xaxis_title="Actual", yaxis_title="Predicted"
        )
        plot_retro(fig)
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=4), name="Residual"))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Residual Plot", height=340, yaxis_title="Predicted − Actual")
        plot_retro(fig2)

    if detail_run.get("feature_importance"):
        st.markdown("##### Feature Importance")
        fi_df = pd.DataFrame(detail_run["feature_importance"], columns=["Feature", "Importance"])
        fig3 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig3.update_layout(title="Top Contributing Features", height=420, yaxis=dict(autorange="reversed"))
        plot_retro(fig3)
    else:
        st.caption("Feature importance not available for this model type (e.g. LSTM/GRU).")

    if detail_run.get("loss_history"):
        st.markdown("##### Training / Validation Loss")
        lh = detail_run["loss_history"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig4.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig4.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig4)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _cached_imf_research_frame(granularity: str = DEFAULT_GRANULARITY):
    """Same redundant-read problem as the Kp lab's Physics Experiments tab
    (see _cached_kp_research_frame) — Feature Engineering and Physics
    Experiments both call load_research_frame() unconditionally on every
    script rerun (every 15s via auto_refresh(), and once per Kp/IMF
    Research Lab sub-tab per rerun regardless of which is visually
    active). Unlike the Kp lab's static historical CSV, "Minute"
    granularity genuinely reads the LIVE ~7-day minute buffer, which does
    update in near-real-time — so this keeps the same REFRESH_SECONDS TTL
    the rest of the dashboard's live data uses (see load_master_data)
    rather than caching indefinitely.
    """
    return load_research_frame(granularity)


def render_imf_feature_engineering_tab() -> None:
    st.caption(
        "The full minute-level feature set this research pipeline trains on — baseline IMF/Solar "
        "Wind variables plus the new physics-informed features below."
    )
    try:
        frame, feature_columns = _cached_imf_research_frame()
    except Exception as exc:
        st.warning(f"Could not load the research feature frame: {exc}")
        return

    latest = frame.iloc[-1]
    st.markdown("##### New Physics Features — Current Live Values")
    physics_cols_display = [
        ("southward_duration_min", "Southward Duration", "min", "Consecutive minutes Bz < 0"),
        ("strong_southward_duration_min", "Strong Southward Duration", "min", "Consecutive minutes Bz < -5 nT"),
        (
            "integrated_southward_bz_60m",
            "Integrated Southward Bz (60m)",
            "nT·min",
            "Rolling cumulative |Bz| (southward only)",
        ),
        ("dbz_gsm", "ΔBz (Magnetic Rotation)", "nT/min", "Minute-to-minute change in Bz"),
        ("clock_angle_deg", "IMF Clock Angle", "°", "atan2(By, Bz) — 180° = purely southward"),
        ("clock_angle_rate_deg", "Clock Angle Rate", "°/min", "How fast the field orientation is rotating"),
        ("bt_persistence_60m_std", "Bt Persistence (60m std)", "nT", "Field-strength volatility over the last hour"),
    ]
    cols = st.columns(4)
    for i, (col, label, unit, desc) in enumerate(physics_cols_display):
        with cols[i % 4]:
            val = latest.get(col)
            metric_card(label, "N/A" if pd.isna(val) else f"{val:.2f} {unit}", desc)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Full Feature List Used for Training")
    st.caption(f"{len(feature_columns)} total input features per model.")
    with st.expander("Show all feature names"):
        st.code("\n".join(feature_columns), language="text")


def render_imf_sequence_models_tab() -> None:
    """Review-only: LSTM/GRU are trained from the unified form in
    Training Runs (same target/granularity/horizon/model selector as
    every other architecture) — this tab exists to inspect what those
    runs actually did, especially the training/validation loss curves
    that only sequence models produce, without duplicating the trainer.
    """
    st.caption(
        "LSTM/GRU never see a single row — each prediction looks back over a window of consecutive "
        "minutes, letting the model learn temporal patterns in how the field evolves, not just its "
        "instantaneous state. Train new ones from the Training Runs tab (same unified form as every "
        "other model); this tab is for reviewing what they learned."
    )
    if not SEQUENCE_MODELS:
        st.warning("TensorFlow/Keras is not installed — sequence models are unavailable in this environment.")
        return

    seq_runs = [r for r in list_runs() if r["model_type"] in SEQUENCE_MODELS]
    if not seq_runs:
        st.info("No LSTM/GRU runs yet — train one from the Training Runs tab.")
        return

    def _run_label(r):
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        return f"{r['model_type']} · {r['target']} · {gran} +{hz} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in seq_runs]
    selected = st.selectbox("Inspect run", labels, key="imf_seq_inspect")
    run = seq_runs[labels.index(selected)]

    m = run["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("R²", f"{m['r2']:.4f}", "")
    with c2:
        metric_card("MAE", f"{m['mae']:.3f}", "")
    with c3:
        metric_card("RMSE", f"{m['rmse']:.3f}", "")
    with c4:
        metric_card("MAPE", "N/A" if m["mape"] is None else f"{m['mape']:.1f}%", "")
    with c5:
        metric_card("Bias", f"{m['bias']:+.3f}", "")

    if run.get("loss_history"):
        lh = run["loss_history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### All Sequence Model Runs")
    for r in seq_runs:
        _imf_research_run_row(r, key_prefix="sequence_models")


def render_imf_physics_experiments_tab() -> None:
    st.caption("Exploratory analysis: how do the new physics features relate to Bz's behavior historically?")
    try:
        frame, _ = _cached_imf_research_frame()
    except Exception as exc:
        st.warning(f"Could not load the research feature frame: {exc}")
        return

    feature_options = {
        "Southward Duration (min)": "southward_duration_min",
        "Integrated Southward Bz (60m)": "integrated_southward_bz_60m",
        "IMF Clock Angle (°)": "clock_angle_deg",
        "Clock Angle Rate (°/min)": "clock_angle_rate_deg",
        "Bt Persistence Std (60m)": "bt_persistence_60m_std",
    }
    label = st.selectbox("Physics feature", list(feature_options), key="imf_physics_feature_select")
    col = feature_options[label]

    recent = frame.tail(24 * 60).dropna(subset=[col, "bz_gsm"])
    if recent.empty:
        st.info("Not enough history to plot yet.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=recent.index, y=recent["bz_gsm"], name="Bz (nT)", line=dict(color="#1f5a7a")), secondary_y=False)
    fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=label, line=dict(color="#7a1f5a")), secondary_y=True)
    fig.update_layout(title=f"Bz vs. {label} — last {len(recent)} minutes", height=380)
    fig.update_yaxes(title_text="Bz (nT)", secondary_y=False)
    fig.update_yaxes(title_text=label, secondary_y=True)
    plot_retro(fig)

    next_bz = frame["bz_gsm"].shift(-1)
    valid = frame[[col]].join(next_bz.rename("next_bz")).dropna()
    if len(valid) > 10:
        corr = valid[col].corr(valid["next_bz"])
        st.caption(f"Correlation between {label} and next-minute Bz across the full history: **{corr:.3f}**")
        fig2 = go.Figure(go.Scattergl(x=valid[col], y=valid["next_bz"], mode="markers", marker=dict(size=3, opacity=0.4)))
        fig2.update_layout(title=f"{label} vs. Next-Minute Bz", height=380, xaxis_title=label, yaxis_title="Next-minute Bz (nT)")
        plot_retro(fig2)


def render_imf_hyperparameter_tuning_tab() -> None:
    st.caption(
        "Manually adjust and train with custom hyperparameters for the tabular models — not an "
        "automated search, a direct way to test a specific configuration and compare it against "
        "other runs in Model Comparison."
    )
    target_label = st.selectbox("Target Variable", list(TARGET_OPTIONS), key="imf_tune_target")
    granularity, horizon = _imf_research_granularity_horizon("tune")
    model_type = st.selectbox("Model", TABULAR_MODELS, key="imf_tune_model")
    hyperparams = _imf_research_hyperparam_inputs(model_type, "tune")

    if st.button("🧪 Train with these Hyperparameters", key="imf_tune_train_btn", type="primary"):
        with st.spinner(f"Training {model_type}..."):
            try:
                run = train_research_model(
                    target_label, model_type, granularity=granularity, horizon=horizon, hyperparams=hyperparams
                )
                st.success(f"Trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.3f}")
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def render_imf_hypothesis_testing_tab() -> None:
    st.caption(
        "Compare two training runs — e.g. a baseline architecture vs. an experimental one — and get "
        "a rule-based Supported/Not Supported/Inconclusive verdict. Adapted from the same philosophy "
        "as the main Research Lab's Hypothesis Testing, but for offline training-run metrics rather "
        "than live verified predictions."
    )
    runs = list_runs()
    if len(runs) < 2:
        st.info("Train at least two models (any target) to compare them here.")
        return

    def _run_label(r):
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        return f"{r['model_type']} — {r['target']} · {gran} +{hz} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    col1, col2 = st.columns(2)
    with col1:
        baseline_label = st.selectbox("Baseline", run_labels, index=0, key="imf_hyp_baseline")
    with col2:
        experimental_label = st.selectbox(
            "Experimental", run_labels, index=min(1, len(run_labels) - 1), key="imf_hyp_experimental"
        )

    if st.button("Compare", key="imf_hyp_compare_btn"):
        baseline_run = runs[run_labels.index(baseline_label)]
        experimental_run = runs[run_labels.index(experimental_label)]
        if baseline_run["run_id"] == experimental_run["run_id"]:
            st.warning("Choose two different runs.")
            return
        result = compare_runs(baseline_run["run_id"], experimental_run["run_id"])
        st.markdown(
            f"### Verdict: <span style='color:{CONCLUSION_COLORS.get(result['verdict'], '#404040')}'>{result['verdict']}</span>",
            unsafe_allow_html=True,
        )
        st.write(result["explanation"])
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("ΔR²", f"{result['delta_r2']:+.4f}", "")
        with c2:
            metric_card("ΔMAE", f"{result['delta_mae']:+.4f}", "")
        with c3:
            metric_card("Samples", f"{experimental_run['n_test_samples']}", "held-out test rows")


def render_imf_horizon_analysis_tab() -> None:
    """Answers the core scientific question this whole redesign exists
    for: how does forecast skill decay with lead time? Trains (or reuses)
    one run per horizon in the chosen granularity's full horizon list and
    plots Horizon vs. R²/MAE/RMSE — the same sweep verified against
    production's own bz_gsm_*h models during this redesign (1h/3h/6h/
    12h/24h Hourly R² landed within ~0.01-0.02 of production's own
    metrics.json values, and decayed in the identical shape).
    """
    st.caption(
        "Automatically trains (or reuses existing runs for) one model per horizon and plots how "
        "R²/MAE/RMSE change with forecast lead time — the central research question this lab exists "
        "to answer."
    )
    target_label = st.selectbox("Target Variable", list(TARGET_OPTIONS), key="imf_horizon_target")
    granularity = st.radio("Forecast Granularity", GRANULARITY_OPTIONS, horizontal=True, key="imf_horizon_granularity")
    horizon_list = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    st.caption(f"Horizons swept: {', '.join(str(h) for h in horizon_list)} {'minutes' if granularity == 'Minute' else 'hours'}")
    model_type = st.selectbox("Model", TABULAR_MODELS, key="imf_horizon_model")
    reuse_existing = st.checkbox(
        "Reuse existing runs where available (faster)", value=True, key="imf_horizon_reuse"
    )

    if st.button("📈 Run Horizon Sweep", key="imf_horizon_sweep_btn", type="primary"):
        with st.spinner(f"Sweeping {model_type} across {len(horizon_list)} horizons for {target_label} ({granularity})..."):
            try:
                runs = train_horizon_sweep(target_label, model_type, granularity=granularity, reuse_existing=reuse_existing)
                st.session_state["imf_horizon_sweep_result"] = {
                    "target": target_label,
                    "granularity": granularity,
                    "model_type": model_type,
                    "run_ids": [r["run_id"] for r in runs],
                }
                st.toast(f"Swept {len(runs)} horizons.")
            except Exception as exc:
                st.error(f"Horizon sweep failed: {exc}")

    result = st.session_state.get("imf_horizon_sweep_result")
    if not result:
        st.info("Pick a target, granularity, and model above, then run the sweep.")
        return

    runs = [get_run(rid) for rid in result["run_ids"]]
    runs = [r for r in runs if r is not None]
    if not runs:
        st.warning("Swept runs are no longer available (were they deleted?). Run the sweep again.")
        return
    runs = sorted(runs, key=lambda r: r.get("horizon", 1))

    unit = "min" if result["granularity"] == "Minute" else "h"
    horizon_labels = [f"{r.get('horizon', 1)}{unit}" for r in runs]
    r2_values = [r["metrics"]["r2"] for r in runs]
    mae_values = [r["metrics"]["mae"] for r in runs]
    rmse_values = [r["metrics"]["rmse"] for r in runs]

    st.markdown(
        f"##### {result['model_type']} — {result['target']} ({result['granularity']}) — skill decay with lead time"
    )
    fig_r2 = go.Figure(go.Scatter(x=horizon_labels, y=r2_values, mode="lines+markers", name="R²"))
    fig_r2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_r2.update_layout(title="Horizon vs. R²", height=340, xaxis_title="Forecast Horizon", yaxis_title="R²")
    plot_retro(fig_r2)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_mae = go.Figure(go.Scatter(x=horizon_labels, y=mae_values, mode="lines+markers", name="MAE"))
        fig_mae.update_layout(title="Horizon vs. MAE", height=320, xaxis_title="Forecast Horizon", yaxis_title="MAE")
        plot_retro(fig_mae)
    with col_b:
        fig_rmse = go.Figure(go.Scatter(x=horizon_labels, y=rmse_values, mode="lines+markers", name="RMSE"))
        fig_rmse.update_layout(title="Horizon vs. RMSE", height=320, xaxis_title="Forecast Horizon", yaxis_title="RMSE")
        plot_retro(fig_rmse)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    table_rows = [
        {
            "Horizon": horizon_labels[i],
            "R²": round(r2_values[i], 4),
            "MAE": round(mae_values[i], 4),
            "RMSE": round(rmse_values[i], 4),
            "Test Samples": runs[i]["n_test_samples"],
        }
        for i in range(len(runs))
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def render_imf_research_laboratory() -> None:
    hdr_col, pause_col = st.columns([3, 1])
    with hdr_col:
        st.markdown("### 🧪 IMF Research Laboratory")
        st.caption(
            "A research environment for comparing Bz/Bt/Bx/By forecasting architectures — fully isolated "
            "from the Production Prediction tab, which continues using the current trained production "
            "model exactly as-is. Models trained here never overwrite it."
        )
    with pause_col:
        st.session_state.setdefault("pause_autorefresh", False)
        if st.toggle("⏸ Pause Live Refresh", value=st.session_state["pause_autorefresh"], key="imf_lab_pause_toggle"):
            st.session_state["pause_autorefresh"] = True
        else:
            st.session_state["pause_autorefresh"] = False
    _imf_research_notes()

    sub = st.tabs(
        [
            "Training Runs",
            "Horizon Analysis",
            "Model Comparison",
            "Feature Engineering",
            "Physics Experiments",
            "Sequence Models",
            "Hyperparameter Tuning",
            "Hypothesis Testing",
        ]
    )
    with sub[0]:
        render_imf_training_runs_tab()
    with sub[1]:
        render_imf_horizon_analysis_tab()
    with sub[2]:
        render_imf_model_comparison_tab()
    with sub[3]:
        render_imf_feature_engineering_tab()
    with sub[4]:
        render_imf_physics_experiments_tab()
    with sub[5]:
        render_imf_sequence_models_tab()
    with sub[6]:
        render_imf_hyperparameter_tuning_tab()
    with sub[7]:
        render_imf_hypothesis_testing_tab()


# ==================== Kp Research Laboratory ====================
# Fully isolated from the Production Prediction tab — see
# swdss.models.kp_research module docstring for the production-safety
# contract. Nothing below ever writes to models/analytics/ or its
# metrics.json. Answers "why does one model perform better than another?"
# and "which physics actually improves Kp prediction?" — never intended
# to replace the operational forecast.


def _kp_research_notes() -> None:
    st.info(
        "**What this lab is for.** Production answers *what is the operational forecast* — this lab "
        "answers *why* one model or feature set performs better than another. Every run trains "
        "against the exact same target (NOAA's next official 3-hour Kp interval) and the same "
        "3-year historical dataset production itself trains on, so results here are genuinely "
        "comparable to production's own R²≈0.68 — not a different, easier problem."
    )


def _kp_research_hyperparam_inputs(model_type: str, key_prefix: str) -> dict:
    schema = kp_research.HYPERPARAM_SCHEMA.get(model_type, {})
    values = {}
    if not schema:
        st.caption("No tunable hyperparameters for this model.")
        return values
    cols = st.columns(min(len(schema), 4))
    for i, (name, spec) in enumerate(schema.items()):
        with cols[i % len(cols)]:
            label = name.replace("_", " ").title()
            if spec["type"] == "int":
                values[name] = st.number_input(
                    label, min_value=spec["min"], max_value=spec["max"], value=spec["default"], step=1,
                    key=f"{key_prefix}_{model_type}_{name}",
                )
            else:
                values[name] = st.number_input(
                    label, min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(spec["default"]),
                    step=0.01, key=f"{key_prefix}_{model_type}_{name}",
                )
    return values


def _kp_research_model_selector(key_prefix: str) -> str:
    options = kp_research.ALL_TRAINABLE_MODELS + kp_research.FUTURE_MODELS

    def _fmt(name):
        return f"{name} (coming soon)" if name in kp_research.FUTURE_MODELS else name

    choice = st.selectbox("Model Architecture", options, format_func=_fmt, key=f"{key_prefix}_model_select")
    if choice in kp_research.FUTURE_MODELS:
        st.warning(f"{choice} is a registered placeholder for future work — not trainable yet.")
    return choice


def _kp_research_feature_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or kp_research.default_feature_toggles()
    toggles = {}
    cols = st.columns(len(kp_research.FEATURE_GROUP_COLUMNS))
    for i, (group, group_cols) in enumerate(kp_research.FEATURE_GROUP_COLUMNS.items()):
        with cols[i]:
            st.markdown(f"**{group}**")
            toggles[group] = {}
            for col in group_cols:
                label = VARIABLE_LABELS.get(col, col.replace("_", " ").title())
                toggles[group][col] = st.checkbox(
                    label, value=defaults.get(group, {}).get(col, True), key=f"{key_prefix}_feat_{group}_{col}"
                )
    return toggles


def _kp_research_engineered_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or kp_research.default_engineered_toggles()
    toggles = {}
    cols = st.columns(len(kp_research.ENGINEERED_GROUPS))
    for i, group in enumerate(kp_research.ENGINEERED_GROUPS):
        with cols[i]:
            toggles[group] = st.checkbox(group, value=defaults.get(group, True), key=f"{key_prefix}_eng_{group}")
    return toggles


def _kp_research_physics_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or {}
    toggles = {}
    cols = st.columns(3)
    for i, name in enumerate(kp_research.PHYSICS_FEATURE_OPTIONS):
        with cols[i % 3]:
            toggles[name] = st.checkbox(name, value=defaults.get(name, False), key=f"{key_prefix}_phys_{name}")
    return toggles


def _kp_research_run_row(run: dict, best_run_id: str = None, key_prefix: str = "kp_runs") -> None:
    """key_prefix disambiguates widget keys when the same run is rendered
    from more than one tab in the same script run — see the identical
    IMF lab pattern (_imf_research_run_row) this was copied from.
    """
    m = run["metrics"]
    is_best = run["run_id"] == best_run_id
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.22, 0.11, 0.11, 0.11, 0.11, 0.11, 0.23])
        with c1:
            star = "⭐ " if is_best else ""
            promoted_tag = " 🚀" if run.get("promoted") else ""
            st.markdown(f"**{star}{run['model_type']}**{promoted_tag}")
            seq_note = f" · seq={run['sequence_length']}h" if run.get("sequence_length") else ""
            st.caption(f"Kp{seq_note} · {pd.Timestamp(run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}")
        with c2:
            metric_card("R²", f"{m['r2']:.4f}", "")
        with c3:
            metric_card("MAE", f"{m['mae']:.3f}", "")
        with c4:
            metric_card("RMSE", f"{m['rmse']:.3f}", "")
        with c5:
            metric_card("Train Time", f"{run.get('training_time_sec', 0):.2f}s", "")
        with c6:
            metric_card("Predict Time", f"{run.get('prediction_time_sec', 0) * 1000:.1f}ms", "")
        with c7:
            st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button(
                    "Promoted" if run.get("promoted") else "Promote",
                    key=f"{key_prefix}_promote_{run['run_id']}",
                    disabled=run.get("promoted", False),
                    use_container_width=True,
                ):
                    kp_research.promote_run(run["run_id"])
                    st.toast("Marked as promoted (label only — production untouched).")
                    st.rerun()
            with b2:
                if st.button(
                    "Load",
                    key=f"{key_prefix}_load_{run['run_id']}",
                    use_container_width=True,
                    disabled=run["model_type"] in kp_research.SEQUENCE_MODELS,
                ):
                    try:
                        model = kp_research.load_trained_model(run["run_id"])
                        st.toast(f"Loaded {run['model_type']} model ({type(model).__name__}) into memory.")
                    except Exception as exc:
                        st.error(f"Load failed: {exc}")
            with b3:
                if st.button("Delete", key=f"{key_prefix}_delete_{run['run_id']}", use_container_width=True):
                    kp_research.delete_run(run["run_id"])
                    st.toast("Run deleted.")
                    st.rerun()


def render_kp_model_comparison_tab() -> None:
    st.caption(
        "Train, evaluate, and compare Kp models — every run auto-saves to its own registry (never "
        "overwrites production) and can be reloaded via each run card's Load button. Target is "
        "always NOAA's next official 3-hour Kp interval, identical to production's own definition."
    )
    model_type = _kp_research_model_selector("kp_compare")
    with st.expander("Feature Groups", expanded=True):
        feature_toggles = _kp_research_feature_toggle_form("kp_compare")
    with st.expander("Engineered Features", expanded=False):
        engineered_groups = _kp_research_engineered_toggle_form("kp_compare")
    with st.expander("Physics Experiment Features (optional)", expanded=False):
        physics_features = _kp_research_physics_toggle_form("kp_compare")

    sequence_length = None
    if model_type in kp_research.SEQUENCE_MODELS:
        sequence_length = st.selectbox(
            "Sequence Length (hours, look-back window)",
            kp_research.SEQUENCE_LENGTH_OPTIONS,
            index=kp_research.SEQUENCE_LENGTH_OPTIONS.index(kp_research.DEFAULT_SEQUENCE_LENGTH),
            key="kp_compare_seqlen",
        )

    st.markdown("**Hyperparameters**")
    hyperparams = (
        _kp_research_hyperparam_inputs(model_type, "kp_compare") if model_type in kp_research.ALL_TRAINABLE_MODELS else {}
    )
    notes = st.text_input("Notes (optional)", key="kp_compare_notes")

    if st.button(
        "🧪 Train Model", key="kp_compare_train_btn", type="primary",
        disabled=model_type not in kp_research.ALL_TRAINABLE_MODELS,
    ):
        with st.spinner(f"Training {model_type}..."):
            try:
                run = kp_research.train_kp_research_model(
                    model_type,
                    feature_toggles=feature_toggles,
                    engineered_groups=engineered_groups,
                    physics_features=physics_features,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes=notes,
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Run Log")
    runs = kp_research.list_runs()
    if not runs:
        st.info("No training runs yet. Train a model above.")
        return
    best_id = max(runs, key=lambda r: r["metrics"]["r2"])["run_id"]
    for run in runs[:10]:
        _kp_research_run_row(run, best_run_id=best_id, key_prefix="kp_compare")
    if len(runs) > 10:
        st.caption(f"Showing 10 most recent of {len(runs)} runs — see Experiment Tracking for the full history.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Compare Models")

    def _run_label(r):
        seq = f" seq={r['sequence_length']}h" if r.get("sequence_length") else ""
        return f"{r['model_type']}{seq} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    default_n = min(4, len(run_labels))
    selected = st.multiselect("Models to compare", run_labels, default=run_labels[:default_n], key="kp_compare_select")
    chosen = [runs[run_labels.index(s)] for s in selected]
    if not chosen:
        st.info("Select at least one model above.")
        return

    best_chosen_id = max(chosen, key=lambda r: r["metrics"]["r2"])["run_id"]
    rows = []
    for r in chosen:
        m = r["metrics"]
        rows.append(
            {
                "Model": r["model_type"] + (" ⭐" if r["run_id"] == best_chosen_id else ""),
                "R²": round(m["r2"], 4),
                "MAE": round(m["mae"], 4),
                "RMSE": round(m["rmse"], 4),
                "MAPE (%)": round(m["mape"], 2) if m["mape"] is not None else None,
                "Bias": round(m["bias"], 4),
                "Training Time (s)": round(r.get("training_time_sec", 0), 3),
                "Prediction Time (ms)": round(r.get("prediction_time_sec", 0) * 1000, 2),
                "Trained": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Predicted vs. Actual / Residuals / Feature Importance")
    detail_label = st.selectbox("Inspect model", selected, key="kp_compare_detail")
    detail_run = chosen[selected.index(detail_label)]
    sample = detail_run["prediction_sample"]
    y_true = sample["y_true"]
    y_pred = sample["y_pred"]
    residuals = [p - t for p, t in zip(y_pred, y_true)]

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=4), name="Predicted vs Actual"))
        lo, hi = min(y_true + y_pred), max(y_true + y_pred)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
        fig.update_layout(
            title="Predicted vs. Actual (held-out test sample)", height=340, xaxis_title="Actual Kp", yaxis_title="Predicted Kp"
        )
        plot_retro(fig, key="kp_compare_pred_actual")
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=4), name="Residual"))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Residual Plot", height=340, yaxis_title="Predicted − Actual")
        plot_retro(fig2, key="kp_compare_residual")

    if detail_run.get("feature_importance"):
        st.markdown("##### Feature Importance (Top 20)")
        fi_df = pd.DataFrame(detail_run["feature_importance"], columns=["Feature", "Importance"])
        fig3 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig3.update_layout(title="Top Contributing Features", height=420, yaxis=dict(autorange="reversed"))
        plot_retro(fig3, key="kp_compare_feature_importance")
    else:
        st.caption("Feature importance not available for this model type (e.g. SVR, LSTM/GRU).")

    if detail_run.get("loss_history"):
        st.markdown("##### Training / Validation Loss")
        lh = detail_run["loss_history"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig4.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig4.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig4, key="kp_compare_loss")


def render_kp_feature_ablation_tab() -> None:
    st.caption(
        "Trains a Full Model (every feature group + every engineered group enabled) then retrains "
        "once per unit with just that one disabled — ranked by how much R² drops when removed, i.e. "
        "how much the full model actually relies on it. This is leave-one-out, not the cumulative "
        "'enable Ey, then enable VBz, ...' style — a cumulative sweep's deltas depend on the order "
        "features are added (whichever goes first tends to look most important merely by going "
        "first), whereas leave-one-out is order-independent. Physics-experiment features are tested "
        "individually in Hypothesis Testing instead."
    )
    model_type = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kp_ablation_model")
    if st.button("🔬 Run Feature Ablation Sweep", key="kp_ablation_run_btn", type="primary"):
        n_units = len(kp_research.FEATURE_ABLATION_UNITS) + 1
        with st.spinner(f"Training {n_units} models ({model_type})..."):
            try:
                result = kp_research.run_feature_ablation_sweep(model_type)
                st.session_state["kp_ablation_result"] = result
                st.toast("Feature ablation sweep complete.")
            except Exception as exc:
                st.error(f"Ablation sweep failed: {exc}")

    result = st.session_state.get("kp_ablation_result")
    if not result:
        st.info("Pick a model above, then run the sweep.")
        return

    st.markdown(f"##### {result['model_type']} — Full Model R² = {result['full_r2']:.4f}")
    ranked = result["ranked"]
    fig = go.Figure(
        go.Bar(
            x=[r["delta_r2"] for r in ranked],
            y=[r["unit"] for r in ranked],
            orientation="h",
            marker_color=["#1f7a3a" if r["delta_r2"] >= 0 else "#7a1f1f" for r in ranked],
        )
    )
    fig.update_layout(
        title="R² Drop When Removed — Ranked Feature Group Contribution",
        height=380,
        xaxis_title="ΔR² (Full Model − Without This Group)",
        yaxis=dict(autorange="reversed"),
    )
    plot_retro(fig, key="kp_ablation_bar")

    table_rows = [
        {
            "Rank": i + 1,
            "Feature Group": r["unit"].replace("Without ", ""),
            "R² Without": round(r["r2"], 4),
            "ΔR² (contribution)": round(r["delta_r2"], 4),
            "MAE Without": round(r["mae"], 4),
        }
        for i, r in enumerate(ranked)
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Swept at {pd.Timestamp(result['swept_at']).strftime('%Y-%m-%d %H:%M UTC')} — a positive ΔR² "
        "means removing that group made the model worse (it was contributing); negative means the "
        "model did slightly better without it on this particular held-out split."
    )


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _cached_kp_research_frame(feature_toggles: dict = None, engineered_groups: dict = None, physics_features: dict = None):
    """Read-only exploratory calls into kp_research.load_kp_research_frame()
    (e.g. the Physics Experiments tab) re-run on every script rerun — every
    15s via the app-wide auto_refresh(), plus once per Kp Research Lab
    sub-tab per rerun since st.tabs() executes every tab body regardless of
    which one is visually active — even though the underlying
    analytics_features.csv is a static historical file that never changes
    within a session. Caching here eliminates that redundant CSV read +
    physics-feature recomputation with zero staleness risk (the source
    data genuinely doesn't change). NEVER used by train_kp_research_model
    itself — training always calls the uncached engine function directly,
    since that call only happens once per explicit Train click, not on
    every idle rerun.
    """
    return kp_research.load_kp_research_frame(feature_toggles, engineered_groups, physics_features)


def render_kp_physics_experiments_tab() -> None:
    st.caption(
        "Exploratory analysis: how does each individual physics-derived feature relate to Kp's "
        "behavior historically? Enable one to inspect it, or quick-train a model with just that one "
        "feature added on top of the full default feature set."
    )
    label = st.selectbox("Physics feature", kp_research.PHYSICS_FEATURE_OPTIONS, key="kp_physics_feature_select")

    try:
        base_frame, base_cols = _cached_kp_research_frame()
        frame, cols_with = _cached_kp_research_frame(physics_features={label: True})
    except Exception as exc:
        st.warning(f"Could not load the research feature frame: {exc}")
        return

    new_cols = [c for c in cols_with if c not in base_cols]
    if not new_cols:
        st.warning("Could not resolve this feature's column name.")
        return
    col = new_cols[-1]

    latest = frame[col].dropna()
    if latest.empty:
        st.info("Not enough history to compute this feature yet.")
        return
    metric_card(label, f"{latest.iloc[-1]:.3f}", f"Latest hourly value ({latest.index[-1].strftime('%Y-%m-%d %H:%M UTC')})")

    recent = frame.tail(24 * 30).dropna(subset=[col, "kp"])
    if not recent.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=recent.index, y=recent["kp"], name="Kp", line=dict(color="#1f5a7a")), secondary_y=False)
        fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=label, line=dict(color="#7a1f5a")), secondary_y=True)
        fig.update_layout(title=f"Kp vs. {label} — last {len(recent)} hours", height=380)
        fig.update_yaxes(title_text="Kp", secondary_y=False)
        fig.update_yaxes(title_text=label, secondary_y=True)
        plot_retro(fig, key=f"kp_physics_timeseries_{col}")

    next_target = kp_research.build_kp_interval_target(frame)
    valid = frame[[col]].join(next_target.rename("next_kp")).dropna()
    if len(valid) > 10:
        corr = valid[col].corr(valid["next_kp"])
        st.caption(f"Correlation between {label} and the next official Kp interval, across the full history: **{corr:.3f}**")
        fig2 = go.Figure(go.Scattergl(x=valid[col], y=valid["next_kp"], mode="markers", marker=dict(size=3, opacity=0.4)))
        fig2.update_layout(title=f"{label} vs. Next Kp Interval", height=380, xaxis_title=label, yaxis_title="Next official Kp interval")
        plot_retro(fig2, key=f"kp_physics_corr_{col}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button(f"🧪 Quick-Train Linear Regression with {label} added", key="kp_physics_quick_train"):
        with st.spinner("Training..."):
            try:
                run = kp_research.train_kp_research_model(
                    "Linear Regression", physics_features={label: True}, notes=f"Physics Experiments — quick test of {label}"
                )
                st.success(
                    f"Trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.3f} "
                    "(compare against the Feature Ablation / Model Comparison full-model baseline)."
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def render_kp_sequence_models_tab() -> None:
    st.caption(
        "LSTM/GRU look back over a window of consecutive hours (Sequence Length) rather than a "
        "single row, letting the model learn how Kp's drivers have been evolving, not just their "
        "instantaneous values. Trains in an isolated subprocess (see swdss.models.kp_research) — "
        "TensorFlow cannot safely share a process with this lab's scikit-learn/XGBoost/LightGBM/"
        "CatBoost imports."
    )
    if not kp_research.SEQUENCE_MODELS:
        st.warning("TensorFlow/Keras is not installed — sequence models are unavailable in this environment.")
        return

    model_type = st.selectbox("Model", kp_research.SEQUENCE_MODELS, key="kp_seq_model")
    sequence_length = st.selectbox(
        "Sequence Length (hours)",
        kp_research.SEQUENCE_LENGTH_OPTIONS,
        index=kp_research.SEQUENCE_LENGTH_OPTIONS.index(kp_research.DEFAULT_SEQUENCE_LENGTH),
        key="kp_seq_seqlen",
    )
    with st.expander("Feature Groups", expanded=False):
        feature_toggles = _kp_research_feature_toggle_form("kp_seq")
    hyperparams = _kp_research_hyperparam_inputs(model_type, "kp_seq")

    if st.button("🧪 Train Sequence Model", key="kp_seq_train_btn", type="primary"):
        with st.spinner(f"Training {model_type} (seq={sequence_length}h) — isolated subprocess, ~10-30s..."):
            try:
                run = kp_research.train_kp_research_model(
                    model_type,
                    feature_toggles=feature_toggles,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes="Sequence Models tab",
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    all_runs = kp_research.list_runs()
    seq_runs = [r for r in all_runs if r["model_type"] in kp_research.SEQUENCE_MODELS]
    tabular_runs = [r for r in all_runs if r["model_type"] not in kp_research.SEQUENCE_MODELS]
    if not seq_runs:
        st.info("No LSTM/GRU runs yet — train one above.")
        return

    best_seq = max(seq_runs, key=lambda r: r["metrics"]["r2"])
    if tabular_runs:
        best_tabular = max(tabular_runs, key=lambda r: r["metrics"]["r2"])
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Best Sequence Model R²", f"{best_seq['metrics']['r2']:.4f}", best_seq["model_type"])
        with c2:
            metric_card("Best Tabular Model R²", f"{best_tabular['metrics']['r2']:.4f}", best_tabular["model_type"])
        with c3:
            diff = best_seq["metrics"]["r2"] - best_tabular["metrics"]["r2"]
            metric_card("Sequence Advantage", f"{diff:+.4f}", "Positive = sequence models outperform tabular")

    def _run_label(r):
        return f"{r['model_type']} seq={r['sequence_length']}h ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in seq_runs]
    selected = st.selectbox("Inspect run", labels, key="kp_seq_inspect")
    run = seq_runs[labels.index(selected)]
    m = run["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("R²", f"{m['r2']:.4f}", "")
    with c2:
        metric_card("MAE", f"{m['mae']:.3f}", "")
    with c3:
        metric_card("RMSE", f"{m['rmse']:.3f}", "")
    with c4:
        metric_card("MAPE", "N/A" if m["mape"] is None else f"{m['mape']:.1f}%", "")
    with c5:
        metric_card("Bias", f"{m['bias']:+.3f}", "")

    if run.get("loss_history"):
        lh = run["loss_history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig, key="kp_seq_loss")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### All Sequence Model Runs")
    for r in seq_runs:
        _kp_research_run_row(r, key_prefix="kp_seq")


def render_kp_experiment_tracking_tab() -> None:
    st.caption(
        "Every experiment ever run in this lab — full configuration and results, for reproducibility. "
        "Filter, inspect, promote, or delete any run."
    )
    runs = kp_research.list_runs()
    if not runs:
        st.info("No experiments recorded yet — train a model in any other tab.")
        return

    model_filter = st.selectbox("Filter by model", ["All"] + kp_research.ALL_TRAINABLE_MODELS, key="kp_track_model_filter")
    filtered = runs if model_filter == "All" else [r for r in runs if r["model_type"] == model_filter]
    st.caption(f"{len(filtered)} of {len(runs)} total experiments")

    rows = []
    for r in filtered:
        feat_summary = ", ".join(g for g, cols in r.get("feature_toggles", {}).items() if all(cols.values())) or "partial"
        physics_on = ", ".join(k for k, v in (r.get("physics_features") or {}).items() if v) or "None"
        rows.append(
            {
                "Timestamp": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
                "Model": r["model_type"],
                "Feature Groups (fully on)": feat_summary,
                "Physics Features": physics_on,
                "Seq Len": str(r["sequence_length"]) + "h" if r.get("sequence_length") else "—",
                "Train N": r["n_train_samples"],
                "Test N": r["n_test_samples"],
                "R²": round(r["metrics"]["r2"], 4),
                "MAE": round(r["metrics"]["mae"], 4),
                "RMSE": round(r["metrics"]["rmse"], 4),
                "MAPE (%)": round(r["metrics"]["mape"], 2) if r["metrics"]["mape"] is not None else None,
                "Bias": round(r["metrics"]["bias"], 4),
                "Train Time (s)": round(r.get("training_time_sec", 0), 3),
                "Notes": r.get("notes", ""),
                "Promoted": "🚀" if r.get("promoted") else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Manage Runs")
    for r in filtered[:15]:
        _kp_research_run_row(r, key_prefix="kp_track")
    if len(filtered) > 15:
        st.caption(f"Showing 15 most recent of {len(filtered)}.")


def render_kp_hypothesis_testing_tab() -> None:
    st.caption(
        "Fixed, reproducible hypotheses: each trains a baseline WITHOUT the tested feature and an "
        "experimental run WITH it (everything else at defaults), then reports ΔR²/ΔMAE/ΔRMSE and an "
        "Accept/Reject verdict (Accept requires ΔR² ≥ "
        f"{kp_research.HYPOTHESIS_ACCEPT_THRESHOLD_R2})."
    )
    hypothesis_label = st.selectbox("Hypothesis", list(kp_research.HYPOTHESIS_DEFINITIONS), key="kp_hyp_select")
    model_type = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kp_hyp_model")

    if st.button("🔬 Run Hypothesis Test", key="kp_hyp_run_btn", type="primary"):
        with st.spinner(f"Testing: {hypothesis_label}..."):
            try:
                result = kp_research.run_hypothesis_test(hypothesis_label, model_type)
                st.session_state["kp_hyp_last_result"] = result
                st.toast(f"{result['verdict']} — ΔR²={result['delta_r2']:+.4f}")
            except Exception as exc:
                st.error(f"Hypothesis test failed: {exc}")

    result = st.session_state.get("kp_hyp_last_result")
    if result and result["hypothesis"] == hypothesis_label:
        color = "#1f7a3a" if result["verdict"] == "Accept" else "#7a1f1f"
        st.markdown(f"### Verdict: <span style='color:{color}'>{result['verdict']}</span>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("ΔR²", f"{result['delta_r2']:+.4f}", "")
        with c2:
            metric_card("ΔMAE", f"{result['delta_mae']:+.4f}", "")
        with c3:
            metric_card("ΔRMSE", f"{result['delta_rmse']:+.4f}", "")
        with c4:
            metric_card("Baseline → Experimental R²", f"{result['baseline_r2']:.4f} → {result['experimental_r2']:.4f}", "")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Hypothesis Test History")
    results = kp_research.list_hypothesis_results()
    if not results:
        st.info("No hypothesis tests run yet.")
        return
    rows = [
        {
            "Hypothesis": r["hypothesis"],
            "Model": r["model_type"],
            "Verdict": r["verdict"],
            "ΔR²": round(r["delta_r2"], 4),
            "ΔMAE": round(r["delta_mae"], 4),
            "ΔRMSE": round(r["delta_rmse"], 4),
            "Tested": pd.Timestamp(r["tested_at"]).strftime("%Y-%m-%d %H:%M UTC"),
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_kp_visualization_tab() -> None:
    st.caption(
        "Cross-experiment views: how R² has moved across every run in this lab, which model "
        "architecture performs best overall, and a full diagnostic breakdown for any single run."
    )
    runs = kp_research.list_runs()
    if not runs:
        st.info("No experiments yet.")
        return

    runs_sorted = sorted(runs, key=lambda r: r["trained_at"])
    fig = go.Figure(
        go.Scatter(
            x=[pd.Timestamp(r["trained_at"]) for r in runs_sorted],
            y=[r["metrics"]["r2"] for r in runs_sorted],
            mode="markers+lines",
            text=[r["model_type"] for r in runs_sorted],
            name="R² over time",
        )
    )
    fig.update_layout(title="Experiment History — R² Over Time", height=360, xaxis_title="Trained At", yaxis_title="R²")
    plot_retro(fig, key="kp_viz_history")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    by_model: dict = {}
    for r in runs:
        by_model.setdefault(r["model_type"], []).append(r["metrics"]["r2"])
    model_avg = {m: sum(v) / len(v) for m, v in by_model.items()}
    fig2 = go.Figure(go.Bar(x=list(model_avg.keys()), y=list(model_avg.values())))
    fig2.update_layout(title="Average R² by Model Architecture (across all runs)", height=360, yaxis_title="Mean R²")
    plot_retro(fig2, key="kp_viz_model_avg")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Inspect a Single Run")

    def _run_label(r):
        seq = f" seq={r['sequence_length']}h" if r.get("sequence_length") else ""
        return f"{r['model_type']}{seq} — R²={r['metrics']['r2']:.4f} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in runs]
    selected = st.selectbox("Run", labels, key="kp_viz_run_select")
    run = runs[labels.index(selected)]
    sample = run["prediction_sample"]
    y_true, y_pred = sample["y_true"], sample["y_pred"]
    residuals = [p - t for p, t in zip(y_pred, y_true)]

    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=4), name="Predicted vs Actual"))
        lo, hi = min(y_true + y_pred), max(y_true + y_pred)
        fig3.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
        fig3.update_layout(title="Predicted vs. Actual", height=320)
        plot_retro(fig3, key="kp_viz_pred_actual")
    with col_b:
        fig4 = go.Figure(go.Scatter(y=residuals, mode="markers", marker=dict(size=4)))
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(title="Residual Plot", height=320)
        plot_retro(fig4, key="kp_viz_residual")

    if run.get("feature_importance"):
        fi_df = pd.DataFrame(run["feature_importance"][:15], columns=["Feature", "Importance"])
        fig5 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig5.update_layout(title="Top 15 Feature Importance", height=380, yaxis=dict(autorange="reversed"))
        plot_retro(fig5, key="kp_viz_feature_importance")

    if run.get("loss_history"):
        lh = run["loss_history"]
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig6.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig6.update_layout(title="Learning Curve", height=320)
        plot_retro(fig6, key="kp_viz_loss")


def render_kp_research_laboratory() -> None:
    st.markdown("### 🧪 Kp Research Laboratory")
    hdr_col, pause_col = st.columns([3, 1])
    with hdr_col:
        st.caption(
            "A scientific experimentation platform for Kp forecasting — fully isolated from the "
            "Production Prediction tab, which continues using the current trained production model "
            "exactly as-is. Nothing trained here overwrites it; Promote only labels a run for your own "
            "tracking, wiring a model into production is always a manual, deliberate step."
        )
    with pause_col:
        st.session_state.setdefault("pause_autorefresh", False)
        if st.toggle("⏸ Pause Live Refresh", value=st.session_state["pause_autorefresh"], key="kp_lab_pause_toggle"):
            st.session_state["pause_autorefresh"] = True
        else:
            st.session_state["pause_autorefresh"] = False
    _kp_research_notes()
    with st.expander("Future Research — planned architecture extensions"):
        st.write(", ".join(kp_research.FUTURE_MODELS))
        st.caption(
            "Registered as disabled entries in the model selector now, so adding a real "
            "implementation later never requires redesigning this interface."
        )

    sub = st.tabs(
        [
            "Model Comparison",
            "Feature Ablation",
            "Physics Experiments",
            "Sequence Models",
            "Experiment Tracking",
            "Hypothesis Testing",
            "Visualization",
        ]
    )
    with sub[0]:
        render_kp_model_comparison_tab()
    with sub[1]:
        render_kp_feature_ablation_tab()
    with sub[2]:
        render_kp_physics_experiments_tab()
    with sub[3]:
        render_kp_sequence_models_tab()
    with sub[4]:
        render_kp_experiment_tracking_tab()
    with sub[5]:
        render_kp_hypothesis_testing_tab()
    with sub[6]:
        render_kp_visualization_tab()


# ==================== AE Research Laboratory ====================
# Fully isolated from the Production Prediction tab — see
# swdss.models.ae_research module docstring for the production-safety
# contract. Nothing below ever writes to models/ae/ or its metrics.json.
# Answers "how can AE prediction be improved?", "what physics governs
# AE?", and "which ML model performs best?" — never intended to replace
# the operational forecast.


def _ae_research_notes() -> None:
    st.info(
        "**What this lab is for.** Production answers *what is the operational AE forecast* — this "
        "lab answers *how can it be improved* and *what physics governs it*. Every run trains against "
        "the exact same `ae_analytics_features.csv` production's own AE model trains on, so results "
        "here are genuinely comparable (production: R²=0.744 @1h, R²=0.398 @3h). Horizons here are "
        "1/2/3 hours only — AE has no minute-level ground truth anywhere in this codebase (the only "
        "historical archive, `ae_processed.parquet`, is itself hourly), so 15/30-minute horizons were "
        "deliberately not built rather than faked via interpolation."
    )


def _ae_research_hyperparam_inputs(model_type: str, key_prefix: str) -> dict:
    schema = ae_research.HYPERPARAM_SCHEMA.get(model_type, {})
    values = {}
    if not schema:
        st.caption("No tunable hyperparameters for this model.")
        return values
    cols = st.columns(min(len(schema), 4))
    for i, (name, spec) in enumerate(schema.items()):
        with cols[i % len(cols)]:
            label = name.replace("_", " ").title()
            if spec["type"] == "int":
                values[name] = st.number_input(
                    label, min_value=spec["min"], max_value=spec["max"], value=spec["default"], step=1,
                    key=f"{key_prefix}_{model_type}_{name}",
                )
            else:
                values[name] = st.number_input(
                    label, min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(spec["default"]),
                    step=0.01, key=f"{key_prefix}_{model_type}_{name}",
                )
    return values


def _ae_research_model_selector(key_prefix: str) -> str:
    options = ae_research.ALL_TRAINABLE_MODELS + ae_research.FUTURE_MODELS

    def _fmt(name):
        return f"{name} (coming soon)" if name in ae_research.FUTURE_MODELS else name

    choice = st.selectbox("Model Architecture", options, format_func=_fmt, key=f"{key_prefix}_model_select")
    if choice in ae_research.FUTURE_MODELS:
        st.warning(f"{choice} is a registered placeholder for future work — not trainable yet.")
    return choice


def _ae_research_horizon_selector(key_prefix: str) -> int:
    return st.radio(
        "Forecast Horizon",
        ae_research.HORIZON_OPTIONS,
        format_func=lambda h: f"{h} hour" + ("s" if h != 1 else ""),
        horizontal=True,
        key=f"{key_prefix}_horizon",
    )


def _ae_research_feature_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or ae_research.default_feature_toggles()
    toggles = {}
    cols = st.columns(len(ae_research.FEATURE_GROUP_COLUMNS))
    for i, (group, group_cols) in enumerate(ae_research.FEATURE_GROUP_COLUMNS.items()):
        with cols[i]:
            st.markdown(f"**{group}**")
            toggles[group] = {}
            for col in group_cols:
                label = VARIABLE_LABELS.get(col, col.replace("_", " ").title())
                toggles[group][col] = st.checkbox(
                    label, value=defaults.get(group, {}).get(col, True), key=f"{key_prefix}_feat_{group}_{col}"
                )
    return toggles


def _ae_research_engineered_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or ae_research.default_engineered_toggles()
    toggles = {}
    cols = st.columns(len(ae_research.ENGINEERED_GROUPS))
    for i, group in enumerate(ae_research.ENGINEERED_GROUPS):
        with cols[i]:
            toggles[group] = st.checkbox(group, value=defaults.get(group, True), key=f"{key_prefix}_eng_{group}")
    return toggles


def _ae_research_physics_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or {}
    toggles = {}
    cols = st.columns(3)
    for i, name in enumerate(ae_research.PHYSICS_FEATURE_OPTIONS):
        with cols[i % 3]:
            toggles[name] = st.checkbox(name, value=defaults.get(name, False), key=f"{key_prefix}_phys_{name}")
    return toggles


def _ae_research_run_row(run: dict, best_run_id: str = None, key_prefix: str = "ae_runs") -> None:
    m = run["metrics"]
    is_best = run["run_id"] == best_run_id
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.22, 0.11, 0.11, 0.11, 0.11, 0.11, 0.23])
        with c1:
            star = "⭐ " if is_best else ""
            promoted_tag = " 🚀" if run.get("promoted") else ""
            st.markdown(f"**{star}{run['model_type']}**{promoted_tag}")
            seq_note = f" · seq={run['sequence_length']}h" if run.get("sequence_length") else ""
            st.caption(f"AE +{run.get('horizon', 1)}h{seq_note} · {pd.Timestamp(run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}")
        with c2:
            metric_card("R²", f"{m['r2']:.4f}", "")
        with c3:
            metric_card("MAE", f"{m['mae']:.3f}", "")
        with c4:
            metric_card("RMSE", f"{m['rmse']:.3f}", "")
        with c5:
            metric_card("Train Time", f"{run.get('training_time_sec', 0):.2f}s", "")
        with c6:
            metric_card("Predict Time", f"{run.get('prediction_time_sec', 0) * 1000:.1f}ms", "")
        with c7:
            st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button(
                    "Promoted" if run.get("promoted") else "Promote",
                    key=f"{key_prefix}_promote_{run['run_id']}",
                    disabled=run.get("promoted", False),
                    use_container_width=True,
                ):
                    ae_research.promote_run(run["run_id"])
                    st.toast("Marked as promoted (label only — production untouched).")
                    st.rerun()
            with b2:
                if st.button(
                    "Load",
                    key=f"{key_prefix}_load_{run['run_id']}",
                    use_container_width=True,
                    disabled=run["model_type"] in ae_research.SEQUENCE_MODELS,
                ):
                    try:
                        model = ae_research.load_trained_model(run["run_id"])
                        st.toast(f"Loaded {run['model_type']} model ({type(model).__name__}) into memory.")
                    except Exception as exc:
                        st.error(f"Load failed: {exc}")
            with b3:
                if st.button("Delete", key=f"{key_prefix}_delete_{run['run_id']}", use_container_width=True):
                    ae_research.delete_run(run["run_id"])
                    st.toast("Run deleted.")
                    st.rerun()


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _cached_ae_research_frame(feature_toggles: dict = None, engineered_groups: dict = None, physics_features: dict = None):
    """Same rationale as _cached_kp_research_frame: ae_analytics_features.csv
    is a static historical file that never changes mid-session, so caching
    the exploratory Physics Experiments tab's reads eliminates redundant
    CSV parsing + physics-feature recomputation on every 15s auto-refresh
    tick with zero staleness risk.
    """
    return ae_research.load_ae_research_frame(feature_toggles, engineered_groups, physics_features)


def render_ae_model_comparison_tab() -> None:
    st.caption(
        "Train, evaluate, and compare AE models — every run auto-saves to its own registry (never "
        "overwrites production) and can be reloaded via each run card's Load button."
    )
    model_type = _ae_research_model_selector("ae_compare")
    horizon = _ae_research_horizon_selector("ae_compare")
    with st.expander("Feature Groups", expanded=True):
        feature_toggles = _ae_research_feature_toggle_form("ae_compare")
    with st.expander("Engineered Features", expanded=False):
        engineered_groups = _ae_research_engineered_toggle_form("ae_compare")
    with st.expander("Physics Feature Experiments (optional)", expanded=False):
        physics_features = _ae_research_physics_toggle_form("ae_compare")

    sequence_length = None
    if model_type in ae_research.SEQUENCE_MODELS:
        sequence_length = st.selectbox(
            "Sequence Length (hours, look-back window)",
            ae_research.SEQUENCE_LENGTH_OPTIONS,
            index=ae_research.SEQUENCE_LENGTH_OPTIONS.index(ae_research.DEFAULT_SEQUENCE_LENGTH),
            key="ae_compare_seqlen",
        )

    st.markdown("**Hyperparameters**")
    hyperparams = (
        _ae_research_hyperparam_inputs(model_type, "ae_compare") if model_type in ae_research.ALL_TRAINABLE_MODELS else {}
    )
    notes = st.text_input("Notes (optional)", key="ae_compare_notes")

    if st.button(
        "🧪 Train Model", key="ae_compare_train_btn", type="primary",
        disabled=model_type not in ae_research.ALL_TRAINABLE_MODELS,
    ):
        with st.spinner(f"Training {model_type}..."):
            try:
                run = ae_research.train_ae_research_model(
                    model_type,
                    horizon=horizon,
                    feature_toggles=feature_toggles,
                    engineered_groups=engineered_groups,
                    physics_features=physics_features,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes=notes,
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Run Log")
    runs = ae_research.list_runs()
    if not runs:
        st.info("No training runs yet. Train a model above.")
        return
    best_id = max(runs, key=lambda r: r["metrics"]["r2"])["run_id"]
    for run in runs[:10]:
        _ae_research_run_row(run, best_run_id=best_id, key_prefix="ae_compare")
    if len(runs) > 10:
        st.caption(f"Showing 10 most recent of {len(runs)} runs — see Experiment Tracking for the full history.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Compare Models")

    def _run_label(r):
        seq = f" seq={r['sequence_length']}h" if r.get("sequence_length") else ""
        return f"{r['model_type']}{seq} +{r.get('horizon', 1)}h ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    default_n = min(4, len(run_labels))
    selected = st.multiselect("Models to compare", run_labels, default=run_labels[:default_n], key="ae_compare_select")
    chosen = [runs[run_labels.index(s)] for s in selected]
    if not chosen:
        st.info("Select at least one model above.")
        return

    best_chosen_id = max(chosen, key=lambda r: r["metrics"]["r2"])["run_id"]
    rows = []
    for r in chosen:
        m = r["metrics"]
        rows.append(
            {
                "Model": r["model_type"] + (" ⭐" if r["run_id"] == best_chosen_id else ""),
                "Horizon": f"{r.get('horizon', 1)}h",
                "R²": round(m["r2"], 4),
                "MAE": round(m["mae"], 4),
                "RMSE": round(m["rmse"], 4),
                "MAPE (%)": round(m["mape"], 2) if m["mape"] is not None else None,
                "Bias": round(m["bias"], 4),
                "Training Time (s)": round(r.get("training_time_sec", 0), 3),
                "Prediction Time (ms)": round(r.get("prediction_time_sec", 0) * 1000, 2),
                "Trained": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Predicted vs. Actual / Residuals / Feature Importance")
    detail_label = st.selectbox("Inspect model", selected, key="ae_compare_detail")
    detail_run = chosen[selected.index(detail_label)]
    sample = detail_run["prediction_sample"]
    y_true = sample["y_true"]
    y_pred = sample["y_pred"]
    residuals = [p - t for p, t in zip(y_pred, y_true)]

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=4), name="Predicted vs Actual"))
        lo, hi = min(y_true + y_pred), max(y_true + y_pred)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
        fig.update_layout(
            title="Predicted vs. Actual (held-out test sample)", height=340, xaxis_title="Actual AE", yaxis_title="Predicted AE"
        )
        plot_retro(fig, key="ae_compare_pred_actual")
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=4), name="Residual"))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Residual Plot", height=340, yaxis_title="Predicted − Actual")
        plot_retro(fig2, key="ae_compare_residual")

    if detail_run.get("feature_importance"):
        st.markdown("##### Feature Importance (Top 20)")
        fi_df = pd.DataFrame(detail_run["feature_importance"], columns=["Feature", "Importance"])
        fig3 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig3.update_layout(title="Top Contributing Features", height=420, yaxis=dict(autorange="reversed"))
        plot_retro(fig3, key="ae_compare_feature_importance")
    else:
        st.caption("Feature importance not available for this model type (e.g. SVR, LSTM/GRU). SHAP support is future work.")

    if detail_run.get("loss_history"):
        st.markdown("##### Training / Validation Loss")
        lh = detail_run["loss_history"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig4.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig4.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig4, key="ae_compare_loss")


def render_ae_feature_ablation_tab() -> None:
    st.caption(
        "Trains a Full Model (every feature group + every engineered group enabled) then retrains "
        "once per unit with just that one disabled — ranked by how much R² drops when removed. "
        "Leave-one-out, not a cumulative sweep, so results are order-independent."
    )
    model_type = st.selectbox("Model", ae_research.TABULAR_MODELS, key="ae_ablation_model")
    horizon = _ae_research_horizon_selector("ae_ablation")
    if st.button("🔬 Run Feature Ablation Sweep", key="ae_ablation_run_btn", type="primary"):
        n_units = len(ae_research.FEATURE_ABLATION_UNITS) + 1
        with st.spinner(f"Training {n_units} models ({model_type}, +{horizon}h)..."):
            try:
                result = ae_research.run_feature_ablation_sweep(model_type, horizon=horizon)
                st.session_state["ae_ablation_result"] = result
                st.toast("Feature ablation sweep complete.")
            except Exception as exc:
                st.error(f"Ablation sweep failed: {exc}")

    result = st.session_state.get("ae_ablation_result")
    if not result:
        st.info("Pick a model and horizon above, then run the sweep.")
        return

    st.markdown(f"##### {result['model_type']} +{result['horizon']}h — Full Model R² = {result['full_r2']:.4f}")
    ranked = result["ranked"]
    fig = go.Figure(
        go.Bar(
            x=[r["delta_r2"] for r in ranked],
            y=[r["unit"] for r in ranked],
            orientation="h",
            marker_color=["#1f7a3a" if r["delta_r2"] >= 0 else "#7a1f1f" for r in ranked],
        )
    )
    fig.update_layout(
        title="R² Drop When Removed — Ranked Feature Group Contribution",
        height=380,
        xaxis_title="ΔR² (Full Model − Without This Group)",
        yaxis=dict(autorange="reversed"),
    )
    plot_retro(fig, key="ae_ablation_bar")

    table_rows = [
        {
            "Rank": i + 1,
            "Feature Group": r["unit"].replace("Without ", ""),
            "R² Without": round(r["r2"], 4),
            "ΔR² (contribution)": round(r["delta_r2"], 4),
            "MAE Without": round(r["mae"], 4),
        }
        for i, r in enumerate(ranked)
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Swept at {pd.Timestamp(result['swept_at']).strftime('%Y-%m-%d %H:%M UTC')} — a positive ΔR² "
        "means removing that group made the model worse (it was contributing)."
    )


def render_ae_physics_experiments_tab() -> None:
    st.caption(
        "Exploratory analysis: how does each individual physics-derived feature relate to AE's "
        "behavior historically? Enable one to inspect it, or quick-train a model with just that one "
        "feature added on top of the full default feature set."
    )
    label = st.selectbox("Physics feature", ae_research.PHYSICS_FEATURE_OPTIONS, key="ae_physics_feature_select")

    try:
        base_frame, base_cols = _cached_ae_research_frame()
        frame, cols_with = _cached_ae_research_frame(physics_features={label: True})
    except Exception as exc:
        st.warning(f"Could not load the research feature frame: {exc}")
        return

    new_cols = [c for c in cols_with if c not in base_cols]
    if not new_cols:
        st.warning("Could not resolve this feature's column name.")
        return
    col = new_cols[-1]

    latest = frame[col].dropna()
    if latest.empty:
        st.info("Not enough history to compute this feature yet.")
        return
    metric_card(label, f"{latest.iloc[-1]:.3f}", f"Latest hourly value ({latest.index[-1].strftime('%Y-%m-%d %H:%M UTC')})")

    recent = frame.tail(24 * 30).dropna(subset=[col, "ae"])
    if not recent.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=recent.index, y=recent["ae"], name="AE", line=dict(color="#1f5a7a")), secondary_y=False)
        fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=label, line=dict(color="#7a1f5a")), secondary_y=True)
        fig.update_layout(title=f"AE vs. {label} — last {len(recent)} hours", height=380)
        fig.update_yaxes(title_text="AE (nT)", secondary_y=False)
        fig.update_yaxes(title_text=label, secondary_y=True)
        plot_retro(fig, key=f"ae_physics_timeseries_{col}")

    next_ae = frame["ae"].shift(-1)
    valid = frame[[col]].join(next_ae.rename("next_ae")).dropna()
    if len(valid) > 10:
        corr = valid[col].corr(valid["next_ae"])
        st.caption(f"Correlation between {label} and next-hour AE, across the full history: **{corr:.3f}**")
        fig2 = go.Figure(go.Scattergl(x=valid[col], y=valid["next_ae"], mode="markers", marker=dict(size=3, opacity=0.4)))
        fig2.update_layout(title=f"{label} vs. Next-Hour AE", height=380, xaxis_title=label, yaxis_title="Next-hour AE (nT)")
        plot_retro(fig2, key=f"ae_physics_corr_{col}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button(f"🧪 Quick-Train Linear Regression with {label} added", key="ae_physics_quick_train"):
        with st.spinner("Training..."):
            try:
                run = ae_research.train_ae_research_model(
                    "Linear Regression", physics_features={label: True}, notes=f"Physics Experiments — quick test of {label}"
                )
                st.success(
                    f"Trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.3f} "
                    "(compare against the Feature Ablation / Model Comparison full-model baseline)."
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def render_ae_sequence_models_tab() -> None:
    st.caption(
        "LSTM/GRU look back over a window of consecutive hours (Sequence Length) rather than a "
        "single row. Trains in an isolated subprocess — TensorFlow cannot safely share a process "
        "with this lab's scikit-learn/XGBoost/LightGBM/CatBoost imports."
    )
    if not ae_research.SEQUENCE_MODELS:
        st.warning("TensorFlow/Keras is not installed — sequence models are unavailable in this environment.")
        return

    model_type = st.selectbox("Model", ae_research.SEQUENCE_MODELS, key="ae_seq_model")
    horizon = _ae_research_horizon_selector("ae_seq")
    sequence_length = st.selectbox(
        "Sequence Length (hours)",
        ae_research.SEQUENCE_LENGTH_OPTIONS,
        index=ae_research.SEQUENCE_LENGTH_OPTIONS.index(ae_research.DEFAULT_SEQUENCE_LENGTH),
        key="ae_seq_seqlen",
    )
    with st.expander("Feature Groups", expanded=False):
        feature_toggles = _ae_research_feature_toggle_form("ae_seq")
    hyperparams = _ae_research_hyperparam_inputs(model_type, "ae_seq")

    if st.button("🧪 Train Sequence Model", key="ae_seq_train_btn", type="primary"):
        with st.spinner(f"Training {model_type} (seq={sequence_length}h) — isolated subprocess, ~10-30s..."):
            try:
                run = ae_research.train_ae_research_model(
                    model_type,
                    horizon=horizon,
                    feature_toggles=feature_toggles,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes="Sequence Models tab",
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    all_runs = ae_research.list_runs()
    seq_runs = [r for r in all_runs if r["model_type"] in ae_research.SEQUENCE_MODELS]
    tabular_runs = [r for r in all_runs if r["model_type"] not in ae_research.SEQUENCE_MODELS]
    if not seq_runs:
        st.info("No LSTM/GRU runs yet — train one above.")
        return

    best_seq = max(seq_runs, key=lambda r: r["metrics"]["r2"])
    if tabular_runs:
        best_tabular = max(tabular_runs, key=lambda r: r["metrics"]["r2"])
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Best Sequence Model R²", f"{best_seq['metrics']['r2']:.4f}", best_seq["model_type"])
        with c2:
            metric_card("Best Tabular Model R²", f"{best_tabular['metrics']['r2']:.4f}", best_tabular["model_type"])
        with c3:
            diff = best_seq["metrics"]["r2"] - best_tabular["metrics"]["r2"]
            metric_card("Sequence Advantage", f"{diff:+.4f}", "Positive = sequence models outperform tabular")

    def _run_label(r):
        return f"{r['model_type']} seq={r['sequence_length']}h +{r.get('horizon', 1)}h ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in seq_runs]
    selected = st.selectbox("Inspect run", labels, key="ae_seq_inspect")
    run = seq_runs[labels.index(selected)]
    m = run["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("R²", f"{m['r2']:.4f}", "")
    with c2:
        metric_card("MAE", f"{m['mae']:.3f}", "")
    with c3:
        metric_card("RMSE", f"{m['rmse']:.3f}", "")
    with c4:
        metric_card("MAPE", "N/A" if m["mape"] is None else f"{m['mape']:.1f}%", "")
    with c5:
        metric_card("Bias", f"{m['bias']:+.3f}", "")

    if run.get("loss_history"):
        lh = run["loss_history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig, key="ae_seq_loss")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### All Sequence Model Runs")
    for r in seq_runs:
        _ae_research_run_row(r, key_prefix="ae_seq")


def render_ae_horizon_analysis_tab() -> None:
    st.caption(
        "Automatically trains (or reuses existing runs for) one model per horizon (1h/2h/3h — the "
        "full range this lab supports, see the lab's own note on why sub-hourly horizons aren't "
        "offered) and plots how R²/MAE/RMSE change with forecast lead time."
    )
    model_type = st.selectbox("Model", ae_research.TABULAR_MODELS, key="ae_horizon_model")
    reuse_existing = st.checkbox("Reuse existing runs where available (faster)", value=True, key="ae_horizon_reuse")

    if st.button("📈 Run Horizon Sweep", key="ae_horizon_sweep_btn", type="primary"):
        with st.spinner(f"Sweeping {model_type} across {len(ae_research.HORIZON_OPTIONS)} horizons..."):
            try:
                runs = ae_research.train_horizon_sweep(model_type, reuse_existing=reuse_existing)
                st.session_state["ae_horizon_sweep_result"] = {
                    "model_type": model_type,
                    "run_ids": [r["run_id"] for r in runs],
                }
                st.toast(f"Swept {len(runs)} horizons.")
            except Exception as exc:
                st.error(f"Horizon sweep failed: {exc}")

    result = st.session_state.get("ae_horizon_sweep_result")
    if not result:
        st.info("Pick a model above, then run the sweep.")
        return

    runs = [ae_research.get_run(rid) for rid in result["run_ids"]]
    runs = [r for r in runs if r is not None]
    if not runs:
        st.warning("Swept runs are no longer available (were they deleted?). Run the sweep again.")
        return
    runs = sorted(runs, key=lambda r: r.get("horizon", 1))

    horizon_labels = [f"{r.get('horizon', 1)}h" for r in runs]
    r2_values = [r["metrics"]["r2"] for r in runs]
    mae_values = [r["metrics"]["mae"] for r in runs]
    rmse_values = [r["metrics"]["rmse"] for r in runs]

    st.markdown(f"##### {result['model_type']} — AE skill decay with lead time")
    fig_r2 = go.Figure(go.Scatter(x=horizon_labels, y=r2_values, mode="lines+markers", name="R²"))
    fig_r2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_r2.update_layout(title="Horizon vs. R²", height=340, xaxis_title="Forecast Horizon", yaxis_title="R²")
    plot_retro(fig_r2, key="ae_horizon_r2")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_mae = go.Figure(go.Scatter(x=horizon_labels, y=mae_values, mode="lines+markers", name="MAE"))
        fig_mae.update_layout(title="Horizon vs. MAE", height=320, xaxis_title="Forecast Horizon", yaxis_title="MAE")
        plot_retro(fig_mae, key="ae_horizon_mae")
    with col_b:
        fig_rmse = go.Figure(go.Scatter(x=horizon_labels, y=rmse_values, mode="lines+markers", name="RMSE"))
        fig_rmse.update_layout(title="Horizon vs. RMSE", height=320, xaxis_title="Forecast Horizon", yaxis_title="RMSE")
        plot_retro(fig_rmse, key="ae_horizon_rmse")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    table_rows = [
        {
            "Horizon": horizon_labels[i],
            "R²": round(r2_values[i], 4),
            "MAE": round(mae_values[i], 4),
            "RMSE": round(rmse_values[i], 4),
            "Test Samples": runs[i]["n_test_samples"],
        }
        for i in range(len(runs))
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def render_ae_experiment_tracking_tab() -> None:
    st.caption(
        "Every experiment ever run in this lab — full configuration and results, for reproducibility. "
        "Filter, inspect, promote, or delete any run."
    )
    runs = ae_research.list_runs()
    if not runs:
        st.info("No experiments recorded yet — train a model in any other tab.")
        return

    model_filter = st.selectbox("Filter by model", ["All"] + ae_research.ALL_TRAINABLE_MODELS, key="ae_track_model_filter")
    filtered = runs if model_filter == "All" else [r for r in runs if r["model_type"] == model_filter]
    st.caption(f"{len(filtered)} of {len(runs)} total experiments")

    rows = []
    for r in filtered:
        feat_summary = ", ".join(g for g, cols in r.get("feature_toggles", {}).items() if all(cols.values())) or "partial"
        physics_on = ", ".join(k for k, v in (r.get("physics_features") or {}).items() if v) or "None"
        rows.append(
            {
                "Timestamp": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
                "Model": r["model_type"],
                "Horizon": f"{r.get('horizon', 1)}h",
                "Feature Groups (fully on)": feat_summary,
                "Physics Features": physics_on,
                "Seq Len": str(r["sequence_length"]) + "h" if r.get("sequence_length") else "—",
                "Train N": r["n_train_samples"],
                "Test N": r["n_test_samples"],
                "R²": round(r["metrics"]["r2"], 4),
                "MAE": round(r["metrics"]["mae"], 4),
                "RMSE": round(r["metrics"]["rmse"], 4),
                "MAPE (%)": round(r["metrics"]["mape"], 2) if r["metrics"]["mape"] is not None else None,
                "Bias": round(r["metrics"]["bias"], 4),
                "Train Time (s)": round(r.get("training_time_sec", 0), 3),
                "Notes": r.get("notes", ""),
                "Promoted": "🚀" if r.get("promoted") else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Manage Runs")
    for r in filtered[:15]:
        _ae_research_run_row(r, key_prefix="ae_track")
    if len(filtered) > 15:
        st.caption(f"Showing 15 most recent of {len(filtered)}.")


def render_ae_hypothesis_testing_tab() -> None:
    st.caption(
        "Fixed, reproducible hypotheses. Most train a baseline WITHOUT the tested feature and an "
        "experimental run WITH it; 'Akasofu Epsilon outperforms Ey' is a head-to-head SWAP instead "
        "(one feature replaces the other, not an additive test). Reports ΔR²/ΔMAE/ΔRMSE and an "
        f"Accept/Reject verdict (Accept requires ΔR² ≥ {ae_research.HYPOTHESIS_ACCEPT_THRESHOLD_R2})."
    )
    hypothesis_label = st.selectbox("Hypothesis", list(ae_research.HYPOTHESIS_DEFINITIONS), key="ae_hyp_select")
    model_type = st.selectbox("Model", ae_research.TABULAR_MODELS, key="ae_hyp_model")
    horizon = _ae_research_horizon_selector("ae_hyp")

    if st.button("🔬 Run Hypothesis Test", key="ae_hyp_run_btn", type="primary"):
        with st.spinner(f"Testing: {hypothesis_label}..."):
            try:
                result = ae_research.run_hypothesis_test(hypothesis_label, model_type, horizon=horizon)
                st.session_state["ae_hyp_last_result"] = result
                st.toast(f"{result['verdict']} — ΔR²={result['delta_r2']:+.4f}")
            except Exception as exc:
                st.error(f"Hypothesis test failed: {exc}")

    result = st.session_state.get("ae_hyp_last_result")
    if result and result["hypothesis"] == hypothesis_label:
        color = "#1f7a3a" if result["verdict"] == "Accept" else "#7a1f1f"
        st.markdown(f"### Verdict: <span style='color:{color}'>{result['verdict']}</span>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("ΔR²", f"{result['delta_r2']:+.4f}", "")
        with c2:
            metric_card("ΔMAE", f"{result['delta_mae']:+.4f}", "")
        with c3:
            metric_card("ΔRMSE", f"{result['delta_rmse']:+.4f}", "")
        with c4:
            metric_card("Baseline → Experimental R²", f"{result['baseline_r2']:.4f} → {result['experimental_r2']:.4f}", "")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Hypothesis Test History")
    results = ae_research.list_hypothesis_results()
    if not results:
        st.info("No hypothesis tests run yet.")
        return
    rows = [
        {
            "Hypothesis": r["hypothesis"],
            "Model": r["model_type"],
            "Horizon": f"{r.get('horizon', 1)}h",
            "Verdict": r["verdict"],
            "ΔR²": round(r["delta_r2"], 4),
            "ΔMAE": round(r["delta_mae"], 4),
            "ΔRMSE": round(r["delta_rmse"], 4),
            "Tested": pd.Timestamp(r["tested_at"]).strftime("%Y-%m-%d %H:%M UTC"),
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_ae_research_laboratory() -> None:
    st.markdown("### 🧪 AE Research Laboratory")
    hdr_col, pause_col = st.columns([3, 1])
    with hdr_col:
        st.caption(
            "A scientific experimentation platform for AE forecasting — fully isolated from the "
            "Production Prediction tab, which continues using the current trained production model "
            "exactly as-is. Nothing trained here overwrites it; Promote only labels a run for your own "
            "tracking, wiring a model into production is always a manual, deliberate step."
        )
    with pause_col:
        st.session_state.setdefault("pause_autorefresh", False)
        if st.toggle("⏸ Pause Live Refresh", value=st.session_state["pause_autorefresh"], key="ae_lab_pause_toggle"):
            st.session_state["pause_autorefresh"] = True
        else:
            st.session_state["pause_autorefresh"] = False
    _ae_research_notes()
    with st.expander("Future Research — planned architecture extensions"):
        st.write(", ".join(ae_research.FUTURE_MODELS))
        st.caption(
            "Registered as disabled entries in the model selector now, so adding a real "
            "implementation later never requires redesigning this interface."
        )

    sub = st.tabs(
        [
            "Model Comparison",
            "Feature Ablation",
            "Physics Experiments",
            "Sequence Models",
            "Horizon Analysis",
            "Experiment Tracking",
            "Hypothesis Testing",
        ]
    )
    with sub[0]:
        render_ae_model_comparison_tab()
    with sub[1]:
        render_ae_feature_ablation_tab()
    with sub[2]:
        render_ae_physics_experiments_tab()
    with sub[3]:
        render_ae_sequence_models_tab()
    with sub[4]:
        render_ae_horizon_analysis_tab()
    with sub[5]:
        render_ae_experiment_tracking_tab()
    with sub[6]:
        render_ae_hypothesis_testing_tab()


_HYPOTHESIS_DATASET_OPTIONS = ["analytics", "ae", "experimental"]
_HYPOTHESIS_VARIABLE_OPTIONS = ["kp", "dst", "ae"]


def _hypothesis_architecture_form(prefix: str, defaults: dict = None) -> dict:
    """Shared baseline/experimental dataset+variable picker, used by both
    the create form and the edit form.
    """
    defaults = defaults or {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Architecture**")
        baseline_dataset = st.selectbox(
            "Baseline Dataset",
            _HYPOTHESIS_DATASET_OPTIONS,
            index=_HYPOTHESIS_DATASET_OPTIONS.index(defaults.get("baseline_dataset", "analytics")),
            key=f"{prefix}_baseline_ds",
        )
        baseline_variable = st.selectbox(
            "Baseline Variable",
            _HYPOTHESIS_VARIABLE_OPTIONS,
            index=_HYPOTHESIS_VARIABLE_OPTIONS.index(defaults.get("baseline_variable", "kp")),
            key=f"{prefix}_baseline_var",
        )
    with col2:
        st.markdown("**Experimental Architecture**")
        experimental_dataset = st.selectbox(
            "Experimental Dataset",
            _HYPOTHESIS_DATASET_OPTIONS,
            index=_HYPOTHESIS_DATASET_OPTIONS.index(defaults.get("experimental_dataset", "experimental")),
            key=f"{prefix}_exp_ds",
        )
        experimental_variable = st.selectbox(
            "Experimental Variable",
            _HYPOTHESIS_VARIABLE_OPTIONS,
            index=_HYPOTHESIS_VARIABLE_OPTIONS.index(defaults.get("experimental_variable", "kp")),
            key=f"{prefix}_exp_var",
        )
    return {
        "baseline_dataset": baseline_dataset,
        "baseline_variable": baseline_variable,
        "experimental_dataset": experimental_dataset,
        "experimental_variable": experimental_variable,
    }


def render_hypothesis_testing_tab() -> None:
    """Experiment management and evaluation system — not a prediction
    page. Every hypothesis pairs a baseline (dataset, variable) against
    an experimental one; conclusions and confidence come entirely from
    swdss.models.hypothesis's fixed statistical rules over measured,
    verified predictions — never an LLM, never a claim of "true."
    """
    st.caption(
        "Experiment management and evaluation system — not a prediction page. Every hypothesis "
        "compares a baseline architecture against an experimental one, using only measured, "
        "verified prediction results. Conclusions are **Supported / Not Supported / Inconclusive** "
        "— never claimed as \"true\" — generated entirely from fixed statistical rules. No LLM."
    )

    with st.expander("➕ Create New Hypothesis"):
        with st.form("new_hypothesis_form", clear_on_submit=True):
            title = st.text_input("Title")
            description = st.text_area("Description", height=70)
            motivation = st.text_area("Scientific Motivation", height=70)
            physics_bg = st.text_area("Physics Background", height=70)
            expected = st.text_input("Expected Improvement")
            arch = _hypothesis_architecture_form("new_hyp")
            notes = st.text_area("Initial Notes (markdown supported)", height=90)

            if st.form_submit_button("Create Hypothesis"):
                if not title:
                    st.error("Title is required.")
                else:
                    created = create_hypothesis(
                        title=title,
                        description=description,
                        scientific_motivation=motivation,
                        physics_background=physics_bg,
                        expected_improvement=expected,
                        notes=notes,
                        **arch,
                    )
                    st.toast(f"Created hypothesis: {created['title']}")
                    st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    show_archived = st.checkbox("Show archived hypotheses", key="hyp_show_archived")
    hyps = list_hypotheses() if show_archived else list_hypotheses(status="active")

    st.markdown("##### Hypotheses")
    if not hyps:
        st.info("No hypotheses yet. Create one above.")
        return

    for h in hyps:
        result = evaluate_hypothesis(h)
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([0.45, 0.18, 0.22, 0.15])
            with c1:
                number_text = f"#{h['number']} — " if h["number"] else ""
                archived_tag = " *(archived)*" if h["status"] == "archived" else ""
                st.markdown(f"**{number_text}{h['title']}**{archived_tag}")
                st.caption(h["description"] or "No description.")
                st.caption(
                    f"Baseline: `{h['baseline_dataset']}/{h['baseline_variable']}` vs. "
                    f"Experimental: `{h['experimental_dataset']}/{h['experimental_variable']}` — "
                    f"created {pd.Timestamp(h['created_at']).strftime('%Y-%m-%d')}"
                )
            with c2:
                metric_card("Verified", str(result["n"]), f"of {result['baseline']['count'] + result['experimental']['count']} total")
            with c3:
                metric_card(
                    "Conclusion",
                    result["conclusion"],
                    f"Confidence: {result['confidence']}",
                    value_color=CONCLUSION_COLORS.get(result["conclusion"], "#404040"),
                )
            with c4:
                st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
                if st.button("View", key=f"view_hyp_{h['hypothesis_id']}", use_container_width=True):
                    open_dialog("hypothesis_detail", h["hypothesis_id"])


@st.dialog("Hypothesis Detail", width="large", dismissible=False)
def show_hypothesis_detail(hypothesis_id: str) -> None:
    render_dialog_close_button("close_hypothesis_detail")

    h = get_hypothesis(hypothesis_id)
    if h is None:
        st.error("This hypothesis could not be found.")
        return

    result = evaluate_hypothesis(h)
    baseline, experimental = result["baseline"], result["experimental"]

    number_text = f"Hypothesis {h['number']} — " if h["number"] else ""
    st.subheader(f"{number_text}{h['title']}")
    st.caption(
        f"Created {pd.Timestamp(h['created_at']).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Status: {h['status'].title()}"
    )

    with st.expander("✏️ Edit Hypothesis Structure"):
        with st.form(f"edit_hyp_{hypothesis_id}"):
            edit_title = st.text_input("Title", value=h["title"])
            edit_description = st.text_area("Description", value=h["description"] or "", height=70)
            edit_motivation = st.text_area("Scientific Motivation", value=h["scientific_motivation"] or "", height=70)
            edit_physics_bg = st.text_area("Physics Background", value=h["physics_background"] or "", height=70)
            edit_expected = st.text_input("Expected Improvement", value=h["expected_improvement"] or "")
            edit_arch = _hypothesis_architecture_form(f"edit_hyp_{hypothesis_id}", defaults=h)
            if st.form_submit_button("Save Changes"):
                update_hypothesis(
                    hypothesis_id,
                    title=edit_title,
                    description=edit_description,
                    scientific_motivation=edit_motivation,
                    physics_background=edit_physics_bg,
                    expected_improvement=edit_expected,
                    **edit_arch,
                )
                st.toast("Hypothesis updated.")
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Hypothesis Structure")
    st.markdown(f"**Description:** {h['description'] or 'N/A'}")
    st.markdown(f"**Scientific Motivation:** {h['scientific_motivation'] or 'N/A'}")
    st.markdown(f"**Physics Background:** {h['physics_background'] or 'N/A'}")
    st.markdown(f"**Expected Improvement:** {h['expected_improvement'] or 'N/A'}")
    st.markdown(f"**Baseline Architecture:** `{h['baseline_dataset']}` / `{h['baseline_variable']}`")
    st.markdown(f"**Experimental Architecture:** `{h['experimental_dataset']}` / `{h['experimental_variable']}`")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Automatic Conclusion")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Conclusion", result["conclusion"], "", value_color=CONCLUSION_COLORS.get(result["conclusion"], "#404040"))
    with c2:
        metric_card("Confidence", result["confidence"], f"{result['n']} verified predictions")
    st.info(result["summary"])
    if h["manual_conclusion"]:
        st.markdown("**Researcher's Manual Conclusion / Addendum:**")
        st.markdown(h["manual_conclusion"])

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Comparison Table")

    def _fmt(v, dp=3):
        return "N/A" if v is None else f"{v:.{dp}f}"

    def _improvement(b, e, higher_is_better=False):
        if b is None or e is None:
            return "N/A"
        if higher_is_better:
            return f"{e - b:+.3f}"
        if b == 0:
            return "N/A"
        return f"{(b - e) / abs(b) * 100:+.1f}%"

    comparison_rows = [
        {"Metric": "MAE", "Baseline": _fmt(baseline["mae"]), "Experimental": _fmt(experimental["mae"]), "Improvement": _improvement(baseline["mae"], experimental["mae"])},
        {"Metric": "RMSE", "Baseline": _fmt(baseline["rmse"]), "Experimental": _fmt(experimental["rmse"]), "Improvement": _improvement(baseline["rmse"], experimental["rmse"])},
        {"Metric": "R²", "Baseline": _fmt(baseline["r2"], 3), "Experimental": _fmt(experimental["r2"], 3), "Improvement": _improvement(baseline["r2"], experimental["r2"], higher_is_better=True)},
        {"Metric": "MAPE (%)", "Baseline": _fmt(baseline["mape"], 1), "Experimental": _fmt(experimental["mape"], 1), "Improvement": _improvement(baseline["mape"], experimental["mape"])},
        {"Metric": "Bias", "Baseline": _fmt(baseline["bias"]), "Experimental": _fmt(experimental["bias"]), "Improvement": "N/A"},
        {"Metric": "Max Error", "Baseline": _fmt(baseline["max_error"]), "Experimental": _fmt(experimental["max_error"]), "Improvement": _improvement(baseline["max_error"], experimental["max_error"])},
        {"Metric": "Median Error", "Baseline": _fmt(baseline["median_error"]), "Experimental": _fmt(experimental["median_error"]), "Improvement": _improvement(baseline["median_error"], experimental["median_error"])},
        {"Metric": "Average Drift", "Baseline": _fmt(baseline["avg_drift"]), "Experimental": _fmt(experimental["avg_drift"]), "Improvement": "N/A"},
        {"Metric": "Forecast Stability (std)", "Baseline": _fmt(baseline["stability"]), "Experimental": _fmt(experimental["stability"]), "Improvement": "N/A"},
        {
            "Metric": "Storm-Time MAE",
            "Baseline": f"{_fmt(baseline['storm_mae'])} (n={baseline['storm_count']})",
            "Experimental": f"{_fmt(experimental['storm_mae'])} (n={experimental['storm_count']})",
            "Improvement": _improvement(baseline["storm_mae"], experimental["storm_mae"]),
        },
        {
            "Metric": "Quiet-Time MAE",
            "Baseline": f"{_fmt(baseline['quiet_mae'])} (n={baseline['quiet_count']})",
            "Experimental": f"{_fmt(experimental['quiet_mae'])} (n={experimental['quiet_count']})",
            "Improvement": _improvement(baseline["quiet_mae"], experimental["quiet_mae"]),
        },
    ]
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Visualizations")

    if baseline["errors"] or experimental["errors"]:
        v1, v2 = st.columns(2)
        with v1:
            fig = go.Figure()
            if baseline["errors"]:
                fig.add_trace(go.Histogram(x=baseline["errors"], name="Baseline", opacity=0.6))
            if experimental["errors"]:
                fig.add_trace(go.Histogram(x=experimental["errors"], name="Experimental", opacity=0.6))
            fig.update_layout(title="Prediction Error Histogram", barmode="overlay", height=320)
            plot_retro(fig)
        with v2:
            fig2 = go.Figure()
            if baseline["predicted_vs_actual"]:
                bp, ba = zip(*baseline["predicted_vs_actual"])
                fig2.add_trace(go.Scatter(x=ba, y=bp, mode="markers", name="Baseline"))
            if experimental["predicted_vs_actual"]:
                ep, ea = zip(*experimental["predicted_vs_actual"])
                fig2.add_trace(go.Scatter(x=ea, y=ep, mode="markers", name="Experimental"))
            fig2.update_layout(title="Predicted vs. Official", height=320, xaxis_title="Official", yaxis_title="Predicted")
            plot_retro(fig2)

        v3, v4 = st.columns(2)
        with v3:
            fig3 = go.Figure()
            if baseline["predicted_vs_actual"]:
                bp, ba = zip(*baseline["predicted_vs_actual"])
                fig3.add_trace(go.Scatter(x=list(ba), y=[p - a for p, a in zip(bp, ba)], mode="markers", name="Baseline"))
            if experimental["predicted_vs_actual"]:
                ep, ea = zip(*experimental["predicted_vs_actual"])
                fig3.add_trace(go.Scatter(x=list(ea), y=[p - a for p, a in zip(ep, ea)], mode="markers", name="Experimental"))
            fig3.update_layout(title="Residual Plot", height=320, xaxis_title="Official", yaxis_title="Residual (Predicted - Official)")
            plot_retro(fig3)
        with v4:
            fig4 = go.Figure()
            if baseline["trend"]:
                times, errs = zip(*baseline["trend"])
                fig4.add_trace(go.Scatter(x=[pd.Timestamp(t) for t in times], y=errs, mode="lines+markers", name="Baseline"))
            if experimental["trend"]:
                times_e, errs_e = zip(*experimental["trend"])
                fig4.add_trace(go.Scatter(x=[pd.Timestamp(t) for t in times_e], y=errs_e, mode="lines+markers", name="Experimental"))
            fig4.update_layout(title="Performance Timeline (Absolute Error)", height=320)
            plot_retro(fig4)

        st.markdown("###### Storm vs. Quiet Performance (MAE)")
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=["Storm", "Quiet"], y=[baseline["storm_mae"], baseline["quiet_mae"]], name="Baseline"))
        fig5.add_trace(go.Bar(x=["Storm", "Quiet"], y=[experimental["storm_mae"], experimental["quiet_mae"]], name="Experimental"))
        fig5.update_layout(height=320, barmode="group", yaxis_title="MAE")
        plot_retro(fig5)
    else:
        st.info("No verified predictions yet for either architecture — visualizations will appear once forecasts complete and are verified.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Experiment Timeline")
    t1, t2, t3 = st.columns(3)
    with t1:
        metric_card("Experiment Started", pd.Timestamp(h["created_at"]).strftime("%Y-%m-%d %H:%M UTC"), "")
    with t2:
        metric_card("Predictions Generated", str(baseline["count"] + experimental["count"]), "Baseline + Experimental")
    with t3:
        metric_card("Current Status", h["status"].title(), "")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Research Notes")
    notes_text = st.text_area("Notes (markdown supported)", value=h["notes"] or "", height=150, key=f"notes_{hypothesis_id}")
    manual_conclusion_text = st.text_area(
        "Manual Conclusion / Addendum (optional)", value=h["manual_conclusion"] or "", height=70, key=f"manual_{hypothesis_id}"
    )
    if st.button("💾 Save Notes", key=f"save_notes_{hypothesis_id}", use_container_width=True):
        update_notes(hypothesis_id, notes_text)
        update_manual_conclusion(hypothesis_id, manual_conclusion_text)
        st.toast("Notes saved.")
        st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    dup_col, archive_col, delete_col = st.columns(3)
    with dup_col:
        if st.button("📋 Duplicate Experiment", key=f"dup_{hypothesis_id}", use_container_width=True):
            new_h = duplicate_hypothesis(hypothesis_id)
            st.toast(f"Duplicated as: {new_h['title']}")
            close_active_dialog()
    with archive_col:
        if h["status"] == "active":
            if st.button("🗄️ Archive", key=f"archive_{hypothesis_id}", use_container_width=True):
                archive_hypothesis(hypothesis_id)
                st.toast("Archived.")
                st.rerun()
        else:
            if st.button("♻️ Reactivate", key=f"reactivate_{hypothesis_id}", use_container_width=True):
                reactivate_hypothesis(hypothesis_id)
                st.toast("Reactivated.")
                st.rerun()
    with delete_col:
        if st.button("🗑️ Delete", key=f"delete_hyp_{hypothesis_id}", use_container_width=True):
            delete_hypothesis(hypothesis_id)
            close_active_dialog()


def research_lab_page(df: pd.DataFrame) -> None:
    st.title("Space Weather Research Lab")
    st.caption(
        "Experimental environment for researchers, students, and developers to evaluate different "
        "space weather forecasting architectures. Nothing here affects the Production Prediction "
        "models — every experiment is isolated from the production pipeline."
    )

    tabs = st.tabs(["Forecasting Architectures", "Physics Interpretation", "Hypothesis Testing"])
    with tabs[0]:
        st.caption(
            "Compare **Independent Models** (each variable predicted separately from observed "
            "history) against **Physics Cascaded Models** (Predicted AE fed forward into Kp/Dst), "
            "using identical live NOAA observations."
        )
        inner = st.tabs(
            ["Independent Models", "Physics Cascaded Models", "Model Comparison", "Prediction Pipeline"]
        )
        with inner[0]:
            render_independent_models_tab()
        with inner[1]:
            render_physics_cascaded_tab()
        with inner[2]:
            render_model_comparison_tab()
        with inner[3]:
            render_prediction_pipeline_tab()
    with tabs[1]:
        render_physics_interpretation_panel(df)
    with tabs[2]:
        render_hypothesis_testing_tab()


apply_retro_windows_style()

if not MASTER_PATH.exists():
    st.error(f"Master file not found: {MASTER_PATH}")
    st.code("PYTHONPATH=src python -m swdss.features.build_master")
    st.stop()

master_df = load_master_data(MASTER_PATH)
df_7d = seven_day_window(master_df)

nav_col, terminal_col = st.columns([1.15, 1])

with nav_col:
    st.markdown("### Navigation")
    page = st.radio(
        "Main sections",
        ["Home Page", "Photosphere", "Heliosphere", "Geospace", "Analytics", "Research Lab"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Home Page":
        with st.container(horizontal=True, gap="small"):
            if st.button("↻ Refresh"):
                st.cache_data.clear()
                st.rerun()
            if st.button("📚 Space Weather Concepts", key="library_open_button"):
                open_dialog("library", None)
    else:
        if st.button("↻ Refresh"):
            st.cache_data.clear()
            st.rerun()

# Auto-refresh fires on all live-data pages except:
#  - "Research Lab" top-level page (IMF/Bz lab) — fully user-driven, no feed
#  - When the user has toggled "Pause Live Refresh" inside a research lab tab
#    (session_state["pause_autorefresh"] = True) — stops the blink while
#    they are actively working in the Kp or AE lab. Toggle it off to resume.
_pause = st.session_state.get("pause_autorefresh", False)
if page != "Research Lab" and not _pause:
    auto_refresh()

with terminal_col:
    if page == "Home Page":
        status_terminal(df_7d)
    elif page == "Photosphere":
        photosphere_reference_window()
    elif page == "Heliosphere":
        heliosphere_reference_window()
    elif page == "Geospace":
        geospace_reference_window()

st.divider()

if page == "Home Page":
    st.sidebar.subheader("Data Freshness")

    sw_status, sw_age = freshness_status(df_7d, "solar_wind_speed", 90)
    imf_status, imf_age = freshness_status(df_7d, "bz", 90)
    kp_status, kp_age = freshness_status(df_7d, "kp", 360)
    dst_status, dst_age = freshness_status(df_7d, "dst", 120)

    solar_events_df = load_processed_data("solar_events")
    cme_df_sidebar = load_processed_data("cme")
    f107_df_sidebar = load_processed_data("f107")

    events_status, events_age = freshness_status(solar_events_df, "event_type", 90)
    cme_status, cme_age = freshness_status(cme_df_sidebar, "speed", 180)
    f107_status, f107_age = freshness_status(f107_df_sidebar, "f107_flux", 1500)

    st.sidebar.write(f"Solar Wind: {sw_status} ({sw_age})")
    st.sidebar.write(f"IMF: {imf_status} ({imf_age})")
    st.sidebar.write(f"Kp: {kp_status} ({kp_age})")
    st.sidebar.write(f"Dst: {dst_status} ({dst_age})")
    st.sidebar.write(f"Solar Events: {events_status} ({events_age})")
    st.sidebar.write(f"CME: {cme_status} ({cme_age})")
    st.sidebar.write(f"F10.7: {f107_status} ({f107_age})")

    st.sidebar.divider()

    st.sidebar.caption("Refresh logic")
    st.sidebar.write("Solar Wind / IMF: every minute")
    st.sidebar.write("Kp: about every 3 hours")
    st.sidebar.write("Dst: about every hour")
    st.sidebar.write("Solar Events: about every 30 minutes")
    st.sidebar.write("CME: about every hour")
    st.sidebar.write("F10.7: about every 24 hours")

    st.sidebar.divider()

    if st.sidebar.button("📁 Saved Events", key="saved_events_sidebar", use_container_width=True):
        open_dialog("saved_events", None)
else:
    if page == "Photosphere":
        if st.sidebar.button("🎬 Animations", key="animations_sidebar", use_container_width=True):
            open_dialog("animations_grid", None)

    if st.sidebar.button("📁 Saved Events", key="saved_events_sidebar_other", use_container_width=True):
        open_dialog("saved_events", None)

    st.sidebar.divider()

active_dialog = st.session_state.get("active_dialog")
if active_dialog is not None:
    kind, payload = active_dialog
    if kind == "event_explorer":
        show_event_explorer(payload)
    elif kind == "reverse_explorer":
        target_time, effect_label = payload
        show_reverse_event_explorer(target_time, effect_label)
    elif kind == "storyboard":
        play_event_animation(payload)
    elif kind == "saved_events":
        show_saved_events()
    elif kind == "animations_grid":
        show_animations_grid()
    elif kind == "library":
        show_space_weather_library()
    elif kind == "prediction_job":
        show_prediction_job(payload)
    elif kind == "saved_predictions":
        show_saved_predictions(payload)
    elif kind == "hypothesis_detail":
        show_hypothesis_detail(payload)

if page == "Home Page":
    home_page(df_7d)
elif page == "Photosphere":
    photosphere_page(df_7d)
elif page == "Heliosphere":
    heliosphere_page(df_7d)
elif page == "Geospace":
    geospace_page(df_7d)
elif page == "Analytics":
    analytics_page(df_7d)
else:
    research_lab_page(df_7d)
