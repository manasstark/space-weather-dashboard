import base64
import json
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


def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
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
    available = [col for col in columns if col in df.columns]
    corr_df = df[available].dropna()

    st.subheader(title)

    if len(available) < 2 or corr_df.empty:
        st.info("Not enough data for correlation analysis.")
        return

    corr = corr_df.corr(numeric_only=True)
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

    scatter_df = df[["timestamp_utc", x_col, y_col]].dropna()
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
    correlation_explorer(df, ["kp", "solar_wind_speed", "proton_density", "bz", "dst"], "Kp With Solar Wind, IMF, Dst")


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
    correlation_explorer(df, ["dst", "kp", "solar_wind_speed", "proton_density", "bz"], "Dst With Solar Wind, IMF, Kp")


def earth_analysis(df: pd.DataFrame) -> None:
    st.subheader("Combined Earth Analysis")

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
        inner = st.tabs(["Current Analysis", "Predictions"])
        with inner[0]:
            current_analysis_solar_wind(df)
        with inner[1]:
            st.info("Prediction module will be added later.")

    with tabs[1]:
        inner = st.tabs(["Current Analysis", "Predictions"])
        with inner[0]:
            current_analysis_imf(df)
        with inner[1]:
            st.info("Prediction module will be added later.")

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
        inner = st.tabs(["Current Analysis", "Predictions"])
        with inner[0]:
            current_analysis_kp(df)
        with inner[1]:
            st.info("Prediction module will be added later.")

    with tabs[1]:
        inner = st.tabs(["Current Analysis", "Predictions"])
        with inner[0]:
            current_analysis_dst(df)
        with inner[1]:
            st.info("Prediction module will be added later.")


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


def analytics_page(df: pd.DataFrame) -> None:
    st.title("Analytics")
    earth_analysis(df)


apply_retro_windows_style()
auto_refresh()

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
        ["Home Page", "Photosphere", "Heliosphere", "Geospace", "Analytics"],
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

if page == "Home Page":
    home_page(df_7d)
elif page == "Photosphere":
    photosphere_page(df_7d)
elif page == "Heliosphere":
    heliosphere_page(df_7d)
elif page == "Geospace":
    geospace_page(df_7d)
else:
    analytics_page(df_7d)
