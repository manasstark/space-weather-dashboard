import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from html import escape

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "features" / "master_df_v1.parquet"

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swdss import physics
from swdss.ingest.kyoto_ae_quicklook import annotate_quicklook_image, fetch_quicklook_image, quicklook_image_url
from swdss.models.physics_interpretation import physics_interpretation
from swdss.models.jobs import (
    classify_quicklook_error,
    get_job_stats,
    get_prediction_statistics,
    get_running_jobs,
    get_saved_jobs,
    poll_jobs,
    quicklook_error,
    quicklook_label,
    quicklook_relative_error,
    refresh_quicklook_estimate,
    start_job,
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
)

from dashboard.lib.ae_research_lab import (
    render_ae_research_laboratory,
    render_hypothesis_testing_tab,
    show_hypothesis_detail,
)
from dashboard.lib.command_centre import render_operational_command_centre
from dashboard.lib.data_helpers import (
    DATASET_LABELS,
    format_value,
    get_base64_image,
    latest_value,
    load_master_data,
    load_processed_data,
    nearest_row_in,
    QUICKLOOK_CONFIDENCE_COLORS,
    recent_window,
    row_at_extreme,
    row_at_extreme_from_source,
    seven_day_window,
    status_badge_html,
    time_caption,
)
from dashboard.lib.event_explorer import (
    _render_chain_box,
    play_event_animation,
    show_animations_grid,
    show_event_explorer,
    show_reverse_event_explorer,
    show_saved_events,
)
from dashboard.lib.forecast_dialogs import _experimental_badge_html, show_prediction_job
from dashboard.lib.imf_research_lab import render_imf_research_laboratory
from dashboard.lib.kp_research_lab import render_kp_research_laboratory
from dashboard.lib.library import show_space_weather_library
from dashboard.lib.project_status import render_project_status_page
from dashboard.lib.solar_activity import (
    cme_analysis,
    cme_predictions,
    f107_analysis,
    f107_predictions,
    heliomap_cme_tab,
    heliomap_solar_events_tab,
    solar_event_news_feed,
    solar_events_analysis,
)
from dashboard.lib.storm_lab import render_storm_backtest_tab, render_storm_learning_tab
from dashboard.lib.shared_ui import (
    apply_retro_chart_style,
    auto_refresh,
    metric_card,
    open_dialog,
    plot_retro,
    render_dialog_close_button,
    render_site_header,
    style_sticky_header,
    style_top_nav,
)


st.set_page_config(
    page_title="Space Weather DSS",
    layout="wide",
)


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


PHOTOSPHERE_REFERENCE_TABLES = [
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

HELIOSPHERE_REFERENCE_TABLES = [
    {
        "title": "Speed Reference",
        "columns": ["Range", "Meaning", "Risk"],
        "data": [
            ["< 400 km/s", "Slow solar wind", "Usually quiet"],
            ["400-500 km/s", "Moderate speed", "Normal/active"],
            ["500-700 km/s", "Fast solar wind", "Storm possible with southward Bz"],
            ["> 700 km/s", "Very fast wind", "Enhanced storm potential"],
        ],
    },
    {
        "title": "Density Reference",
        "columns": ["Range", "Meaning", "Risk"],
        "data": [
            ["< 5 p/cm3", "Low density", "Weak pressure"],
            ["5-10 p/cm3", "Moderate density", "Normal solar wind"],
            ["10-30 p/cm3", "High density", "Compression possible"],
            ["> 30 p/cm3", "Very high density", "Shock/CME sheath possible"],
        ],
    },
    {
        "title": "Temperature Reference",
        "columns": ["Range", "Meaning", "Risk"],
        "data": [
            ["< 50,000 K", "Cool wind", "Usually quiet"],
            ["50,000-150,000 K", "Typical wind", "Normal"],
            ["150,000-500,000 K", "Hot wind", "Disturbed flow possible"],
            ["> 500,000 K", "Very hot plasma", "Shock/CME heating possible"],
        ],
    },
    {
        "title": "Bz Reference",
        "columns": ["Range", "Meaning", "Risk"],
        "data": [
            ["Bz > 0 nT", "Northward IMF", "Low coupling"],
            ["0 to -5 nT", "Weak southward IMF", "Minor activity possible"],
            ["-5 to -10 nT", "Moderate southward IMF", "Storm possible"],
            ["-10 to -20 nT", "Strong southward IMF", "Strong storm coupling"],
            ["< -20 nT", "Extreme southward IMF", "Severe storm potential"],
        ],
    },
]

GEOSPACE_REFERENCE_TABLES = [
    {
        "title": "Kp Reference",
        "columns": ["Range", "Meaning", "Risk"],
        "data": [
            ["0-3", "Quiet", "Normal"],
            ["4", "Active", "Unsettled field"],
            ["5", "G1 storm", "Minor storm"],
            ["6", "G2 storm", "Moderate storm"],
            ["7", "G3 storm", "Strong storm"],
            ["8-9", "G4-G5 storm", "Severe/extreme storm"],
        ],
    },
    {
        "title": "Dst Reference",
        "columns": ["Range", "Meaning", "Risk"],
        "data": [
            ["Dst > -30 nT", "Quiet", "Low storm activity"],
            ["-30 to -50 nT", "Weak storm", "Minor ring current"],
            ["-50 to -100 nT", "Moderate storm", "Storm underway"],
            ["-100 to -200 nT", "Intense storm", "Strong disturbance"],
            ["< -200 nT", "Superstorm", "Extreme disturbance"],
        ],
    },
]


def _references_terminal_section_html(section_name: str, tables: list[dict]) -> str:
    """One section's worth of reference tables, formatted as aligned
    monospace columns (fixed-width padded so the terminal box stays
    readable without an actual HTML <table>).
    """
    html = f'<div style="margin-top:18px; color:#ffcc66; font-weight:800;">=== {escape(section_name)} ===</div>'
    for table in tables:
        columns = table["columns"]
        rows = table["data"]
        widths = [
            max(len(str(columns[i])), max((len(str(row[i])) for row in rows), default=0))
            for i in range(len(columns))
        ]
        header_line = "  ".join(str(columns[i]).ljust(widths[i]) for i in range(len(columns)))
        html += f'<div style="margin-top:10px; color:#9adfff;">&gt; {escape(table["title"])}</div>'
        html += f"<div>{escape(header_line)}</div>"
        html += f"<div>{'-' * len(header_line)}</div>"
        for row in rows:
            line = "  ".join(str(row[i]).ljust(widths[i]) for i in range(len(columns)))
            html += f"<div>{escape(line)}</div>"
    return html


@st.dialog("References", width="large", dismissible=False)
def show_references_terminal() -> None:
    """All Photosphere/Heliosphere/Geospace reference tables in one
    scrollable terminal window, opened from the "📋 References" toolbar
    button — replaces the three separate rotating reference-table panels
    that used to sit inline on those pages (UI simplification, 2026-07).
    """
    render_dialog_close_button("close_references")

    body_html = (
        _references_terminal_section_html("PHOTOSPHERE", PHOTOSPHERE_REFERENCE_TABLES)
        + _references_terminal_section_html("HELIOSPHERE", HELIOSPHERE_REFERENCE_TABLES)
        + _references_terminal_section_html("GEOSPACE", GEOSPACE_REFERENCE_TABLES)
    )

    st.markdown(
        f"""
        <style>
        .references-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 14px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
            height: 600px;
            box-sizing: border-box;
            overflow-y: auto;
        }}
        </style>
        <div class="references-terminal">
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render_architecture_status_panel(dataset: str) -> None:
    """Read-only counterpart to prediction_panel(), for the Research Lab's
    Forecasting Architectures tabs. Those tabs exist to COMPARE
    architectures using jobs that already exist — not to be a second place
    to start one. Reusing prediction_panel() there (as this project did
    before) put the exact same "Start Prediction" control in two separate
    navigation locations for the same underlying jobs, which read as a
    duplicated feature rather than a deliberate comparison view. This
    shows the same live queue stats and job tiles (including the same
    "Open" dialog every other job view uses) without a second Start
    Prediction button — new jobs are always started from Analytics/
    Heliosphere, and appear here automatically since it's the same
    dataset/job data underneath.
    """
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
    st.caption(
        "Start new predictions from the Analytics page — jobs started there appear below "
        "automatically, for comparison against the other architecture."
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Running Predictions")
    render_prediction_job_tiles(
        get_running_jobs(dataset),
        "No predictions running for this architecture right now.",
    )


@st.dialog("Saved Predictions", width="large", dismissible=False)
def show_saved_predictions(dataset: str) -> None:
    render_dialog_close_button("close_saved_predictions")

    label = DATASET_LABELS.get(dataset, dataset.title())
    st.subheader(f"{label} — Saved Predictions")

    jobs = get_saved_jobs(dataset)
    render_prediction_job_tiles(jobs, "No saved predictions yet. Save a completed job to keep it here permanently.")


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
    """Compute derived physics columns used by the Geospace scatter explorer.

    Ey/Dynamic Pressure/Clock Angle are computed via swdss.physics — the
    Physics Engine and this project's single canonical implementation of
    these formulas — rather than recomputed inline.
    """
    d = df.copy().sort_values("timestamp_utc").reset_index(drop=True)

    if "solar_wind_speed" in d.columns and "bz" in d.columns:
        d["ey"] = physics.ey_series(d["solar_wind_speed"], d["bz"])

    if "solar_wind_speed" in d.columns and "proton_density" in d.columns:
        d["dynamic_pressure"] = physics.dynamic_pressure_series(d["proton_density"], d["solar_wind_speed"])

    if "by" in d.columns and "bz" in d.columns:
        d["clock_angle"] = physics.clock_angle_series(d["by"], d["bz"])

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
        # Deliberately NOT swdss.physics.core.southward_duration_series
        # (a consecutive-streak count) — this is a distinct quantity, a
        # rolling COUNT of southward hours within the trailing 24h
        # window, regardless of whether they're consecutive. Kept as its
        # own formula so this column's displayed values don't change.
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
    st.caption("7-day NOAA-based summary. Page refreshes every minute.")

    render_overview_chart()

    st.divider()

    solar_event_news_feed()

    st.divider()

    # Solar Events and CME heliomaps side by side as two independent
    # panels (previously two tabs under one "Heliomap" component, in the
    # column "Top 5 Recorded Conditions" used to occupy — removed
    # 2026-07 per user request).
    solar_events_map_col, cme_map_col = st.columns(2)

    with solar_events_map_col:
        with st.container(height=620, border=False):
            st.markdown("#### Heliomap — Solar Events")
            st.caption("Where on the Sun recent solar events originated.")
            heliomap_solar_events_tab()

    with cme_map_col:
        with st.container(height=620, border=False):
            st.markdown("#### Heliomap — CME")
            st.caption("Where on the Sun recent CMEs originated.")
            heliomap_cme_tab()


def heliosphere_page(df: pd.DataFrame) -> None:
    """Derived Parameters and Travel Time were removed from this page's
    tabs (UI audit, 2026-07) — both were permanent stub tabs ("will be
    added in a future version") with no content behind them at all,
    which reads as broken rather than "coming soon" when clicked. Dynamic
    Pressure is fully functional (a real chart) and was kept — it was
    initially miscategorized as another stub during the audit because its
    chart renders below the fold on a short viewport, not because it's
    actually empty.
    """
    st.title("Heliosphere")
    tabs = st.tabs(["Solar Wind", "IMF", "Dynamic Pressure"])

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
        st.subheader("Dynamic Pressure")
        if {"proton_density", "solar_wind_speed"}.issubset(df.columns):
            pressure = df.copy()
            pressure["dynamic_pressure"] = physics.dynamic_pressure_series(pressure["proton_density"], pressure["solar_wind_speed"])
            line_chart(pressure, ["dynamic_pressure"], "Estimated Solar Wind Dynamic Pressure")
        else:
            st.info("Need proton_density and solar_wind_speed columns.")


def geospace_page(df: pd.DataFrame) -> None:
    st.title("Geospace")
    tabs = st.tabs(["Kp", "Dst"])

    with tabs[0]:
        current_analysis_kp(df)

    with tabs[1]:
        current_analysis_dst(df)




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
    prediction. Shows the same underlying jobs as the Analytics page in
    read-only form (render_architecture_status_panel) — this tab is for
    comparing architectures, not a second place to start a job.
    """
    st.markdown(
        "**Architecture:** Live NOAA &rarr; Feature Engineering &rarr; AE Model &rarr; Kp Model &rarr; "
        "Dst Model. None of the models use another model's prediction — only observed historical "
        "values are used.",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### AE (independent)")
    render_architecture_status_panel("ae")
    st.divider()
    st.markdown("##### Kp / Dst (independent — combined Sun-Earth model, observed AE)")
    render_architecture_status_panel("analytics")


def render_physics_cascaded_tab() -> None:
    """The experimental cascade: Predicted AE (never observed AE) feeds
    forward into the experimental Kp/Dst models as an extra feature. Shows
    the same underlying jobs as the Analytics page's Experimental
    Predictions tab in read-only form (render_architecture_status_panel)
    — this tab is for comparing architectures, not a second place to
    start a job.
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
    render_architecture_status_panel("experimental")


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


def research_lab_page(df: pd.DataFrame) -> None:
    st.title("Space Weather Research Lab")
    st.caption(
        "Experimental environment for researchers, students, and developers to evaluate different "
        "space weather forecasting architectures. Nothing here affects the Production Prediction "
        "models — every experiment is isolated from the production pipeline."
    )

    tabs = st.tabs(
        ["Forecasting Architectures", "Physics Interpretation", "Hypothesis Testing", "Storm Backtest", "Storm Learning"]
    )
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
    with tabs[3]:
        render_storm_backtest_tab()
    with tabs[4]:
        render_storm_learning_tab()


apply_retro_windows_style()

if not MASTER_PATH.exists():
    st.error(f"Master file not found: {MASTER_PATH}")
    st.code("PYTHONPATH=src python -m swdss.features.build_master")
    st.stop()

master_df = load_master_data(MASTER_PATH)
df_7d = seven_day_window(master_df)

# Site-wide masthead + nav bar (NOAA SWPC-style — 2026-07 redesign,
# pinned to the top of the viewport in a later 2026-07 pass): previously
# the title only appeared on the Home page and navigation was a per-page
# st.radio with default circular buttons, both inside the main content
# column, and both scrolled away with the page. Now the masthead, the
# page links, and the Refresh/Concepts/References actions are all one
# combined header — sticky at the top like a traditional website's fixed
# nav — with the actions living in the same dark bar as the page links
# (via st.columns) rather than a separate button row underneath.
with st.container(key="sticky_header"):
    render_site_header()

    with st.container(key="topnav"):
        nav_col, actions_col = st.columns([6.5, 3.5], vertical_alignment="center")
        with nav_col:
            page = st.radio(
                "Main sections",
                ["Home Page", "Photosphere", "Heliosphere", "Geospace", "Analytics", "Research Lab", "Project Status"],
                horizontal=True,
                label_visibility="collapsed",
            )
        with actions_col:
            with st.container(key="topnav_actions", horizontal=True, gap="small"):
                if st.button("↻ Refresh", key="refresh_button"):
                    st.cache_data.clear()
                    st.rerun()
                if st.button("📚 Concepts", key="library_open_button"):
                    open_dialog("library", None)
                if st.button("📋 References", key="references_open_button"):
                    open_dialog("references", None)
    style_top_nav()
style_sticky_header()

# Auto-refresh fires on all live-data pages except:
#  - "Research Lab" top-level page (IMF/Bz lab) — fully user-driven, no feed
#  - "Project Status" — an internal progress-notes page (dashboard/lib/
#    project_status.py), reads/writes nothing live; a rerun mid-typing
#    would be actively disruptive, not just wasted, so it's suppressed
#    unconditionally here rather than needing its own pause toggle.
#  - When the user has toggled "Pause Live Refresh" inside a research lab tab
#    (session_state["pause_autorefresh"] = True) — stops the blink while
#    they are actively working in the Kp or AE lab. Toggle it off to resume.
_pause = st.session_state.get("pause_autorefresh", False)
if page not in ("Research Lab", "Project Status") and not _pause:
    auto_refresh()

# The Photosphere/Heliosphere/Geospace rotating reference-table panels
# were removed from inline page display (UI simplification, 2026-07) —
# all of their tables now live in one scrollable terminal window behind
# the "📋 References" toolbar button above (show_references_terminal).
# The Home page's live status terminal was replaced (2026-07) by the
# SW Operational Command Centre — a full-width operational terminal
# reading only from the Operational Forecast Engine's stored snapshot
# (swdss.engine.storage), never computing anything live itself. Its own
# System tab now covers what the sidebar "Data Freshness" panel used to
# show, so that panel was removed rather than kept as a second,
# potentially-inconsistent source of the same information.
if page == "Home Page":
    render_operational_command_centre()

st.divider()

if page == "Home Page":
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
    elif kind == "references":
        show_references_terminal()
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
elif page == "Research Lab":
    research_lab_page(df_7d)
else:
    render_project_status_page()
