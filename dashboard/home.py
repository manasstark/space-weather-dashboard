from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
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


def apply_retro_windows_style() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="stApp"] {
            background: #c0c0c0;
            color: #000000;
            font-family: "MS Sans Serif", "Tahoma", sans-serif;
        }

        section[data-testid="stSidebar"] {
            background: #c0c0c0;
            border-right: 2px solid #808080;
        }

        div[data-testid="stMetric"] {
            background: #dcdcdc;
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            padding: 12px;
        }

        .stRadio > div {
            background: #dcdcdc;
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            padding: 8px;
        }

        div[data-testid="stDataFrame"] {
            border-top: 2px solid #808080;
            border-left: 2px solid #808080;
            border-right: 2px solid #ffffff;
            border-bottom: 2px solid #ffffff;
        }

        h1, h2, h3 {
            color: #000080;
        }

                div.stButton > button {
            background: #dcdcdc;
            color: #000000;
            border-radius: 0px;
            border-top: 2px solid #ffffff;
            border-left: 2px solid #ffffff;
            border-right: 2px solid #808080;
            border-bottom: 2px solid #808080;
            min-width: 34px;
            height: 32px;
            padding: 2px 8px;
            font-weight: 700;
            font-family: "MS Sans Serif", Tahoma, sans-serif;
            box-shadow: none;
        }

        div.stButton > button:active {
            border-top: 2px solid #808080;
            border-left: 2px solid #808080;
            border-right: 2px solid #ffffff;
            border-bottom: 2px solid #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def card_note(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            min-height: 28px;
            margin-top: 6px;
            margin-bottom: 20px;
            color: #404040;
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


def status_terminal(df: pd.DataFrame) -> None:
    speed, speed_time = latest_value(df, "solar_wind_speed", "solar_wind")
    density, density_time = latest_value(df, "proton_density", "solar_wind")
    temperature, temperature_time = latest_value(df, "temperature", "solar_wind")
    bz, bz_time = latest_value(df, "bz", "imf")
    kp, kp_time = latest_value(df, "kp", "kp")
    dst, dst_time = latest_value(df, "dst", "dst")

    lines = [
        "SW-DSS STATUS TERMINAL",
        f"DATASET WINDOW: {date_window_label(df)}",
        "-" * 34,
        f"SPEED : {format_value(speed, ' km/s', 1)} | {speed_reference(speed)} | {latest_label_time(speed_time)}",
        f"DENS  : {format_value(density, ' p/cm3', 2)} | {density_reference(density)} | {latest_label_time(density_time)}",
        f"TEMP  : {format_value(temperature, ' K', 0)} | {temperature_reference(temperature)} | {latest_label_time(temperature_time)}",
        f"Bz    : {format_value(bz, ' nT', 2)} | {bz_reference_short(bz)} | {latest_label_time(bz_time)}",
        f"Kp    : {format_value(kp, '', 1)} | {kp_reference_short(kp)} | {latest_label_time(kp_time)}",
        f"Dst   : {format_value(dst, ' nT', 1)} | {dst_reference_short(dst)} | {latest_label_time(dst_time)}",
    ]

    terminal_text = "\n".join(lines)

    st.markdown(
        f"""
        <div style="
            background: #050505;
            color: #f2f2f2;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.82rem;
            line-height: 1.35;
            white-space: pre-wrap;
            min-height: 190px;
        ">{terminal_text}</div>
        """,
        unsafe_allow_html=True,
    )


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


def speed_reference(value) -> str:
    if pd.isna(value):
        return "No data"
    if value < 400:
        return "slow wind / quiet"
    if value < 500:
        return "moderate wind"
    if value < 700:
        return "fast wind"
    return "very fast wind"


def density_reference(value) -> str:
    if pd.isna(value):
        return "No data"
    if value < 5:
        return "low density"
    if value < 10:
        return "moderate density"
    if value < 30:
        return "high density"
    return "very high density"


def temperature_reference(value) -> str:
    if pd.isna(value):
        return "No data"
    if value < 50000:
        return "cool plasma"
    if value < 150000:
        return "typical solar wind"
    if value < 500000:
        return "hot solar wind"
    return "very hot plasma"


def bz_reference_short(value) -> str:
    if pd.isna(value):
        return "No data"
    if value > 0:
        return "northward / quiet coupling"
    if value >= -5:
        return "weak southward"
    if value >= -10:
        return "moderate southward"
    if value >= -20:
        return "strong southward"
    return "severe storm potential"


def kp_reference_short(value) -> str:
    if pd.isna(value):
        return "No data"
    if value <= 3:
        return "quiet"
    if value < 5:
        return "active"
    if value < 6:
        return "G1 minor storm"
    if value < 7:
        return "G2 moderate storm"
    if value < 8:
        return "G3 strong storm"
    return "G4/G5 severe storm"


def dst_reference_short(value) -> str:
    if pd.isna(value):
        return "No data"
    if value > -30:
        return "quiet"
    if value > -50:
        return "weak storm"
    if value > -100:
        return "moderate storm"
    if value > -200:
        return "intense storm"
    return "superstorm"


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


def reference_window() -> None:
    reference_tables = [
        {
            "title": "Bz Reference",
            "data": [
                {"Range": "Bz > 0 nT", "Meaning": "Northward IMF", "Risk": "Low coupling"},
                {"Range": "0 to -5 nT", "Meaning": "Weak southward IMF", "Risk": "Minor activity possible"},
                {"Range": "-5 to -10 nT", "Meaning": "Moderate southward IMF", "Risk": "Storm possible"},
                {"Range": "-10 to -20 nT", "Meaning": "Strong southward IMF", "Risk": "Strong storm coupling"},
                {"Range": "< -20 nT", "Meaning": "Extreme southward IMF", "Risk": "Severe storm potential"},
            ],
        },
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
        {
            "title": "Speed Reference",
            "data": [
                {"Range": "< 400 km/s", "Meaning": "Slow solar wind", "Risk": "Usually quiet"},
                {"Range": "400-500 km/s", "Meaning": "Moderate speed", "Risk": "Normal/active"},
                {"Range": "500-700 km/s", "Meaning": "Fast solar wind", "Risk": "Storm possible with southward Bz"},
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
    ]

    if "reference_idx" not in st.session_state:
        st.session_state.reference_idx = 0

    current = reference_tables[st.session_state.reference_idx]
    table_df = pd.DataFrame(current["data"])

    header_col, prev_col, next_col = st.columns([10, 0.6, 0.6])

    with header_col:
        st.subheader(current["title"])

    with prev_col:
        if st.button("<", help="Previous reference table"):
            st.session_state.reference_idx = (st.session_state.reference_idx - 1) % len(reference_tables)
            st.rerun()

    with next_col:
        if st.button(">", help="Next reference table"):
            st.session_state.reference_idx = (st.session_state.reference_idx + 1) % len(reference_tables)
            st.rerun()

    st.dataframe(
        table_df,
        hide_index=True,
        use_container_width=False,
        height=(len(table_df) + 1) * 35 + 10,
        column_config={
            "Range": st.column_config.TextColumn(width="small"),
            "Meaning": st.column_config.TextColumn(width="medium"),
            "Risk": st.column_config.TextColumn(width="large"),
        },
    )


def time_caption(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return f"Recorded at {row['timestamp_utc']}"


def line_chart(df: pd.DataFrame, columns: list[str], title: str) -> None:
    available = [col for col in columns if col in df.columns]
    if not available:
        st.info("No data available for this chart yet.")
        return
    chart_df = df[["timestamp_utc", *available]].dropna(how="all", subset=available)
    fig = px.line(chart_df, x="timestamp_utc", y=available, title=title)
    fig.update_layout(height=390, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(scatter, use_container_width=True)


def current_analysis_solar_wind(df: pd.DataFrame) -> None:
    st.subheader("Solar Wind Current Analysis")

    speed_row = row_at_extreme(df, "solar_wind_speed", "max")
    density_row = row_at_extreme(df, "proton_density", "max")
    temp_row = row_at_extreme(df, "temperature", "max")

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
            st.caption(
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

    bz_row = row_at_extreme(df, "bz", "min")
    bt_row = row_at_extreme(df, "bt", "max")
    bx_row = row_at_extreme(df, "bx", "max")
    by_row = row_at_extreme(df, "by", "max")

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

    bz_row = row_at_extreme(df, "bz", "min")
    kp_row = row_at_extreme(df, "kp", "max")
    dst_row = row_at_extreme(df, "dst", "min")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Lowest Bz", format_value(None if bz_row is None else bz_row["bz"], " nT", 2), time_caption(bz_row))
        if bz_row is not None:
            st.caption(
                f"Speed: {format_value(bz_row.get('solar_wind_speed'), ' km/s', 1)} | "
                f"Density: {format_value(bz_row.get('proton_density'), ' p/cm3', 2)} | "
                f"Temp: {format_value(bz_row.get('temperature'), ' K', 0)}"
            )
    with c2:
        metric_card("Highest Kp", format_value(None if kp_row is None else kp_row["kp"], "", 1), time_caption(kp_row))
        if kp_row is not None:
            st.caption(
                f"Speed: {format_value(kp_row.get('solar_wind_speed'), ' km/s', 1)} | "
                f"Density: {format_value(kp_row.get('proton_density'), ' p/cm3', 2)} | "
                f"Bz: {format_value(kp_row.get('bz'), ' nT', 2)}"
            )
    with c3:
        metric_card("Lowest Dst", format_value(None if dst_row is None else dst_row["dst"], " nT", 1), time_caption(dst_row))
        if dst_row is not None:
            st.caption(
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

    speed_row = row_at_extreme(df, "solar_wind_speed", "max")
    density_row = row_at_extreme(df, "proton_density", "max")
    bz_row = row_at_extreme(df, "bz", "min")
    kp_row = row_at_extreme(df, "kp", "max")
    dst_row = row_at_extreme(df, "dst", "min")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Highest Speed", format_value(None if speed_row is None else speed_row["solar_wind_speed"], " km/s", 1), time_caption(speed_row))
    with c2:
        metric_card("Highest Density", format_value(None if density_row is None else density_row["proton_density"], " p/cm3", 2), time_caption(density_row))
    with c3:
        metric_card("Lowest Bz", format_value(None if bz_row is None else bz_row["bz"], " nT", 2), time_caption(bz_row))
    with c4:
        metric_card("Highest Kp", format_value(None if kp_row is None else kp_row["kp"], "", 1), time_caption(kp_row))
    with c5:
        metric_card("Lowest Dst", format_value(None if dst_row is None else dst_row["dst"], " nT", 1), time_caption(dst_row))

    st.divider()

    st.subheader("Latest Values")

    speed, speed_time = latest_value(df, "solar_wind_speed", "solar_wind")
    density, density_time = latest_value(df, "proton_density", "solar_wind")
    temperature, temperature_time = latest_value(df, "temperature", "solar_wind")
    bz, bz_time = latest_value(df, "bz", "imf")
    kp, kp_time = latest_value(df, "kp", "kp")
    dst, dst_time = latest_value(df, "dst", "dst")  

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Speed", format_value(speed, " km/s", 1))
    c1.caption(f"At {speed_time}" if speed_time is not None else "No data")

    c2.metric("Density", format_value(density, " p/cm3", 2))
    c2.caption(f"At {density_time}" if density_time is not None else "No data")

    c3.metric("Temperature", format_value(temperature, " K", 0))
    c3.caption(f"At {temperature_time}" if temperature_time is not None else "No data")

    c4.metric("Bz", format_value(bz, " nT", 2))
    c4.caption(f"At {bz_time}" if bz_time is not None else "No data")

    c5.metric("Kp", format_value(kp, "", 1))
    c5.caption(f"At {kp_time}" if kp_time is not None else "No data")

    c6.metric("Dst", format_value(dst, " nT", 1))
    c6.caption(f"At {dst_time}" if dst_time is not None else "No data")

    st.divider()

    

    st.subheader("Top 5 Recorded Conditions")

    tab1, tab2, tab3 = st.tabs(["Lowest Bz", "Highest Kp", "Lowest Dst"])

    with tab1:
        top_event_table(df, "bz", "lowest", "Top 5 Most Negative Bz Events")

    with tab2:
        top_event_table(df, "kp", "highest", "Top 5 Highest Kp Events")

    with tab3:
        top_event_table(df, "dst", "lowest", "Top 5 Lowest Dst Events")

    st.divider()

    reference_window()


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
        ["Home Page", "Heliosphere", "Geospace", "Analytics"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if st.button("↻ Refresh", help="Refresh dashboard data"):
        st.cache_data.clear()
        st.rerun()

with terminal_col:
    status_terminal(df_7d)

st.divider()



st.sidebar.subheader("Data Freshness")

sw_status, sw_age = freshness_status(df_7d, "solar_wind_speed", 90)
imf_status, imf_age = freshness_status(df_7d, "bz", 90)
kp_status, kp_age = freshness_status(df_7d, "kp", 360)
dst_status, dst_age = freshness_status(df_7d, "dst", 120)

st.sidebar.write(f"Solar Wind: {sw_status} ({sw_age})")
st.sidebar.write(f"IMF: {imf_status} ({imf_age})")
st.sidebar.write(f"Kp: {kp_status} ({kp_age})")
st.sidebar.write(f"Dst: {dst_status} ({dst_age})")

st.sidebar.divider()




st.sidebar.caption("Refresh logic")
st.sidebar.write("Solar Wind / IMF: every minute")
st.sidebar.write("Kp: about every 3 hours")
st.sidebar.write("Dst: about every hour")

if page == "Home Page":
    home_page(df_7d)
elif page == "Heliosphere":
    heliosphere_page(df_7d)
elif page == "Geospace":
    geospace_page(df_7d)
else:
    analytics_page(df_7d)
