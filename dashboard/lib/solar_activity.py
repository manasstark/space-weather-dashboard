"""Solar-side analysis: the News Feed's chronological event list, Solar
Events / CME statistical analysis tabs, the two Heliomap tabs (event and
CME source positions on the solar disk), and F10.7 radio flux
classification/analysis. Extracted verbatim from dashboard/home.py.

Everything here reads event/CME/F10.7 data and renders charts — none of
it drives a prediction or writes state, so unlike dashboard.lib.library
or event_explorer, this module has no CRUD or dialog-dispatch surface of
its own; it's pure analysis + Plotly rendering on top of
dashboard.lib.event_explorer's event-classification helpers.
"""

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.lib.data_helpers import (
    format_value,
    get_base64_image,
    latest_label_time,
    latest_non_null,
    load_processed_data,
    recent_window,
    render_simple_retro_table,
    row_at_extreme,
    time_caption,
)
from dashboard.lib.event_explorer import (
    EVENT_TYPE_CATEGORY,
    RADIO_BURST_TYPES,
    _event_severity,
    count_associated_cmes,
    dedupe_near_duplicate_events,
    get_notable_solar_events,
    render_event_button_list,
    render_event_terminal_panel,
    render_event_timeline,
)
from dashboard.lib.shared_ui import plot_retro, terminal_metric
from swdss.paths import PROJECT_ROOT


def solar_event_news_feed() -> None:
    """A single chronological list (previously two parallel lists — "By
    Severity" and "Latest Recorded" — that showed the same underlying
    events twice, just reordered; UI audit + user request, 2026-07) next
    to a live detail panel where "By Severity" used to sit. Severity is
    shown per-event via a colored marker (SEVERITY_BADGES) rather than
    via a second, separately-sorted list. Clicking an event updates the
    detail panel in place (render_event_terminal_panel) instead of
    opening the Event Explorer popup; with nothing selected yet, the
    panel defaults to the most recent event.
    """
    st.subheader("Solar Activity News Feed")

    notable = get_notable_solar_events()

    if notable.empty:
        st.info("No notable solar events in the recent window.")
        return

    by_latest = notable.sort_values("timestamp_utc", ascending=False)

    if "news_feed_expanded" not in st.session_state:
        st.session_state.news_feed_expanded = False

    list_col, detail_col = st.columns(2)

    with list_col:
        st.caption("🔴 Severe (X-class / Type II) · 🟠 Moderate (M-class) · 🟡 Minor (Type III) · ⚪ Notable")

        expanded = st.session_state.news_feed_expanded
        limit = len(by_latest) if expanded else 5
        render_event_button_list(by_latest.head(limit), "news_feed", scrollable=expanded)

        if len(by_latest) > 5:
            button_label = "Show fewer" if st.session_state.news_feed_expanded else "More"
            if st.button(button_label, key="news_feed_more"):
                st.session_state.news_feed_expanded = not st.session_state.news_feed_expanded
                st.rerun()

    with detail_col:
        selected_event = st.session_state.get("news_feed_selected_event")
        if selected_event is None:
            selected_event = by_latest.iloc[0]
        render_event_terminal_panel(selected_event)


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
        terminal_metric("Today's Solar Events", str(total_events_today))
    with c2:
        terminal_metric("Total Solar Flares (7d)", str(total_flares))
    with c3:
        terminal_metric("Total Radio Bursts (7d)", str(total_radio_bursts))
    with c4:
        terminal_metric("X-Class Flares (7d)", str(x_class_count))
    with c5:
        terminal_metric("Associated CMEs (7d)", str(associated_cme_count))
    with c6:
        terminal_metric("Most Active Region", str(most_active_region))

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

    # Only the metrics NOT already shown in the cards above (Total Events,
    # M-Class Count, Avg Events/Day) — this used to be a full second
    # "Statistics (7 Days)" row that repeated Total Flares/Total Radio
    # Bursts/X-Class Count/Most Active Region verbatim from the cards
    # above (UI audit, 2026-07: same numbers shown twice on one tab).
    average_events_per_day = len(recent) / 7 if not recent.empty else 0
    st1, st2, st3 = st.columns(3)
    with st1:
        terminal_metric("Total Events (7d)", str(len(recent)))
    with st2:
        terminal_metric("M-Class Flares (7d)", str(m_class_count))
    with st3:
        terminal_metric("Avg Events/Day (7d)", format_value(average_events_per_day, "", 1))

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
        terminal_metric("Latest CME Speed", format_value(latest_speed, " km/s", 1), latest_label_time(latest_speed_time))
    with c2:
        terminal_metric(
            "Fastest CME (7d)",
            format_value(None if fastest_row is None else fastest_row["speed"], " km/s", 1),
            time_caption(fastest_row),
        )
    with c3:
        terminal_metric("Average Speed (7d)", format_value(avg_speed, " km/s", 1))
    with c4:
        terminal_metric("Total CMEs This Week", str(total_cmes))
    with c5:
        terminal_metric(
            "Earth-Directed CMEs (7d)",
            str(earth_directed_count),
            f"|longitude| ≤ {EARTH_DIRECTED_LONGITUDE_DEG}°",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Only the metrics NOT already shown in the cards above (Min/Median
    # Speed, Std Deviation) — this used to be a full second "CME
    # Statistics" row that repeated Average Speed/Total CMEs verbatim,
    # plus a "Max Speed" duplicating "Fastest CME" above in all but name
    # (UI audit, 2026-07: same numbers shown twice on one tab).
    if not speeds.empty:
        s1, s2, s3 = st.columns(3)
        with s1:
            terminal_metric("Min Speed (7d)", format_value(speeds.min(), " km/s", 1))
        with s2:
            terminal_metric("Median Speed (7d)", format_value(speeds.median(), " km/s", 1))
        with s3:
            terminal_metric("Std Deviation (7d)", format_value(speeds.std(), " km/s", 1))
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
        terminal_metric("Today's Flux", format_value(latest_flux, "", 1), latest_label_time(latest_flux_time))
    with c2:
        terminal_metric(
            "Highest Flux (7d)",
            format_value(None if highest_row is None else highest_row["f107_flux"], "", 1),
            time_caption(highest_row),
        )
    with c3:
        terminal_metric(
            "Lowest Flux (7d)",
            format_value(None if lowest_row is None else lowest_row["f107_flux"], "", 1),
            time_caption(lowest_row),
        )
    with c4:
        terminal_metric("Average Flux (7d)", format_value(avg_flux, "", 1))

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




