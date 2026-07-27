"""Event Explorer — the Sun-to-Earth causal-chain trace for one solar
event (flare/burst -> associated CME -> estimated Earth arrival ->
Solar Wind/IMF/Kp/Dst response), plus everything built around that same
trace: the News Feed's inline detail panel, the reverse lookup ("what
solar event explains this Kp/Dst/Bz reading?"), the animated storyboard,
and the Saved Events dialog. Extracted verbatim from dashboard/home.py.

Everything here shares one building block, build_event_chain_steps —
written once so the Event Explorer dialog (grid layout) and the News
Feed's inline terminal panel never drift out of sync by maintaining the
trace logic twice.
"""

from html import escape

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from dashboard.lib.data_helpers import (
    format_value,
    latest_label_time,
    load_processed_data,
    nearest_master_row,
    recent_window,
    variable_meaning_and_risk,
)
from dashboard.lib.library import load_saved_events, record_to_series, remove_saved_event, save_event_record
from dashboard.lib.shared_ui import open_dialog, render_dialog_close_button


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




def event_title(row: pd.Series) -> str:
    flare_class = row.get("flare_class")
    burst_type = row.get("radio_burst_type")
    event_type = str(row.get("event_type") or "")

    if pd.notna(flare_class):
        return f"{flare_class} Flare"
    if pd.notna(burst_type):
        return f"Type {burst_type} Radio Burst"
    return EVENT_TYPE_CATEGORY.get(event_type, event_type or "Solar Event")


def build_event_chain_steps(event: pd.Series) -> list[tuple[str, list[str]]]:
    """Builds the Sun-to-Earth causal-chain trace for one solar event —
    shared by the Event Explorer dialog (grid layout, show_event_explorer)
    and the News Feed's inline detail panel (single scrollable terminal
    box, render_event_terminal_panel), so both stay in sync automatically
    instead of maintaining the trace logic twice.
    """
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
        return steps

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
        return steps

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

    return steps


@st.dialog("Event Explorer", width="large", dismissible=False)
def show_event_explorer(event: pd.Series) -> None:
    render_dialog_close_button("close_event_explorer")
    render_chain_grid(build_event_chain_steps(event))


def render_event_terminal_panel(event: pd.Series, height: int = 560) -> None:
    """Inline, always-visible counterpart to the Event Explorer dialog —
    used by the News Feed's detail column so clicking an event updates
    this panel in place instead of opening a popup. Same underlying
    trace (build_event_chain_steps) as show_event_explorer, rendered as
    one scrollable terminal-style box instead of a grid of separate
    cards, since it has to fit in a single narrow column rather than the
    full dialog width.
    """
    steps = build_event_chain_steps(event)

    lines_html = ""
    for title, lines in steps:
        lines_html += f'<div style="margin-top:10px; color:#9adfff;">&gt; {escape(title)}</div>'
        for line in lines:
            lines_html += f"<div>&nbsp;&nbsp;{escape(line)}</div>"

    st.markdown(
        f"""
        <style>
        .event-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
            height: {height}px;
            box-sizing: border-box;
            overflow-y: auto;
        }}
        </style>
        <div class="event-terminal">
            <div style="color:#ffffff; font-weight:700;">
                {escape(event_title(event))} &mdash; {escape(latest_label_time(event["timestamp_utc"]))}
            </div>
            {lines_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


SEVERITY_BADGES = {3: "🔴", 2: "🟠", 1: "🟡", 0: "⚪"}
SEVERITY_LABELS = {3: "Severe (X-class / Type II)", 2: "Moderate (M-class)", 1: "Minor (Type III)", 0: "Notable"}


def _render_event_buttons(events: pd.DataFrame, key_prefix: str) -> None:
    """Clicking an event selects it into `news_feed_selected_event`
    (rendered by render_event_terminal_panel in the News Feed's detail
    column) rather than opening the Event Explorer popup — this helper
    is only used by the News Feed's single list now. The Saved Events
    dialog still uses the popup directly (see show_saved_events).
    """
    for idx, row in events.iterrows():
        ts = row["timestamp_utc"]
        severity = row["severity"] if "severity" in row.index else _event_severity(row)
        badge = SEVERITY_BADGES.get(severity, "⚪")
        label = f"{badge} {event_title(row)} - {ts.strftime('%d %b %Y')}"

        btn_col, save_col = st.columns([0.85, 0.15])
        with btn_col:
            if st.button(
                label, key=f"{key_prefix}_{idx}", use_container_width=True,
                help=SEVERITY_LABELS.get(severity, "Notable"),
            ):
                st.session_state["news_feed_selected_event"] = row
                st.rerun()
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
