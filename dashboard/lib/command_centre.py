"""SW Operational Command Centre — the homepage's operational terminal.

Design philosophy: this is an operational instrument, not a SaaS
dashboard. No cards, no rounded tiles, no KPI widgets, no floating
panels. Every tab renders dense, aligned, monospace terminal-report
tables — the visual language of a Bloomberg Terminal / mission-control
console, not a web analytics product.

Reads ONLY from swdss.engine.storage (the Operational Forecast Engine's
stored products) — this module never calls a prediction model, never
touches jobs.py's SQLite DB directly, and never recomputes physics. If
the engine hasn't produced a snapshot yet, it shows one line of plain
text and nothing else.

Eleven fixed tabs: Forecast (default), Current, Physics, Timeline,
Verification, Logs, Downloads, Alerts, System, Search, Solar Forecast.
Solar Forecast is the odd one out architecturally — F10.7 and CME
arrival aren't PRODUCTION_MATRIX jobs, so it reads
swdss.engine.storage.load_solar_forecast() (written by
orchestrator._solar_forecast_snapshot) instead of the main
`snapshot` dict every other tab here reads, but follows the exact same
"engine computes, dashboard only reads" rule.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.design_tokens import ACCENT, AMBER, BG, BLUE, BORDER, MONO, MUTED, PANEL_BG, RED, TEXT
from dashboard.lib.shared_ui import open_dialog
from swdss.engine import calibration, packages, skill, storage
from swdss.engine.labels import classify_current_reading
from swdss.paths import PROCESSED_DIR

# ==================== Terminal design tokens ====================
# (BG/PANEL_BG/BORDER/TEXT/MUTED/ACCENT/AMBER/RED/BLUE/MONO now live in
# dashboard.lib.design_tokens — single source of truth shared with
# shared_ui.terminal_metric, imported above rather than redefined here.)

STATUS_COLORS = {
    "CREATED": MUTED,
    "LIVE": BLUE,
    "ACTIVE": ACCENT,
    "AWAITING EVALUATION": AMBER,
    "WAITING FOR VERIFICATION": AMBER,
    "VERIFIED": ACCENT,
    "ARCHIVED": MUTED,
}
PACKAGE_LIFECYCLE_STAGES = ["CREATED", "LIVE", "ACTIVE", "WAITING FOR VERIFICATION", "VERIFIED", "ARCHIVED"]
CONFIDENCE_COLORS = {
    "Very High": ACCENT, "High": ACCENT,
    "Moderate": AMBER,
    "Low": RED, "Very Low": RED,
}
SEVERITY_COLORS = {"critical": RED, "warning": AMBER, "info": BLUE}
LOG_LEVEL_COLORS = {"ERROR": RED, "WARN": AMBER, "INFO": ACCENT}
SERVICE_STATUS_COLORS = {
    "RUNNING": ACCENT, "ONLINE": ACCENT,
    "DEGRADED": AMBER, "STALE": AMBER, "DELAYED": AMBER,
    "OFFLINE": RED, "UNKNOWN": MUTED,
}

VARIABLE_LABELS_DISPLAY = {
    "speed": "SPEED", "density": "DENSITY", "temperature": "TEMPERATURE",
    "bt": "BT", "bx_gsm": "BX", "by_gsm": "BY", "bz_gsm": "BZ",
    "kp": "KP", "dst": "DST", "ae": "AE",
}
UNITS = {
    "speed": "km/s", "density": "p/cm3", "temperature": "K",
    "bt": "nT", "bx_gsm": "nT", "by_gsm": "nT", "bz_gsm": "nT",
    "kp": "", "dst": "nT", "ae": "nT",
}
# Decimal precision per variable's natural scale — Temperature in
# particular has no business displaying as "251169.00 K"; a whole-Kelvin
# reading isn't a rounding of anything more precise the model actually
# knows. Used everywhere a forecast/current/range value is rendered so
# precision stays consistent across the Forecast, Current, and Search tabs.
VARIABLE_DECIMALS = {
    "speed": 1, "density": 2, "temperature": 0,
    "bt": 2, "bx_gsm": 2, "by_gsm": 2, "bz_gsm": 2,
    "kp": 1, "dst": 1, "ae": 1,
}


def _horizon_sort_key(horizon_label) -> int:
    """Sorts '1h' < '3h' < ... < '24h', with Kp's 'interval' label last —
    used only to order the Search tab's multi-horizon match table by lead
    time, since horizon_label strings don't sort correctly as plain text
    ('12h' would otherwise sort before '3h').
    """
    if horizon_label == "interval":
        return 10_000
    try:
        return int(str(horizon_label).rstrip("h"))
    except ValueError:
        return 9_999


def _css() -> None:
    st.markdown(
        f"""
        <style>
        .term-root, .term-root * {{ font-family: {MONO} !important; }}
        .term-banner {{
            background: {PANEL_BG};
            border: 1px solid {BORDER};
            border-top: 2px solid {ACCENT};
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 4px;
        }}
        .term-banner .term-title {{
            color: {ACCENT};
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 1.5px;
        }}
        .term-banner .term-meta {{
            color: {TEXT};
            font-size: 0.74rem;
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .term-section {{
            color: {ACCENT};
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            border-bottom: 1px solid {BORDER};
            padding: 10px 2px 4px 2px;
            margin-top: 2px;
        }}
        .term-rule {{
            border: none;
            border-top: 1px solid {BORDER};
            margin: 4px 0 10px 0;
        }}
        table.term-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.74rem;
            background: {BG};
        }}
        table.term-table th, table.term-table td {{
            border: 1px solid {BORDER};
            padding: 4px 8px;
            text-align: left;
            color: {TEXT};
            white-space: nowrap;
        }}
        table.term-table th {{
            background: {PANEL_BG};
            color: {MUTED};
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.66rem;
            letter-spacing: 0.6px;
        }}
        table.term-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .term-wrap {{ overflow-x: auto; background: {BG}; border: 1px solid {BORDER}; padding: 2px; }}
        .term-scroll {{
            max-height: 460px;
            overflow-y: auto;
            background: {BG};
            border: 1px solid {BORDER};
            padding: 6px 10px;
        }}
        .term-logline {{
            font-size: 0.74rem;
            padding: 2px 0;
            border-bottom: 1px solid {BORDER};
            color: {TEXT};
        }}
        .term-caption {{
            color: {MUTED};
            font-size: 0.72rem;
            margin: 2px 0 8px 0;
        }}
        div[data-testid="stExpander"] {{
            border: 1px solid {BORDER} !important;
            border-radius: 0 !important;
            background: {BG} !important;
        }}
        div[data-testid="stExpander"] summary {{
            font-family: {MONO} !important;
            font-size: 0.76rem !important;
            color: {TEXT} !important;
        }}
        .term-pkg-bar {{
            background: {PANEL_BG};
            border: 1px solid {BORDER};
            border-left: 3px solid {ACCENT};
            padding: 6px 14px;
            font-size: 0.76rem;
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 8px;
        }}
        .term-lifecycle {{
            font-size: 0.7rem;
            letter-spacing: 0.5px;
        }}
        .term-lifecycle .stage {{ color: {MUTED}; }}
        .term-lifecycle .stage.current {{ font-weight: 700; }}
        .term-lifecycle .arrow {{ color: {BORDER}; margin: 0 4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _dark_plotly_layout(fig, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family=MONO, size=12, color=TEXT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=MONO, size=10, color=TEXT),
        height=height,
        margin=dict(t=36, b=30, l=44, r=16),
        legend=dict(bgcolor=BG, font=dict(size=9, color=TEXT), orientation="h", y=-0.18),
    )
    axis_style = dict(
        showline=True, linecolor=BORDER, linewidth=1,
        gridcolor=BORDER, gridwidth=0.5, zeroline=False,
        tickfont=dict(family=MONO, size=9, color=MUTED),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _table(headers: list, rows: list, numeric_cols: set = None) -> str:
    numeric_cols = numeric_cols or set()
    html = '<div class="term-wrap"><table class="term-table"><thead><tr>'
    html += "".join(f"<th>{escape(h)}</th>" for h in headers)
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>"
        for i, cell in enumerate(row):
            cls = ' class="num"' if i in numeric_cols else ""
            html += f"<td{cls}>{cell}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html


def _is_missing(value) -> bool:
    """pd.isna() on a bare Python scalar (None/NaN/NaT/pd.NA) always
    returns a plain bool; it only returns an array for list-likes, which
    every caller here avoids by construction. Checked this way (rather
    than `isinstance(value, float) and pd.isna(value)`) because pd.NA
    fails that isinstance check yet still isn't safe to compare/format —
    `pd.NA >= 0` returns pd.NA itself, not a bool, and using it directly
    in an `if` raises "boolean value of NA is ambiguous".
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _fmt(value, decimals: int = 2, suffix: str = "") -> str:
    if _is_missing(value):
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def _fmt_signed(value, decimals: int = 2, suffix: str = "") -> str:
    if _is_missing(value):
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}{suffix}"


def _fmt_time(iso_ts) -> str:
    if _is_missing(iso_ts):
        return "—"
    ts = pd.Timestamp(iso_ts)
    if pd.isna(ts):
        return "—"
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.strftime("%d %b %H:%M UTC")


def _fmt_period(start_iso, end_iso) -> str:
    if _is_missing(start_iso) or _is_missing(end_iso):
        return "—"
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    if pd.isna(start) or pd.isna(end):
        return "—"
    return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')} UTC"


def _fmt_lead(minutes) -> str:
    if _is_missing(minutes):
        return "—"
    if minutes <= 0:
        return "In window"
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} hr"


def _status_span(status) -> str:
    if status is None or (isinstance(status, float) and pd.isna(status)):
        status = "—"
    status = str(status)
    color = STATUS_COLORS.get(status, MUTED)
    return f'<span style="color:{color}; font-weight:700;">[{escape(status)}]</span>'


def _confidence_span(category) -> str:
    if category is None or (isinstance(category, float) and pd.isna(category)):
        category = "—"
    category = str(category)
    color = CONFIDENCE_COLORS.get(category, MUTED)
    return f'<span style="color:{color};">{escape(category)}</span>'


def _age_text(iso_ts: str) -> str:
    if not iso_ts:
        return "N/A"
    ts = pd.Timestamp(iso_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age_seconds = (pd.Timestamp.now(tz="UTC") - ts).total_seconds()
    if age_seconds < 90:
        return f"{age_seconds:.0f}s ago"
    if age_seconds < 3600:
        return f"{age_seconds / 60:.1f} min ago"
    return f"{age_seconds / 3600:.1f} hr ago"


def _health_color(iso_ts: str) -> str:
    if not iso_ts:
        return MUTED
    ts = pd.Timestamp(iso_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age_minutes = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    if age_minutes <= 3:
        return ACCENT
    if age_minutes <= 10:
        return AMBER
    return RED


def _render_banner(snapshot: dict) -> None:
    generated_at = snapshot.get("generated_at")
    color = _health_color(generated_at)
    now_text = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(
        f"""
        <div class="term-banner term-root">
            <div class="term-title">SW OPERATIONAL COMMAND CENTRE</div>
            <div class="term-meta">
                <span style="color:{color};">● CYCLE {escape(_age_text(generated_at))}</span>
                <span>UTC {escape(now_text)}</span>
                <span>ENGINE v{escape(str(snapshot.get('engine_version', '—')))}</span>
                <span>ACTIVE JOBS {snapshot.get('active_jobs_total', 0)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _lifecycle_html(current_stage: str) -> str:
    parts = []
    for i, stage in enumerate(PACKAGE_LIFECYCLE_STAGES):
        cls = "stage current" if stage == current_stage else "stage"
        color = STATUS_COLORS.get(stage, MUTED) if stage == current_stage else MUTED
        parts.append(f'<span class="{cls}" style="color:{color};">{escape(stage)}</span>')
    return '<span class="arrow">→</span>'.join(parts)


def _render_package_banner(package: dict) -> None:
    """Always-visible, one-line package summary + an expander for full
    metadata — placed above the tabs so the current operational product
    is visible without scrolling into the Forecast tab.
    """
    if package is None:
        st.markdown(
            '<div class="term-root term-caption">NO FORECAST PACKAGE YET — the first engine cycle builds one automatically.</div>',
            unsafe_allow_html=True,
        )
        return

    status = package.get("status", "—")
    status_color = STATUS_COLORS.get(status, MUTED)
    completeness = package.get("completeness", "—")
    completeness_color = ACCENT if completeness == "COMPLETE" else AMBER
    variables_complete = package.get("variables_complete")
    variables_total = package.get("variables_total")
    completeness_fraction = f"{variables_complete}/{variables_total}" if variables_complete is not None else "—"
    conf = package.get("overall_confidence", {})

    st.markdown(
        f"""
        <div class="term-pkg-bar term-root">
            <b style="color:{ACCENT};">{escape(package.get('package_id', '—'))}</b>
            <span>CYCLE #{package.get('forecast_cycle', '—')}</span>
            <span>ISSUED {escape(_fmt_time(package.get('generated_at')))}</span>
            <span>OBS THROUGH {escape(_fmt_time(package.get('observations_used_through')))}</span>
            <span>VALID {escape(_fmt_period(package.get('valid_start'), package.get('valid_end')))}</span>
            <span style="color:{status_color}; font-weight:700;">[{escape(status)}]</span>
            <span style="color:{completeness_color};">{completeness_fraction} {escape(completeness)}</span>
            <span>CONF {_confidence_span(conf.get('category'))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("▸ Forecast Package Details", expanded=False):
        st.markdown(f'<div class="term-root term-lifecycle">{_lifecycle_html(status)}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        headers = ["FIELD", "VALUE"]
        rows = [
            ["Package ID", package.get("package_id", "—")],
            ["Forecast Cycle", str(package.get("forecast_cycle", "—"))],
            ["Generated", _fmt_time(package.get("generated_at"))],
            ["Observations Used Through", _fmt_time(package.get("observations_used_through"))],
            ["Valid Period", _fmt_period(package.get("valid_start"), package.get("valid_end"))],
            ["Status", _status_span(status)],
            ["Completeness", f'<span style="color:{completeness_color};">{completeness_fraction} {escape(completeness)}</span>'],
            ["Evaluation Status", escape(package.get("evaluation_status", "—"))],
            ["Overall Confidence", f"{_confidence_span(conf.get('category'))} ({_fmt(conf.get('score'), 2)})"],
            ["Physics Engine Version", escape(package.get("physics_engine_version", "—"))],
            ["Production Version", escape(package.get("production_version", "—"))],
            ["Kp Carried Over", "Yes — NOAA hasn't published a new 3h interval since last package" if package.get("kp_carried_over") else "No — refreshed this cycle"],
            ["Package Archive Path", escape(package.get("package_archive_path", "—"))],
        ]
        st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="term-root term-caption">{escape(package.get("package_summary", ""))}</div>', unsafe_allow_html=True)

        if package.get("missing_variables"):
            st.markdown(
                f'<div class="term-root" style="color:{AMBER}; font-size:0.76rem;">MISSING: {escape(", ".join(package["missing_variables"]))}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="term-root term-section">Models Used</div>', unsafe_allow_html=True)
        model_rows = [[k.upper(), v or "—"] for k, v in (package.get("models_used") or {}).items()]
        st.markdown(f'<div class="term-root">{_table(["VARIABLE", "MODEL"], model_rows)}</div>', unsafe_allow_html=True)


def render_operational_command_centre() -> None:
    snapshot = storage.load_current_snapshot()
    _css()

    if snapshot is None:
        st.markdown(
            '<div class="term-root" style="color:#6e7681; font-size:0.8rem;">'
            "NO ENGINE SNAPSHOT YET — run `python -m swdss.features.live_update` "
            "or call run_forecast_cycle() / evaluate_due_forecasts() / refresh_dashboard_products() manually."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    _render_banner(snapshot)
    _render_package_banner(storage.load_current_package())

    tabs = st.tabs([
        "Forecast", "Current", "Physics", "Timeline", "Verification",
        "Logs", "Downloads", "Alerts", "System", "Search", "Solar Forecast",
    ])
    with tabs[0]:
        _render_forecast_tab(snapshot)
    with tabs[1]:
        _render_current_tab(snapshot)
    with tabs[2]:
        _render_physics_tab(snapshot)
    with tabs[3]:
        _render_timeline_tab(snapshot)
    with tabs[4]:
        _render_verification_tab(snapshot)
    with tabs[5]:
        _render_logs_tab(snapshot)
    with tabs[6]:
        _render_downloads_tab(snapshot)
    with tabs[7]:
        _render_alerts_tab(snapshot)
    with tabs[8]:
        _render_system_tab(snapshot)
    with tabs[9]:
        _render_search_tab(snapshot)
    with tabs[10]:
        _render_solar_forecast_tab()


# ==================== Forecast ====================

def _headline_horizon_key(dataset: str, variable: str) -> str:
    return "interval" if (dataset == "analytics" and variable == "kp") else "1h"


def _activity_badge_html(variable: str, value) -> str:
    """NOAA-style G-scale / storm-class badge for a predicted Kp/Dst/AE
    value, reusing swdss.engine.labels.classify_current_reading — the
    same thresholds already shown for the Current tab's observed values —
    so a forecast reads as an operator-recognizable storm class, not
    just a bare decimal.
    """
    if variable not in ("kp", "dst", "ae") or value is None:
        return ""
    meaning, risk = classify_current_reading(variable, value)
    if not meaning:
        return ""
    color = RISK_COLORS.get(risk, MUTED)
    return f' <span style="color:{color};">({escape(meaning)})</span>'


def _forecast_value_html(entry: dict, variable: str) -> str:
    if entry is None or entry.get("predicted_value") is None:
        return "—"
    unit = UNITS.get(variable, "")
    decimals = VARIABLE_DECIMALS.get(variable, 2)
    badge = _activity_badge_html(variable, entry["predicted_value"])
    if entry.get("range_low") is not None and entry.get("range_high") is not None:
        return f"{entry['range_low']:.{decimals}f} &rarr; {entry['range_high']:.{decimals}f}{(' ' + unit) if unit else ''}{badge}"
    return f"{entry['predicted_value']:.{decimals}f}{(' ' + unit) if unit else ''}{badge}"


def _delta_trend_html(entry: dict, variable: str) -> str:
    if entry is None:
        return "—"
    unit = UNITS.get(variable, "")
    decimals = VARIABLE_DECIMALS.get(variable, 2)
    if variable in ("bz_gsm", "kp"):
        return escape(entry.get("trend", "—"))
    delta = entry.get("delta")
    if delta is None:
        return "—"
    return escape(_fmt_signed(delta, decimals, (" " + unit) if unit else ""))


def _forecast_row(variable: str, entry: dict) -> list:
    unit = UNITS.get(variable, "")
    decimals = VARIABLE_DECIMALS.get(variable, 2)
    current_text = _fmt(entry.get("current_value") if entry else None, decimals, (" " + unit) if unit else "")
    if entry is None:
        return [
            f"<b>{escape(VARIABLE_LABELS_DISPLAY.get(variable, variable))}</b>", "—",
            current_text, "—", "—", "—", "—", "—", "—", "—",
        ]
    conf = entry.get("confidence") or {}
    return [
        f"<b>{escape(VARIABLE_LABELS_DISPLAY.get(variable, variable))}</b>",
        _status_span(entry.get("status", "—")),
        current_text,
        _forecast_value_html(entry, variable),
        _delta_trend_html(entry, variable),
        _confidence_span(conf.get("category")),
        _fmt_time(entry.get("generated_at")),
        _fmt_period(entry.get("valid_start"), entry.get("valid_end")),
        _fmt_lead(entry.get("lead_time_minutes")),
        escape(entry.get("model_name", "—")),
    ]


EXPLAINED_VARIABLES = {"kp", "dst", "ae"}


def _render_explanation_expander(variable: str, exp: dict) -> None:
    """The Forecast Explanation Engine's output — connects this
    forecast back to the Physics tab's live readings, rule-based (see
    swdss.engine.explanation), never fabricated.
    """
    with st.expander(f"▸ Why this {VARIABLE_LABELS_DISPLAY.get(variable, variable)} forecast?"):
        st.markdown(f'<div class="term-root term-caption">{escape(exp.get("sentence", ""))}</div>', unsafe_allow_html=True)
        headers = ["RANK", "DRIVER"]
        rows = []
        for rank, key in (("PRIMARY", "primary"), ("SECONDARY", "secondary"), ("SUPPORTING", "supporting")):
            value = exp.get(key)
            if value:
                rows.append([rank, escape(value)])
        if rows:
            st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)
        for detail in exp.get("detail", []):
            st.markdown(f'<div class="term-root term-caption">— {escape(detail)}</div>', unsafe_allow_html=True)


def _render_section_report(title: str, dataset: str, variables: list, snapshot: dict) -> None:
    st.markdown(f'<div class="term-root"><div class="term-section">{escape(title)}</div></div>', unsafe_allow_html=True)
    headers = ["VAR", "STATUS", "CURRENT", "FORECAST", "Δ / TREND", "CONF", "GENERATED", "VALID PERIOD", "LEAD", "MODEL"]
    rows = []
    for variable in variables:
        headline_key = _headline_horizon_key(dataset, variable)
        entry = (snapshot.get("forecasts", {}).get(dataset, {}).get(variable, {}) or {}).get(headline_key)
        rows.append(_forecast_row(variable, entry))
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

    for variable in variables:
        headline_key = _headline_horizon_key(dataset, variable)
        headline_entry = (snapshot.get("forecasts", {}).get(dataset, {}).get(variable, {}) or {}).get(headline_key)
        if variable in EXPLAINED_VARIABLES and headline_entry is not None and headline_entry.get("explanation"):
            _render_explanation_expander(variable, headline_entry["explanation"])

        variable_forecasts = snapshot.get("forecasts", {}).get(dataset, {}).get(variable, {}) or {}
        with st.expander(f"{VARIABLE_LABELS_DISPLAY.get(variable, variable)} — full horizon set"):
            sub_headers = ["HORIZON", "STATUS", "FORECAST", "CONF", "MODEL", "FORECAST ID"]
            sub_rows = []
            for horizon_label, entry in variable_forecasts.items():
                if entry is None:
                    sub_rows.append([escape(horizon_label), "—", "—", "—", "—", "—"])
                    continue
                conf = entry.get("confidence") or {}
                sub_rows.append([
                    escape(horizon_label),
                    _status_span(entry.get("status", "—")),
                    _forecast_value_html(entry, variable),
                    _confidence_span(conf.get("category")),
                    escape(entry.get("model_name", "—")),
                    escape(entry.get("job_id", "—")[:8]),
                ])
            st.markdown(f'<div class="term-root">{_table(sub_headers, sub_rows)}</div>', unsafe_allow_html=True)


def _render_forecast_tab(snapshot: dict) -> None:
    generated_at = snapshot.get("generated_at")
    outlook = snapshot.get("overall_outlook", {})
    level = outlook.get("level", "Quiet")
    level_color = {"Quiet": ACCENT, "Unsettled": AMBER, "Minor Storm": AMBER, "Moderate Storm": RED, "Strong Storm": RED}.get(level, TEXT)

    st.markdown('<div class="term-root"><div class="term-section">Forecast Cycle</div></div>', unsafe_allow_html=True)
    headers = ["CYCLE GENERATED", "ENGINE VERSION", "ACTIVE JOBS", "STATUS"]
    rows = [[_fmt_time(generated_at), str(snapshot.get("engine_version", "—")), str(snapshot.get("active_jobs_total", 0)), _status_span("ACTIVE")]]
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

    _render_section_report("Solar Wind", "solar_wind", ["speed", "density", "temperature"], snapshot)
    _render_section_report("IMF", "imf", ["bt", "bx_gsm", "by_gsm", "bz_gsm"], snapshot)
    _render_section_report("Geomagnetic", "analytics", ["dst", "kp"], snapshot)
    _render_section_report("Auroral Electrojet", "ae", ["ae"], snapshot)

    st.markdown('<div class="term-root"><div class="term-section">Overall Space Weather Outlook</div></div>', unsafe_allow_html=True)
    headers = ["EXPECTED ACTIVITY", "PRIMARY DRIVER", "CONFIDENCE BASIS"]
    primary_driver = outlook.get("reasoning", ["—"])[0] if outlook.get("reasoning") else "—"
    rows = [[f'<span style="color:{level_color}; font-weight:700;">{escape(level.upper())}</span>', escape(primary_driver), f"{len(outlook.get('reasoning', []))} contributing indices"]]
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)
    for reason in outlook.get("reasoning", []):
        st.markdown(f'<div class="term-root term-caption">— {escape(reason)}</div>', unsafe_allow_html=True)


# ==================== Current ====================

CURRENT_ROWS = [
    ("speed", "SPD", " km/s", 1),
    ("density", "DENS", " p/cm3", 2),
    ("temperature", "TEMP", " K", 0),
    ("bt", "BT", " nT", 2),
    ("bx_gsm", "BX", " nT", 2),
    ("by_gsm", "BY", " nT", 2),
    ("bz_gsm", "BZ", " nT", 2),
    ("kp", "KP", "", 1),
    ("dst", "DST", " nT", 1),
    ("ae", "AE", " nT", 1),
]
RISK_COLORS = {
    "Usually quiet": ACCENT, "Normal": ACCENT, "Normal/active": ACCENT, "Low coupling": ACCENT,
    "Normal solar wind": ACCENT, "Weak pressure": ACCENT, "Low storm activity": ACCENT,
    "Low auroral activity": ACCENT, "Unsettled field": AMBER, "Minor activity possible": AMBER,
    "Compression possible": AMBER, "Disturbed flow possible": AMBER, "Minor storm": AMBER,
    "Minor ring current": AMBER, "Moderate auroral activity": AMBER, "Storm possible": RED,
    "Strong storm coupling": RED, "Shock/CME sheath possible": RED, "Severe storm potential": RED,
    "Enhanced storm potential": RED, "Shock/CME heating possible": RED, "Moderate storm": RED,
    "Strong storm": RED, "Storm underway": RED, "Elevated auroral activity": RED,
    "Severe/extreme storm": RED, "Strong disturbance": RED, "Extreme disturbance": RED,
    "High auroral activity": RED,
}


def _render_current_tab(snapshot: dict) -> None:
    conditions = snapshot.get("current_conditions", {})
    headers = ["VAR", "VALUE", "TIME (UTC)", "MEANING", "RISK"]
    rows = []
    for variable, tag, suffix, decimals in CURRENT_ROWS:
        c = conditions.get(variable) or {}
        value_text = _fmt(c.get("value"), decimals, suffix)
        risk = c.get("risk", "")
        risk_color = RISK_COLORS.get(risk, MUTED)
        rows.append([
            f"<b>{tag}</b>", value_text, _fmt_time(c.get("observed_at")),
            escape(c.get("meaning", "")), f'<span style="color:{risk_color};">{escape(risk)}</span>',
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

    anchor_time = conditions.get("anchor_time")
    if st.button("Solar Event ▸", key="cc_current_solar_event"):
        open_dialog("reverse_explorer", (pd.Timestamp(anchor_time) if anchor_time else None, "Current Conditions"))


# ==================== Physics ====================

PHYSICS_ROWS = [
    ("dynamic_pressure", "Dynamic Pressure", " nPa", "dynamic_pressure_label"),
    ("clock_angle_deg", "Clock Angle", "°", "clock_angle_label"),
    ("clock_angle_rate", "Clock Angle Rate", "°/h", None),
    ("southward_duration_hr", "Southward Duration", " h", None),
    ("integrated_southward_bz", "Integrated Southward Bz", " nT", None),
    ("integrated_vbz", "Integrated VBz", " nT·km/s", None),
    ("integrated_ey", "Integrated Ey", " mV/m", None),
    ("newell_coupling", "Newell Coupling", "", None),
    ("akasofu_epsilon_watts", "Akasofu ε", " W", None),
    ("boyle_index_kv", "Boyle Index", " kV", None),
    ("plasma_beta", "Plasma Beta", "", "plasma_beta_label"),
    ("alfven_mach_number", "Alfvén Mach Number", "", None),
    ("magnetopause_standoff_re", "Magnetopause Distance", " Re", None),
    ("estimated_compression_pct", "Estimated Compression", "%", None),
    ("magnetic_shear", "Magnetic Shear", " nT/h", None),
    ("imf_rotation_rate", "IMF Rotation Rate", "°", None),
]


def _render_physics_tab(snapshot: dict) -> None:
    physics = snapshot.get("physics_summary", {}) or {}
    if not physics:
        st.markdown('<div class="term-root term-caption">Physics summary not available yet.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="term-root"><div class="term-section">Engineering Readout</div></div>', unsafe_allow_html=True)
    headers = ["QUANTITY", "VALUE", "INTERPRETATION"]
    rows = []
    for key, label, suffix, label_key in PHYSICS_ROWS:
        value = physics.get(key)
        decimals = 0 if key == "akasofu_epsilon_watts" else 2
        value_text = _fmt(value, decimals, suffix)
        interpretation = physics.get(label_key, "") if label_key else ""
        rows.append([label, f'<span class="num">{value_text}</span>', escape(interpretation)])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={1})}</div>', unsafe_allow_html=True)

    coupling = physics.get("overall_coupling", "Unknown")
    coupling_color = {"Strong": RED, "Moderate": AMBER, "Weak": ACCENT}.get(coupling, MUTED)
    st.markdown(
        f'<div class="term-root term-caption">OVERALL COUPLING: '
        f'<span style="color:{coupling_color}; font-weight:700;">{escape(coupling.upper())}</span></div>',
        unsafe_allow_html=True,
    )


# ==================== Timeline ====================

def _render_timeline_tab(snapshot: dict) -> None:
    history = storage.load_forecast_history(days=14)
    if history.empty:
        st.markdown('<div class="term-root term-caption">No forecast history yet.</div>', unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)
    with col1:
        variable = st.selectbox("Variable", sorted(history["variable"].unique()), key="cc_timeline_var")
    horizon_options = sorted(history[history["variable"] == variable]["horizon"].unique())
    with col2:
        horizon = st.selectbox("Horizon", horizon_options, key="cc_timeline_horizon")

    subset = history[(history["variable"] == variable) & (history["horizon"] == horizon)].copy()
    subset["generated_at"] = pd.to_datetime(subset["generated_at"], utc=True, errors="coerce")
    subset = subset.sort_values("generated_at")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["generated_at"], y=subset["predicted_value"], mode="lines+markers", name="Forecast", line=dict(color=BLUE, width=1.4), marker=dict(size=4)))
    if subset["actual_value"].notna().any():
        fig.add_trace(go.Scatter(x=subset["generated_at"], y=subset["actual_value"], mode="markers", name="Observed", marker=dict(size=7, symbol="diamond", color=ACCENT)))
    _dark_plotly_layout(fig, f"{variable.upper()} @ {horizon} — Forecast vs. Observed", height=300)
    st.plotly_chart(fig, use_container_width=True, key="cc_timeline_chart")

    st.markdown('<div class="term-root"><div class="term-section">Lifecycle Log</div></div>', unsafe_allow_html=True)
    tail = subset.tail(40).iloc[::-1]
    headers = ["CYCLE GENERATED", "VALID PERIOD", "STATUS", "FORECAST", "OBSERVED", "CONF"]
    rows = []
    last_job = None
    for _, r in tail.iterrows():
        status = r.get("status", "—")
        if last_job is not None and r.get("job_id") != last_job and status in ("VERIFIED",):
            status = "ARCHIVED"
        last_job = r.get("job_id")
        rows.append([
            _fmt_time(r["generated_at"]),
            _fmt_period(r.get("valid_start"), r.get("valid_end")),
            _status_span(status),
            _fmt(r.get("predicted_value"), 2),
            _fmt(r.get("actual_value"), 2),
            _confidence_span(r.get("confidence_category")),
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)


# ==================== Verification ====================

def _verification_status(abs_error, mae) -> tuple:
    if abs_error is None or mae is None or mae == 0:
        return "—", MUTED
    ratio = abs_error / mae
    if ratio <= 1.0:
        return "WITHIN TOLERANCE", ACCENT
    if ratio <= 2.0:
        return "MODERATE DEVIATION", AMBER
    return "LARGE DEVIATION", RED


# The exact same "10 headline forecasts" definition the Forecast Package
# itself uses (packages.HEADLINE_KEYS) — reused here so "operational"
# means the same thing in the Verification tab as it does everywhere
# else in this app (the Forecast tab, the Package banner, the Package
# Verification Summary), rather than a second, subtly different
# definition of "headline" living in just this one tab.
_HEADLINE_SET = set(packages.HEADLINE_KEYS)
_OPERATIONAL_LABEL = "Operational (1h / Kp interval)"
_EXTENDED_LABEL = "Extended Horizons (3h / 6h / 12h / 24h)"


def _is_headline(dataset: str, variable: str, horizon) -> bool:
    return (dataset, variable, horizon) in _HEADLINE_SET


def _split_by_headline(df: pd.DataFrame) -> tuple:
    if df.empty:
        return df, df
    mask = df.apply(lambda r: _is_headline(r["dataset"], r["variable"], r["horizon"]), axis=1)
    return df[mask], df[~mask]


def _evaluation_caption(evals: pd.DataFrame) -> str:
    """The same 'EVALUATED: N · SUCCESS RATE (<=1.5x MAE): P%' summary
    line, computed for whatever subset is passed in — every sub-table in
    this tab gets its own honest count rather than inheriting the
    other's.
    """
    if evals.empty:
        return '<div class="term-root term-caption">No evaluated forecasts in this group yet.</div>'
    mae_by_key = evals.groupby(["variable", "horizon"])["abs_error"].mean().to_dict()
    total = len(evals)
    success = sum(
        1 for _, r in evals.iterrows()
        if mae_by_key.get((r["variable"], r["horizon"])) and r.get("abs_error", 1e9) <= 1.5 * mae_by_key[(r["variable"], r["horizon"])]
    )
    return (
        f'<div class="term-root term-caption">EVALUATED: {total}  ·  SUCCESS RATE (≤1.5x MAE): '
        f'{(success / total * 100):.0f}%</div>'
    )


def _verified_forecasts_subtable(evals: pd.DataFrame, subtitle: str) -> None:
    st.markdown(f'<div class="term-root term-section">{escape(subtitle)}</div>', unsafe_allow_html=True)
    st.markdown(_evaluation_caption(evals), unsafe_allow_html=True)
    if evals.empty:
        return

    grouped = evals.groupby(["variable", "horizon"])
    mae_by_key = grouped["abs_error"].mean().to_dict()
    rmse_by_key = grouped["squared_error"].mean().pow(0.5).to_dict()
    bias_by_key = grouped["error"].mean().to_dict()

    headers = ["VARIABLE", "HORIZON", "FORECAST", "OBSERVED", "ERROR", "MAE", "RMSE", "BIAS", "STATUS"]
    rows = []
    for _, r in evals.sort_values("completed_at", ascending=False).head(60).iterrows():
        key = (r["variable"], r["horizon"])
        mae = mae_by_key.get(key)
        status_text, status_color = _verification_status(r.get("abs_error"), mae)
        rows.append([
            f"<b>{escape(r['variable'].upper())}</b>",
            escape(str(r["horizon"])),
            f'<span class="num">{_fmt(r.get("final_predicted_value"), 2)}</span>',
            f'<span class="num">{_fmt(r.get("actual_value"), 2)}</span>',
            f'<span class="num">{_fmt_signed(r.get("error"), 2)}</span>',
            f'<span class="num">{_fmt(mae, 3)}</span>',
            f'<span class="num">{_fmt(rmse_by_key.get(key), 3)}</span>',
            f'<span class="num">{_fmt_signed(bias_by_key.get(key), 3)}</span>',
            f'<span style="color:{status_color}; font-weight:700;">{status_text}</span>',
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={2,3,4,5,6,7})}</div>', unsafe_allow_html=True)


def _render_verification_tab(snapshot: dict) -> None:
    evaluations = storage.load_evaluation_history()
    if evaluations.empty:
        st.markdown('<div class="term-root term-caption">No evaluated forecasts yet.</div>', unsafe_allow_html=True)
        return

    # Rows appended before this engine version added signed error/squared
    # error columns won't have them — backfill so older evaluations still
    # render (with a symmetric-error assumption for the missing bias/RMSE
    # contribution) rather than crashing on the newer schema.
    if "error" not in evaluations.columns:
        evaluations["error"] = pd.NA
    if "squared_error" not in evaluations.columns:
        evaluations["squared_error"] = evaluations["abs_error"] ** 2

    st.markdown(_evaluation_caption(evaluations), unsafe_allow_html=True)

    headline_evals, extended_evals = _split_by_headline(evaluations)

    with st.expander(f"▸ Verified Forecasts (showing up to 60 per table, {len(evaluations)} total)", expanded=False):
        _verified_forecasts_subtable(headline_evals, _OPERATIONAL_LABEL)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        _verified_forecasts_subtable(extended_evals, _EXTENDED_LABEL)

    package_verifications = storage.load_package_verification_history()
    if not package_verifications.empty:
        # Not split — a Forecast Package is BY DEFINITION built only from
        # the headline members (see packages.HEADLINE_KEYS), so there is
        # no "extended horizons" equivalent of a package to show here.
        with st.expander(f"▸ Package Verification Summary ({len(package_verifications)})", expanded=False):
            headers = ["PACKAGE ID", "VERIFIED AT", "VARS VERIFIED", "AVG ERROR", "WORST VAR", "BEST VAR", "ACCURACY"]
            rows = []
            for _, r in package_verifications.sort_values("verified_at", ascending=False).head(20).iterrows():
                rows.append([
                    r.get("package_id", "—"),
                    _fmt_time(r.get("verified_at")),
                    str(int(r.get("variables_verified", 0))),
                    f'<span class="num">{_fmt(r.get("average_abs_error"), 3)}</span>',
                    f"{r.get('worst_variable', '—')} ({_fmt(r.get('worst_variable_error'), 3)})",
                    f"{r.get('best_variable', '—')} ({_fmt(r.get('best_variable_error'), 3)})",
                    f'<span class="num">{_fmt(r.get("overall_package_accuracy_pct"), 0)}%</span>',
                ])
            st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={3,6})}</div>', unsafe_allow_html=True)

    _render_skill_section(headline_evals, extended_evals)
    _render_calibration_sections(headline_evals, extended_evals)


def _insufficient_data_note(min_samples: int) -> None:
    st.markdown(
        f'<div class="term-root term-caption">Not enough evaluated history yet — needs at least {min_samples} '
        f"evaluated forecasts per group before reporting anything (a handful of points is noise, not a finding). "
        f"This fills in on its own as the engine accumulates more verified forecasts.</div>",
        unsafe_allow_html=True,
    )


def _skill_subtable(scores: list, evals: pd.DataFrame, subtitle: str) -> None:
    st.markdown(f'<div class="term-root term-section">{escape(subtitle)}</div>', unsafe_allow_html=True)
    st.markdown(_evaluation_caption(evals), unsafe_allow_html=True)
    if not scores:
        _insufficient_data_note(skill.MIN_SAMPLES)
        return

    beating = sum(1 for s in scores if s["verdict"] == "Beats Persistence")
    st.markdown(
        f'<div class="term-root term-caption">GROUPS REPORTED: {len(scores)}  ·  BEATING PERSISTENCE: '
        f'{(beating / len(scores) * 100):.0f}%</div>',
        unsafe_allow_html=True,
    )
    headers = ["VARIABLE", "HORIZON", "SAMPLES", "MODEL RMSE", "PERSISTENCE RMSE", "SKILL SCORE", "VERDICT"]
    rows = []
    for s in scores:
        beats = s["verdict"] == "Beats Persistence"
        verdict_color = ACCENT if beats else RED
        rows.append([
            f"<b>{escape(s['variable'].upper())}</b>",
            s["horizon"],
            str(s["sample_n"]),
            f'<span class="num">{_fmt(s["model_rmse"], 3)}</span>',
            f'<span class="num">{_fmt(s["persistence_rmse"], 3)}</span>',
            f'<span class="num" style="color:{verdict_color}; font-weight:700;">{_fmt_signed(s["skill_score"], 3)}</span>',
            f'<span style="color:{verdict_color}; font-weight:700;">{escape(s["verdict"])}</span>',
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={2,3,4,5})}</div>', unsafe_allow_html=True)


def _render_skill_section(headline_evals: pd.DataFrame, extended_evals: pd.DataFrame) -> None:
    """Forecast Skill vs. Persistence — does the model actually beat the
    naive "assume no change" forecast, not just its own training
    benchmark (see swdss.engine.skill). AE never appears here: it has no
    live feed, so there is no persistence baseline to compare against.
    """
    scores = storage.load_skill_scores()
    headline_scores = [s for s in scores if _is_headline(s["dataset"], s["variable"], s["horizon"])]
    extended_scores = [s for s in scores if not _is_headline(s["dataset"], s["variable"], s["horizon"])]

    with st.expander(f"▸ Forecast Skill vs. Persistence ({len(scores)})", expanded=len(scores) > 0):
        st.markdown(
            '<div class="term-root term-caption">Skill score = 1 − (model MSE / persistence MSE). '
            "Positive means the model beats assuming no change between issuance and the target hour; "
            "zero or negative means it currently does not.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        _skill_subtable(headline_scores, headline_evals, _OPERATIONAL_LABEL)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        _skill_subtable(extended_scores, extended_evals, _EXTENDED_LABEL)


def _confidence_subtable(confidence_rows: list, evals: pd.DataFrame, subtitle: str) -> None:
    st.markdown(f'<div class="term-root term-section">{escape(subtitle)}</div>', unsafe_allow_html=True)
    st.markdown(_evaluation_caption(evals), unsafe_allow_html=True)
    reportable = [r for r in confidence_rows if r.get("status") == "OK"]
    if not reportable:
        _insufficient_data_note(calibration.MIN_SAMPLES)
        return
    headers = ["CONFIDENCE CATEGORY", "SAMPLES", "SUCCESS RATE (≤1.5x MAE)"]
    rows = []
    for r in confidence_rows:
        if r.get("status") != "OK":
            rows.append([escape(r["confidence_category"]), str(r["sample_n"]), f'<span style="color:{MUTED};">Insufficient Data</span>'])
            continue
        rows.append([
            f"<b>{escape(r['confidence_category'])}</b>",
            str(r["sample_n"]),
            f'<span class="num">{_fmt(r["success_rate_pct"], 1)}%</span>',
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={1,2})}</div>', unsafe_allow_html=True)


def _regime_subtable(regime_rows: list, evals: pd.DataFrame, subtitle: str) -> None:
    st.markdown(f'<div class="term-root term-section">{escape(subtitle)}</div>', unsafe_allow_html=True)
    st.markdown(_evaluation_caption(evals), unsafe_allow_html=True)
    if not regime_rows:
        _insufficient_data_note(calibration.MIN_SAMPLES)
        return
    headers = ["VARIABLE", "HORIZON", "REGIME", "SAMPLES", "MEAN ABS ERROR"]
    rows = []
    for r in regime_rows:
        regime_color = {"Quiet": ACCENT, "Active": AMBER, "Storm": RED}.get(r["activity_regime"], MUTED)
        rows.append([
            f"<b>{escape(r['variable'].upper())}</b>",
            r["horizon"],
            f'<span style="color:{regime_color}; font-weight:700;">{escape(r["activity_regime"])}</span>',
            str(r["sample_n"]),
            f'<span class="num">{_fmt(r["mean_abs_error"], 3)}</span>',
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={3,4})}</div>', unsafe_allow_html=True)


def _render_calibration_sections(headline_evals: pd.DataFrame, extended_evals: pd.DataFrame) -> None:
    """Confidence calibration (is a "Very High" forecast actually more
    reliable than a "Moderate" one?) and activity-regime error bands (is
    accuracy worse specifically during storms, when it matters most?) —
    see swdss.engine.calibration. Pure analysis over already-collected
    evaluation history; neither result feeds back into how confidence or
    activity regime are computed.

    Confidence calibration is recomputed here directly on the headline/
    extended split (rather than read pre-split from storage) because it
    groups by confidence category, not by variable/horizon — splitting
    it can't be done by filtering already-computed rows the way Skill
    and Regime below can; it needs its own grouping pass per subset.
    This is the same pure pandas aggregation swdss.engine.calibration
    always does — nothing here recomputes a model or a forecast, exactly
    like the Verified Forecasts table above already aggregates inline.
    """
    headline_confidence = calibration.compute_confidence_calibration(headline_evals)
    extended_confidence = calibration.compute_confidence_calibration(extended_evals)

    with st.expander(f"▸ Confidence Calibration ({len(headline_confidence) + len(extended_confidence)})", expanded=False):
        st.markdown(
            '<div class="term-root term-caption">A well-calibrated confidence score shows success rate '
            "decreasing monotonically from Very High down to Low. A flat or inverted ordering means the "
            "heuristic's weights (swdss.engine.confidence) don't track real accuracy and should be revisited.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        _confidence_subtable(headline_confidence, headline_evals, _OPERATIONAL_LABEL)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        _confidence_subtable(extended_confidence, extended_evals, _EXTENDED_LABEL)

    report = storage.load_calibration_report()
    regime_rows_all = report.get("regime_error_bands") or []
    headline_regime = [r for r in regime_rows_all if _is_headline(r["dataset"], r["variable"], r["horizon"])]
    extended_regime = [r for r in regime_rows_all if not _is_headline(r["dataset"], r["variable"], r["horizon"])]

    with st.expander(f"▸ Error by Activity Regime ({len(regime_rows_all)})", expanded=False):
        st.markdown(
            '<div class="term-root term-caption">The classic space-weather ML failure mode: a model that looks '
            "strong on a quiet-time-dominated aggregate and degrades exactly when a storm makes the forecast "
            "matter most. This segments already-observed error by regime — it never changes a model's training "
            "or its dashboard-reported aggregate MAE.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        _regime_subtable(headline_regime, headline_evals, _OPERATIONAL_LABEL)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        _regime_subtable(extended_regime, extended_evals, _EXTENDED_LABEL)


# ==================== Logs ====================

def _render_logs_tab(snapshot: dict) -> None:
    query = st.text_input("Filter", "", key="cc_logs_filter", placeholder="filter by stage, level, or message text")
    logs = storage.load_recent_logs(500)
    if query:
        q = query.lower()
        logs = [
            entry
            for entry in logs
            if q in str(entry.get("stage", "")).lower()
            or q in str(entry.get("message", "")).lower()
            or q in str(entry.get("level", "")).lower()
        ]

    if not logs:
        st.markdown('<div class="term-root term-caption">No matching log lines.</div>', unsafe_allow_html=True)
        return

    lines_html = ""
    for entry in logs[:300]:
        level = entry.get("level", "INFO")
        color = LOG_LEVEL_COLORS.get(level, TEXT)
        ts = entry.get("ts", "")
        stage = entry.get("stage", "")
        message = entry.get("message", "")
        lines_html += (
            f'<div class="term-logline"><span style="color:{color};">{escape(level):>5}</span> '
            f'{escape(str(ts))}  <b>{escape(stage)}</b>  {escape(str(message))}</div>'
        )
    st.markdown(f'<div class="term-root term-scroll">{lines_html}</div>', unsafe_allow_html=True)


# ==================== Downloads ====================

def _file_row(label: str, path, fmt: str, mime: str) -> None:
    if not path.exists():
        st.markdown(f'<div class="term-root term-caption">{escape(label)} — not yet generated</div>', unsafe_allow_html=True)
        return
    stat = path.stat()
    size_kb = stat.st_size / 1024
    mtime = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
    headers = ["FILENAME", "CREATED", "SIZE", "FORMAT"]
    rows = [[path.name, _fmt_time(mtime.isoformat()), f"{size_kb:.1f} KB", fmt]]
    col_table, col_btn = st.columns([4, 1])
    with col_table:
        st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)
    with col_btn:
        st.download_button("↓ FETCH", data=path.read_bytes(), file_name=path.name, mime=mime, key=f"dl_{path.name}")


def _render_downloads_tab(snapshot: dict) -> None:
    st.markdown('<div class="term-root"><div class="term-section">Forecast Package</div></div>', unsafe_allow_html=True)
    _file_row("Current Package", storage.CURRENT_PACKAGE_PATH, "JSON", "application/json")
    _file_row("Package History", storage.PACKAGE_HISTORY_PATH, "PARQUET", "application/octet-stream")
    _file_row("Package Verification History", storage.PACKAGE_VERIFICATION_HISTORY_PATH, "PARQUET", "application/octet-stream")

    package_history = storage.load_package_history()
    if not package_history.empty:
        flat_cols = [c for c in package_history.columns if c != "snapshot_json"]
        st.download_button(
            "↓ FETCH package_history.csv", data=package_history[flat_cols].to_csv(index=False),
            file_name="package_history.csv", mime="text/csv", key="dl_pkg_history_csv",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="term-root"><div class="term-section">Forecast Products</div></div>', unsafe_allow_html=True)
    _file_row("Current Snapshot", storage.CURRENT_SNAPSHOT_PATH, "JSON", "application/json")
    _file_row("Forecast History", storage.FORECAST_HISTORY_PATH, "PARQUET", "application/octet-stream")
    _file_row("Evaluation History", storage.EVALUATION_HISTORY_PATH, "PARQUET", "application/octet-stream")
    _file_row("Engine Log", storage.ENGINE_LOG_PATH, "JSONL", "application/json")

    history = storage.load_forecast_history()
    if not history.empty:
        st.download_button("↓ FETCH forecast_history.csv", data=history.to_csv(index=False), file_name="forecast_history.csv", mime="text/csv", key="dl_history_csv")
    evaluations = storage.load_evaluation_history()
    if not evaluations.empty:
        st.download_button("↓ FETCH evaluation_history.csv", data=evaluations.to_csv(index=False), file_name="evaluation_history.csv", mime="text/csv", key="dl_eval_csv")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="term-root"><div class="term-section">Verification Science</div></div>', unsafe_allow_html=True)
    _file_row("Skill Scores", storage.SKILL_SCORES_PATH, "JSON", "application/json")
    _file_row("Calibration Report", storage.CALIBRATION_REPORT_PATH, "JSON", "application/json")


# ==================== Alerts ====================

def _render_alerts_tab(snapshot: dict) -> None:
    alerts = snapshot.get("alerts", [])
    if not alerts:
        st.markdown(f'<div class="term-root" style="color:{ACCENT}; font-size:0.8rem;">NO ACTIVE ALERTS</div>', unsafe_allow_html=True)
        return

    headers = ["TIMESTAMP", "SEVERITY", "SOURCE", "DESCRIPTION", "STATUS"]
    rows = []
    for alert in alerts:
        severity = alert.get("severity", "info")
        color = SEVERITY_COLORS.get(severity, MUTED)
        since = alert.get("since", "")
        rows.append([
            _fmt_time(since),
            f'<span style="color:{color}; font-weight:700;">{escape(severity.upper())}</span>',
            escape(alert.get("source", "—")),
            escape(alert.get("message", "")),
            f'<span style="color:{color};">OPEN</span>',
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)


# ==================== System ====================

def _render_system_tab(snapshot: dict) -> None:
    services = snapshot.get("system_health", [])
    with st.expander(f"▸ Service Status ({len(services)})", expanded=True):
        headers = ["SERVICE", "STATUS", "DETAIL"]
        rows = []
        for svc in services:
            color = SERVICE_STATUS_COLORS.get(svc.get("status"), MUTED)
            rows.append([svc.get("name", "—"), f'<span style="color:{color}; font-weight:700;">{escape(svc.get("status", "—"))}</span>', escape(svc.get("detail", ""))])
        st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

    freshness = snapshot.get("freshness", {})
    with st.expander(f"▸ Data Source Freshness ({len(freshness)})", expanded=True):
        headers = ["SOURCE", "STATUS", "AGE"]
        rows = []
        for name, status in freshness.items():
            color = ACCENT if status.get("status") == "Fresh" else (RED if status.get("status") == "No data" else AMBER)
            rows.append([name.replace("_", " ").upper(), f'<span style="color:{color}; font-weight:700;">{escape(status.get("status", "—").upper())}</span>', escape(status.get("age", "—"))])
        st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

    model_rows = []
    for dataset, variables in snapshot.get("forecasts", {}).items():
        for variable, horizons in variables.items():
            for horizon_label, entry in horizons.items():
                if entry is None:
                    continue
                model_rows.append([
                    dataset.upper(), variable.upper(), horizon_label,
                    escape(entry.get("model_name", "—")),
                    _fmt_time(entry.get("model_trained_at")),
                    f'<span class="num">{_fmt(entry.get("metrics", {}).get("r2"), 3)}</span>',
                    _status_span(entry.get("status", "—")),
                ])
    with st.expander(f"▸ Model Versions In Use ({len(model_rows)})", expanded=False):
        headers = ["DATASET", "VARIABLE", "HORIZON", "ALGORITHM", "TRAINED", "R²", "STATUS"]
        if model_rows:
            st.markdown(f'<div class="term-root">{_table(headers, model_rows, numeric_cols={5})}</div>', unsafe_allow_html=True)


# ==================== Search ====================

# (dataset, column, mode, label, decimals, suffix) — reads the minute-
# level processed parquet directly (never master_df_v1.parquet) so a
# real spike isn't smoothed away by the hourly-merged table. Bx/By are
# deliberately excluded: neither has a canonical "interesting direction"
# the way Bz (southward = geoeffective) or Bt (magnitude) do.
EXTREME_SPECS = [
    ("solar_wind", "solar_wind_speed", "max", "Highest Speed", 1, " km/s"),
    ("solar_wind", "proton_density", "max", "Highest Density", 2, " p/cm3"),
    ("solar_wind", "temperature", "max", "Highest Temperature", 0, " K"),
    ("imf", "bt", "max", "Highest Bt", 2, " nT"),
    ("imf", "bz", "min", "Lowest Bz", 2, " nT"),
    ("kp", "kp", "max", "Highest Kp", 1, ""),
    ("dst", "dst", "min", "Lowest Dst", 1, " nT"),
    ("ae", "ae", "max", "Highest AE", 1, " nT"),
]


def _load_7day_extremes() -> list:
    """One row per EXTREME_SPECS entry: the true minute-level extreme
    over the trailing 7 days, with its timestamp. Independent of the
    Operational Forecast Engine — this is observed data, not a forecast
    product — so it reads swdss.paths.PROCESSED_DIR directly rather than
    swdss.engine.storage.
    """
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    rows = []
    for dataset, column, mode, label, decimals, suffix in EXTREME_SPECS:
        path = PROCESSED_DIR / dataset / f"{dataset}_processed.parquet"
        try:
            df = pd.read_parquet(path, columns=["timestamp_utc", column])
        except Exception:
            rows.append([label, "—", "—"])
            continue
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc", column])
        df = df[df["timestamp_utc"] >= cutoff]
        if df.empty:
            rows.append([label, "—", "—"])
            continue
        idx = df[column].idxmax() if mode == "max" else df[column].idxmin()
        best = df.loc[idx]
        rows.append([label, f'<span class="num">{_fmt(best[column], decimals, suffix)}</span>', _fmt_time(best["timestamp_utc"])])
    return rows


def _render_search_tab(snapshot: dict) -> None:
    """Lookup over swdss.engine.storage.load_forecast_history() — no new
    computation, purely a filter over the same append-only history the
    Timeline tab already reads. Given a variable and a target date/hour,
    shows every horizon's forecast whose VALID WINDOW covers that moment
    — since every horizon's valid window is the same 1-hour (or Kp's 3-
    hour) target slot (see orchestrator._valid_window), a 1h-ahead and a
    24h-ahead forecast for the identical historical moment both show up
    side by side, letting you compare how much lead time actually cost
    in accuracy for that specific event.
    """
    st.markdown('<div class="term-root term-caption">Look up what was forecast for a specific past moment — e.g. "what did we predict for the hour Speed hit its 7-day high?"</div>', unsafe_allow_html=True)

    st.markdown('<div class="term-root"><div class="term-section">7-Day Extremes (observed)</div></div>', unsafe_allow_html=True)
    extreme_rows = _load_7day_extremes()
    st.markdown(f'<div class="term-root">{_table(["METRIC", "VALUE", "OBSERVED AT"], extreme_rows, numeric_cols={1})}</div>', unsafe_allow_html=True)

    st.markdown('<div class="term-root term-caption">Search a specific moment below — plug a date/hour here in and pull up its forecast.</div>', unsafe_allow_html=True)
    variables = list(VARIABLE_LABELS_DISPLAY.keys())
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        variable = st.selectbox("Variable", variables, format_func=lambda v: VARIABLE_LABELS_DISPLAY.get(v, v), key="cc_search_var")
    with col2:
        search_date = st.date_input("Date (UTC)", value=pd.Timestamp.now(tz="UTC").date(), key="cc_search_date")
    with col3:
        search_hour = st.selectbox("Hour (UTC)", list(range(24)), key="cc_search_hour")

    target_time = pd.Timestamp(search_date, tz="UTC") + pd.Timedelta(hours=search_hour)

    history = storage.load_forecast_history()
    if history.empty:
        st.markdown('<div class="term-root term-caption">No forecast history recorded yet.</div>', unsafe_allow_html=True)
        return

    subset = history[history["variable"] == variable].copy()
    if subset.empty:
        st.markdown('<div class="term-root term-caption">No history for this variable yet.</div>', unsafe_allow_html=True)
        return

    subset["generated_at"] = pd.to_datetime(subset["generated_at"], utc=True, errors="coerce")
    subset["valid_start"] = pd.to_datetime(subset["valid_start"], utc=True, errors="coerce")
    subset["valid_end"] = pd.to_datetime(subset["valid_end"], utc=True, errors="coerce")

    covering = subset[(subset["valid_start"] <= target_time) & (target_time < subset["valid_end"])]
    if covering.empty:
        st.markdown(
            f'<div class="term-root term-caption">No forecast found covering {escape(target_time.strftime("%Y-%m-%d %H:%M UTC"))} '
            f'for {escape(VARIABLE_LABELS_DISPLAY.get(variable, variable))}.</div>',
            unsafe_allow_html=True,
        )
        return

    # One settled record per (horizon, valid_start): the last-logged row
    # for that combination, since the same forecast is re-logged every
    # cycle while it's LIVE/ACTIVE and its status (and eventually its
    # actual_value) keeps advancing — the last one logged reflects its
    # most up-to-date state.
    settled = (
        covering.sort_values("generated_at")
        .drop_duplicates(subset=["horizon", "valid_start"], keep="last")
        .sort_values("horizon", key=lambda s: s.map(_horizon_sort_key))
    )

    st.markdown(f'<div class="term-root"><div class="term-section">Matches — {escape(VARIABLE_LABELS_DISPLAY.get(variable, variable))} @ {escape(target_time.strftime("%Y-%m-%d %H:%M UTC"))}</div></div>', unsafe_allow_html=True)
    headers = ["HORIZON", "ISSUED (UTC)", "VALID PERIOD", "PREDICTED", "ACTUAL", "STATUS", "CONF", "MODEL"]
    rows = []
    for _, r in settled.iterrows():
        rows.append([
            r["horizon"],
            _fmt_time(r["generated_at"]),
            _fmt_period(r["valid_start"].isoformat(), r["valid_end"].isoformat()),
            f'<span class="num">{_fmt(r.get("predicted_value"), 2)}</span>',
            f'<span class="num">{_fmt(r.get("actual_value"), 2)}</span>',
            _status_span(r.get("status", "—")),
            _confidence_span(r.get("confidence_category")),
            escape(r.get("model_name", "—")),
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={3,4})}</div>', unsafe_allow_html=True)

    horizon_options = list(settled["horizon"])
    if horizon_options:
        with st.expander("▸ Full cycle history for one horizon"):
            chosen_horizon = st.selectbox("Horizon", horizon_options, key="cc_search_detail_horizon")
            detail = covering[covering["horizon"] == chosen_horizon].sort_values("generated_at", ascending=False)
            detail_headers = ["CYCLE GENERATED", "STATUS", "PREDICTED", "ACTUAL", "CONF"]
            detail_rows = [
                [
                    _fmt_time(r["generated_at"]),
                    _status_span(r.get("status", "—")),
                    _fmt(r.get("predicted_value"), 2),
                    _fmt(r.get("actual_value"), 2),
                    _confidence_span(r.get("confidence_category")),
                ]
                for _, r in detail.iterrows()
            ]
            st.markdown(f'<div class="term-root">{_table(detail_headers, detail_rows)}</div>', unsafe_allow_html=True)


# ==================== Solar Forecast ====================

_F107_METHOD_SPAN = {
    "harmonic": (ACCENT, "HARMONIC (VALIDATED)"),
    "seasonal_naive_fallback_underperformed": (AMBER, "27-DAY SEASONAL-NAIVE (HARMONIC UNDERPERFORMED)"),
    "seasonal_naive_fallback_insufficient_history": (MUTED, "27-DAY SEASONAL-NAIVE (INSUFFICIENT HISTORY)"),
    "seasonal_naive_fallback_unvalidated": (MUTED, "27-DAY SEASONAL-NAIVE (HARMONIC UNVALIDATED)"),
}


def _render_f107_section(f107: dict | None) -> None:
    st.markdown('<div class="term-root"><div class="term-section">F10.7 Outlook</div></div>', unsafe_allow_html=True)
    if f107 is None:
        st.markdown('<div class="term-root term-caption">No F10.7 data available yet.</div>', unsafe_allow_html=True)
        return

    color, method_label = _F107_METHOD_SPAN.get(f107.get("tomorrow_method"), (MUTED, f107.get("tomorrow_method", "—").upper()))
    headers = ["LAST OBSERVED", "TOMORROW FORECAST", "27D SEASONAL-NAIVE", "METHOD", "HISTORY"]
    rows = [[
        f'{_fmt(f107.get("last_observed_value"), 0)} <span class="term-caption">({escape(str(f107.get("last_observed_date", "—"))[:10])})</span>',
        f'<span class="num" style="font-weight:700;">{_fmt(f107.get("tomorrow_forecast"), 0)}</span>',
        f'<span class="num">{_fmt(f107.get("tomorrow_naive_baseline"), 0)}</span>',
        f'<span style="color:{color}; font-weight:700;">{escape(method_label)}</span>',
        f'{f107.get("n_days_history", 0)} days <span class="term-caption">(need {f107.get("min_days_for_harmonic", "—")}+ for harmonic)</span>',
    ]]
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

    skill = f107.get("skill")
    if skill is not None:
        s_headers = ["HARMONIC MAE (HOLDOUT)", "NAIVE MAE (HOLDOUT)", "SKILL SCORE", "HOLDOUT SAMPLES"]
        skill_score = skill.get("skill_score")
        skill_color = ACCENT if (skill_score or 0) > 0 else RED
        s_rows = [[
            f'<span class="num">{_fmt(skill.get("model_mae"), 2)}</span>',
            f'<span class="num">{_fmt(skill.get("naive_mae"), 2)}</span>',
            f'<span class="num" style="color:{skill_color}; font-weight:700;">{_fmt_signed(skill_score, 2)}</span>',
            str(skill.get("n_samples", "—")),
        ]]
        st.markdown(f'<div class="term-root">{_table(s_headers, s_rows)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="term-root term-caption">Not enough history yet to run a holdout skill check against the naive baseline.</div>', unsafe_allow_html=True)

    week_trend = f107.get("week_trend") or []
    if week_trend:
        with st.expander("▸ 7-Day Trend (Harmonic Model vs. Seasonal-Naive)"):
            t_headers = ["DAYS AHEAD", "HARMONIC MODEL", "27D SEASONAL-NAIVE"]
            t_rows = [
                [str(day.get("day")), f'<span class="num">{_fmt(day.get("model"), 1)}</span>', f'<span class="num">{_fmt(day.get("naive"), 1)}</span>']
                for day in week_trend
            ]
            st.markdown(f'<div class="term-root">{_table(t_headers, t_rows, numeric_cols={1, 2})}</div>', unsafe_allow_html=True)


def _render_cme_arrival_section(cme_arrivals: list) -> None:
    st.markdown('<div class="term-root"><div class="term-section">CME Arrival — Drag-Based Model</div></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="term-root term-caption">Vršnak et al. (2013) — each CME decelerates or accelerates toward the '
        'ambient solar wind speed rather than travelling at a fixed launch speed the whole way; arrival is reported '
        'as a window (fastest/typical/slowest plausible transit), not a single number.</div>',
        unsafe_allow_html=True,
    )
    if not cme_arrivals:
        st.markdown('<div class="term-root term-caption">No CME with a usable speed reading in the recent lookback window.</div>', unsafe_allow_html=True)
        return

    headers = ["LAUNCHED", "SPEED", "TYPE", "ARRIVAL (MEDIAN)", "ARRIVAL WINDOW", "AMBIENT WIND USED", "TRAVEL TIME (MEDIAN)", "SOURCE REGION HELICITY"]
    rows = []
    for cme in cme_arrivals:
        helicity = cme.get("source_region_helicity")
        if helicity and helicity.get("mean_current_helicity") is not None:
            helicity_cell = f'{_fmt(helicity["mean_current_helicity"], 4)} <span class="term-caption">({escape(helicity["sign_label"])})</span>'
        else:
            helicity_cell = '<span class="term-caption">no matched active region</span>'
        rows.append([
            _fmt_time(cme.get("launch_time")),
            f'<span class="num">{_fmt(cme.get("launch_speed_km_s"), 0)} km/s</span>',
            escape(str(cme.get("cme_type") or "—")),
            _fmt_time(cme.get("arrival_median")),
            f'{_fmt_time(cme.get("arrival_min"))} – {_fmt_time(cme.get("arrival_max"))}',
            f'<span class="num">{_fmt(cme.get("ambient_speed_used_km_s"), 0)} km/s</span>',
            f'<span class="num">{_fmt(cme.get("travel_hours_median"), 1)} h</span>',
            helicity_cell,
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={1, 5, 6})}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="term-root term-caption">Source Region Helicity is descriptive, not predictive — it\'s the '
        'source active region\'s measured magnetic helicity sign (SDO/HMI SHARP), relevant to the hemispheric '
        'helicity rule research on CME flux-rope chirality. It is NOT a Bz-sign forecast, and does not replace the '
        'production Bz model (Heliosphere → IMF), which forecasts ambient solar wind Bz — a different, continuous '
        'question from this CME-specific, descriptive note. See CME Outlook below for occurrence-probability '
        'scoring; storm-strength/geoeffectiveness scoring (how strong an impact, not whether one occurs) is still '
        'a separate, unbuilt capability.</div>',
        unsafe_allow_html=True,
    )


_OUTLOOK_STATUS_MESSAGES = {
    "not_trained": "Model not yet trained — insufficient historical flare/CME event history at the time this pipeline was built.",
    "insufficient_positive_examples": "Not enough historical positive examples (flares/CMEs) yet to train a validated model.",
    "degenerate_split": "Historical positive examples exist but aren't spread across a valid train/test split yet — needs more history.",
    "trained_no_measurable_skill": "A model was trained, but scored no better than chance (TSS ≤ 0) on a holdout test — not shown as a live forecast until it demonstrates real skill. Expected outcome for a first pass on ~2 months of history across ~45 active regions, not a bug.",
    "fetch_failed": "Live SHARP data fetch from JSOC failed this cycle — showing the last successful result if any.",
    "no_active_regions": "No active regions with usable SHARP data in the current live snapshot.",
    "no_scorable_regions": "Active regions present, but none have enough SHARP history yet (need 24h+ per region) to score.",
}


def _render_outlook_section(title: str, outlook: dict | None, caption: str) -> None:
    st.markdown(f'<div class="term-root"><div class="term-section">{escape(title)}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="term-root term-caption">{caption}</div>', unsafe_allow_html=True)

    if not outlook or outlook.get("status") != "ok":
        status = (outlook or {}).get("status", "not_trained")
        message = _OUTLOOK_STATUS_MESSAGES.get(status, status)
        st.markdown(f'<div class="term-root term-caption">{escape(message)}</div>', unsafe_allow_html=True)
        return

    headers = ["ACTIVE REGION", "PROBABILITY (NEXT 24H)"]
    rows = []
    for region in outlook.get("regions", []):
        prob = region["probability"]
        color = RED if prob >= 0.5 else (AMBER if prob >= 0.2 else ACCENT)
        rows.append([
            f"AR{region['noaa_ar']}",
            f'<span class="num" style="color:{color}; font-weight:700;">{prob:.0%}</span>',
        ])
    st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={1})}</div>', unsafe_allow_html=True)


def _render_solar_forecast_tab() -> None:
    """Reads swdss.engine.storage.load_solar_forecast() only — F10.7's
    harmonic-regression outlook, CME Drag-Based Model arrival estimates,
    and (as of the SHARP pipeline) Flare Outlook / CME Outlook
    probabilities are all computed once per engine cycle in
    orchestrator._solar_forecast_snapshot, never recomputed here, the
    same "engine computes, dashboard only reads" rule every other tab in
    this module follows.
    """
    solar = storage.load_solar_forecast()
    if solar is None:
        st.markdown(
            '<div class="term-root" style="color:#6e7681; font-size:0.8rem;">'
            "NO SOLAR FORECAST SNAPSHOT YET — run `python -m swdss.features.live_update` "
            "or call refresh_dashboard_products() manually."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    _render_outlook_section(
        "Flare Outlook",
        solar.get("flare_outlook"),
        "Probability of an M-class+ flare in the next 24h, per active region — trained on SDO/HMI SHARP magnetic-"
        "complexity features (USFLUX, R_VALUE, TOTPOT, MEANJZH, and related keywords) via RandomForestClassifier, "
        "scored by True Skill Statistic and Brier score against a chronological holdout, not accuracy (which a "
        "model that always predicts \"no flare\" would already win at, since flares are rare).",
    )
    _render_outlook_section(
        "CME Outlook",
        solar.get("cme_outlook"),
        "Probability of an Earth-directed CME in the next 24h, per active region — same SHARP features and "
        "methodology as Flare Outlook, trained as a separate model since flare occurrence and CME occurrence are "
        "related but distinct events.",
    )
    _render_f107_section(solar.get("f107"))
    _render_cme_arrival_section(solar.get("cme_arrivals") or [])
