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

Ten fixed tabs: Forecast (default), Current, Physics, Timeline,
Verification, Logs, Downloads, Alerts, System, Search.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.shared_ui import open_dialog
from swdss.engine import storage
from swdss.paths import PROCESSED_DIR

# ==================== Terminal design tokens ====================

BG = "#090c10"
PANEL_BG = "#0d1117"
BORDER = "#1c2530"
TEXT = "#c9d1d9"
MUTED = "#6e7681"
ACCENT = "#39d98a"      # section headers, positive/online status
AMBER = "#e3b341"       # warnings, moderate deviation
RED = "#f85149"         # critical, large deviation
BLUE = "#58a6ff"        # informational, links, LIVE status
MONO = "'JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', monospace"

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
    return f'<span class="arrow">→</span>'.join(parts)


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
    conf = package.get("overall_confidence", {})

    st.markdown(
        f"""
        <div class="term-pkg-bar term-root">
            <b style="color:{ACCENT};">{escape(package.get('package_id', '—'))}</b>
            <span>CYCLE #{package.get('forecast_cycle', '—')}</span>
            <span>ISSUED {escape(_fmt_time(package.get('generated_at')))}</span>
            <span>VALID {escape(_fmt_period(package.get('valid_start'), package.get('valid_end')))}</span>
            <span style="color:{status_color}; font-weight:700;">[{escape(status)}]</span>
            <span style="color:{completeness_color};">{escape(completeness)}</span>
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
            ["Valid Period", _fmt_period(package.get("valid_start"), package.get("valid_end"))],
            ["Status", _status_span(status)],
            ["Completeness", f'<span style="color:{completeness_color};">{escape(completeness)}</span>'],
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
        "Logs", "Downloads", "Alerts", "System", "Search",
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


# ==================== Forecast ====================

def _headline_horizon_key(dataset: str, variable: str) -> str:
    return "interval" if (dataset == "analytics" and variable == "kp") else "1h"


def _forecast_value_html(entry: dict, variable: str) -> str:
    if entry is None or entry.get("predicted_value") is None:
        return "—"
    unit = UNITS.get(variable, "")
    decimals = 1 if variable == "kp" else 2
    if entry.get("range_low") is not None and entry.get("range_high") is not None:
        return f"{entry['range_low']:.{decimals}f} &rarr; {entry['range_high']:.{decimals}f}{(' ' + unit) if unit else ''}"
    return f"{entry['predicted_value']:.{decimals}f}{(' ' + unit) if unit else ''}"


def _delta_trend_html(entry: dict, variable: str) -> str:
    if entry is None:
        return "—"
    unit = UNITS.get(variable, "")
    decimals = 1 if variable == "kp" else 2
    if variable in ("bz_gsm", "kp"):
        return escape(entry.get("trend", "—"))
    delta = entry.get("delta")
    if delta is None:
        return "—"
    return escape(_fmt_signed(delta, decimals, (" " + unit) if unit else ""))


def _forecast_row(variable: str, entry: dict) -> list:
    unit = UNITS.get(variable, "")
    decimals = 1 if variable == "kp" else 2
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

    st.markdown(f'<div class="term-root"><div class="term-section">Forecast Cycle</div></div>', unsafe_allow_html=True)
    headers = ["CYCLE GENERATED", "ENGINE VERSION", "ACTIVE JOBS", "STATUS"]
    rows = [[_fmt_time(generated_at), str(snapshot.get("engine_version", "—")), str(snapshot.get("active_jobs_total", 0)), _status_span("ACTIVE")]]
    st.markdown(f'<div class="term-root">{_table(headers, rows)}</div>', unsafe_allow_html=True)

    _render_section_report("Solar Wind", "solar_wind", ["speed", "density", "temperature"], snapshot)
    _render_section_report("IMF", "imf", ["bt", "bx_gsm", "by_gsm", "bz_gsm"], snapshot)
    _render_section_report("Geomagnetic", "analytics", ["dst", "kp"], snapshot)
    _render_section_report("Auroral Electrojet", "ae", ["ae"], snapshot)

    st.markdown(f'<div class="term-root"><div class="term-section">Overall Space Weather Outlook</div></div>', unsafe_allow_html=True)
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

    st.markdown(f'<div class="term-root"><div class="term-section">Engineering Readout</div></div>', unsafe_allow_html=True)
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

    st.markdown(f'<div class="term-root"><div class="term-section">Lifecycle Log</div></div>', unsafe_allow_html=True)
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

    mae_by_var = evaluations.groupby("variable")["abs_error"].mean().to_dict()
    rmse_by_var = {v: (evaluations.loc[evaluations["variable"] == v, "squared_error"].mean() ** 0.5) for v in mae_by_var}
    bias_by_var = evaluations.groupby("variable")["error"].mean().to_dict()

    total = len(evaluations)
    success = sum(1 for _, r in evaluations.iterrows() if mae_by_var.get(r["variable"]) and r.get("abs_error", 1e9) <= 1.5 * mae_by_var[r["variable"]])
    st.markdown(
        f'<div class="term-root term-caption">EVALUATED: {total}  ·  SUCCESS RATE (≤1.5x MAE): '
        f'{(success / total * 100):.0f}%</div>',
        unsafe_allow_html=True,
    )

    with st.expander(f"▸ Verified Forecasts (showing up to 60 of {total})", expanded=False):
        headers = ["VARIABLE", "FORECAST", "OBSERVED", "ERROR", "MAE", "RMSE", "BIAS", "STATUS"]
        rows = []
        for _, r in evaluations.sort_values("completed_at", ascending=False).head(60).iterrows():
            variable = r["variable"]
            mae = mae_by_var.get(variable)
            status_text, status_color = _verification_status(r.get("abs_error"), mae)
            rows.append([
                f"<b>{escape(variable.upper())}</b>",
                f'<span class="num">{_fmt(r.get("final_predicted_value"), 2)}</span>',
                f'<span class="num">{_fmt(r.get("actual_value"), 2)}</span>',
                f'<span class="num">{_fmt_signed(r.get("error"), 2)}</span>',
                f'<span class="num">{_fmt(mae, 3)}</span>',
                f'<span class="num">{_fmt(rmse_by_var.get(variable), 3)}</span>',
                f'<span class="num">{_fmt_signed(bias_by_var.get(variable), 3)}</span>',
                f'<span style="color:{status_color}; font-weight:700;">{status_text}</span>',
            ])
        st.markdown(f'<div class="term-root">{_table(headers, rows, numeric_cols={1,2,3,4,5,6})}</div>', unsafe_allow_html=True)

    package_verifications = storage.load_package_verification_history()
    if not package_verifications.empty:
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


# ==================== Logs ====================

def _render_logs_tab(snapshot: dict) -> None:
    query = st.text_input("Filter", "", key="cc_logs_filter", placeholder="filter by stage, level, or message text")
    logs = storage.load_recent_logs(500)
    if query:
        q = query.lower()
        logs = [l for l in logs if q in str(l.get("stage", "")).lower() or q in str(l.get("message", "")).lower() or q in str(l.get("level", "")).lower()]

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
    st.markdown(f'<div class="term-root"><div class="term-section">Forecast Package</div></div>', unsafe_allow_html=True)
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
    st.markdown(f'<div class="term-root"><div class="term-section">Forecast Products</div></div>', unsafe_allow_html=True)
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
