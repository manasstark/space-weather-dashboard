"""Project Status — an internal, dashboard-native progress tracker: one
sub-tab per major subsystem, each holding a short Current Status line, a
Next Milestone line, and a dated Progress Log. Exists purely as a fast,
no-README-editing way to keep track of where things stand session to
session — typed directly here, or updated by Claude at the end of a
session when explicitly asked to (Claude just edits the same JSON file
this page reads; nothing special is wired up beyond that).

Deliberately temporary: one isolated module (this file), one new nav
entry (dashboard/home.py), one local JSON store (data/project_status.json,
gitignored like every other data/ file) — meant to be deleted entirely
before public deployment, not shipped as a feature.
"""

import json
from datetime import datetime, timezone
from html import escape

import streamlit as st

from swdss.paths import DATA_DIR

STATUS_PATH = DATA_DIR / "project_status.json"

SECTIONS = {
    "forecast_engine": "Forecast Engine",
    "prediction_models": "Prediction Models",
    "research_optimization": "Research & Optimization",
    "home_ui": "Home Page / UI",
    "data_pipeline": "Data Pipeline",
}

BADGE_STYLES = {
    "Stable": ("#1f5a2e", "#d4f4dd"),
    "In Progress": ("#7a5a1f", "#fff3cd"),
    "Needs Attention": ("#7a1f1f", "#fddede"),
    "Not Started": ("#4a4a4a", "#e2e2e2"),
}
BADGE_OPTIONS = list(BADGE_STYLES)


def _badge_html(label: str) -> str:
    border_color, bg_color = BADGE_STYLES.get(label, ("#808080", "#dcdcdc"))
    return (
        f'<span style="display:inline-block; padding:2px 10px; border-radius:10px; '
        f'background:{bg_color}; border:1px solid {border_color}; color:{border_color}; '
        f'font-size:0.75rem; font-weight:700;">{escape(label)}</span>'
    )


def _default_section() -> dict:
    return {"badge": "Not Started", "current_status": "", "next_milestone": "", "log": []}


def _load_status() -> dict:
    if not STATUS_PATH.exists():
        return {key: _default_section() for key in SECTIONS}
    with open(STATUS_PATH) as f:
        data = json.load(f)
    for key in SECTIONS:
        data.setdefault(key, _default_section())
    return data


def _save_status(data: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def render_project_status_page() -> None:
    st.title("Project Status")
    st.caption(
        "🚧 Internal tracker, not part of the public dashboard — meant to be removed before deployment. "
        "One place to see where each subsystem stands, what changed recently, and what's next."
    )

    status = _load_status()
    tab_labels = ["Overview"] + list(SECTIONS.values())
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_overview(status)

    for tab, key in zip(tabs[1:], SECTIONS):
        with tab:
            _render_section(key, SECTIONS[key], status)


def _render_overview(status: dict) -> None:
    st.caption("At-a-glance status across every section — click a tab above to update one.")
    for key, label in SECTIONS.items():
        section = status[key]
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(_badge_html(section["badge"]), unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{label}**")
        st.markdown(f"**Status:** {escape(section['current_status']) if section['current_status'] else '_(not set)_'}")
        st.markdown(f"**Next:** {escape(section['next_milestone']) if section['next_milestone'] else '_(not set)_'}")
        if section["log"]:
            latest = section["log"][-1]
            st.caption(f"Last logged: {latest['date']} — {latest['note']}")
        st.divider()


def _render_section(key: str, label: str, status: dict) -> None:
    section = status[key]
    st.markdown(f"#### {label}")

    badge_idx = BADGE_OPTIONS.index(section["badge"]) if section["badge"] in BADGE_OPTIONS else BADGE_OPTIONS.index("Not Started")
    new_badge = st.selectbox("Status", BADGE_OPTIONS, index=badge_idx, key=f"{key}_badge_select")
    st.markdown(_badge_html(new_badge), unsafe_allow_html=True)

    new_current = st.text_area(
        "Current Status — where are we at?", value=section["current_status"], key=f"{key}_current", height=100
    )
    new_milestone = st.text_area(
        "Next Milestone — what's next?", value=section["next_milestone"], key=f"{key}_milestone", height=80
    )

    if st.button("💾 Save", key=f"{key}_save"):
        section["badge"] = new_badge
        section["current_status"] = new_current
        section["next_milestone"] = new_milestone
        _save_status(status)
        st.success("Saved.")
        st.rerun()

    st.markdown("##### Progress Log")
    st.caption("What we did — add a dated note each session. Newest first.")
    new_note = st.text_input("Add a note", key=f"{key}_new_note", placeholder="e.g. Wired regime-conditioned confidence bands into the Forecast tab")
    if st.button("➕ Add Note", key=f"{key}_add_note"):
        if new_note.strip():
            section["log"].append({"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "note": new_note.strip()})
            _save_status(status)
            st.rerun()

    if section["log"]:
        for entry in reversed(section["log"]):
            st.markdown(f"**{entry['date']}** — {escape(entry['note'])}")
    else:
        st.caption("No log entries yet.")
