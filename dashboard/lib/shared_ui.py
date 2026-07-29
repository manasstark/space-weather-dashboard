"""Shared, presentation-only helpers used by dashboard/home.py and every
dashboard/lib/*_research_lab.py module.

Extracted verbatim from dashboard/home.py (no formula, styling, or
behavior changes) so the ~4700 lines of Research Laboratory code could
move into their own modules without duplicating the small set of retro
UI/dialog helpers and constants they and home.py both depend on.
"""

from html import escape

import pandas as pd
import streamlit as st

from dashboard.lib.design_tokens import BORDER, MONO, MUTED, PANEL_BG, TEXT

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


REFRESH_SECONDS = 15

RETRO_CHART_COLORWAY = ["#0000FF", "#008000", "#FF0000", "#00BFBF", "#BF00BF", "#BFBF00", "#404040"]
RETRO_CHART_FONT = "Courier New, Consolas, monospace"

# Bright terminal-palette values (dashboard.lib.design_tokens' ACCENT/RED/AMBER) —
# the previous dark, desaturated values were chosen for the retired metric_card's
# light-grey Win-95 background and would be near-invisible on terminal_metric's
# dark panel.
CONCLUSION_COLORS = {"Supported": "#39d98a", "Not Supported": "#f85149", "Inconclusive": "#e3b341"}


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

def terminal_metric(label: str, value: str, caption: str = "", tooltip: str = "", value_color: str | None = None) -> None:
    """Terminal-styled metric tile — the Command Centre's actual dark,
    monospace design language (dashboard.lib.design_tokens), replacing
    the retired metric_card's literal Windows-95 beveled grey box.
    Resolves the "three competing visual identities" product-review
    finding (Command Centre's Bloomberg-terminal look vs. metric_card's
    Win-95 box vs. Project Status's plainer third look) by making this
    the one consistent metric tile used everywhere.

    Same content contract as the retired metric_card (label/value/
    caption/tooltip/value_color) so every call site only needed a
    rename, not a rewrite. Fixed-height so a row of tiles always lines
    up evenly, even when one value wraps to two lines or one caption is
    longer than its neighbors — nothing is clamped or hidden, a tile
    with unusually long text simply grows past the fixed height.
    """
    color = value_color or TEXT
    title_attr = f' title="{escape(tooltip)}"' if tooltip else ""
    info_icon = " ⓘ" if tooltip else ""
    st.markdown(
        f"""
        <div{title_attr} style="
            border: 1px solid {BORDER};
            padding: 14px 16px;
            min-height: 120px;
            box-sizing: border-box;
            background: {PANEL_BG};
            font-family: {MONO};
            display: flex;
            flex-direction: column;
        ">
            <div style="font-size: 0.7rem; letter-spacing: 0.03em; text-transform: uppercase; color: {MUTED}; line-height: 1.3; flex-shrink: 0;">{escape(label)}{info_icon}</div>
            <div style="
                font-size: 1.5rem;
                font-weight: 700;
                color: {color};
                line-height: 1.2;
                min-height: 1.6em;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                overflow-wrap: break-word;
                word-break: break-word;
            ">{escape(str(value))}</div>
            <div style="font-size: 0.68rem; color: {MUTED}; line-height: 1.3; flex-grow: 1;">{escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


def render_site_header() -> None:
    """NOAA Space Weather Prediction Center-style masthead — white
    background, project title + one-line subtitle on the left, current
    UTC date/time on the right. Shown once, site-wide, above the top
    navigation bar (style_top_nav) rather than per-page inside home_page()
    — matching NOAA SWPC's own site layout, where the masthead and nav
    bar appear identically on every page (2026-07 redesign).
    """
    now = pd.Timestamp.now(tz="UTC")
    time_text = now.strftime("%A, %B ") + str(now.day) + now.strftime(", %Y at %H:%M:%S UTC")
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            padding:16px 28px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            flex-wrap:wrap;
            gap:8px;
            border-bottom:4px solid #0a2a5e;
            border-radius:4px 4px 0 0;
        ">
            <div>
                <div style="font-size:1.55rem; font-weight:800; letter-spacing:0.3px; line-height:1.15; color:#0a2a5e;">
                    SPACE WEATHER DECISION SUPPORT SYSTEM
                </div>
                <div style="font-size:0.82rem; color:#4a5568; margin-top:2px;">
                    Physics-Informed Sun-to-Earth Forecasting, Research &amp; Verification Platform
                </div>
            </div>
            <div style="font-size:0.85rem; color:#333333; white-space:nowrap;">
                {escape(time_text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_top_nav() -> None:
    """CSS turning the st.radio wrapped in st.container(key="topnav")
    into a flat, dark-blue horizontal nav bar (NOAA SWPC style) instead
    of Streamlit's default circular radio buttons. Hides each option's
    circle indicator and restyles its label as a nav-bar item; the
    currently-selected page is detected via the CSS :has()/:checked
    combinator rather than any Streamlit-internal class name, since
    those are auto-generated and not a stable thing to depend on.

    The dark-blue background lives on the OUTER topnav container (not
    just the radiogroup) so the page links and the Refresh/Concepts/
    References buttons — laid out as siblings in home.py via st.columns
    inside this same container — read as one continuous bar, matching a
    traditional website's combined nav-and-actions header row, rather
    than a colored nav strip with mismatched white buttons floating next
    to it (2026-07 sticky-header redesign — see style_sticky_header).
    """
    st.markdown(
        """
        <style>
        div[class*="st-key-topnav"] {
            background:#0a2a5e;
            padding:0 8px;
        }
        /* stColumn defaults to min-width:auto (a standard flexbox gotcha),
        so it refuses to shrink below its content's natural width — the
        actual cause of the nav bleeding into the buttons' column despite
        every content-size reduction above. Forcing min-width:0 here lets
        the column honor its intended fractional width, which is what
        makes the inner overflow-x:auto scroll (rather than the whole
        column silently growing past its share) actually take effect. */
        div[class*="st-key-topnav"] div[data-testid="stColumn"] {
            min-width:0 !important;
        }
        div[class*="st-key-topnav"] div[data-testid="stRadio"] {
            border:none !important;
            background:transparent !important;
            width:100%;
            max-width:100%;
            overflow-x:auto;
            overflow-y:hidden;
            scrollbar-width:thin;
        }
        div[class*="st-key-topnav"] div[data-testid="stRadio"] > div[role="radiogroup"] {
            background:transparent;
            border:none !important;
            display:flex;
            flex-wrap:nowrap;
            gap:0;
            width:max-content;
        }
        div[class*="st-key-topnav"] label[data-baseweb="radio"] {
            margin:0 !important;
            padding:13px 1px !important;
            border-radius:0 !important;
            cursor:pointer;
            flex-shrink:0;
        }
        div[class*="st-key-topnav"] label[data-baseweb="radio"] > div:first-child {
            display:none !important;
        }
        div[class*="st-key-topnav"] label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p {
            color:#ffffff !important;
            font-weight:600;
            text-transform:uppercase;
            font-size:0.62rem;
            letter-spacing:0px;
            margin:0;
            white-space:nowrap;
        }
        div[class*="st-key-topnav"] label[data-baseweb="radio"]:hover {
            background:#13396e;
        }
        div[class*="st-key-topnav"] label[data-baseweb="radio"]:has(input:checked) {
            background:#1c56a3;
            box-shadow: inset 0 -3px 0 0 #ffffff;
        }

        /* Refresh / Space Weather Concepts / References — restyled as
        ghost buttons so they read as part of the same dark nav bar
        instead of Streamlit's default white rounded buttons. */
        div[class*="st-key-topnav_actions"] {
            justify-content:flex-end !important;
            flex-wrap:nowrap !important;
            overflow:hidden;
        }
        div[class*="st-key-topnav"] div[data-testid="stButton"] > button {
            background:transparent !important;
            border:1px solid rgba(255,255,255,0.35) !important;
            color:#ffffff !important;
            font-size:0.64rem !important;
            font-weight:600 !important;
            padding:5px 6px !important;
            white-space:nowrap;
        }
        div[class*="st-key-topnav"] div[data-testid="stButton"] > button:hover {
            background:rgba(255,255,255,0.15) !important;
            border-color:#ffffff !important;
            color:#ffffff !important;
        }
        div[class*="st-key-topnav"] div[data-testid="stButton"] > button p {
            color:#ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_automl_shell(
    *,
    caption: str,
    run_study_fn,
    list_studies_fn,
    render_detail_fn,
    study_label_fn,
    success_message_fn,
    session_key_prefix: str,
    button_label: str = "🚀 Run Complete Optimization Study",
    warning: str | None = None,
    extra_caption: str | None = None,
) -> None:
    """Shared AutoML orchestration shell for the Bz/Kp/AE Optimization
    Studies — button, live progress callback, success message, and
    study-history selector are mechanically identical across all three
    (confirmed by reading all three call sites, not assumed); only the
    underlying study/listing/detail functions and how a study's result
    is summarized differ, passed in by the caller.

    Deliberately NOT consolidated: the actual experiment-generation
    logic (run_study_fn — Bz's 8 steps, Kp's 10, AE's 10-per-horizon
    plus cross-horizon synthesis) and each study's detailed result
    breakdown (render_detail_fn) stay as three separate, existing
    functions, since the underlying science genuinely differs per
    variable (AE alone runs across 5 horizons) — only the UI shell that
    wraps them was duplicated for no functional reason, so only the
    shell is shared here.
    """
    st.markdown("### 🤖 Automated Optimization (AutoML)")
    st.caption(caption)
    if warning:
        st.warning(warning)
    if extra_caption:
        st.caption(extra_caption)

    run_key = f"{session_key_prefix}_run_study"
    last_id_key = f"{session_key_prefix}_last_study_id"
    select_key = f"{session_key_prefix}_study_select"

    if st.button(button_label, key=run_key, type="primary"):
        status_box = st.status("Running complete optimization study…", expanded=True)

        def _cb(step, total, msg):
            status_box.update(label=f"Step {step}/{total} — {msg}")
            status_box.write(f"**Step {step}/{total}:** {msg}")

        try:
            study = run_study_fn(progress_cb=_cb)
            status_box.update(label="Optimization study complete.", state="complete", expanded=False)
            st.session_state[last_id_key] = study["study_id"]
            st.success(success_message_fn(study))
            st.rerun()
        except Exception as exc:
            status_box.update(label="Optimization study failed.", state="error")
            st.error(f"Study failed: {exc}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Study History")
    studies = list_studies_fn()
    if not studies:
        st.info("No optimization studies run yet — click the button above to run the first one.")
        return

    study_labels = [study_label_fn(s) for s in studies]
    default_idx = 0
    last_id = st.session_state.get(last_id_key)
    if last_id:
        for i, s in enumerate(studies):
            if s["study_id"] == last_id:
                default_idx = i
                break
    chosen_label = st.selectbox("Select a study to inspect", study_labels, index=default_idx, key=select_key)
    study = studies[study_labels.index(chosen_label)]
    render_detail_fn(study)


def style_sticky_header() -> None:
    """Pins the masthead + top navigation bar — rendered together inside
    st.container(key="sticky_header") in home.py — to the top of the
    viewport while the rest of the page scrolls beneath it. A standard
    website fixed-header pattern; Streamlit has no native equivalent, so
    this is plain CSS position:sticky on the outer wrapper Streamlit
    already gives every keyed container.
    """
    st.markdown(
        """
        <style>
        div[class*="st-key-sticky_header"] {
            position: sticky;
            top: 0;
            z-index: 999;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
