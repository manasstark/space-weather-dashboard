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

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


REFRESH_SECONDS = 15

RETRO_CHART_COLORWAY = ["#0000FF", "#008000", "#FF0000", "#00BFBF", "#BF00BF", "#BFBF00", "#404040"]
RETRO_CHART_FONT = "Courier New, Consolas, monospace"

CONCLUSION_COLORS = {"Supported": "#1f7a3a", "Not Supported": "#7a1f1f", "Inconclusive": "#7a5a1f"}


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
            min-height: 158px;
            box-sizing: border-box;
            background: #dcdcdc;
            color: #000000;
            font-family: 'MS Sans Serif', Tahoma, sans-serif;
            display: flex;
            flex-direction: column;
        ">
            <div style="font-size: 0.85rem; color: #000080; line-height: 1.2; flex-shrink: 0;">{label}{info_icon}</div>
            <div style="
                font-size: 1.6rem;
                font-weight: 700;
                color: {value_color};
                line-height: 1.15;
                min-height: 2.3em;
                flex-shrink: 0;
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
