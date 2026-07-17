"""Shared, presentation-only helpers used by dashboard/home.py and every
dashboard/lib/*_research_lab.py module.

Extracted verbatim from dashboard/home.py (no formula, styling, or
behavior changes) so the ~4700 lines of Research Laboratory code could
move into their own modules without duplicating the small set of retro
UI/dialog helpers and constants they and home.py both depend on.
"""

from html import escape

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
            height: 158px;
            box-sizing: border-box;
            background: #dcdcdc;
            color: #000000;
            font-family: 'MS Sans Serif', Tahoma, sans-serif;
            display: flex;
            flex-direction: column;
        ">
            <div style="font-size: 0.85rem; color: #000080; line-height: 1.2;">{label}{info_icon}</div>
            <div style="
                font-size: 1.6rem;
                font-weight: 700;
                color: {value_color};
                line-height: 1.15;
                min-height: 2.3em;
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
