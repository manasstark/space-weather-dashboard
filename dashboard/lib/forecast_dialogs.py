"""The four per-dataset forecast dialogs — Kp, Dst, AE, and Experimental
— plus the prediction-job dispatcher and shared helpers only these
dialogs use (format_analytics_inputs, render_explainability_section,
_cv_terminal_line). Extracted verbatim from dashboard/home.py.

Kp and Dst live on the Analytics (combined Sun-Earth) page; AE has its
own Analytics sub-page; Experimental is the AE-cascade research product
shown alongside AE's production dialog for direct comparison. All four
follow the same shape (current job state -> operational context ->
confidence/stability -> explainability) but are kept as four separate
functions rather than one parametrized one, since each dataset's dialog
surfaces genuinely different fields (e.g. Kp's Mode 1/Mode 2 dual
product, AE's Kyoto quicklook estimate) that don't collapse cleanly into
shared branching logic.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from swdss.models.explainability import explain_prediction
from swdss.models.jobs import (
    average_prediction,
    delete_job,
    final_percentage_error,
    find_matching_job,
    forecast_drift,
    forecast_evaluation_label,
    get_job,
    job_mae,
    model_quality_label,
    poll_jobs,
    production_bias,
    production_error,
    rolling_final_error,
    save_job,
    stability_metric,
    stop_job,
)
from swdss.models.predict import latest_minute_observation
from swdss.models.registry import VARIABLE_LABELS, VARIABLE_UNITS

from dashboard.lib.data_helpers import format_value, status_badge_html
from dashboard.lib.shared_ui import close_active_dialog, metric_card, plot_retro, render_dialog_close_button


def format_analytics_inputs(inputs: dict) -> str:
    """Renders the Solar Wind + IMF readings captured for one Analytics
    tick as two terminal lines — the live upstream conditions feeding the
    combined Kp/Dst model, distinct from the single self-referential
    reading the standalone Solar Wind/IMF/Kp/Dst tabs show.
    """
    if not inputs:
        return ""
    sw_parts = []
    if inputs.get("speed") is not None:
        sw_parts.append(f"Speed {inputs['speed']:.1f} km/s")
    if inputs.get("density") is not None:
        sw_parts.append(f"Density {inputs['density']:.2f} p/cm3")
    if inputs.get("temperature") is not None:
        sw_parts.append(f"Temp {inputs['temperature']:.0f} K")

    imf_parts = []
    for key, lbl in [("bt", "Bt"), ("bx_gsm", "Bx"), ("by_gsm", "By"), ("bz_gsm", "Bz")]:
        if inputs.get(key) is not None:
            imf_parts.append(f"{lbl} {inputs[key]:.2f} nT")

    lines = []
    if sw_parts:
        lines.append(f"<div>Solar Wind: {escape(' | '.join(sw_parts))}</div>")
    if imf_parts:
        lines.append(f"<div>IMF: {escape(' | '.join(imf_parts))}</div>")
    return "".join(lines)


def render_explainability_section(dataset: str, variable: str, horizon) -> None:
    """"Why did the model predict this value?" — shared across every job
    dialog (production and Research Lab alike). Purely diagnostic: reads
    the model and the most recently available live feature row, never
    retrains anything or affects the live prediction itself.
    """
    with st.expander("🔍 Why did the model predict this? (Feature Importance)"):
        try:
            result = explain_prediction(dataset, variable, horizon)
        except Exception as exc:
            st.warning(f"Could not compute explainability right now: {exc}")
            return

        if result["method"] == "unavailable" or not result["contributions"]:
            st.info("Not enough live feature history yet to explain this model's current prediction.")
            return

        method_label = "SHAP (exact Shapley values)" if result["method"] == "shap" else "Permutation sensitivity (SHAP unavailable for this model type)"
        st.caption(f"Method: {method_label} — Model: {result['model_name']}. Based on the most recently available live feature row.")

        contrib_df = pd.DataFrame(
            [
                {"Feature": feat, "Current Value": round(val, 3), "Contribution": round(contrib, 4)}
                for feat, val, contrib in result["contributions"]
            ]
        )
        fig = go.Figure()
        colors = ["#1f7a3a" if c >= 0 else "#7a1f1f" for c in contrib_df["Contribution"]]
        fig.add_trace(
            go.Bar(
                x=contrib_df["Contribution"],
                y=contrib_df["Feature"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title="Top Contributing Variables",
            height=320,
            xaxis_title="Contribution to prediction",
            yaxis=dict(autorange="reversed"),
        )
        plot_retro(fig)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        st.caption(
            "Positive contribution pushes the prediction up; negative pushes it down. "
            f"Predicted value: {result['predicted_value']:.2f}." if result["predicted_value"] is not None else ""
        )


def _cv_terminal_line(metrics: dict, unit: str = "") -> str:
    """One extra terminal-style line for the live job-terminal blocks
    (render_*_forecast_dialog / show_prediction_job) showing walk-forward
    CV stability alongside the single-holdout R²/MAE/RMSE line those
    already print — empty string (no line at all) for any model trained
    before this field existed, so old jobs render exactly as before.
    """
    if metrics.get("cv_r2_mean") is None:
        return ""
    unit_suffix = f" {escape(unit)}" if unit else ""
    return (
        f"<div>CV R&sup2;: {metrics['cv_r2_mean']:.4f} &plusmn; {metrics['cv_r2_std']:.4f} "
        f"({metrics.get('cv_n_folds')} folds) &nbsp;|&nbsp; CV MAE: {metrics['cv_mae_mean']:.3f}{unit_suffix} "
        f"&plusmn; {metrics['cv_mae_std']:.3f}{unit_suffix}</div>"
    )


def render_kp_forecast_dialog(job: dict) -> None:
    """Kp on the Analytics page runs TWO independent products from one job
    record — see predict.predict_kp_interval / predict_kp_rolling for the
    full reasoning:

    - Mode 1, Production Forecast: `production_prediction`, frozen at job
      creation from data strictly BEFORE the target interval (matching
      how the model was trained — features from block B predict block
      B+1, never block B's own value). This is the number used for
      official model evaluation and never changes after creation.
    - Mode 2, Operational Rolling Estimate: every tick logged after
      creation (job["ticks"]), computed from whatever data is freshest
      right now — deliberately off the training distribution, an
      "operational situational awareness" read, always labeled
      Experimental, never used for official evaluation.

    They're rendered in clearly separate sections below and never allowed
    to overwrite each other, matching how they're stored.
    """
    ticks = job["ticks"]
    metrics = job.get("metrics", {})

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])
    target_interval_end = target_hour + pd.Timedelta(hours=3)
    production_prediction = job.get("production_prediction")
    production_observed_at = job.get("production_observed_at")

    st.subheader("Kp Forecast")
    st.caption(
        "Two independent products: a frozen Production Forecast (used for official model "
        "evaluation) and an Experimental Rolling Estimate (situational awareness only). "
        "Neither ever overwrites the other."
    )

    # ==================== Production Forecast (Frozen) ====================
    st.markdown("##### Production Forecast (Frozen)")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        metric_card("Prediction Generated", created_at.strftime("%H:%M:%S UTC"), "")
    with pc2:
        metric_card(
            "Prediction Time",
            "N/A" if production_observed_at is None else pd.Timestamp(production_observed_at).strftime("%H:%M UTC"),
            "Input data timestamp — strictly before the interval, matching training",
        )
    with pc3:
        metric_card(
            "Forecast Interval", f"{target_hour.strftime('%H:%M')}–{target_interval_end.strftime('%H:%M UTC')}", ""
        )
    with pc4:
        metric_card(
            "Frozen Forecast Value",
            "N/A" if production_prediction is None else f"{production_prediction:.2f}",
            "Never recomputed — used for official model evaluation",
        )

    badge_key = "kp_waiting_official" if job["status"] in ("in_progress", "evaluating") else job["status"]
    st.markdown(status_badge_html(badge_key), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f}
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f}</div>
            {_cv_terminal_line(metrics)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================== Operational Rolling Estimate (Experimental) ==================
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Operational Rolling Estimate (Experimental)")
    st.warning(
        "⚠️ Uses whatever upstream data is freshest right now, including data from inside the "
        "forecast interval itself — a situational-awareness read, not a calibrated forecast. "
        "Never used for official model evaluation; see Production Forecast above for that."
    )

    latest_tick = ticks[-1] if ticks else None
    latest_inputs = (latest_tick.get("inputs") or {}) if latest_tick else {}
    latest_minute_at = (
        pd.Timestamp(latest_tick["minute_at"]) if latest_tick and latest_tick.get("minute_at") else None
    )
    rolling_value = latest_tick["predicted_value"] if latest_tick else None
    drift = forecast_drift(job)

    if len(ticks) >= 2:
        rolling_diff = ticks[-1]["predicted_value"] - ticks[-2]["predicted_value"]
        if abs(rolling_diff) < 1e-9:
            rolling_trend = "Stable"
        elif rolling_diff > 0:
            rolling_trend = "Increasing"
        else:
            rolling_trend = "Decreasing"
    else:
        rolling_trend = "N/A"

    def _fmt(value, dp, suffix=""):
        return "N/A" if value is None else f"{value:.{dp}f}{suffix}"

    r1, r2c, r3, r4, r5 = st.columns(5)
    with r1:
        metric_card("Current Time", pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S UTC"), "")
    with r2c:
        metric_card(
            "Latest Solar Wind Update",
            _fmt(latest_inputs.get("speed"), 1, " km/s"),
            "" if latest_minute_at is None else latest_minute_at.strftime("%H:%M:%S UTC"),
        )
    with r3:
        metric_card(
            "Latest IMF Update",
            "N/A" if latest_inputs.get("bz_gsm") is None else f"Bz {latest_inputs['bz_gsm']:.2f} nT",
            "" if latest_minute_at is None else latest_minute_at.strftime("%H:%M:%S UTC"),
        )
    with r4:
        metric_card("Latest Dst", _fmt(latest_inputs.get("dst"), 1, " nT"), "")
    with r5:
        metric_card("Latest AE", _fmt(latest_inputs.get("ae"), 1, " nT"), "")

    r6, r7, r8, r9, r10 = st.columns(5)
    with r6:
        metric_card(
            "Current Rolling Estimate",
            "N/A" if rolling_value is None else f"{rolling_value:.2f}",
            "Experimental — not the production forecast",
        )
    with r7:
        metric_card(
            "Last Updated", "N/A" if latest_minute_at is None else latest_minute_at.strftime("%H:%M:%S UTC"), ""
        )
    with r8:
        metric_card("Trend", rolling_trend, "")
    with r9:
        if drift is None:
            metric_card("Prediction Drift", "N/A", "")
        else:
            sign = "+" if drift >= 0 else ""
            metric_card(
                "Prediction Drift",
                f"{sign}{drift:.2f}",
                "First rolling estimate to latest",
                tooltip="How much the rolling estimate has moved as new data arrived.",
            )
    with r10:
        r2_val = metrics.get("r2")
        metric_card(
            "Confidence",
            model_quality_label(r2_val),
            "Reflects the underlying model only — not calibrated for off-distribution input",
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Prediction History — Rolling Estimate Timeline")

    if ticks:
        estimate_col = f"Rolling Estimate ({target_hour.strftime('%H:%M')}–{target_interval_end.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Latest Dst (nT)",
            "Latest Official Kp",
            "Latest AE (nT)",
            estimate_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("dst"), 1),
                _fmt(inputs.get("kp"), 2),
                _fmt(inputs.get("ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .kp-console {{
                background: #050505;
                border: 2px solid #ffffff;
                box-shadow: 3px 3px 0px #808080;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.kp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.kp-console-table th, table.kp-console-table td {{
                border: 1px solid #1a3a2a;
                padding: 3px 8px;
                text-align: right;
                color: #d8ffe8;
            }}
            table.kp-console-table th {{
                color: #00ff88;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.kp-console-table td:first-child, table.kp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="kp-console">
                <table class="kp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name="Rolling Estimate (Experimental)",
            )
        )
        fig.update_layout(
            title="Rolling Estimate Timeline (Experimental)",
            height=360,
            legend_title_text="",
            yaxis_title="Rolling Kp Estimate",
        )
        plot_retro(fig)

    # ==================== Verification ====================
    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Verification")

        actual = job["actual_value"]
        prod_err = production_error(job)
        prod_bias = production_bias(job)
        roll_err = rolling_final_error(job)
        final_rolling_value = ticks[-1]["predicted_value"] if ticks else None

        st.markdown("**Production Forecast → Official Kp** (official model evaluation)")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            metric_card(
                "Frozen Forecast", "N/A" if production_prediction is None else f"{production_prediction:.2f}", ""
            )
        with v2:
            metric_card(
                "Official Kp",
                "Pending" if actual is None else f"{actual:.2f}",
                "" if job["status"] == "completed" else "Stopped before the interval closed",
            )
        with v3:
            metric_card(
                "Production Error", "N/A" if prod_err is None else f"{prod_err:.2f}", "Primary official accuracy metric"
            )
        with v4:
            metric_card(
                "Production Bias",
                "N/A" if prod_bias is None else f"{prod_bias:+.2f}",
                "Signed — positive means the model runs high",
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("**Final Rolling Estimate → Official Kp** (research comparison only, never official)")
        v5, v6, v7 = st.columns(3)
        with v5:
            metric_card(
                "Final Rolling Estimate",
                "N/A" if final_rolling_value is None else f"{final_rolling_value:.2f}",
                "Experimental",
            )
        with v6:
            metric_card("Official Kp", "Pending" if actual is None else f"{actual:.2f}", "")
        with v7:
            metric_card("Rolling Error", "N/A" if roll_err is None else f"{roll_err:.2f}", "Research only")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            r2 = metrics.get("r2")
            metric_card("Model Quality", model_quality_label(r2), "N/A" if r2 is None else f"R² = {r2:.4f}")
        with m2:
            metric_card("MAE", f"{metrics.get('mae', float('nan')):.3f}", "Model's typical training error")
        with m3:
            metric_card("RMSE", f"{metrics.get('rmse', float('nan')):.3f}", "")
        if metrics.get("cv_r2_mean") is not None:
            st.caption(
                f"🔁 Walk-forward CV ({metrics.get('cv_n_folds')} folds): "
                f"R² = {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f} · "
                f"MAE = {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f} — "
                "a stability check across several time periods, not just one holdout split."
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section(job["dataset"], "kp", "interval")
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


def render_dst_forecast_dialog(job: dict) -> None:
    """Dst on the Analytics page is still horizon-based (1h/3h/6h/12h/24h,
    not NOAA's fixed publishing cadence like Kp), but otherwise gets the
    exact same live console + completion summary architecture as Kp's
    dedicated dialog, since both are driven by the same combined Solar
    Wind + IMF + geomagnetic feature set.
    """
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    horizon = job["horizon"]

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])

    st.subheader("Dst Forecast")
    st.markdown(
        f"**Started:** {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
        f"**Target:** {target_hour.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Horizon:** {horizon}h"
    )
    st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} nT
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} nT</div>
            {_cv_terminal_line(metrics, "nT")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Live Forecast Console")

    def _fmt(value, dp):
        return "N/A" if value is None else f"{value:.{dp}f}"

    if ticks:
        forecast_col = f"Forecast ({target_hour.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Latest Dst (nT)",
            "Latest Official Kp",
            "Latest AE (nT)",
            forecast_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("dst"), 1),
                _fmt(inputs.get("kp"), 2),
                _fmt(inputs.get("ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .kp-console {{
                background: #050505;
                border: 2px solid #ffffff;
                box-shadow: 3px 3px 0px #808080;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.kp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.kp-console-table th, table.kp-console-table td {{
                border: 1px solid #1a3a2a;
                padding: 3px 8px;
                text-align: right;
                color: #d8ffe8;
            }}
            table.kp-console-table th {{
                color: #00ff88;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.kp-console-table td:first-child, table.kp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="kp-console">
                <table class="kp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name="Predicted Dst",
            )
        )
        fig.update_layout(
            title="Dst Forecast Drift Toward Target",
            height=360,
            legend_title_text="",
            yaxis_title="Predicted Dst (nT)",
        )
        plot_retro(fig)

    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Forecast Summary")

        final_pred = ticks[-1]["predicted_value"] if ticks else None
        avg_pred = average_prediction(job)
        actual = job["actual_value"]
        final_error = None if (final_pred is None or actual is None) else abs(final_pred - actual)
        avg_error = None if (avg_pred is None or actual is None) else abs(avg_pred - actual)
        drift = forecast_drift(job)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("Started", created_at.strftime("%H:%M:%S UTC"), "")
        with s2:
            metric_card("Target", target_hour.strftime("%H:%M UTC"), "")
        with s3:
            metric_card("Horizon", f"{horizon}h", "")
        with s4:
            metric_card(
                "Final Prediction",
                "N/A" if final_pred is None else f"{final_pred:.2f} nT",
                "Last forecast before the target arrived — the operational forecast",
            )

        s5, s6, s7, s8 = st.columns(4)
        with s5:
            metric_card(
                "Average Prediction",
                "N/A" if avg_pred is None else f"{avg_pred:.2f} nT",
                "Stability indicator, not the operational forecast",
                tooltip="Mean of every prediction generated during the session.",
            )
        with s6:
            metric_card(
                "Actual Dst",
                "Pending" if actual is None else f"{actual:.2f} nT",
                "" if job["status"] == "completed" else "Stopped before the target arrived",
            )
        with s7:
            metric_card(
                "Final Prediction Error",
                "N/A" if final_error is None else f"{final_error:.2f} nT",
                "Primary operational accuracy metric",
            )
        with s8:
            metric_card(
                "Average Prediction Error",
                "N/A" if avg_error is None else f"{avg_error:.2f} nT",
                "Secondary stability metric",
            )

        s9, s10, s11, s12 = st.columns(4)
        with s9:
            if drift is None:
                metric_card("Forecast Drift", "N/A", "")
            else:
                sign = "+" if drift >= 0 else ""
                metric_card(
                    "Forecast Drift",
                    f"{sign}{drift:.2f} nT",
                    "Final minus initial prediction",
                    tooltip="How much the forecast moved from the first tick to the last.",
                )
        with s10:
            r2 = metrics.get("r2")
            metric_card("Model Quality", model_quality_label(r2), "N/A" if r2 is None else f"R² = {r2:.4f}")
        with s11:
            metric_card("MAE", f"{metrics.get('mae', float('nan')):.3f} nT", "Model's typical training error")
        with s12:
            metric_card("RMSE", f"{metrics.get('rmse', float('nan')):.3f} nT", "")
        if metrics.get("cv_r2_mean") is not None:
            st.caption(
                f"🔁 Walk-forward CV ({metrics.get('cv_n_folds')} folds): "
                f"R² = {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f} · "
                f"MAE = {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f} nT — "
                "a stability check across several time periods, not just one holdout split."
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section(job["dataset"], "dst", job["horizon"])
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


def render_ae_forecast_dialog(job: dict) -> None:
    """AE V1: independent target driven by Solar Wind + IMF + derived
    physics only (no Kp/Dst inputs, no chained predictions — see README's
    staged AE plan). Otherwise the same live console + completion summary
    architecture as Dst's dedicated dialog.

    AE has no live NOAA/DONKI feed (only a historical file), so "Latest
    AE" below is the last known value, not a per-minute live reading.

    Prediction and verification are two separate, independent phases (see
    swdss.models.jobs._advance_predicting / _verify_static_jobs):
    completion (status='completed') happens purely because the target
    hour arrived — it never waits on an AE observation that may never
    come. Verification (verification_status) is a separate, ongoing check
    against Kyoto WDC's published real-time AE digital data (never NOAA,
    which has no AE product) of whether that source has since been
    published to cover
    that hour; until then it stays 'pending', shown as "Awaiting Official
    AE Verification" rather than blocking the job from completing.
    """
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    horizon = job["horizon"]

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])
    r2 = metrics.get("r2")
    verification_status = job.get("verification_status")

    st.subheader("AE Forecast")
    st.markdown(
        f"**Started:** {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
        f"**Target:** {target_hour.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Horizon:** {horizon}h  \n"
        f"**Model Quality:** {model_quality_label(r2)}"
    )
    badge_col, verify_col = st.columns([0.3, 0.7])
    with badge_col:
        st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    with verify_col:
        if job["status"] == "completed" and verification_status:
            st.markdown(status_badge_html(verification_status), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} nT
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} nT</div>
            {_cv_terminal_line(metrics, "nT")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Live Forecast Console")

    def _fmt(value, dp):
        return "N/A" if value is None else f"{value:.{dp}f}"

    if ticks:
        forecast_col = f"Forecast ({target_hour.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Ey (mV/m)",
            "VBz",
            "Dynamic Pressure (nPa)",
            "Latest AE (nT)",
            forecast_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("ey"), 3),
                _fmt(inputs.get("vbz"), 1),
                _fmt(inputs.get("dynamic_pressure"), 3),
                _fmt(inputs.get("ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .kp-console {{
                background: #050505;
                border: 2px solid #ffffff;
                box-shadow: 3px 3px 0px #808080;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.kp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.kp-console-table th, table.kp-console-table td {{
                border: 1px solid #1a3a2a;
                padding: 3px 8px;
                text-align: right;
                color: #d8ffe8;
            }}
            table.kp-console-table th {{
                color: #00ff88;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.kp-console-table td:first-child, table.kp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="kp-console">
                <table class="kp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name="Predicted AE",
            )
        )
        fig.update_layout(
            title="AE Forecast Drift Toward Target",
            height=360,
            legend_title_text="",
            yaxis_title="Predicted AE (nT)",
        )
        plot_retro(fig)

    if job["status"] == "completed" and verification_status == "pending":
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        checked_at = job.get("verification_checked_at")
        checked_caption = (
            f" Last checked {pd.Timestamp(checked_at).strftime('%Y-%m-%d %H:%M:%S UTC')}." if checked_at else ""
        )
        st.markdown("**Prediction Complete — Verification Pending**")
        st.info(
            "Official Kyoto AE data has not yet been published. "
            "Estimated publication delay: approximately 10-20 days.\n\n"
            "The target hour has arrived and the forecast is frozen below. NOAA/DONKI publish no AE "
            "product, so **verification** (confirming the official AE value) is a completely separate, "
            "ongoing check against **Kyoto World Data Center**'s published digital AE data — once per "
            "day is sufficient, since Kyoto WDC only publishes in batches." + checked_caption
        )

    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Forecast Summary")

        final_pred = ticks[-1]["predicted_value"] if ticks else None
        avg_pred = average_prediction(job)
        actual = job["actual_value"]
        final_error = None if (final_pred is None or actual is None) else abs(final_pred - actual)
        avg_error = None if (avg_pred is None or actual is None) else abs(avg_pred - actual)
        drift = forecast_drift(job)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("Started", created_at.strftime("%H:%M:%S UTC"), "")
        with s2:
            metric_card("Target", target_hour.strftime("%H:%M UTC"), "")
        with s3:
            metric_card("Horizon", f"{horizon}h", "")
        with s4:
            metric_card(
                "Final Prediction",
                "N/A" if final_pred is None else f"{final_pred:.2f} nT",
                "Last forecast before the target arrived — the operational forecast",
            )

        s5, s6, s7, s8 = st.columns(4)
        with s5:
            metric_card(
                "Average Prediction",
                "N/A" if avg_pred is None else f"{avg_pred:.2f} nT",
                "Stability indicator, not the operational forecast",
                tooltip="Mean of every prediction generated during the session.",
            )
        with s6:
            if job["status"] == "stopped":
                actual_caption = "Stopped before the target arrived"
            elif verification_status == "verified":
                actual_caption = "Verified against Kyoto WDC's published AE digital data"
            else:
                actual_caption = "Awaiting Official AE (Kyoto WDC checked automatically)"
            metric_card(
                "Actual AE",
                "Pending" if actual is None else f"{actual:.2f} nT",
                actual_caption,
            )
        with s7:
            metric_card(
                "Final Prediction Error",
                "N/A" if final_error is None else f"{final_error:.2f} nT",
                "Primary operational accuracy metric",
            )
        with s8:
            metric_card(
                "Average Prediction Error",
                "N/A" if avg_error is None else f"{avg_error:.2f} nT",
                "Secondary stability metric",
            )

        s9, s10, s11, s12 = st.columns(4)
        with s9:
            if drift is None:
                metric_card("Forecast Drift", "N/A", "")
            else:
                sign = "+" if drift >= 0 else ""
                metric_card(
                    "Forecast Drift",
                    f"{sign}{drift:.2f} nT",
                    "Final minus initial prediction",
                    tooltip="How much the forecast moved from the first tick to the last.",
                )
        with s10:
            metric_card("Model Quality", model_quality_label(r2), "N/A" if r2 is None else f"R² = {r2:.4f}")
        with s11:
            metric_card("MAE", f"{metrics.get('mae', float('nan')):.3f} nT", "Model's typical training error")
        with s12:
            metric_card("RMSE", f"{metrics.get('rmse', float('nan')):.3f} nT", "")

        s13, s14, s15, s16 = st.columns(4)
        with s13:
            pct_error = final_percentage_error(job)
            metric_card(
                "Percentage Error",
                "N/A" if pct_error is None else f"{pct_error:.1f}%",
                "Final prediction error as % of the official AE value",
            )
        with s14:
            verified_at = job.get("verified_at")
            metric_card(
                "Verification Date",
                "N/A" if verified_at is None else pd.Timestamp(verified_at).strftime("%Y-%m-%d %H:%M UTC"),
                "When Kyoto WDC's data first covered this target hour",
            )
        with s15:
            metric_card(
                "Verification Status",
                "Verified" if verification_status == "verified" else "Pending Official Kyoto Data",
                "Kyoto WDC digital AE" if verification_status == "verified" else "Checked ~daily",
            )
        with s16:
            cv_r2_mean = metrics.get("cv_r2_mean")
            if cv_r2_mean is not None:
                metric_card(
                    "Walk-Forward CV",
                    f"R² = {cv_r2_mean:.3f} ± {metrics['cv_r2_std']:.3f}",
                    f"{metrics.get('cv_n_folds')} folds",
                    tooltip="Stability across several time periods, not just one holdout split.",
                )
            else:
                st.empty()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section("ae", "ae", job["horizon"])
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


def _experimental_badge_html() -> str:
    return (
        "<div style=\"display:inline-block;background:#5a1f8a;color:#f2e8ff;"
        "border:2px solid #d8b8ff;border-radius:4px;padding:2px 10px;"
        "font-weight:700;font-size:0.78rem;letter-spacing:0.03em;\">"
        "🧪 EXPERIMENTAL — RESEARCH ONLY, NOT PRODUCTION</div>"
    )


def render_experimental_forecast_dialog(job: dict) -> None:
    """AE V3 (research/experimental — see README's staged AE plan): the
    cascaded pipeline that feeds Predicted AE (never observed AE) into
    Kp/Dst as an extra feature. Completely separate models/training data
    from the production "analytics" pipeline — this dialog only ever
    reads production jobs (via find_matching_job) for side-by-side
    comparison, it never influences them.
    """
    variable = job["variable"]
    horizon = job["horizon"]
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    label = VARIABLE_LABELS.get(variable, variable)
    unit = VARIABLE_UNITS.get(variable, "")
    decimals = 2 if variable == "kp" else 1

    target_hour = pd.Timestamp(job["target_hour"])
    created_at = pd.Timestamp(job["created_at"])
    is_kp_interval = variable == "kp"

    st.markdown(_experimental_badge_html(), unsafe_allow_html=True)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.subheader(f"Experimental {label} Forecast — Cascaded via Predicted AE")
    st.markdown(
        f"**Started:** {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
        f"**Target:** {target_hour.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Horizon:** {'Next official NOAA interval' if is_kp_interval else f'{horizon}h'}  \n"
        f"**Model Quality:** {model_quality_label(metrics.get('r2'))}"
    )
    if metrics.get("cv_r2_mean") is not None:
        st.caption(
            f"🔁 Walk-forward CV ({metrics.get('cv_n_folds')} folds): "
            f"R² = {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f} · "
            f"MAE = {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f} — "
            "a stability check across several time periods, not just one holdout split."
        )
    st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #d8b8ff;
            box-shadow: 3px 3px 0px #5a1f8a;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #e8caff;
        }}
        </style>
        <div class="job-terminal">
            <div>EXPERIMENTAL MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} {escape(unit)}
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} {escape(unit)}</div>
            {_cv_terminal_line(metrics, unit)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Live Forecast Console (Experimental — Cascaded)")

    def _fmt(value, dp):
        return "N/A" if value is None else f"{value:.{dp}f}"

    if ticks:
        forecast_col = f"Experimental Forecast ({target_hour.strftime('%H:%M UTC')})"
        columns = [
            "Time UTC",
            "Speed (km/s)",
            "Density (p/cm3)",
            "Temperature (K)",
            "Bt (nT)",
            "Bx (nT)",
            "By (nT)",
            "Bz (nT)",
            "Previous Kp",
            "Previous Dst (nT)",
            "Predicted AE (nT)",
            forecast_col,
        ]

        row_html = []
        for t in ticks:  # chronological — new rows append at the bottom, like a real console
            inputs = t.get("inputs") or {}
            minute_at = t["minute_at"]
            cells = [
                pd.Timestamp(minute_at).strftime("%H:%M:%S") if minute_at else "N/A",
                _fmt(inputs.get("speed"), 1),
                _fmt(inputs.get("density"), 2),
                _fmt(inputs.get("temperature"), 0),
                _fmt(inputs.get("bt"), 2),
                _fmt(inputs.get("bx_gsm"), 2),
                _fmt(inputs.get("by_gsm"), 2),
                _fmt(inputs.get("bz_gsm"), 2),
                _fmt(inputs.get("kp"), 2),
                _fmt(inputs.get("dst"), 1),
                _fmt(inputs.get("predicted_ae"), 1),
                f"{t['predicted_value']:.2f}",
            ]
            row_html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")

        header_html = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr>"

        st.markdown(
            f"""
            <style>
            .exp-console {{
                background: #050505;
                border: 2px solid #d8b8ff;
                box-shadow: 3px 3px 0px #5a1f8a;
                padding: 10px;
                max-height: 380px;
                overflow: auto;
            }}
            table.exp-console-table {{
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.72rem;
                white-space: nowrap;
            }}
            table.exp-console-table th, table.exp-console-table td {{
                border: 1px solid #3a1a4a;
                padding: 3px 8px;
                text-align: right;
                color: #f0e0ff;
            }}
            table.exp-console-table th {{
                color: #e8caff;
                font-weight: 700;
                position: sticky;
                top: 0;
                background: #0a0a0a;
            }}
            table.exp-console-table td:first-child, table.exp-console-table th:first-child {{
                text-align: left;
            }}
            </style>
            <div class="exp-console">
                <table class="exp-console-table">
                    <thead>{header_html}</thead>
                    <tbody>{''.join(row_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for the first NOAA reading...")

    # Look up a comparable production job — same variable, same exact
    # target hour — "whenever possible", per the research spec. Only ever
    # reads production's job table; never writes to or influences it.
    production_job = find_matching_job("analytics", variable, target_hour)

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name=f"Experimental {label}",
            )
        )
        if production_job and production_job["ticks"]:
            prod_plotted = [t for t in production_job["ticks"] if t["minute_at"] is not None]
            if prod_plotted:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.Timestamp(t["minute_at"]) for t in prod_plotted],
                        y=[t["predicted_value"] for t in prod_plotted],
                        mode="lines+markers",
                        name=f"Production {label}",
                    )
                )
        fig.update_layout(
            title=f"{label} Forecast Drift — Production vs. Experimental",
            height=360,
            legend_title_text="",
        )
        plot_retro(fig)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Production vs. Experimental — Side by Side")
    if production_job is None:
        st.info(
            f"No Production {label} job found targeting {target_hour.strftime('%Y-%m-%d %H:%M UTC')}. "
            "Start one from the Prediction tab (same horizon, similar start time) to compare live."
        )
    else:
        prod_final = production_job["ticks"][-1]["predicted_value"] if production_job["ticks"] else None
        exp_final = ticks[-1]["predicted_value"] if ticks else None
        actual = job["actual_value"] if job["actual_value"] is not None else production_job["actual_value"]

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card(
                "Production (latest)",
                "N/A" if prod_final is None else f"{prod_final:.2f} {unit}",
                f"Model: {production_job['model_name']}",
            )
        with c2:
            metric_card(
                "Experimental (latest)",
                "N/A" if exp_final is None else f"{exp_final:.2f} {unit}",
                f"Model: {job['model_name']}",
            )
        with c3:
            metric_card(
                "Actual",
                "Pending" if actual is None else f"{actual:.2f} {unit}",
                "Shared ground truth for both pipelines",
            )

    if job["status"] in ("completed", "stopped"):
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Evaluation")

        exp_final = ticks[-1]["predicted_value"] if ticks else None
        actual = job["actual_value"]
        if production_job is not None and actual is None:
            actual = production_job["actual_value"]
        exp_error = None if (exp_final is None or actual is None) else abs(exp_final - actual)

        prod_final = None
        prod_error = None
        prod_mae = None
        prod_r2 = None
        if production_job is not None:
            prod_final = production_job["ticks"][-1]["predicted_value"] if production_job["ticks"] else None
            prod_error = None if (prod_final is None or actual is None) else abs(prod_final - actual)
            prod_mae = production_job.get("metrics", {}).get("mae")
            prod_r2 = production_job.get("metrics", {}).get("r2")

        comparison_df = pd.DataFrame(
            [
                {
                    "Pipeline": "Production",
                    f"Final Prediction ({unit})": "N/A" if prod_final is None else round(prod_final, decimals),
                    f"Absolute Error ({unit})": "N/A" if prod_error is None else round(prod_error, decimals),
                    "Model MAE (offline, training)": "N/A" if prod_mae is None else round(prod_mae, 3),
                    "Model R² (offline, training)": "N/A" if prod_r2 is None else round(prod_r2, 4),
                },
                {
                    "Pipeline": "Experimental",
                    f"Final Prediction ({unit})": "N/A" if exp_final is None else round(exp_final, decimals),
                    f"Absolute Error ({unit})": "N/A" if exp_error is None else round(exp_error, decimals),
                    "Model MAE (offline, training)": round(metrics.get("mae", float("nan")), 3),
                    "Model R² (offline, training)": round(metrics.get("r2", float("nan")), 4),
                },
            ]
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        if actual is None:
            st.caption("Actual observation still Pending — errors above will populate once it's available.")
        elif prod_error is not None and exp_error is not None:
            if exp_error < prod_error:
                st.success(
                    f"For this target, the experimental cascade performed better "
                    f"({exp_error:.{decimals}f} {unit} vs {prod_error:.{decimals}f} {unit} absolute error)."
                )
            elif exp_error > prod_error:
                st.warning(
                    f"For this target, the production pipeline performed better "
                    f"({prod_error:.{decimals}f} {unit} vs {exp_error:.{decimals}f} {unit} absolute error)."
                )
            else:
                st.info("Both pipelines produced the same absolute error for this target.")
        else:
            st.caption("No matching Production job to compare error against for this target.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    render_explainability_section("experimental", job["variable"], "interval" if job["variable"] == "kp" else job["horizon"])
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


@st.dialog("Prediction Job", width="large", dismissible=False)
def show_prediction_job(job_id: str) -> None:
    render_dialog_close_button("close_prediction_job")

    job = get_job(job_id)
    if job is None:
        st.error("This prediction job could not be found.")
        return

    poll_jobs(job["dataset"])
    job = get_job(job_id)

    dataset = job["dataset"]
    variable = job["variable"]

    if dataset == "analytics" and variable == "kp":
        render_kp_forecast_dialog(job)
        return
    if dataset == "analytics" and variable == "dst":
        render_dst_forecast_dialog(job)
        return
    if dataset == "ae":
        render_ae_forecast_dialog(job)
        return
    if dataset == "experimental":
        render_experimental_forecast_dialog(job)
        return

    horizon = job["horizon"]
    label = VARIABLE_LABELS.get(variable, variable)
    unit = VARIABLE_UNITS.get(variable, "")
    decimals = 0 if unit == "K" else 2

    start_hour = pd.Timestamp(job["start_hour"])
    target_hour = pd.Timestamp(job["target_hour"])
    ticks = job["ticks"]
    metrics = job.get("metrics", {})
    is_kp_interval = dataset == "analytics" and variable == "kp"
    horizon_display = "Next NOAA Interval" if is_kp_interval else f"{horizon}h"

    st.subheader(f"{label} — {horizon_display} Forecast")
    badge_col, caption_col = st.columns([0.15, 0.85])
    with badge_col:
        st.markdown(status_badge_html(job["status"]), unsafe_allow_html=True)
    with caption_col:
        st.caption(
            f"Started {start_hour.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Target: {target_hour.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    live_ts, live_val = latest_minute_observation(dataset, variable)
    latest_predicted = ticks[-1]["predicted_value"] if ticks else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Current NOAA",
            format_value(live_val, f" {unit}", decimals),
            "N/A" if live_ts is None else f"As of {live_ts.strftime('%H:%M:%S UTC')}",
        )
    with c2:
        r2 = metrics.get("r2")
        quality = model_quality_label(r2)
        metric_card(
            "Model Quality",
            quality,
            "N/A" if r2 is None else f"R² = {r2:.4f}",
            tooltip="How well this model performed on held-out training data, categorized from its R² score.",
        )
    with c3:
        if latest_predicted is not None and live_val is not None:
            change = latest_predicted - live_val
            if abs(change) < 1e-9:
                trend, color = "No Change", "#404040"
            elif change > 0:
                trend, color = "Increase", "#1f7a3a"
            else:
                trend, color = "Decrease", "#a31f1f"
            sign = "+" if change >= 0 else ""
            metric_card(
                "Expected Change",
                f"{sign}{format_value(change, f' {unit}', decimals)}",
                f"Trend: {trend}",
                tooltip="Difference between the latest forecast and the current live NOAA reading.",
                value_color=color,
            )
        else:
            metric_card("Expected Change", "N/A", "", tooltip="Difference between the latest forecast and the current live NOAA reading.")
    with c4:
        stability_label, stability_delta = stability_metric(job)
        if stability_label is None:
            metric_card(
                "Stability",
                "N/A",
                "Not enough updates to evaluate stability",
                tooltip="How much the prediction has varied across the most recent updates in this session.",
            )
        else:
            metric_card(
                "Stability",
                stability_label,
                f"Δ Prediction = {stability_delta:.2f} {unit}",
                tooltip="How much the prediction has varied across the most recent updates in this session.",
            )

    if metrics.get("cv_r2_mean") is not None:
        st.caption(
            f"🔁 Walk-forward CV ({metrics.get('cv_n_folds')} folds): "
            f"R² = {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f} · "
            f"MAE = {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f} — "
            "a stability check across several time periods, not just one holdout split."
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if job["status"] == "completed" and job["actual_value"] is not None:
        final_pred = ticks[-1]["predicted_value"] if ticks else None
        error = None if final_pred is None else abs(final_pred - job["actual_value"])
        model_mae = metrics.get("mae")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            metric_card(
                "Final Prediction",
                format_value(final_pred, f" {unit}", decimals),
                f"Target {target_hour.strftime('%H:%M UTC')}",
            )
        with a2:
            metric_card(
                "Actual NOAA",
                format_value(job["actual_value"], f" {unit}", decimals),
                f"At {target_hour.strftime('%H:%M UTC')}",
            )
        with a3:
            metric_card(
                "Absolute Error",
                "N/A" if error is None else format_value(error, f" {unit}", decimals),
                "",
                tooltip="Difference between the final prediction and the actual NOAA observation, in native units.",
            )
        with a4:
            eval_label = forecast_evaluation_label(error, model_mae)
            metric_card(
                "Forecast Evaluation",
                eval_label,
                "" if model_mae is None else f"Model's typical error: {model_mae:.{decimals}f} {unit}",
                tooltip="How this forecast's error compares to the model's typical error (MAE) from training.",
            )
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .job-terminal {{
            background: #050505;
            border: 2px solid #ffffff;
            box-shadow: 3px 3px 0px #808080;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.78rem;
            color: #00ff88;
        }}
        .job-terminal .pipeline-step {{ color: #9adfff; margin-left: 12px; }}
        .job-terminal div {{ margin-bottom: 3px; }}
        .job-terminal.scroll {{ max-height: 320px; overflow-y: auto; }}
        </style>
        <div class="job-terminal">
            <div>MODEL: {escape(job['model_name'])}</div>
            <div>R&sup2;: {metrics.get('r2', float('nan')):.4f} &nbsp;|&nbsp; MAE: {metrics.get('mae', float('nan')):.3f} {escape(unit)}
            &nbsp;|&nbsp; RMSE: {metrics.get('rmse', float('nan')):.3f} {escape(unit)}</div>
            {_cv_terminal_line(metrics, unit)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if not ticks:
        st.info("No ticks logged yet.")
        return

    blocks = []
    for i, tick in enumerate(reversed(ticks)):
        minute_at = tick["minute_at"]
        minute_label = "N/A" if minute_at is None else pd.Timestamp(minute_at).strftime("%H:%M:%S UTC")
        pred_text = f"{tick['predicted_value']:.{decimals}f} {unit}"
        used_horizon = tick.get("used_horizon", horizon)
        used_horizon_text = "Next Interval" if used_horizon == "interval" else f"{used_horizon}h"
        next_step = "Waiting For Next NOAA Update..." if i == 0 and job["status"] == "in_progress" else "Superseded"

        if dataset == "analytics" and tick.get("inputs"):
            header = f"<div>[{escape(minute_label)}] Multi-Source Update</div>" + format_analytics_inputs(
                tick["inputs"]
            )
            features_step = "Features Generated (Solar Wind + IMF + Geomagnetic Lags)"
        else:
            noaa_text = "N/A" if tick["noaa_value"] is None else f"{tick['noaa_value']:.{decimals}f} {unit}"
            header = f"<div>[{escape(minute_label)}] NOAA {escape(label)}: {escape(noaa_text)}</div>"
            features_step = "Features Generated"

        blocks.append(
            header
            + f"<div class='pipeline-step'>&rarr; {features_step}</div>"
            + f"<div class='pipeline-step'>&rarr; Model Loaded (Horizon: {used_horizon_text})</div>"
            + f"<div class='pipeline-step'>&rarr; Prediction (Target {target_hour.strftime('%H:%M UTC')}): {escape(pred_text)}</div>"
            + f"<div class='pipeline-step'>&rarr; {next_step}</div>"
        )

    st.markdown(
        f"""
        <div class="job-terminal scroll">
            {''.join(blocks)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    plotted = [t for t in ticks if t["minute_at"] is not None]
    if plotted:
        chart_times = [pd.Timestamp(t["minute_at"]) for t in plotted]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_times,
                y=[t["predicted_value"] for t in plotted],
                mode="lines+markers",
                name=f"Predicted ({target_hour.strftime('%H:%M UTC')})",
            )
        )
        fig.update_layout(
            title=f"{label} — Prediction Drift Toward Target",
            height=360,
            legend_title_text="",
        )
        plot_retro(fig)

    if job["status"] == "evaluating" and dataset == "analytics":
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.info(f"Waiting for NOAA to publish the official {label} value for this target. The job will complete automatically once it's available.")
    elif job["status"] == "evaluating":
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Evaluation In Progress — Target Hour Filling")
        st.caption(
            f"Collecting NOAA minute readings for {target_hour.strftime('%H:%M')}–"
            f"{(target_hour + pd.Timedelta(hours=1)).strftime('%H:%M UTC')}. Once the hour closes, "
            "this collapses into the final Job Summary below."
        )
        eval_ticks = job.get("eval_ticks", [])
        if eval_ticks:
            eval_df = pd.DataFrame(
                [
                    {
                        "Minute": pd.Timestamp(t["minute_at"]).strftime("%H:%M:%S UTC") if t["minute_at"] else "N/A",
                        f"NOAA Value ({unit})": "N/A" if t["noaa_value"] is None else round(t["noaa_value"], decimals),
                        f"Running Average ({unit})": (
                            "N/A" if t["running_avg"] is None else round(t["running_avg"], decimals)
                        ),
                    }
                    for t in reversed(eval_ticks)
                ]
            )
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
        else:
            st.info("Waiting for the first NOAA reading in this hour...")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Job Summary")

    mae = job_mae(job)
    summary_df = pd.DataFrame(
        [
            {
                "Variable": label,
                "Horizon": horizon_display,
                "Started": start_hour.strftime("%Y-%m-%d %H:%M UTC"),
                "Target": target_hour.strftime("%Y-%m-%d %H:%M UTC"),
                f"Initial Prediction ({unit})": round(ticks[0]["predicted_value"], decimals),
                f"Final Prediction ({unit})": round(ticks[-1]["predicted_value"], decimals),
                f"Actual NOAA ({unit})": "Pending" if job["actual_value"] is None else round(job["actual_value"], decimals),
                f"Mean Error - All Ticks ({unit})": "N/A" if mae is None else round(mae, decimals),
                "Model": job["model_name"],
                "R²": round(metrics.get("r2", float("nan")), 4),
            }
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if job["status"] in ("in_progress", "evaluating"):
        if st.button("⏹ Stop Prediction", key=f"stop_{job['job_id']}", use_container_width=True):
            stop_job(job["job_id"])
            st.toast("Prediction stopped.")
            st.rerun()
    else:
        already_saved = job.get("saved", False)
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button(
                "💾 Saved" if already_saved else "💾 Save",
                key=f"save_{job['job_id']}",
                use_container_width=True,
                disabled=already_saved,
            ):
                save_job(job["job_id"])
                st.toast("Prediction saved.")
                st.rerun()
        with delete_col:
            if st.button("🗑️ Delete", key=f"delete_{job['job_id']}", use_container_width=True):
                delete_job(job["job_id"])
                close_active_dialog()


