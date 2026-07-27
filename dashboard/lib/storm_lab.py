"""Storm Backtest & Storm Learning — two research tools built to answer
one honest gap the engine's own verification history couldn't cover on
its own: every accuracy number this project has ever reported (R², skill
scores, confidence calibration) was computed against live 2026 NOAA data,
which has been geomagnetically quiet the entire time (Kp never exceeded
~3, Dst never dropped below -14 nT). A forecasting engine's hardest real
job is a storm, and this one had genuinely never been checked against
one. Dev-only: nothing here writes to data/forecasts/ or predictions.db,
and nothing here is deployed — it exists purely to answer "can my engine
do that" with real historical evidence instead of hope.

Two deliberately separate tools, because a first design (train a tiny
model on the ~100 hours right before a storm, test it on the storm) would
have conflated two different questions and answered neither honestly:

  Storm Backtest  — does the EXISTING, already-trained production model
                    generalize to a storm it has never seen? No retraining.
  Storm Learning  — if a NEW model is actually trained on real storm data,
                    does that measurably help on a storm it still never
                    saw? Evaluated only, never deployed to production.

Both pull real historical data for named, independently-documented storms
from NASA's public OMNI2 archive (swdss.models.storm_data) — the same
source this project's own v2 training-data refresh already uses.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.shared_ui import metric_card, plot_retro
from swdss.models.registry import HORIZONS
from swdss.models.storm_backtest import BACKTESTABLE, list_backtest_runs, record_backtest_run, run_storm_backtest
from swdss.models.storm_data import NAMED_STORMS
from swdss.models.storm_learning import LEARNABLE, list_learning_runs, record_learning_run, run_storm_learning_experiment

VARIABLE_LABELS_DISPLAY = {
    "speed": "Solar Wind Speed",
    "density": "Solar Wind Density",
    "temperature": "Solar Wind Temperature",
    "bt": "IMF Bt",
    "bx_gsm": "IMF Bx",
    "by_gsm": "IMF By",
    "bz_gsm": "IMF Bz",
    "dst": "Dst",
    "kp": "Kp",
    "ae": "AE",
}

DATASET_LABELS_DISPLAY = {
    "solar_wind": "Solar Wind",
    "imf": "IMF",
    "analytics": "Geomagnetic (production Kp/Dst model)",
    "ae": "Auroral Electrojet (AE)",
}


def _storm_options() -> dict:
    return {key: storm["label"] for key, storm in NAMED_STORMS.items()}


def _dataset_variable_options(scope: dict) -> list:
    return [(dataset, variable) for dataset, variables in scope.items() for variable in variables]


def _format_dataset_variable(pair: tuple) -> str:
    dataset, variable = pair
    return f"{DATASET_LABELS_DISPLAY.get(dataset, dataset)} — {VARIABLE_LABELS_DISPLAY.get(variable, variable)}"


def _format_horizon(horizon) -> str:
    """Kp's production model always targets NOAA's next official 3-hour
    interval — never a fixed hourly horizon — so both storm_backtest.py and
    storm_learning.py force horizon="interval" for it regardless of what's
    selected below; this just renders that consistently everywhere.
    """
    return "interval" if horizon == "interval" else f"{horizon}h"


# ==================== Storm Backtest ====================


def render_storm_backtest_tab() -> None:
    st.markdown("#### Storm Backtest")
    with st.expander("Why this tab exists, and what we're checking", expanded=True):
        st.markdown(
            "**The observation that led here:** every R², skill score, and confidence-calibration "
            "number this engine has ever reported was computed against live NOAA data from 2026 — "
            "a stretch that has been geomagnetically quiet the entire time. A forecasting engine's "
            "hardest real job is a storm, and this one had genuinely never been checked against one.\n\n"
            "**What this tool does:** loads the actual, already-trained production model — unchanged, "
            "not retrained — and feeds it real historical hourly data from a named, independently "
            "documented geomagnetic storm (via NASA's public OMNI2 archive). It compares what the "
            "model predicted against what actually happened that hour, and against a naive "
            "'assume nothing changes' persistence forecast.\n\n"
            "**What we're hoping to find:** that accuracy holds up reasonably well even during a "
            "storm, or at minimum that it still beats persistence — real evidence the engine is more "
            "than a fair-weather system, instead of just assuming so. If it doesn't hold up, that's a "
            "real, useful finding too — see the Storm Learning tab for the next question that raises."
        )

    storm_options = _storm_options()
    col1, col2, col3 = st.columns(3)
    with col1:
        storm_key = st.selectbox("Storm", list(storm_options), format_func=lambda k: storm_options[k], key="backtest_storm")
    with col2:
        pairs = _dataset_variable_options(BACKTESTABLE)
        pair = st.selectbox("Dataset / Variable", pairs, format_func=_format_dataset_variable, key="backtest_pair")
    with col3:
        horizon = st.selectbox("Horizon", HORIZONS, format_func=lambda h: f"{h}h", key="backtest_horizon")

    storm = NAMED_STORMS[storm_key]
    badge = (
        "🟡 May have been seen during production training"
        if storm["in_training_range"]
        else "🟢 Genuinely blind test — predates the production training window"
    )
    st.caption(f"{storm['g_scale']} · Dst min ≈ {storm['dst_min_nT']} nT · {storm['notes']} — {badge}")
    if pair[1] == "kp":
        st.caption("ℹ️ Kp's production model always targets NOAA's next official 3-hour interval — the Horizon selector above is ignored for it.")

    if st.button("▶ Run Backtest", key="run_storm_backtest"):
        dataset_key, variable = pair
        with st.spinner(f"Downloading OMNI2 data and scoring the production {variable} model against {storm['label']}..."):
            try:
                result = run_storm_backtest(dataset_key, variable, horizon, storm_key)
                record_backtest_run(result)
                st.session_state["last_backtest_result"] = result
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")

    result = st.session_state.get("last_backtest_result")
    if result:
        _render_backtest_result(result)

    st.markdown("##### Run History")
    runs = list_backtest_runs()
    if runs:
        rows = [
            {
                "Storm": r["storm_label"],
                "Dataset": r["dataset"],
                "Variable": r["variable"],
                "Horizon": _format_horizon(r["horizon"]),
                "MAE": round(r["mae"], 3),
                "Skill vs Persistence": round(r["skill_score"], 3) if r["skill_score"] is not None else "—",
                "Blind Test?": "No (in training range)" if r["in_training_range"] else "Yes",
                "Run At": pd.Timestamp(r["run_at"]).strftime("%Y-%m-%d %H:%M UTC"),
            }
            for r in runs
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No backtests run yet.")


def _render_backtest_result(result: dict) -> None:
    variable_label = VARIABLE_LABELS_DISPLAY.get(result["variable"], result["variable"])
    st.markdown(f"##### Result: {result['storm_label']} — {variable_label} +{_format_horizon(result['horizon'])}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        caption = f"vs. {result['production_mae']:.2f} on ordinary (quiet-dominated) test data" if result["production_mae"] else ""
        metric_card("Storm-Period MAE", f"{result['mae']:.2f}", caption)
    with c2:
        skill = result["skill_score"]
        metric_card(
            "Skill vs. Persistence",
            f"{skill:.3f}" if skill is not None else "—",
            "Beats naive 'assume no change'" if (skill or 0) > 0 else "No skill over persistence",
            value_color="#1f7a3a" if (skill or 0) > 0 else "#7a1f1f",
        )
    with c3:
        band = result["within_production_mae_band_rate"]
        metric_card(
            "Within Normal Error Band",
            f"{band * 100:.0f}%" if band is not None else "—",
            "Share of storm-hours within 1.5x the model's usual MAE",
        )
    with c4:
        metric_card("Persistence MAE", f"{result['persistence_mae']:.2f}", "Naive baseline for comparison")

    if result["per_regime"]:
        st.markdown("###### Error by Activity Regime (this storm)")
        rows = [{"Regime": regime, "Samples": v["n"], "MAE": round(v["mae"], 3)} for regime, v in result["per_regime"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["timestamps"], y=result["actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=result["timestamps"], y=result["predicted"], mode="lines", name="Production Model Predicted"))
    fig.add_trace(go.Scatter(x=result["timestamps"], y=result["persistence"], mode="lines", name="Persistence (naive)", line=dict(dash="dot")))
    fig.update_layout(title=f"{result['storm_label']} — Actual vs. Predicted", height=380)
    plot_retro(fig)

    if result["in_training_range"]:
        st.caption(
            "⚠️ This storm's dates fall inside the production model's own training window — a good "
            "result here is weaker evidence than a genuinely blind storm (September 2017, August 2018, "
            "or March 2015), since the model may have already seen this exact event."
        )


# ==================== Storm Learning ====================


def render_storm_learning_tab() -> None:
    st.markdown("#### Storm Learning")
    with st.expander("Why this tab exists, and what we're checking", expanded=True):
        st.markdown(
            "**The observation that led here:** an earlier, tempting version of this idea was to take "
            "the ~100 hours right before a storm, train a small model on just that, and test it on the "
            "storm itself. That's the wrong experiment — those 100 hours are, by construction, quiet "
            "time, so a tiny model trained on them would fail during the storm regardless of whether "
            "the underlying approach is any good. It would only prove 'a model can't predict what it's "
            "never seen,' which we already know.\n\n"
            "**What this tool actually does instead:** trains a genuinely new model (never touching the "
            "real production model files) on the existing multi-year production training data plus "
            "several real historical storms — with one storm held out completely, never seen during "
            "training — then scores it on that held-out storm. The result is compared side-by-side "
            "against both a naive persistence baseline and the actual currently-deployed production "
            "model's own performance on that same held-out storm (via the Storm Backtest tool).\n\n"
            "**What we're hoping to find:** that showing a model real storm examples during training "
            "measurably improves its accuracy on a storm it still hasn't seen — a real, concrete case "
            "for including storm-period data in the next production retrain. If it doesn't help, that's "
            "just as useful to know: it means the gap isn't simply about data volume, and the real fix "
            "has to be looked for elsewhere (features, model choice, or the held-out storm may already "
            "resemble something in the model's existing training range)."
        )

    storm_options = _storm_options()
    col1, col2, col3 = st.columns(3)
    with col1:
        pairs = _dataset_variable_options(LEARNABLE)
        pair = st.selectbox("Dataset / Variable", pairs, format_func=_format_dataset_variable, key="learning_pair")
    with col2:
        horizon = st.selectbox("Horizon", HORIZONS, format_func=lambda h: f"{h}h", key="learning_horizon")
    with col3:
        held_out_storm = st.selectbox(
            "Held-out storm (never trained on)", list(storm_options), format_func=lambda k: storm_options[k], key="learning_held_out"
        )
    if pair[1] == "kp":
        st.caption("ℹ️ Kp's production model always targets NOAA's next official 3-hour interval — the Horizon selector above is ignored for it.")

    training_candidates = [k for k in storm_options if k != held_out_storm]
    training_storms = st.multiselect(
        "Storms to add to training data",
        training_candidates,
        default=training_candidates,
        format_func=lambda k: storm_options[k],
        key="learning_training_storms",
    )

    if st.button("▶ Run Storm Learning Experiment", key="run_storm_learning"):
        if not training_storms:
            st.warning("Pick at least one storm to add to training data.")
        else:
            dataset_key, variable = pair
            with st.spinner(
                "Training a new model on quiet-time data + selected storms, then evaluating on the "
                "held-out storm — this trains real models, so it may take a minute..."
            ):
                try:
                    result = run_storm_learning_experiment(dataset_key, variable, horizon, held_out_storm, training_storms)
                    record_learning_run(result)
                    st.session_state["last_learning_result"] = result
                except Exception as exc:
                    st.error(f"Experiment failed: {exc}")

    result = st.session_state.get("last_learning_result")
    if result:
        _render_learning_result(result)

    st.markdown("##### Run History")
    runs = list_learning_runs()
    if runs:
        rows = [
            {
                "Held-Out Storm": r["held_out_storm_label"],
                "Dataset": r["dataset"],
                "Variable": r["variable"],
                "Horizon": _format_horizon(r["horizon"]),
                "New Model MAE": round(r["new_model_mae"], 3),
                "Production MAE": round(r["production_model_mae"], 3),
                "Verdict": r["verdict"],
                "Run At": pd.Timestamp(r["run_at"]).strftime("%Y-%m-%d %H:%M UTC"),
            }
            for r in runs
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No storm-learning experiments run yet.")


def _render_learning_result(result: dict) -> None:
    variable_label = VARIABLE_LABELS_DISPLAY.get(result["variable"], result["variable"])
    st.markdown(f"##### Result: held out {result['held_out_storm_label']} — {variable_label} +{_format_horizon(result['horizon'])}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("New Model MAE", f"{result['new_model_mae']:.2f}", f"Algorithm: {result['new_model_algorithm']}")
    with c2:
        metric_card("Production Model MAE", f"{result['production_model_mae']:.2f}", "Same held-out storm, current deployed model")
    with c3:
        metric_card("Persistence MAE", f"{result['persistence_mae']:.2f}", "Naive baseline")
    with c4:
        metric_card(
            "Training Samples", f"{result['n_train_samples']:,}", f"{len(result['training_storms'])} storm(s) + quiet-time history"
        )

    verdict_lower = result["verdict"].lower()
    if "improved" in verdict_lower:
        color = "#1f7a3a"
    elif "not" in verdict_lower:
        color = "#7a1f1f"
    else:
        color = "#7a5a1f"
    st.markdown(
        f"<div style='padding:10px 14px; background:#f4f1ea; border-left:4px solid {color}; margin:10px 0;'>{escape(result['verdict'])}</div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["timestamps"], y=result["actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=result["timestamps"], y=result["new_model_predicted"], mode="lines", name="New (Storm-Trained) Model"))
    fig.add_trace(
        go.Scatter(x=result["timestamps"], y=result["production_predicted"], mode="lines", name="Current Production Model", line=dict(dash="dash"))
    )
    fig.update_layout(title=f"{result['held_out_storm_label']} — Held-Out Comparison", height=380)
    plot_retro(fig)

    st.caption(
        "The new model was trained on the multi-year production data plus the storms selected above, "
        "with the held-out storm fully excluded from training — a genuine blind test, not a "
        "retrospective fit. No walk-forward cross-validation was run here (a deliberate scope cut for "
        "an interactive research tool, not an oversight)."
    )
