"""AE Research Laboratory — extracted verbatim from dashboard/home.py.
Also contains render_hypothesis_testing_tab/show_hypothesis_detail, which
predate this module's boundary but sit in this same source range; both
are re-exported for dashboard/home.py's research_lab_page and dialog
dispatcher.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.lib.kp_research_lab import render_ae_optimization_study
from dashboard.lib.shared_ui import (
    CONCLUSION_COLORS,
    REFRESH_SECONDS,
    close_active_dialog,
    metric_card,
    open_dialog,
    plot_retro,
    render_dialog_close_button,
)
from swdss.models import ae_research
from swdss.models.hypothesis import (
    archive_hypothesis,
    create_hypothesis,
    delete_hypothesis,
    duplicate_hypothesis,
    evaluate_hypothesis,
    get_hypothesis,
    list_hypotheses,
    reactivate_hypothesis,
    update_hypothesis,
    update_manual_conclusion,
    update_notes,
)
from swdss.models.registry import VARIABLE_LABELS

# ==================== AE Research Laboratory ====================
# Fully isolated from the Production Prediction tab — see
# swdss.models.ae_research module docstring for the production-safety
# contract. Nothing below ever writes to models/ae/ or its metrics.json.
# Answers "how can AE prediction be improved?", "what physics governs
# AE?", and "which ML model performs best?" — never intended to replace
# the operational forecast.


def _ae_research_notes() -> None:
    st.info(
        "**What this lab is for.** Production answers *what is the operational AE forecast* — this "
        "lab answers *how can it be improved* and *what physics governs it*. Every run trains against "
        "the exact same `ae_analytics_features.csv` production's own AE model trains on, so results "
        "here are genuinely comparable (production: R²=0.744 @1h, R²=0.398 @3h). Horizons here are "
        "1/2/3 hours only — AE has no minute-level ground truth anywhere in this codebase (the only "
        "historical archive, `ae_processed.parquet`, is itself hourly), so 15/30-minute horizons were "
        "deliberately not built rather than faked via interpolation."
    )


def _ae_research_hyperparam_inputs(model_type: str, key_prefix: str) -> dict:
    schema = ae_research.HYPERPARAM_SCHEMA.get(model_type, {})
    values = {}
    if not schema:
        st.caption("No tunable hyperparameters for this model.")
        return values
    cols = st.columns(min(len(schema), 4))
    for i, (name, spec) in enumerate(schema.items()):
        with cols[i % len(cols)]:
            label = name.replace("_", " ").title()
            if spec["type"] == "int":
                values[name] = st.number_input(
                    label, min_value=spec["min"], max_value=spec["max"], value=spec["default"], step=1,
                    key=f"{key_prefix}_{model_type}_{name}",
                )
            else:
                values[name] = st.number_input(
                    label, min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(spec["default"]),
                    step=0.01, key=f"{key_prefix}_{model_type}_{name}",
                )
    return values


def _ae_research_model_selector(key_prefix: str) -> str:
    options = ae_research.ALL_TRAINABLE_MODELS + ae_research.FUTURE_MODELS

    def _fmt(name):
        return f"{name} (coming soon)" if name in ae_research.FUTURE_MODELS else name

    choice = st.selectbox("Model Architecture", options, format_func=_fmt, key=f"{key_prefix}_model_select")
    if choice in ae_research.FUTURE_MODELS:
        st.warning(f"{choice} is a registered placeholder for future work — not trainable yet.")
    return choice


def _ae_research_horizon_selector(key_prefix: str) -> int:
    return st.radio(
        "Forecast Horizon",
        ae_research.HORIZON_OPTIONS,
        format_func=lambda h: f"{h} hour" + ("s" if h != 1 else ""),
        horizontal=True,
        key=f"{key_prefix}_horizon",
    )


def _ae_research_feature_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or ae_research.default_feature_toggles()
    toggles = {}
    cols = st.columns(len(ae_research.FEATURE_GROUP_COLUMNS))
    for i, (group, group_cols) in enumerate(ae_research.FEATURE_GROUP_COLUMNS.items()):
        with cols[i]:
            st.markdown(f"**{group}**")
            toggles[group] = {}
            for col in group_cols:
                label = VARIABLE_LABELS.get(col, col.replace("_", " ").title())
                toggles[group][col] = st.checkbox(
                    label, value=defaults.get(group, {}).get(col, True), key=f"{key_prefix}_feat_{group}_{col}"
                )
    return toggles


def _ae_research_engineered_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or ae_research.default_engineered_toggles()
    toggles = {}
    cols = st.columns(len(ae_research.ENGINEERED_GROUPS))
    for i, group in enumerate(ae_research.ENGINEERED_GROUPS):
        with cols[i]:
            toggles[group] = st.checkbox(group, value=defaults.get(group, True), key=f"{key_prefix}_eng_{group}")
    return toggles


def _ae_research_physics_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or {}
    toggles = {}
    cols = st.columns(3)
    for i, name in enumerate(ae_research.PHYSICS_FEATURE_OPTIONS):
        with cols[i % 3]:
            toggles[name] = st.checkbox(name, value=defaults.get(name, False), key=f"{key_prefix}_phys_{name}")
    return toggles


def _ae_research_run_row(run: dict, best_run_id: str = None, key_prefix: str = "ae_runs") -> None:
    m = run["metrics"]
    is_best = run["run_id"] == best_run_id
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.22, 0.11, 0.11, 0.11, 0.11, 0.11, 0.23])
        with c1:
            star = "⭐ " if is_best else ""
            promoted_tag = " 🚀" if run.get("promoted") else ""
            st.markdown(f"**{star}{run['model_type']}**{promoted_tag}")
            seq_note = f" · seq={run['sequence_length']}h" if run.get("sequence_length") else ""
            st.caption(f"AE +{run.get('horizon', 1)}h{seq_note} · {pd.Timestamp(run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}")
        with c2:
            metric_card("R²", f"{m['r2']:.4f}", "")
        with c3:
            metric_card("MAE", f"{m['mae']:.3f}", "")
        with c4:
            metric_card("RMSE", f"{m['rmse']:.3f}", "")
        with c5:
            metric_card("Train Time", f"{run.get('training_time_sec', 0):.2f}s", "")
        with c6:
            metric_card("Predict Time", f"{run.get('prediction_time_sec', 0) * 1000:.1f}ms", "")
        with c7:
            st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button(
                    "Promoted" if run.get("promoted") else "Promote",
                    key=f"{key_prefix}_promote_{run['run_id']}",
                    disabled=run.get("promoted", False),
                    use_container_width=True,
                ):
                    ae_research.promote_run(run["run_id"])
                    st.toast("Marked as promoted (label only — production untouched).")
                    st.rerun()
            with b2:
                if st.button(
                    "Load",
                    key=f"{key_prefix}_load_{run['run_id']}",
                    use_container_width=True,
                    disabled=run["model_type"] in ae_research.SEQUENCE_MODELS,
                ):
                    try:
                        model = ae_research.load_trained_model(run["run_id"])
                        st.toast(f"Loaded {run['model_type']} model ({type(model).__name__}) into memory.")
                    except Exception as exc:
                        st.error(f"Load failed: {exc}")
            with b3:
                if st.button("Delete", key=f"{key_prefix}_delete_{run['run_id']}", use_container_width=True):
                    ae_research.delete_run(run["run_id"])
                    st.toast("Run deleted.")
                    st.rerun()


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _cached_ae_research_frame(feature_toggles: dict = None, engineered_groups: dict = None, physics_features: dict = None):
    """Same rationale as _cached_kp_research_frame: ae_analytics_features.csv
    is a static historical file that never changes mid-session, so caching
    the exploratory Physics Experiments tab's reads eliminates redundant
    CSV parsing + physics-feature recomputation on every 15s auto-refresh
    tick with zero staleness risk.
    """
    return ae_research.load_ae_research_frame(feature_toggles, engineered_groups, physics_features)


def render_ae_model_comparison_tab() -> None:
    st.caption(
        "Train, evaluate, and compare AE models — every run auto-saves to its own registry (never "
        "overwrites production) and can be reloaded via each run card's Load button."
    )
    model_type = _ae_research_model_selector("ae_compare")
    horizon = _ae_research_horizon_selector("ae_compare")
    with st.expander("Feature Groups", expanded=True):
        feature_toggles = _ae_research_feature_toggle_form("ae_compare")
    with st.expander("Engineered Features", expanded=False):
        engineered_groups = _ae_research_engineered_toggle_form("ae_compare")
    with st.expander("Physics Feature Experiments (optional)", expanded=False):
        physics_features = _ae_research_physics_toggle_form("ae_compare")

    sequence_length = None
    if model_type in ae_research.SEQUENCE_MODELS:
        sequence_length = st.selectbox(
            "Sequence Length (hours, look-back window)",
            ae_research.SEQUENCE_LENGTH_OPTIONS,
            index=ae_research.SEQUENCE_LENGTH_OPTIONS.index(ae_research.DEFAULT_SEQUENCE_LENGTH),
            key="ae_compare_seqlen",
        )

    st.markdown("**Hyperparameters**")
    hyperparams = (
        _ae_research_hyperparam_inputs(model_type, "ae_compare") if model_type in ae_research.ALL_TRAINABLE_MODELS else {}
    )
    notes = st.text_input("Notes (optional)", key="ae_compare_notes")

    if st.button(
        "🧪 Train Model", key="ae_compare_train_btn", type="primary",
        disabled=model_type not in ae_research.ALL_TRAINABLE_MODELS,
    ):
        with st.spinner(f"Training {model_type}..."):
            try:
                run = ae_research.train_ae_research_model(
                    model_type,
                    horizon=horizon,
                    feature_toggles=feature_toggles,
                    engineered_groups=engineered_groups,
                    physics_features=physics_features,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes=notes,
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Run Log")
    runs = ae_research.list_runs()
    if not runs:
        st.info("No training runs yet. Train a model above.")
        return
    best_id = max(runs, key=lambda r: r["metrics"]["r2"])["run_id"]
    for run in runs[:10]:
        _ae_research_run_row(run, best_run_id=best_id, key_prefix="ae_compare")
    if len(runs) > 10:
        st.caption(f"Showing 10 most recent of {len(runs)} runs — see Experiment Tracking for the full history.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Compare Models")

    def _run_label(r):
        seq = f" seq={r['sequence_length']}h" if r.get("sequence_length") else ""
        return f"{r['model_type']}{seq} +{r.get('horizon', 1)}h ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    default_n = min(4, len(run_labels))
    selected = st.multiselect("Models to compare", run_labels, default=run_labels[:default_n], key="ae_compare_select")
    chosen = [runs[run_labels.index(s)] for s in selected]
    if not chosen:
        st.info("Select at least one model above.")
        return

    best_chosen_id = max(chosen, key=lambda r: r["metrics"]["r2"])["run_id"]
    rows = []
    for r in chosen:
        m = r["metrics"]
        rows.append(
            {
                "Model": r["model_type"] + (" ⭐" if r["run_id"] == best_chosen_id else ""),
                "Horizon": f"{r.get('horizon', 1)}h",
                "R²": round(m["r2"], 4),
                "MAE": round(m["mae"], 4),
                "RMSE": round(m["rmse"], 4),
                "MAPE (%)": round(m["mape"], 2) if m["mape"] is not None else None,
                "Bias": round(m["bias"], 4),
                "Training Time (s)": round(r.get("training_time_sec", 0), 3),
                "Prediction Time (ms)": round(r.get("prediction_time_sec", 0) * 1000, 2),
                "Trained": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Predicted vs. Actual / Residuals / Feature Importance")
    detail_label = st.selectbox("Inspect model", selected, key="ae_compare_detail")
    detail_run = chosen[selected.index(detail_label)]
    sample = detail_run["prediction_sample"]
    y_true = sample["y_true"]
    y_pred = sample["y_pred"]
    residuals = [p - t for p, t in zip(y_pred, y_true)]

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=4), name="Predicted vs Actual"))
        lo, hi = min(y_true + y_pred), max(y_true + y_pred)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
        fig.update_layout(
            title="Predicted vs. Actual (held-out test sample)", height=340, xaxis_title="Actual AE", yaxis_title="Predicted AE"
        )
        plot_retro(fig, key="ae_compare_pred_actual")
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=4), name="Residual"))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Residual Plot", height=340, yaxis_title="Predicted − Actual")
        plot_retro(fig2, key="ae_compare_residual")

    if detail_run.get("feature_importance"):
        st.markdown("##### Feature Importance (Top 20)")
        fi_df = pd.DataFrame(detail_run["feature_importance"], columns=["Feature", "Importance"])
        fig3 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig3.update_layout(title="Top Contributing Features", height=420, yaxis=dict(autorange="reversed"))
        plot_retro(fig3, key="ae_compare_feature_importance")
    else:
        st.caption("Feature importance not available for this model type (e.g. SVR, LSTM/GRU). SHAP support is future work.")

    if detail_run.get("loss_history"):
        st.markdown("##### Training / Validation Loss")
        lh = detail_run["loss_history"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig4.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig4.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig4, key="ae_compare_loss")


def render_ae_feature_ablation_tab() -> None:
    st.caption(
        "Trains a Full Model (every feature group + every engineered group enabled) then retrains "
        "once per unit with just that one disabled — ranked by how much R² drops when removed. "
        "Leave-one-out, not a cumulative sweep, so results are order-independent."
    )
    model_type = st.selectbox("Model", ae_research.TABULAR_MODELS, key="ae_ablation_model")
    horizon = _ae_research_horizon_selector("ae_ablation")
    if st.button("🔬 Run Feature Ablation Sweep", key="ae_ablation_run_btn", type="primary"):
        n_units = len(ae_research.FEATURE_ABLATION_UNITS) + 1
        with st.spinner(f"Training {n_units} models ({model_type}, +{horizon}h)..."):
            try:
                result = ae_research.run_feature_ablation_sweep(model_type, horizon=horizon)
                st.session_state["ae_ablation_result"] = result
                st.toast("Feature ablation sweep complete.")
            except Exception as exc:
                st.error(f"Ablation sweep failed: {exc}")

    result = st.session_state.get("ae_ablation_result")
    if not result:
        st.info("Pick a model and horizon above, then run the sweep.")
        return

    st.markdown(f"##### {result['model_type']} +{result['horizon']}h — Full Model R² = {result['full_r2']:.4f}")
    ranked = result["ranked"]
    fig = go.Figure(
        go.Bar(
            x=[r["delta_r2"] for r in ranked],
            y=[r["unit"] for r in ranked],
            orientation="h",
            marker_color=["#1f7a3a" if r["delta_r2"] >= 0 else "#7a1f1f" for r in ranked],
        )
    )
    fig.update_layout(
        title="R² Drop When Removed — Ranked Feature Group Contribution",
        height=380,
        xaxis_title="ΔR² (Full Model − Without This Group)",
        yaxis=dict(autorange="reversed"),
    )
    plot_retro(fig, key="ae_ablation_bar")

    table_rows = [
        {
            "Rank": i + 1,
            "Feature Group": r["unit"].replace("Without ", ""),
            "R² Without": round(r["r2"], 4),
            "ΔR² (contribution)": round(r["delta_r2"], 4),
            "MAE Without": round(r["mae"], 4),
        }
        for i, r in enumerate(ranked)
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Swept at {pd.Timestamp(result['swept_at']).strftime('%Y-%m-%d %H:%M UTC')} — a positive ΔR² "
        "means removing that group made the model worse (it was contributing)."
    )


def render_ae_physics_experiments_tab() -> None:
    st.caption(
        "Exploratory analysis: how does each individual physics-derived feature relate to AE's "
        "behavior historically? Enable one to inspect it, or quick-train a model with just that one "
        "feature added on top of the full default feature set."
    )
    label = st.selectbox("Physics feature", ae_research.PHYSICS_FEATURE_OPTIONS, key="ae_physics_feature_select")

    try:
        base_frame, base_cols = _cached_ae_research_frame()
        frame, cols_with = _cached_ae_research_frame(physics_features={label: True})
    except Exception as exc:
        st.warning(f"Could not load the research feature frame: {exc}")
        return

    new_cols = [c for c in cols_with if c not in base_cols]
    if not new_cols:
        st.warning("Could not resolve this feature's column name.")
        return
    col = new_cols[-1]

    latest = frame[col].dropna()
    if latest.empty:
        st.info("Not enough history to compute this feature yet.")
        return
    metric_card(label, f"{latest.iloc[-1]:.3f}", f"Latest hourly value ({latest.index[-1].strftime('%Y-%m-%d %H:%M UTC')})")

    recent = frame.tail(24 * 30).dropna(subset=[col, "ae"])
    if not recent.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=recent.index, y=recent["ae"], name="AE", line=dict(color="#1f5a7a")), secondary_y=False)
        fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=label, line=dict(color="#7a1f5a")), secondary_y=True)
        fig.update_layout(title=f"AE vs. {label} — last {len(recent)} hours", height=380)
        fig.update_yaxes(title_text="AE (nT)", secondary_y=False)
        fig.update_yaxes(title_text=label, secondary_y=True)
        plot_retro(fig, key=f"ae_physics_timeseries_{col}")

    next_ae = frame["ae"].shift(-1)
    valid = frame[[col]].join(next_ae.rename("next_ae")).dropna()
    if len(valid) > 10:
        corr = valid[col].corr(valid["next_ae"])
        st.caption(f"Correlation between {label} and next-hour AE, across the full history: **{corr:.3f}**")
        fig2 = go.Figure(go.Scattergl(x=valid[col], y=valid["next_ae"], mode="markers", marker=dict(size=3, opacity=0.4)))
        fig2.update_layout(title=f"{label} vs. Next-Hour AE", height=380, xaxis_title=label, yaxis_title="Next-hour AE (nT)")
        plot_retro(fig2, key=f"ae_physics_corr_{col}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button(f"🧪 Quick-Train Linear Regression with {label} added", key="ae_physics_quick_train"):
        with st.spinner("Training..."):
            try:
                run = ae_research.train_ae_research_model(
                    "Linear Regression", physics_features={label: True}, notes=f"Physics Experiments — quick test of {label}"
                )
                st.success(
                    f"Trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.3f} "
                    "(compare against the Feature Ablation / Model Comparison full-model baseline)."
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def render_ae_sequence_models_tab() -> None:
    st.caption(
        "LSTM/GRU look back over a window of consecutive hours (Sequence Length) rather than a "
        "single row. Trains in an isolated subprocess — TensorFlow cannot safely share a process "
        "with this lab's scikit-learn/XGBoost/LightGBM/CatBoost imports."
    )
    if not ae_research.SEQUENCE_MODELS:
        st.warning("TensorFlow/Keras is not installed — sequence models are unavailable in this environment.")
        return

    model_type = st.selectbox("Model", ae_research.SEQUENCE_MODELS, key="ae_seq_model")
    horizon = _ae_research_horizon_selector("ae_seq")
    sequence_length = st.selectbox(
        "Sequence Length (hours)",
        ae_research.SEQUENCE_LENGTH_OPTIONS,
        index=ae_research.SEQUENCE_LENGTH_OPTIONS.index(ae_research.DEFAULT_SEQUENCE_LENGTH),
        key="ae_seq_seqlen",
    )
    with st.expander("Feature Groups", expanded=False):
        feature_toggles = _ae_research_feature_toggle_form("ae_seq")
    hyperparams = _ae_research_hyperparam_inputs(model_type, "ae_seq")

    if st.button("🧪 Train Sequence Model", key="ae_seq_train_btn", type="primary"):
        with st.spinner(f"Training {model_type} (seq={sequence_length}h) — isolated subprocess, ~10-30s..."):
            try:
                run = ae_research.train_ae_research_model(
                    model_type,
                    horizon=horizon,
                    feature_toggles=feature_toggles,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes="Sequence Models tab",
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    all_runs = ae_research.list_runs()
    seq_runs = [r for r in all_runs if r["model_type"] in ae_research.SEQUENCE_MODELS]
    tabular_runs = [r for r in all_runs if r["model_type"] not in ae_research.SEQUENCE_MODELS]
    if not seq_runs:
        st.info("No LSTM/GRU runs yet — train one above.")
        return

    best_seq = max(seq_runs, key=lambda r: r["metrics"]["r2"])
    if tabular_runs:
        best_tabular = max(tabular_runs, key=lambda r: r["metrics"]["r2"])
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Best Sequence Model R²", f"{best_seq['metrics']['r2']:.4f}", best_seq["model_type"])
        with c2:
            metric_card("Best Tabular Model R²", f"{best_tabular['metrics']['r2']:.4f}", best_tabular["model_type"])
        with c3:
            diff = best_seq["metrics"]["r2"] - best_tabular["metrics"]["r2"]
            metric_card("Sequence Advantage", f"{diff:+.4f}", "Positive = sequence models outperform tabular")

    def _run_label(r):
        return f"{r['model_type']} seq={r['sequence_length']}h +{r.get('horizon', 1)}h ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in seq_runs]
    selected = st.selectbox("Inspect run", labels, key="ae_seq_inspect")
    run = seq_runs[labels.index(selected)]
    m = run["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("R²", f"{m['r2']:.4f}", "")
    with c2:
        metric_card("MAE", f"{m['mae']:.3f}", "")
    with c3:
        metric_card("RMSE", f"{m['rmse']:.3f}", "")
    with c4:
        metric_card("MAPE", "N/A" if m["mape"] is None else f"{m['mape']:.1f}%", "")
    with c5:
        metric_card("Bias", f"{m['bias']:+.3f}", "")

    if run.get("loss_history"):
        lh = run["loss_history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig, key="ae_seq_loss")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### All Sequence Model Runs")
    for r in seq_runs:
        _ae_research_run_row(r, key_prefix="ae_seq")


def render_ae_horizon_analysis_tab() -> None:
    st.caption(
        "Automatically trains (or reuses existing runs for) one model per horizon (1h/2h/3h — the "
        "full range this lab supports, see the lab's own note on why sub-hourly horizons aren't "
        "offered) and plots how R²/MAE/RMSE change with forecast lead time."
    )
    model_type = st.selectbox("Model", ae_research.TABULAR_MODELS, key="ae_horizon_model")
    reuse_existing = st.checkbox("Reuse existing runs where available (faster)", value=True, key="ae_horizon_reuse")

    if st.button("📈 Run Horizon Sweep", key="ae_horizon_sweep_btn", type="primary"):
        with st.spinner(f"Sweeping {model_type} across {len(ae_research.HORIZON_OPTIONS)} horizons..."):
            try:
                runs = ae_research.train_horizon_sweep(model_type, reuse_existing=reuse_existing)
                st.session_state["ae_horizon_sweep_result"] = {
                    "model_type": model_type,
                    "run_ids": [r["run_id"] for r in runs],
                }
                st.toast(f"Swept {len(runs)} horizons.")
            except Exception as exc:
                st.error(f"Horizon sweep failed: {exc}")

    result = st.session_state.get("ae_horizon_sweep_result")
    if not result:
        st.info("Pick a model above, then run the sweep.")
        return

    runs = [ae_research.get_run(rid) for rid in result["run_ids"]]
    runs = [r for r in runs if r is not None]
    if not runs:
        st.warning("Swept runs are no longer available (were they deleted?). Run the sweep again.")
        return
    runs = sorted(runs, key=lambda r: r.get("horizon", 1))

    horizon_labels = [f"{r.get('horizon', 1)}h" for r in runs]
    r2_values = [r["metrics"]["r2"] for r in runs]
    mae_values = [r["metrics"]["mae"] for r in runs]
    rmse_values = [r["metrics"]["rmse"] for r in runs]

    st.markdown(f"##### {result['model_type']} — AE skill decay with lead time")
    fig_r2 = go.Figure(go.Scatter(x=horizon_labels, y=r2_values, mode="lines+markers", name="R²"))
    fig_r2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_r2.update_layout(title="Horizon vs. R²", height=340, xaxis_title="Forecast Horizon", yaxis_title="R²")
    plot_retro(fig_r2, key="ae_horizon_r2")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_mae = go.Figure(go.Scatter(x=horizon_labels, y=mae_values, mode="lines+markers", name="MAE"))
        fig_mae.update_layout(title="Horizon vs. MAE", height=320, xaxis_title="Forecast Horizon", yaxis_title="MAE")
        plot_retro(fig_mae, key="ae_horizon_mae")
    with col_b:
        fig_rmse = go.Figure(go.Scatter(x=horizon_labels, y=rmse_values, mode="lines+markers", name="RMSE"))
        fig_rmse.update_layout(title="Horizon vs. RMSE", height=320, xaxis_title="Forecast Horizon", yaxis_title="RMSE")
        plot_retro(fig_rmse, key="ae_horizon_rmse")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    table_rows = [
        {
            "Horizon": horizon_labels[i],
            "R²": round(r2_values[i], 4),
            "MAE": round(mae_values[i], 4),
            "RMSE": round(rmse_values[i], 4),
            "Test Samples": runs[i]["n_test_samples"],
        }
        for i in range(len(runs))
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def render_ae_experiment_tracking_tab() -> None:
    st.caption(
        "Every experiment ever run in this lab — full configuration and results, for reproducibility. "
        "Filter, inspect, promote, or delete any run."
    )
    runs = ae_research.list_runs()
    if not runs:
        st.info("No experiments recorded yet — train a model in any other tab.")
        return

    model_filter = st.selectbox("Filter by model", ["All"] + ae_research.ALL_TRAINABLE_MODELS, key="ae_track_model_filter")
    filtered = runs if model_filter == "All" else [r for r in runs if r["model_type"] == model_filter]
    st.caption(f"{len(filtered)} of {len(runs)} total experiments")

    rows = []
    for r in filtered:
        feat_summary = ", ".join(g for g, cols in r.get("feature_toggles", {}).items() if all(cols.values())) or "partial"
        physics_on = ", ".join(k for k, v in (r.get("physics_features") or {}).items() if v) or "None"
        rows.append(
            {
                "Timestamp": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
                "Model": r["model_type"],
                "Horizon": f"{r.get('horizon', 1)}h",
                "Feature Groups (fully on)": feat_summary,
                "Physics Features": physics_on,
                "Seq Len": str(r["sequence_length"]) + "h" if r.get("sequence_length") else "—",
                "Train N": r["n_train_samples"],
                "Test N": r["n_test_samples"],
                "R²": round(r["metrics"]["r2"], 4),
                "MAE": round(r["metrics"]["mae"], 4),
                "RMSE": round(r["metrics"]["rmse"], 4),
                "MAPE (%)": round(r["metrics"]["mape"], 2) if r["metrics"]["mape"] is not None else None,
                "Bias": round(r["metrics"]["bias"], 4),
                "Train Time (s)": round(r.get("training_time_sec", 0), 3),
                "Notes": r.get("notes", ""),
                "Promoted": "🚀" if r.get("promoted") else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Manage Runs")
    for r in filtered[:15]:
        _ae_research_run_row(r, key_prefix="ae_track")
    if len(filtered) > 15:
        st.caption(f"Showing 15 most recent of {len(filtered)}.")


def render_ae_hypothesis_testing_tab() -> None:
    st.caption(
        "Fixed, reproducible hypotheses. Most train a baseline WITHOUT the tested feature and an "
        "experimental run WITH it; 'Akasofu Epsilon outperforms Ey' is a head-to-head SWAP instead "
        "(one feature replaces the other, not an additive test). Reports ΔR²/ΔMAE/ΔRMSE and an "
        f"Accept/Reject verdict (Accept requires ΔR² ≥ {ae_research.HYPOTHESIS_ACCEPT_THRESHOLD_R2})."
    )
    hypothesis_label = st.selectbox("Hypothesis", list(ae_research.HYPOTHESIS_DEFINITIONS), key="ae_hyp_select")
    model_type = st.selectbox("Model", ae_research.TABULAR_MODELS, key="ae_hyp_model")
    horizon = _ae_research_horizon_selector("ae_hyp")

    if st.button("🔬 Run Hypothesis Test", key="ae_hyp_run_btn", type="primary"):
        with st.spinner(f"Testing: {hypothesis_label}..."):
            try:
                result = ae_research.run_hypothesis_test(hypothesis_label, model_type, horizon=horizon)
                st.session_state["ae_hyp_last_result"] = result
                st.toast(f"{result['verdict']} — ΔR²={result['delta_r2']:+.4f}")
            except Exception as exc:
                st.error(f"Hypothesis test failed: {exc}")

    result = st.session_state.get("ae_hyp_last_result")
    if result and result["hypothesis"] == hypothesis_label:
        color = "#1f7a3a" if result["verdict"] == "Accept" else "#7a1f1f"
        st.markdown(f"### Verdict: <span style='color:{color}'>{result['verdict']}</span>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("ΔR²", f"{result['delta_r2']:+.4f}", "")
        with c2:
            metric_card("ΔMAE", f"{result['delta_mae']:+.4f}", "")
        with c3:
            metric_card("ΔRMSE", f"{result['delta_rmse']:+.4f}", "")
        with c4:
            metric_card("Baseline → Experimental R²", f"{result['baseline_r2']:.4f} → {result['experimental_r2']:.4f}", "")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Hypothesis Test History")
    results = ae_research.list_hypothesis_results()
    if not results:
        st.info("No hypothesis tests run yet.")
        return
    rows = [
        {
            "Hypothesis": r["hypothesis"],
            "Model": r["model_type"],
            "Horizon": f"{r.get('horizon', 1)}h",
            "Verdict": r["verdict"],
            "ΔR²": round(r["delta_r2"], 4),
            "ΔMAE": round(r["delta_mae"], 4),
            "ΔRMSE": round(r["delta_rmse"], 4),
            "Tested": pd.Timestamp(r["tested_at"]).strftime("%Y-%m-%d %H:%M UTC"),
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_ae_research_laboratory() -> None:
    st.markdown("### 🧪 AE Research Laboratory")
    hdr_col, pause_col = st.columns([3, 1])
    with hdr_col:
        st.caption(
            "A scientific experimentation platform for AE forecasting — fully isolated from the "
            "Production Prediction tab, which continues using the current trained production model "
            "exactly as-is. Nothing trained here overwrites it; Promote only labels a run for your own "
            "tracking, wiring a model into production is always a manual, deliberate step."
        )
    with pause_col:
        st.session_state.setdefault("pause_autorefresh", False)
        if st.toggle("⏸ Pause Live Refresh", value=st.session_state["pause_autorefresh"], key="ae_lab_pause_toggle"):
            st.session_state["pause_autorefresh"] = True
        else:
            st.session_state["pause_autorefresh"] = False
    _ae_research_notes()
    with st.expander("Future Research — planned architecture extensions"):
        st.write(", ".join(ae_research.FUTURE_MODELS))
        st.caption(
            "Registered as disabled entries in the model selector now, so adding a real "
            "implementation later never requires redesigning this interface."
        )

    sub = st.tabs(
        [
            "🔬 AE Optimization Study",
            "Model Comparison",
            "Feature Ablation",
            "Physics Experiments",
            "Sequence Models",
            "Horizon Analysis",
            "Experiment Tracking",
            "Hypothesis Testing",
        ]
    )
    with sub[0]:
        render_ae_optimization_study()
    with sub[1]:
        render_ae_model_comparison_tab()
    with sub[2]:
        render_ae_feature_ablation_tab()
    with sub[3]:
        render_ae_physics_experiments_tab()
    with sub[4]:
        render_ae_sequence_models_tab()
    with sub[5]:
        render_ae_horizon_analysis_tab()
    with sub[6]:
        render_ae_experiment_tracking_tab()
    with sub[7]:
        render_ae_hypothesis_testing_tab()


_HYPOTHESIS_DATASET_OPTIONS = ["analytics", "ae", "experimental"]
_HYPOTHESIS_VARIABLE_OPTIONS = ["kp", "dst", "ae"]


def _hypothesis_architecture_form(prefix: str, defaults: dict = None) -> dict:
    """Shared baseline/experimental dataset+variable picker, used by both
    the create form and the edit form.
    """
    defaults = defaults or {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Architecture**")
        baseline_dataset = st.selectbox(
            "Baseline Dataset",
            _HYPOTHESIS_DATASET_OPTIONS,
            index=_HYPOTHESIS_DATASET_OPTIONS.index(defaults.get("baseline_dataset", "analytics")),
            key=f"{prefix}_baseline_ds",
        )
        baseline_variable = st.selectbox(
            "Baseline Variable",
            _HYPOTHESIS_VARIABLE_OPTIONS,
            index=_HYPOTHESIS_VARIABLE_OPTIONS.index(defaults.get("baseline_variable", "kp")),
            key=f"{prefix}_baseline_var",
        )
    with col2:
        st.markdown("**Experimental Architecture**")
        experimental_dataset = st.selectbox(
            "Experimental Dataset",
            _HYPOTHESIS_DATASET_OPTIONS,
            index=_HYPOTHESIS_DATASET_OPTIONS.index(defaults.get("experimental_dataset", "experimental")),
            key=f"{prefix}_exp_ds",
        )
        experimental_variable = st.selectbox(
            "Experimental Variable",
            _HYPOTHESIS_VARIABLE_OPTIONS,
            index=_HYPOTHESIS_VARIABLE_OPTIONS.index(defaults.get("experimental_variable", "kp")),
            key=f"{prefix}_exp_var",
        )
    return {
        "baseline_dataset": baseline_dataset,
        "baseline_variable": baseline_variable,
        "experimental_dataset": experimental_dataset,
        "experimental_variable": experimental_variable,
    }


def render_hypothesis_testing_tab() -> None:
    """Experiment management and evaluation system — not a prediction
    page. Every hypothesis pairs a baseline (dataset, variable) against
    an experimental one; conclusions and confidence come entirely from
    swdss.models.hypothesis's fixed statistical rules over measured,
    verified predictions — never an LLM, never a claim of "true."
    """
    st.caption(
        "Experiment management and evaluation system — not a prediction page. Every hypothesis "
        "compares a baseline architecture against an experimental one, using only measured, "
        "verified prediction results. Conclusions are **Supported / Not Supported / Inconclusive** "
        "— never claimed as \"true\" — generated entirely from fixed statistical rules. No LLM."
    )

    with st.expander("➕ Create New Hypothesis"):
        with st.form("new_hypothesis_form", clear_on_submit=True):
            title = st.text_input("Title")
            description = st.text_area("Description", height=70)
            motivation = st.text_area("Scientific Motivation", height=70)
            physics_bg = st.text_area("Physics Background", height=70)
            expected = st.text_input("Expected Improvement")
            arch = _hypothesis_architecture_form("new_hyp")
            notes = st.text_area("Initial Notes (markdown supported)", height=90)

            if st.form_submit_button("Create Hypothesis"):
                if not title:
                    st.error("Title is required.")
                else:
                    created = create_hypothesis(
                        title=title,
                        description=description,
                        scientific_motivation=motivation,
                        physics_background=physics_bg,
                        expected_improvement=expected,
                        notes=notes,
                        **arch,
                    )
                    st.toast(f"Created hypothesis: {created['title']}")
                    st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    show_archived = st.checkbox("Show archived hypotheses", key="hyp_show_archived")
    hyps = list_hypotheses() if show_archived else list_hypotheses(status="active")

    st.markdown("##### Hypotheses")
    if not hyps:
        st.info("No hypotheses yet. Create one above.")
        return

    for h in hyps:
        result = evaluate_hypothesis(h)
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([0.45, 0.18, 0.22, 0.15])
            with c1:
                number_text = f"#{h['number']} — " if h["number"] else ""
                archived_tag = " *(archived)*" if h["status"] == "archived" else ""
                st.markdown(f"**{number_text}{h['title']}**{archived_tag}")
                st.caption(h["description"] or "No description.")
                st.caption(
                    f"Baseline: `{h['baseline_dataset']}/{h['baseline_variable']}` vs. "
                    f"Experimental: `{h['experimental_dataset']}/{h['experimental_variable']}` — "
                    f"created {pd.Timestamp(h['created_at']).strftime('%Y-%m-%d')}"
                )
            with c2:
                metric_card("Verified", str(result["n"]), f"of {result['baseline']['count'] + result['experimental']['count']} total")
            with c3:
                metric_card(
                    "Conclusion",
                    result["conclusion"],
                    f"Confidence: {result['confidence']}",
                    value_color=CONCLUSION_COLORS.get(result["conclusion"], "#404040"),
                )
            with c4:
                st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
                if st.button("View", key=f"view_hyp_{h['hypothesis_id']}", use_container_width=True):
                    open_dialog("hypothesis_detail", h["hypothesis_id"])


@st.dialog("Hypothesis Detail", width="large", dismissible=False)
def show_hypothesis_detail(hypothesis_id: str) -> None:
    render_dialog_close_button("close_hypothesis_detail")

    h = get_hypothesis(hypothesis_id)
    if h is None:
        st.error("This hypothesis could not be found.")
        return

    result = evaluate_hypothesis(h)
    baseline, experimental = result["baseline"], result["experimental"]

    number_text = f"Hypothesis {h['number']} — " if h["number"] else ""
    st.subheader(f"{number_text}{h['title']}")
    st.caption(
        f"Created {pd.Timestamp(h['created_at']).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Status: {h['status'].title()}"
    )

    with st.expander("✏️ Edit Hypothesis Structure"):
        with st.form(f"edit_hyp_{hypothesis_id}"):
            edit_title = st.text_input("Title", value=h["title"])
            edit_description = st.text_area("Description", value=h["description"] or "", height=70)
            edit_motivation = st.text_area("Scientific Motivation", value=h["scientific_motivation"] or "", height=70)
            edit_physics_bg = st.text_area("Physics Background", value=h["physics_background"] or "", height=70)
            edit_expected = st.text_input("Expected Improvement", value=h["expected_improvement"] or "")
            edit_arch = _hypothesis_architecture_form(f"edit_hyp_{hypothesis_id}", defaults=h)
            if st.form_submit_button("Save Changes"):
                update_hypothesis(
                    hypothesis_id,
                    title=edit_title,
                    description=edit_description,
                    scientific_motivation=edit_motivation,
                    physics_background=edit_physics_bg,
                    expected_improvement=edit_expected,
                    **edit_arch,
                )
                st.toast("Hypothesis updated.")
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Hypothesis Structure")
    st.markdown(f"**Description:** {h['description'] or 'N/A'}")
    st.markdown(f"**Scientific Motivation:** {h['scientific_motivation'] or 'N/A'}")
    st.markdown(f"**Physics Background:** {h['physics_background'] or 'N/A'}")
    st.markdown(f"**Expected Improvement:** {h['expected_improvement'] or 'N/A'}")
    st.markdown(f"**Baseline Architecture:** `{h['baseline_dataset']}` / `{h['baseline_variable']}`")
    st.markdown(f"**Experimental Architecture:** `{h['experimental_dataset']}` / `{h['experimental_variable']}`")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Automatic Conclusion")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Conclusion", result["conclusion"], "", value_color=CONCLUSION_COLORS.get(result["conclusion"], "#404040"))
    with c2:
        metric_card("Confidence", result["confidence"], f"{result['n']} verified predictions")
    st.info(result["summary"])
    if h["manual_conclusion"]:
        st.markdown("**Researcher's Manual Conclusion / Addendum:**")
        st.markdown(h["manual_conclusion"])

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Comparison Table")

    def _fmt(v, dp=3):
        return "N/A" if v is None else f"{v:.{dp}f}"

    def _improvement(b, e, higher_is_better=False):
        if b is None or e is None:
            return "N/A"
        if higher_is_better:
            return f"{e - b:+.3f}"
        if b == 0:
            return "N/A"
        return f"{(b - e) / abs(b) * 100:+.1f}%"

    comparison_rows = [
        {"Metric": "MAE", "Baseline": _fmt(baseline["mae"]), "Experimental": _fmt(experimental["mae"]), "Improvement": _improvement(baseline["mae"], experimental["mae"])},
        {"Metric": "RMSE", "Baseline": _fmt(baseline["rmse"]), "Experimental": _fmt(experimental["rmse"]), "Improvement": _improvement(baseline["rmse"], experimental["rmse"])},
        {"Metric": "R²", "Baseline": _fmt(baseline["r2"], 3), "Experimental": _fmt(experimental["r2"], 3), "Improvement": _improvement(baseline["r2"], experimental["r2"], higher_is_better=True)},
        {"Metric": "MAPE (%)", "Baseline": _fmt(baseline["mape"], 1), "Experimental": _fmt(experimental["mape"], 1), "Improvement": _improvement(baseline["mape"], experimental["mape"])},
        {"Metric": "Bias", "Baseline": _fmt(baseline["bias"]), "Experimental": _fmt(experimental["bias"]), "Improvement": "N/A"},
        {"Metric": "Max Error", "Baseline": _fmt(baseline["max_error"]), "Experimental": _fmt(experimental["max_error"]), "Improvement": _improvement(baseline["max_error"], experimental["max_error"])},
        {"Metric": "Median Error", "Baseline": _fmt(baseline["median_error"]), "Experimental": _fmt(experimental["median_error"]), "Improvement": _improvement(baseline["median_error"], experimental["median_error"])},
        {"Metric": "Average Drift", "Baseline": _fmt(baseline["avg_drift"]), "Experimental": _fmt(experimental["avg_drift"]), "Improvement": "N/A"},
        {"Metric": "Forecast Stability (std)", "Baseline": _fmt(baseline["stability"]), "Experimental": _fmt(experimental["stability"]), "Improvement": "N/A"},
        {
            "Metric": "Storm-Time MAE",
            "Baseline": f"{_fmt(baseline['storm_mae'])} (n={baseline['storm_count']})",
            "Experimental": f"{_fmt(experimental['storm_mae'])} (n={experimental['storm_count']})",
            "Improvement": _improvement(baseline["storm_mae"], experimental["storm_mae"]),
        },
        {
            "Metric": "Quiet-Time MAE",
            "Baseline": f"{_fmt(baseline['quiet_mae'])} (n={baseline['quiet_count']})",
            "Experimental": f"{_fmt(experimental['quiet_mae'])} (n={experimental['quiet_count']})",
            "Improvement": _improvement(baseline["quiet_mae"], experimental["quiet_mae"]),
        },
    ]
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Visualizations")

    if baseline["errors"] or experimental["errors"]:
        v1, v2 = st.columns(2)
        with v1:
            fig = go.Figure()
            if baseline["errors"]:
                fig.add_trace(go.Histogram(x=baseline["errors"], name="Baseline", opacity=0.6))
            if experimental["errors"]:
                fig.add_trace(go.Histogram(x=experimental["errors"], name="Experimental", opacity=0.6))
            fig.update_layout(title="Prediction Error Histogram", barmode="overlay", height=320)
            plot_retro(fig)
        with v2:
            fig2 = go.Figure()
            if baseline["predicted_vs_actual"]:
                bp, ba = zip(*baseline["predicted_vs_actual"])
                fig2.add_trace(go.Scatter(x=ba, y=bp, mode="markers", name="Baseline"))
            if experimental["predicted_vs_actual"]:
                ep, ea = zip(*experimental["predicted_vs_actual"])
                fig2.add_trace(go.Scatter(x=ea, y=ep, mode="markers", name="Experimental"))
            fig2.update_layout(title="Predicted vs. Official", height=320, xaxis_title="Official", yaxis_title="Predicted")
            plot_retro(fig2)

        v3, v4 = st.columns(2)
        with v3:
            fig3 = go.Figure()
            if baseline["predicted_vs_actual"]:
                bp, ba = zip(*baseline["predicted_vs_actual"])
                fig3.add_trace(go.Scatter(x=list(ba), y=[p - a for p, a in zip(bp, ba)], mode="markers", name="Baseline"))
            if experimental["predicted_vs_actual"]:
                ep, ea = zip(*experimental["predicted_vs_actual"])
                fig3.add_trace(go.Scatter(x=list(ea), y=[p - a for p, a in zip(ep, ea)], mode="markers", name="Experimental"))
            fig3.update_layout(title="Residual Plot", height=320, xaxis_title="Official", yaxis_title="Residual (Predicted - Official)")
            plot_retro(fig3)
        with v4:
            fig4 = go.Figure()
            if baseline["trend"]:
                times, errs = zip(*baseline["trend"])
                fig4.add_trace(go.Scatter(x=[pd.Timestamp(t) for t in times], y=errs, mode="lines+markers", name="Baseline"))
            if experimental["trend"]:
                times_e, errs_e = zip(*experimental["trend"])
                fig4.add_trace(go.Scatter(x=[pd.Timestamp(t) for t in times_e], y=errs_e, mode="lines+markers", name="Experimental"))
            fig4.update_layout(title="Performance Timeline (Absolute Error)", height=320)
            plot_retro(fig4)

        st.markdown("###### Storm vs. Quiet Performance (MAE)")
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=["Storm", "Quiet"], y=[baseline["storm_mae"], baseline["quiet_mae"]], name="Baseline"))
        fig5.add_trace(go.Bar(x=["Storm", "Quiet"], y=[experimental["storm_mae"], experimental["quiet_mae"]], name="Experimental"))
        fig5.update_layout(height=320, barmode="group", yaxis_title="MAE")
        plot_retro(fig5)
    else:
        st.info("No verified predictions yet for either architecture — visualizations will appear once forecasts complete and are verified.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Experiment Timeline")
    t1, t2, t3 = st.columns(3)
    with t1:
        metric_card("Experiment Started", pd.Timestamp(h["created_at"]).strftime("%Y-%m-%d %H:%M UTC"), "")
    with t2:
        metric_card("Predictions Generated", str(baseline["count"] + experimental["count"]), "Baseline + Experimental")
    with t3:
        metric_card("Current Status", h["status"].title(), "")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Research Notes")
    notes_text = st.text_area("Notes (markdown supported)", value=h["notes"] or "", height=150, key=f"notes_{hypothesis_id}")
    manual_conclusion_text = st.text_area(
        "Manual Conclusion / Addendum (optional)", value=h["manual_conclusion"] or "", height=70, key=f"manual_{hypothesis_id}"
    )
    if st.button("💾 Save Notes", key=f"save_notes_{hypothesis_id}", use_container_width=True):
        update_notes(hypothesis_id, notes_text)
        update_manual_conclusion(hypothesis_id, manual_conclusion_text)
        st.toast("Notes saved.")
        st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    dup_col, archive_col, delete_col = st.columns(3)
    with dup_col:
        if st.button("📋 Duplicate Experiment", key=f"dup_{hypothesis_id}", use_container_width=True):
            new_h = duplicate_hypothesis(hypothesis_id)
            st.toast(f"Duplicated as: {new_h['title']}")
            close_active_dialog()
    with archive_col:
        if h["status"] == "active":
            if st.button("🗄️ Archive", key=f"archive_{hypothesis_id}", use_container_width=True):
                archive_hypothesis(hypothesis_id)
                st.toast("Archived.")
                st.rerun()
        else:
            if st.button("♻️ Reactivate", key=f"reactivate_{hypothesis_id}", use_container_width=True):
                reactivate_hypothesis(hypothesis_id)
                st.toast("Reactivated.")
                st.rerun()
    with delete_col:
        if st.button("🗑️ Delete", key=f"delete_hyp_{hypothesis_id}", use_container_width=True):
            delete_hypothesis(hypothesis_id)
            close_active_dialog()
