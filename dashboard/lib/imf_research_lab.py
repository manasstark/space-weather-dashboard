"""IMF Research Laboratory (Bz experimentation) — extracted verbatim from
dashboard/home.py. See dashboard/home.py's research_lab_page and the
Heliosphere page for where render_imf_research_laboratory() is called.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.lib.shared_ui import (
    CONCLUSION_COLORS,
    REFRESH_SECONDS,
    metric_card,
    plot_retro,
)
from swdss.models.imf_research import (
    ALL_TRAINABLE_MODELS,
    DEFAULT_FEATURE_SET,
    DEFAULT_GRANULARITY,
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_SET_OPTIONS,
    FUTURE_MODELS,
    GRANULARITY_OPTIONS,
    HOURLY_HORIZONS,
    HYPERPARAM_SCHEMA,
    MINUTE_HORIZONS,
    SEQUENCE_LENGTH_OPTIONS,
    SEQUENCE_MODELS,
    TABULAR_MODELS,
    TARGET_OPTIONS,
    compare_runs,
    compute_persistence_benchmark,
    delete_run,
    get_run,
    list_runs,
    list_studies,
    load_research_frame,
    mark_study_promoted,
    mark_study_rejected,
    promote_run,
    promote_to_production,
    run_complete_optimization_study,
    train_horizon_sweep,
    train_research_model,
    train_research_model_exp,
)

# ==================== IMF Research Laboratory (Bz experimentation) ====================
# Fully isolated from the Production Prediction tab — see
# swdss.models.imf_research module docstring for the production-safety
# contract. Nothing below ever writes to the production model files.


def _imf_research_notes() -> None:
    st.info(
        "**Why Bz is hard to forecast.** Bz is fundamentally different from Solar Wind Speed or "
        "Density: it's the *orientation* of the interplanetary magnetic field, not a scalar plasma "
        "quantity. Its evolution depends on evolving magnetic structures — CME flux-rope rotation, "
        "turbulence, current-sheet crossings — not just how fast or dense the plasma is. Sequence "
        "models (LSTM/GRU) may capture this temporal magnetic evolution better than single-row "
        "regression models, since they see how the field has been *rotating* over time, not just its "
        "instantaneous value."
    )


def _imf_research_hyperparam_inputs(model_type: str, key_prefix: str) -> dict:
    schema = HYPERPARAM_SCHEMA.get(model_type, {})
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
                    label,
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=spec["default"],
                    step=1,
                    key=f"{key_prefix}_{model_type}_{name}",
                )
            else:
                values[name] = st.number_input(
                    label,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    step=0.01,
                    key=f"{key_prefix}_{model_type}_{name}",
                )
    return values


def _imf_research_model_selector(key_prefix: str) -> str:
    options = ALL_TRAINABLE_MODELS + FUTURE_MODELS

    def _fmt(name):
        return f"{name} (coming soon)" if name in FUTURE_MODELS else name

    choice = st.selectbox("Model Architecture", options, format_func=_fmt, key=f"{key_prefix}_model_select")
    if choice in FUTURE_MODELS:
        st.warning(f"{choice} is a registered placeholder for future work — not trainable yet.")
    return choice


def _imf_research_granularity_horizon(key_prefix: str) -> tuple:
    """Shared Forecast Granularity + Forecast Horizon control pair — the
    axis that was missing before this redesign (see imf_research.py
    module docstring): "Minute" targets shift(-horizon) minutes on live
    minute data; "Hourly" targets shift(-horizon) hours on the SAME
    historical CSVs production trains on, via the identical
    swdss.models.features functions. Returns (granularity, horizon).
    """
    granularity = st.radio(
        "Forecast Granularity", GRANULARITY_OPTIONS, horizontal=True, key=f"{key_prefix}_granularity"
    )
    horizon_options = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    unit = "min" if granularity == "Minute" else "hour"
    horizon = st.radio(
        "Forecast Horizon",
        horizon_options,
        format_func=lambda h: f"{h} {unit}" + ("s" if h != 1 else ""),
        horizontal=True,
        key=f"{key_prefix}_horizon",
    )
    if granularity == "Hourly":
        st.caption("Sourced from the same 3-year historical CSVs production trains on — genuinely comparable R².")
    else:
        st.caption("Sourced from the live ~7-day minute-level buffer — includes the new physics features.")
    return granularity, horizon


def _imf_research_run_row(run: dict, best_run_id: str = None, key_prefix: str = "runs") -> None:
    """key_prefix disambiguates widget keys when the same run is rendered
    from more than one tab in the same script run (e.g. a trained LSTM/GRU
    run appears in both Training Runs' full log and Sequence Models' own
    list) — Streamlit renders every tab's content on every rerun (inactive
    tabs are just CSS-hidden), so two calls with the same run_id would
    otherwise register duplicate widget keys and crash.
    """
    m = run["metrics"]
    is_best = run["run_id"] == best_run_id
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns([0.24, 0.13, 0.13, 0.13, 0.13, 0.24])
        with c1:
            star = "⭐ " if is_best else ""
            promoted_tag = " 🚀" if run.get("promoted") else ""
            st.markdown(f"**{star}{run['model_type']}**{promoted_tag}")
            granularity = run.get("granularity", "Minute")
            horizon_label = run.get("horizon_label") or (f"{run.get('horizon', 1)}m" if granularity == "Minute" else f"{run.get('horizon', 1)}h")
            seq_note = f" · seq={run['sequence_length']}{'m' if granularity == 'Minute' else 'h'}" if run.get("sequence_length") else ""
            st.caption(
                f"{run['target']} · {granularity} · +{horizon_label}{seq_note} · "
                f"{pd.Timestamp(run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            cv = run.get("cv_metrics")
            if cv:
                st.caption(
                    f"🔁 Walk-forward CV ({cv['n_folds']} folds): R²={cv['r2_mean']:.3f} ± {cv['r2_std']:.3f} · "
                    f"MAE={cv['mae_mean']:.3f} ± {cv['mae_std']:.3f}"
                )
        with c2:
            metric_card("R²", f"{m['r2']:.4f}", "")
        with c3:
            metric_card("MAE", f"{m['mae']:.3f}", "")
        with c4:
            metric_card("RMSE", f"{m['rmse']:.3f}", "")
        with c5:
            metric_card("Bias", f"{m['bias']:+.3f}", "")
        with c6:
            st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button(
                    "Promoted" if run.get("promoted") else "Promote",
                    key=f"{key_prefix}_promote_{run['run_id']}",
                    disabled=run.get("promoted", False),
                    use_container_width=True,
                ):
                    promote_run(run["run_id"])
                    st.toast("Marked as promoted (label only — production untouched).")
                    st.rerun()
            with b2:
                if st.button("Delete", key=f"{key_prefix}_delete_{run['run_id']}", use_container_width=True):
                    delete_run(run["run_id"])
                    st.toast("Run deleted.")
                    st.rerun()


def render_imf_training_runs_tab() -> None:
    st.caption(
        "Train a new experimental model — any target, any granularity/horizon, any architecture "
        "including LSTM/GRU. Every run is stored separately from the production predictor and NEVER "
        "overwrites it — Promote only labels a run for your own tracking; wiring a model into "
        "production is always a manual, deliberate step."
    )
    target_label = st.selectbox("Target Variable", list(TARGET_OPTIONS), index=0, key="imf_research_target")
    granularity, horizon = _imf_research_granularity_horizon("train_runs")
    model_type = _imf_research_model_selector("train_runs")

    sequence_length = None
    if model_type in SEQUENCE_MODELS:
        unit = "minutes" if granularity == "Minute" else "hours"
        sequence_length = st.selectbox(
            f"Sequence Length ({unit}, look-back window — independent of the forecast horizon above)",
            SEQUENCE_LENGTH_OPTIONS,
            index=SEQUENCE_LENGTH_OPTIONS.index(DEFAULT_SEQUENCE_LENGTH),
            key="train_runs_seqlen",
        )

    st.markdown("**Hyperparameters**")
    hyperparams = _imf_research_hyperparam_inputs(model_type, "train_runs") if model_type in ALL_TRAINABLE_MODELS else {}

    if st.button(
        "🧪 Train Model", key="train_runs_train_btn", type="primary", disabled=model_type not in ALL_TRAINABLE_MODELS
    ):
        with st.spinner(f"Training {model_type} on {target_label} ({granularity}, +{horizon})..."):
            try:
                run = train_research_model(
                    target_label,
                    model_type,
                    granularity=granularity,
                    horizon=horizon,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    run_cv=True,
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Training Run Log")
    runs = list_runs()
    if not runs:
        st.info("No training runs yet. Train a model above.")
        return

    filt1, filt2 = st.columns(2)
    with filt1:
        target_filter = st.selectbox("Filter by target", ["All"] + list(TARGET_OPTIONS), key="train_runs_filter")
    with filt2:
        granularity_filter = st.selectbox(
            "Filter by granularity", ["All"] + GRANULARITY_OPTIONS, key="train_runs_granularity_filter"
        )
    filtered = runs
    if target_filter != "All":
        filtered = [r for r in filtered if r["target"] == target_filter]
    if granularity_filter != "All":
        filtered = [r for r in filtered if r.get("granularity", "Minute") == granularity_filter]
    best_id = max(filtered, key=lambda r: r["metrics"]["r2"])["run_id"] if filtered else None
    for run in filtered:
        _imf_research_run_row(run, best_run_id=best_id, key_prefix="training_runs")


def render_imf_model_comparison_tab() -> None:
    st.caption(
        "Compare trained models for a single target side by side — R²/MAE/RMSE/MAPE/Bias, best "
        "model highlighted."
    )
    target_label = st.selectbox("Target", list(TARGET_OPTIONS), key="imf_compare_target")
    granularity_filter = st.selectbox("Granularity", ["All"] + GRANULARITY_OPTIONS, key="imf_compare_granularity")
    runs = list_runs(target_label)
    if granularity_filter != "All":
        runs = [r for r in runs if r.get("granularity", "Minute") == granularity_filter]
    if not runs:
        st.info(f"No trained models for {target_label} yet — train one in Training Runs.")
        return

    def _run_label(r):
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        return f"{r['model_type']} · {gran} +{hz} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    default_n = min(4, len(run_labels))
    selected = st.multiselect("Models to compare", run_labels, default=run_labels[:default_n], key="imf_compare_select")
    chosen = [runs[run_labels.index(s)] for s in selected]
    if not chosen:
        st.info("Select at least one model above.")
        return

    best_id = max(chosen, key=lambda r: r["metrics"]["r2"])["run_id"]

    rows = []
    for r in chosen:
        m = r["metrics"]
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        rows.append(
            {
                "Model": r["model_type"] + (" ⭐" if r["run_id"] == best_id else ""),
                "Granularity": gran,
                "Horizon": hz,
                "R²": round(m["r2"], 4),
                "MAE": round(m["mae"], 4),
                "RMSE": round(m["rmse"], 4),
                "MAPE (%)": round(m["mape"], 2) if m["mape"] is not None else None,
                "Bias": round(m["bias"], 4),
                "Trained": pd.Timestamp(r["trained_at"]).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Predicted vs. Actual / Residuals")
    detail_label = st.selectbox("Inspect model", selected, key="imf_compare_detail")
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
            title="Predicted vs. Actual (held-out test sample)", height=340, xaxis_title="Actual", yaxis_title="Predicted"
        )
        plot_retro(fig)
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=4), name="Residual"))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Residual Plot", height=340, yaxis_title="Predicted − Actual")
        plot_retro(fig2)

    if detail_run.get("feature_importance"):
        st.markdown("##### Feature Importance")
        fi_df = pd.DataFrame(detail_run["feature_importance"], columns=["Feature", "Importance"])
        fig3 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig3.update_layout(title="Top Contributing Features", height=420, yaxis=dict(autorange="reversed"))
        plot_retro(fig3)
    else:
        st.caption("Feature importance not available for this model type (e.g. LSTM/GRU).")

    if detail_run.get("loss_history"):
        st.markdown("##### Training / Validation Loss")
        lh = detail_run["loss_history"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig4.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig4.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig4)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _cached_imf_research_frame(granularity: str = DEFAULT_GRANULARITY):
    """Same redundant-read problem as the Kp lab's Physics Experiments tab
    (see _cached_kp_research_frame) — Feature Engineering and Physics
    Experiments both call load_research_frame() unconditionally on every
    script rerun (every 15s via auto_refresh(), and once per Kp/IMF
    Research Lab sub-tab per rerun regardless of which is visually
    active). Unlike the Kp lab's static historical CSV, "Minute"
    granularity genuinely reads the LIVE ~7-day minute buffer, which does
    update in near-real-time — so this keeps the same REFRESH_SECONDS TTL
    the rest of the dashboard's live data uses (see load_master_data)
    rather than caching indefinitely.
    """
    return load_research_frame(granularity)


def render_imf_feature_engineering_tab() -> None:
    st.caption(
        "The full minute-level feature set this research pipeline trains on — baseline IMF/Solar "
        "Wind variables plus the new physics-informed features below."
    )
    try:
        frame, feature_columns = _cached_imf_research_frame()
    except Exception as exc:
        st.warning(f"Could not load the research feature frame: {exc}")
        return

    latest = frame.iloc[-1]
    st.markdown("##### New Physics Features — Current Live Values")
    physics_cols_display = [
        ("southward_duration_min", "Southward Duration", "min", "Consecutive minutes Bz < 0"),
        ("strong_southward_duration_min", "Strong Southward Duration", "min", "Consecutive minutes Bz < -5 nT"),
        (
            "integrated_southward_bz_60m",
            "Integrated Southward Bz (60m)",
            "nT·min",
            "Rolling cumulative |Bz| (southward only)",
        ),
        ("dbz_gsm", "ΔBz (Magnetic Rotation)", "nT/min", "Minute-to-minute change in Bz"),
        ("clock_angle_deg", "IMF Clock Angle", "°", "atan2(By, Bz) — 180° = purely southward"),
        ("clock_angle_rate_deg", "Clock Angle Rate", "°/min", "How fast the field orientation is rotating"),
        ("bt_persistence_60m_std", "Bt Persistence (60m std)", "nT", "Field-strength volatility over the last hour"),
    ]
    cols = st.columns(4)
    for i, (col, label, unit, desc) in enumerate(physics_cols_display):
        with cols[i % 4]:
            val = latest.get(col)
            metric_card(label, "N/A" if pd.isna(val) else f"{val:.2f} {unit}", desc)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Full Feature List Used for Training")
    st.caption(f"{len(feature_columns)} total input features per model.")
    with st.expander("Show all feature names"):
        st.code("\n".join(feature_columns), language="text")


def render_imf_sequence_models_tab() -> None:
    """Review-only: LSTM/GRU are trained from the unified form in
    Training Runs (same target/granularity/horizon/model selector as
    every other architecture) — this tab exists to inspect what those
    runs actually did, especially the training/validation loss curves
    that only sequence models produce, without duplicating the trainer.
    """
    st.caption(
        "LSTM/GRU never see a single row — each prediction looks back over a window of consecutive "
        "minutes, letting the model learn temporal patterns in how the field evolves, not just its "
        "instantaneous state. Train new ones from the Training Runs tab (same unified form as every "
        "other model); this tab is for reviewing what they learned."
    )
    if not SEQUENCE_MODELS:
        st.warning("TensorFlow/Keras is not installed — sequence models are unavailable in this environment.")
        return

    seq_runs = [r for r in list_runs() if r["model_type"] in SEQUENCE_MODELS]
    if not seq_runs:
        st.info("No LSTM/GRU runs yet — train one from the Training Runs tab.")
        return

    def _run_label(r):
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        return f"{r['model_type']} · {r['target']} · {gran} +{hz} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in seq_runs]
    selected = st.selectbox("Inspect run", labels, key="imf_seq_inspect")
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
        # Keyed on run_id: without a stable key, two runs whose loss curves
        # happen to serialize identically (or the same run re-rendered
        # across the 15s auto-refresh rerun cycle) collide on Streamlit's
        # auto-generated element ID (element type + params only).
        plot_retro(fig, key=f"imf_seq_loss_{run['run_id']}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### All Sequence Model Runs")
    for r in seq_runs:
        _imf_research_run_row(r, key_prefix="sequence_models")


def render_imf_physics_experiments_tab() -> None:
    st.caption("Exploratory analysis: how do the new physics features relate to Bz's behavior historically?")
    try:
        frame, _ = _cached_imf_research_frame()
    except Exception as exc:
        st.warning(f"Could not load the research feature frame: {exc}")
        return

    feature_options = {
        "Southward Duration (min)": "southward_duration_min",
        "Integrated Southward Bz (60m)": "integrated_southward_bz_60m",
        "IMF Clock Angle (°)": "clock_angle_deg",
        "Clock Angle Rate (°/min)": "clock_angle_rate_deg",
        "Bt Persistence Std (60m)": "bt_persistence_60m_std",
    }
    label = st.selectbox("Physics feature", list(feature_options), key="imf_physics_feature_select")
    col = feature_options[label]

    recent = frame.tail(24 * 60).dropna(subset=[col, "bz_gsm"])
    if recent.empty:
        st.info("Not enough history to plot yet.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=recent.index, y=recent["bz_gsm"], name="Bz (nT)", line=dict(color="#1f5a7a")), secondary_y=False)
    fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=label, line=dict(color="#7a1f5a")), secondary_y=True)
    fig.update_layout(title=f"Bz vs. {label} — last {len(recent)} minutes", height=380)
    fig.update_yaxes(title_text="Bz (nT)", secondary_y=False)
    fig.update_yaxes(title_text=label, secondary_y=True)
    plot_retro(fig)

    next_bz = frame["bz_gsm"].shift(-1)
    valid = frame[[col]].join(next_bz.rename("next_bz")).dropna()
    if len(valid) > 10:
        corr = valid[col].corr(valid["next_bz"])
        st.caption(f"Correlation between {label} and next-minute Bz across the full history: **{corr:.3f}**")
        fig2 = go.Figure(go.Scattergl(x=valid[col], y=valid["next_bz"], mode="markers", marker=dict(size=3, opacity=0.4)))
        fig2.update_layout(title=f"{label} vs. Next-Minute Bz", height=380, xaxis_title=label, yaxis_title="Next-minute Bz (nT)")
        plot_retro(fig2)


def render_imf_hyperparameter_tuning_tab() -> None:
    st.caption(
        "Manually adjust and train with custom hyperparameters for the tabular models — not an "
        "automated search, a direct way to test a specific configuration and compare it against "
        "other runs in Model Comparison."
    )
    target_label = st.selectbox("Target Variable", list(TARGET_OPTIONS), key="imf_tune_target")
    granularity, horizon = _imf_research_granularity_horizon("tune")
    model_type = st.selectbox("Model", TABULAR_MODELS, key="imf_tune_model")
    hyperparams = _imf_research_hyperparam_inputs(model_type, "tune")

    if st.button("🧪 Train with these Hyperparameters", key="imf_tune_train_btn", type="primary"):
        with st.spinner(f"Training {model_type}..."):
            try:
                run = train_research_model(
                    target_label, model_type, granularity=granularity, horizon=horizon, hyperparams=hyperparams
                )
                st.success(f"Trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.3f}")
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def render_imf_hypothesis_testing_tab() -> None:
    st.caption(
        "Compare two training runs — e.g. a baseline architecture vs. an experimental one — and get "
        "a rule-based Supported/Not Supported/Inconclusive verdict. Adapted from the same philosophy "
        "as the main Research Lab's Hypothesis Testing, but for offline training-run metrics rather "
        "than live verified predictions."
    )
    runs = list_runs()
    if len(runs) < 2:
        st.info("Train at least two models (any target) to compare them here.")
        return

    def _run_label(r):
        gran = r.get("granularity", "Minute")
        hz = r.get("horizon_label") or f"{r.get('horizon', 1)}{'m' if gran == 'Minute' else 'h'}"
        return f"{r['model_type']} — {r['target']} · {gran} +{hz} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    col1, col2 = st.columns(2)
    with col1:
        baseline_label = st.selectbox("Baseline", run_labels, index=0, key="imf_hyp_baseline")
    with col2:
        experimental_label = st.selectbox(
            "Experimental", run_labels, index=min(1, len(run_labels) - 1), key="imf_hyp_experimental"
        )

    if st.button("Compare", key="imf_hyp_compare_btn"):
        baseline_run = runs[run_labels.index(baseline_label)]
        experimental_run = runs[run_labels.index(experimental_label)]
        if baseline_run["run_id"] == experimental_run["run_id"]:
            st.warning("Choose two different runs.")
            return
        result = compare_runs(baseline_run["run_id"], experimental_run["run_id"])
        st.markdown(
            f"### Verdict: <span style='color:{CONCLUSION_COLORS.get(result['verdict'], '#404040')}'>{result['verdict']}</span>",
            unsafe_allow_html=True,
        )
        st.write(result["explanation"])
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("ΔR²", f"{result['delta_r2']:+.4f}", "")
        with c2:
            metric_card("ΔMAE", f"{result['delta_mae']:+.4f}", "")
        with c3:
            metric_card("Samples", f"{experimental_run['n_test_samples']}", "held-out test rows")


def render_imf_horizon_analysis_tab() -> None:
    """Answers the core scientific question this whole redesign exists
    for: how does forecast skill decay with lead time? Trains (or reuses)
    one run per horizon in the chosen granularity's full horizon list and
    plots Horizon vs. R²/MAE/RMSE — the same sweep verified against
    production's own bz_gsm_*h models during this redesign (1h/3h/6h/
    12h/24h Hourly R² landed within ~0.01-0.02 of production's own
    metrics.json values, and decayed in the identical shape).
    """
    st.caption(
        "Automatically trains (or reuses existing runs for) one model per horizon and plots how "
        "R²/MAE/RMSE change with forecast lead time — the central research question this lab exists "
        "to answer."
    )
    target_label = st.selectbox("Target Variable", list(TARGET_OPTIONS), key="imf_horizon_target")
    granularity = st.radio("Forecast Granularity", GRANULARITY_OPTIONS, horizontal=True, key="imf_horizon_granularity")
    horizon_list = MINUTE_HORIZONS if granularity == "Minute" else HOURLY_HORIZONS
    st.caption(f"Horizons swept: {', '.join(str(h) for h in horizon_list)} {'minutes' if granularity == 'Minute' else 'hours'}")
    model_type = st.selectbox("Model", TABULAR_MODELS, key="imf_horizon_model")
    reuse_existing = st.checkbox(
        "Reuse existing runs where available (faster)", value=True, key="imf_horizon_reuse"
    )

    if st.button("📈 Run Horizon Sweep", key="imf_horizon_sweep_btn", type="primary"):
        with st.spinner(f"Sweeping {model_type} across {len(horizon_list)} horizons for {target_label} ({granularity})..."):
            try:
                runs = train_horizon_sweep(target_label, model_type, granularity=granularity, reuse_existing=reuse_existing)
                st.session_state["imf_horizon_sweep_result"] = {
                    "target": target_label,
                    "granularity": granularity,
                    "model_type": model_type,
                    "run_ids": [r["run_id"] for r in runs],
                }
                st.toast(f"Swept {len(runs)} horizons.")
            except Exception as exc:
                st.error(f"Horizon sweep failed: {exc}")

    result = st.session_state.get("imf_horizon_sweep_result")
    if not result:
        st.info("Pick a target, granularity, and model above, then run the sweep.")
        return

    runs = [get_run(rid) for rid in result["run_ids"]]
    runs = [r for r in runs if r is not None]
    if not runs:
        st.warning("Swept runs are no longer available (were they deleted?). Run the sweep again.")
        return
    runs = sorted(runs, key=lambda r: r.get("horizon", 1))

    unit = "min" if result["granularity"] == "Minute" else "h"
    horizon_labels = [f"{r.get('horizon', 1)}{unit}" for r in runs]
    r2_values = [r["metrics"]["r2"] for r in runs]
    mae_values = [r["metrics"]["mae"] for r in runs]
    rmse_values = [r["metrics"]["rmse"] for r in runs]

    st.markdown(
        f"##### {result['model_type']} — {result['target']} ({result['granularity']}) — skill decay with lead time"
    )
    fig_r2 = go.Figure(go.Scatter(x=horizon_labels, y=r2_values, mode="lines+markers", name="R²"))
    fig_r2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_r2.update_layout(title="Horizon vs. R²", height=340, xaxis_title="Forecast Horizon", yaxis_title="R²")
    plot_retro(fig_r2)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_mae = go.Figure(go.Scatter(x=horizon_labels, y=mae_values, mode="lines+markers", name="MAE"))
        fig_mae.update_layout(title="Horizon vs. MAE", height=320, xaxis_title="Forecast Horizon", yaxis_title="MAE")
        plot_retro(fig_mae)
    with col_b:
        fig_rmse = go.Figure(go.Scatter(x=horizon_labels, y=rmse_values, mode="lines+markers", name="RMSE"))
        fig_rmse.update_layout(title="Horizon vs. RMSE", height=320, xaxis_title="Forecast Horizon", yaxis_title="RMSE")
        plot_retro(fig_rmse)

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


def _render_automl_section() -> None:
    """AutoML orchestration layer — automatically runs Experiments 1-10
    end-to-end (the manual Exp 1-8 tabs below stay untouched; this only
    sequences the same underlying functions those tabs call). Promotion
    is never automatic — the study always ends with a leaderboard and a
    Promote/Keep recommendation for a human to act on.
    """
    st.markdown("### 🤖 Automated Optimization (AutoML)")
    st.caption(
        "Runs the entire pipeline below — baseline, persistence benchmark, structured feature "
        "search, full model sweep, feature importance, SHAP, leaderboard, and a production "
        "recommendation — with a single click and no manual tab-clicking. Every model trained "
        "here lands in the same registry as the manual experiments and can be inspected there. "
        "**Promotion always requires your explicit confirmation** at the end."
    )

    if st.button("🚀 Run Complete Optimization Study", key="automl_run_study", type="primary"):
        status_box = st.status("Running complete Bz optimization study…", expanded=True)

        def _cb(step, total, msg):
            status_box.update(label=f"Step {step}/{total} — {msg}")
            status_box.write(f"**Step {step}/{total}:** {msg}")

        try:
            study = run_complete_optimization_study(progress_cb=_cb)
            status_box.update(label="Optimization study complete.", state="complete", expanded=False)
            st.session_state["automl_last_study_id"] = study["study_id"]
            st.success(
                f"Study complete — winner: **{study['winner_model_type']}** "
                f"(R²={study['winner_metrics']['r2']:.4f}) — recommendation: **{study['recommendation']}**"
            )
            st.rerun()
        except Exception as exc:
            status_box.update(label="Optimization study failed.", state="error")
            st.error(f"Study failed: {exc}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Study History")
    studies = list_studies()
    if not studies:
        st.info("No optimization studies run yet — click the button above to run the first one.")
        return

    def _study_label(s):
        rec_icon = "✅" if s["recommendation"] == "Promote" else "⏸"
        status_icon = {"pending": "🕓", "promoted": "🚀", "rejected": "❌"}.get(s.get("promotion_status", "pending"), "")
        return (
            f"{rec_icon} {pd.Timestamp(s['started_at']).strftime('%Y-%m-%d %H:%M UTC')} · "
            f"Winner: {s['winner_model_type']} (R²={s['winner_metrics']['r2']:.4f}) · "
            f"{status_icon} {s.get('promotion_status', 'pending')}"
        )

    study_labels = [_study_label(s) for s in studies]
    default_idx = 0
    last_id = st.session_state.get("automl_last_study_id")
    if last_id:
        for i, s in enumerate(studies):
            if s["study_id"] == last_id:
                default_idx = i
                break
    chosen_label = st.selectbox("Select a study to inspect", study_labels, index=default_idx, key="automl_study_select")
    study = studies[study_labels.index(chosen_label)]
    _render_study_detail(study)


def _render_study_detail(study: dict) -> None:
    st.markdown(
        f"**Study ID:** `{study['study_id'][:8]}…`  ·  "
        f"**Started:** {pd.Timestamp(study['started_at']).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
        f"**Completed:** {pd.Timestamp(study['completed_at']).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    step_cols = st.columns(5)
    step_cols[0].metric("1. Baseline R²", f"{study['baseline_metrics']['r2']:.4f}")
    step_cols[1].metric("2. Persistence R²", f"{study['persistence_metrics']['r2']:.4f}")
    step_cols[2].metric(
        "3-5. Best Feature Set", study["best_feature_combo"]["name"],
        help=f"R²={study['best_feature_combo']['r2']:.4f}",
    )
    step_cols[3].metric("6. Winner", study["winner_model_type"], help=f"R²={study['winner_metrics']['r2']:.4f}")
    step_cols[4].metric("10. Recommendation", study["recommendation"])

    detail_tabs = st.tabs(
        ["Feature Search", "Leaderboard", "Feature Importance", "SHAP", "Production Comparison", "Promotion"]
    )

    with detail_tabs[0]:
        st.markdown("**Experiments 3-5 — Structured Feature Search Results**")
        fs_rows = [
            {
                "Combination": r["name"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4),
                "RMSE": round(r["rmse"], 4), "Features": r["feature_count"],
                "Winner": "⭐" if r["name"] == study["best_feature_combo"]["name"] else "",
            }
            for r in study["feature_search_results"]
        ]
        st.dataframe(pd.DataFrame(fs_rows), use_container_width=True, hide_index=True)

    with detail_tabs[1]:
        st.markdown("**Experiment 6 — Model Search Leaderboard**")
        lb_rows = [
            {
                "Rank": r["rank"], "Model": r["model_type"], "Feature Set": r["feature_set_name"],
                "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4), "RMSE": round(r["rmse"], 4),
                "MAPE (%)": round(r["mape"], 2) if r["mape"] is not None else None,
                "Bias": round(r["bias"], 4),
                "Train Time (s)": round(r["training_time_sec"], 3) if r["training_time_sec"] is not None else None,
                "Inference (ms/sample)": (
                    round(r["inference_time_ms_per_sample"], 4) if r["inference_time_ms_per_sample"] is not None else None
                ),
                "Model Size (KB)": round(r["model_size_kb"], 1) if r["model_size_kb"] is not None else None,
                "Features": r["feature_count"],
            }
            for r in study["leaderboard"]
        ]
        st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)
        if study.get("failed_candidates"):
            with st.expander(f"⚠️ {len(study['failed_candidates'])} candidate(s) failed to train"):
                for f in study["failed_candidates"]:
                    st.caption(f"**{f['model_type']}**: {f['error']}")

    with detail_tabs[2]:
        st.markdown("**Experiment 7 — Feature Importance (winning / best-available candidate)**")
        if study.get("feature_importance"):
            fi_df = pd.DataFrame(study["feature_importance"], columns=["Feature", "Importance"]).head(25)
            fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h", marker_color="steelblue"))
            fig.update_layout(title="Top 25 Features by Importance", height=520, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
        else:
            st.info("Feature importance not available for any candidate in this study.")

    with detail_tabs[3]:
        st.markdown("**Experiment 8 — SHAP Analysis**")
        shap_result = study.get("shap_result") or {}
        if shap_result.get("supported"):
            shap_df = pd.DataFrame(shap_result["shap_importance"], columns=["Feature", "Mean |SHAP|"]).head(25)
            fig = go.Figure(go.Bar(x=shap_df["Mean |SHAP|"], y=shap_df["Feature"], orientation="h", marker_color="indianred"))
            fig.update_layout(title="Top 25 Features by Mean |SHAP Value|", height=520, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
        else:
            st.info(shap_result.get("skipped_reason", "SHAP analysis not available for this study."))

    with detail_tabs[4]:
        st.markdown("**Production Comparison**")
        comp = study["production_comparison"]
        if comp["current"] is None:
            st.warning("No production Bz model currently on disk to compare against.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Current Production**")
                st.metric("Algorithm", comp["current"]["algorithm"])
                st.metric("R²", f"{comp['current']['r2']:.4f}")
                st.metric("MAE", f"{comp['current']['mae']:.4f} nT")
                st.metric("RMSE", f"{comp['current']['rmse']:.4f} nT")
            with c2:
                st.markdown("**Candidate**")
                st.metric("Algorithm", comp["candidate"]["algorithm"])
                st.metric("R²", f"{comp['candidate']['r2']:.4f}", delta=f"{comp['delta_r2']:+.4f}")
                st.metric("MAE", f"{comp['candidate']['mae']:.4f} nT", delta=f"{comp['delta_mae']:+.4f}", delta_color="inverse")
                st.metric("RMSE", f"{comp['candidate']['rmse']:.4f} nT", delta=f"{comp['delta_rmse']:+.4f}", delta_color="inverse")
            with c3:
                st.markdown("**Verdict**")
                if study["recommendation"] == "Promote":
                    st.success("✅ Promote")
                else:
                    st.warning("⏸ Keep Current Production")

    with detail_tabs[5]:
        st.markdown("**Promotion Criteria Checklist**")
        for item in study["promotion_check"]["checklist"]:
            icon = "✅" if item["passed"] else "❌"
            st.markdown(f"{icon} **{item['criterion']}** — {item['detail']}")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        status = study.get("promotion_status", "pending")
        if status == "promoted":
            st.success(
                f"This study's winning candidate was already promoted to production at "
                f"{pd.Timestamp(study['promoted_at']).strftime('%Y-%m-%d %H:%M UTC')}."
            )
        elif status == "rejected":
            rejected_at = study.get("rejected_at")
            st.info(
                "This study was reviewed and Kept Current Production"
                + (f" at {pd.Timestamp(rejected_at).strftime('%Y-%m-%d %H:%M UTC')}." if rejected_at else ".")
            )
        else:
            eligible = study["promotion_check"]["eligible"]
            notes = st.text_area("Promotion notes (optional)", key=f"automl_promo_notes_{study['study_id']}")
            pc1, pc2 = st.columns(2)
            with pc1:
                confirm = st.checkbox(
                    "I understand this will overwrite the production model (a rollback archive will be created)",
                    key=f"automl_promo_confirm_{study['study_id']}",
                    disabled=not eligible,
                )
                if st.button(
                    "🚀 Promote Winning Candidate to Production", key=f"automl_promote_{study['study_id']}",
                    type="primary", disabled=not (eligible and confirm),
                ):
                    with st.spinner("Promoting to production…"):
                        try:
                            result = promote_to_production(study["winner_run_id"], notes=notes)
                            mark_study_promoted(study["study_id"], study["winner_run_id"])
                            st.success(f"Promoted! Archive: `{result['archive_path']}`")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Promotion failed: {exc}")
                if not eligible:
                    st.caption("Promotion disabled — one or more criteria above failed.")
            with pc2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("⏸ Keep Current Production", key=f"automl_reject_{study['study_id']}"):
                    mark_study_rejected(study["study_id"], notes=notes)
                    st.rerun()


def render_imf_bz_optimization_study() -> None:
    """8-experiment structured study to find the best production Bz model.

    Experiments:
      1  Baseline — reproduce current production model (Hourly, Linear Reg, 1h)
      2  Persistence Benchmark — naïve lower bound
      3  Solar Wind Inputs — IMF Only → +Speed → +Speed+Density → +All SW
      4  Short-term Dynamics — add rolling min/max, slope, acceleration
      5  Physics Variables — Ey, VBz, DynPressure, ClockAngle, SouthwardHours
      6  ML Model Sweep — all 12 model types on best feature set
      7  Feature Importance — tree-based feature ranking
      8  Promote — archive production, install best model, update metrics.json
    """
    st.info(
        "**Bz Production Model Optimization Study** — 8 structured experiments to answer: "
        "can the current production Bz 1h model (R²≈0.50, Linear Regression) be improved, "
        "and if so, what features/model should replace it?  All experiments use Hourly "
        "granularity and 1h horizon so results are directly comparable to production."
    )

    _render_automl_section()
    st.markdown("---")
    st.markdown(
        "### 🔬 Manual Experiments — Exp 1 through Exp 8\n"
        "Run any experiment individually below for hands-on investigation. This is exactly what "
        "**Run Complete Optimization Study** above orchestrates automatically — nothing here changes."
    )

    exp_tabs = st.tabs([
        "Exp 1 · Baseline",
        "Exp 2 · Persistence",
        "Exp 3 · Solar Wind Inputs",
        "Exp 4 · Dynamics Features",
        "Exp 5 · Physics Variables",
        "Exp 6 · Model Sweep",
        "Exp 7 · Feature Importance",
        "Exp 8 · Promote to Production",
    ])

    # ── Exp 1 · Baseline ────────────────────────────────────────────────────
    with exp_tabs[0]:
        st.markdown("##### Experiment 1 — Reproduce Production Baseline")
        st.caption(
            "Train Linear Regression, Hourly, 1h horizon, IMF + All Solar Wind features — "
            "the exact configuration of the current production bz_gsm_1h.joblib.  "
            "If this reproduces R²≈0.50, the study is calibrated correctly."
        )
        if st.button("▶ Run Baseline", key="opt_exp1_run", type="primary"):
            with st.spinner("Training baseline (Linear Regression · Hourly · 1h)…"):
                try:
                    run = train_research_model_exp(
                        "Bz", "Linear Regression",
                        granularity="Hourly", horizon=1,
                        feature_set="IMF + All Solar Wind",
                        add_dynamics=False, add_physics=False,
                        experiment_tag="exp1_baseline",
                    )
                    st.success(f"Baseline trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} nT")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        baseline_runs = [r for r in list_runs("Bz") if r.get("experiment_tag") == "exp1_baseline"]
        if baseline_runs:
            st.markdown("**Baseline Runs**")
            rows = []
            for r in baseline_runs:
                m = r["metrics"]
                rows.append({"Model": r["model_type"], "R²": round(m["r2"], 4), "MAE": round(m["mae"], 4),
                              "RMSE": round(m["rmse"], 4), "Bias": round(m["bias"], 4),
                              "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No baseline runs yet — click Run Baseline above.")

    # ── Exp 2 · Persistence Benchmark ───────────────────────────────────────
    with exp_tabs[1]:
        st.markdown("##### Experiment 2 — Persistence Benchmark")
        st.caption(
            "Naïve forecast: next Bz = current Bz.  Any real model that cannot beat "
            "this provides no value.  Stored permanently in the registry."
        )
        bench_run = next((r for r in list_runs("Bz") if r.get("experiment_tag") == "persistence_benchmark"), None)
        if st.button("▶ Compute Persistence Benchmark", key="opt_exp2_run", type="primary"):
            with st.spinner("Computing persistence benchmark…"):
                try:
                    bench_run = compute_persistence_benchmark("Bz", horizon=1, granularity="Hourly")
                    st.success(f"Persistence baseline — R²={bench_run['metrics']['r2']:.4f}, MAE={bench_run['metrics']['mae']:.4f} nT")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")
        if bench_run:
            m = bench_run["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{m['r2']:.4f}")
            c2.metric("MAE (nT)", f"{m['mae']:.4f}")
            c3.metric("RMSE (nT)", f"{m['rmse']:.4f}")
            c4.metric("Bias (nT)", f"{m['bias']:.4f}")
            st.caption(f"Test samples: {bench_run['n_test_samples']:,}  ·  Computed: {pd.Timestamp(bench_run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            st.info("Click above to compute and store the persistence benchmark.")

    # ── Exp 3 · Solar Wind Inputs ────────────────────────────────────────────
    with exp_tabs[2]:
        st.markdown("##### Experiment 3 — Solar Wind Input Investigation")
        st.caption(
            "Train Linear Regression four times — adding one solar-wind group at a time — "
            "to quantify how much each adds over IMF-only features."
        )
        col_fs, col_model3 = st.columns(2)
        with col_fs:
            fs3 = st.selectbox("Feature Set", list(FEATURE_SET_OPTIONS), key="opt_exp3_fs",
                               help="Choose which feature group to add next")
        with col_model3:
            model3 = st.selectbox("Model", TABULAR_MODELS, key="opt_exp3_model")
        if st.button("▶ Train This Feature Set", key="opt_exp3_run", type="primary"):
            with st.spinner(f"Training {model3} with feature set '{fs3}'…"):
                try:
                    run = train_research_model_exp(
                        "Bz", model3, granularity="Hourly", horizon=1,
                        feature_set=fs3, add_dynamics=False, add_physics=False,
                        experiment_tag="exp3_solar_wind_inputs",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} nT  ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        exp3_runs = [r for r in list_runs("Bz") if r.get("experiment_tag") == "exp3_solar_wind_inputs"]
        if exp3_runs:
            rows3 = []
            for r in exp3_runs:
                m = r["metrics"]
                rows3.append({"Feature Set": r.get("feature_set", "—"), "Model": r["model_type"],
                               "R²": round(m["r2"], 4), "MAE": round(m["mae"], 4), "RMSE": round(m["rmse"], 4),
                               "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")})
            st.dataframe(pd.DataFrame(rows3), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 3 runs yet.")

    # ── Exp 4 · Short-term Dynamics ──────────────────────────────────────────
    with exp_tabs[3]:
        st.markdown("##### Experiment 4 — Short-term Dynamics Features")
        st.caption(
            "Adds 3h/6h/12h rolling min, max, std; linear slope over last 6h; and second-difference "
            "acceleration for each base column.  Compare to the best Exp-3 feature set to isolate "
            "what the dynamics add."
        )
        col4a, col4b = st.columns(2)
        with col4a:
            fs4 = st.selectbox("Base Feature Set", list(FEATURE_SET_OPTIONS), key="opt_exp4_fs",
                                index=list(FEATURE_SET_OPTIONS).index(DEFAULT_FEATURE_SET))
        with col4b:
            model4 = st.selectbox("Model", TABULAR_MODELS, key="opt_exp4_model")
        if st.button("▶ Train with Dynamics Features", key="opt_exp4_run", type="primary"):
            with st.spinner("Training with short-term dynamics…"):
                try:
                    run = train_research_model_exp(
                        "Bz", model4, granularity="Hourly", horizon=1,
                        feature_set=fs4, add_dynamics=True, add_physics=False,
                        experiment_tag="exp4_dynamics",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} nT  ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        exp4_runs = [r for r in list_runs("Bz") if r.get("experiment_tag") == "exp4_dynamics"]
        if exp4_runs:
            rows4 = []
            for r in exp4_runs:
                m = r["metrics"]
                rows4.append({"Feature Set": r.get("feature_set", "—"), "Model": r["model_type"],
                               "R²": round(m["r2"], 4), "MAE": round(m["mae"], 4), "RMSE": round(m["rmse"], 4),
                               "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")})
            st.dataframe(pd.DataFrame(rows4), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 4 runs yet.")

    # ── Exp 5 · Physics Variables ─────────────────────────────────────────────
    with exp_tabs[4]:
        st.markdown("##### Experiment 5 — Physics Variables")
        st.caption(
            "Adds hourly-computable physics features: Ey, VBz, Dynamic Pressure, Clock Angle, "
            "Clock Angle Rate, Southward Hours (24h), Strong Southward Hours (24h), "
            "Integrated Southward Bz (24h).  Requires 'IMF + All Solar Wind' to unlock Ey/VBz/DynPressure."
        )
        col5a, col5b = st.columns(2)
        with col5a:
            fs5 = st.selectbox("Base Feature Set", list(FEATURE_SET_OPTIONS), key="opt_exp5_fs",
                                index=list(FEATURE_SET_OPTIONS).index(DEFAULT_FEATURE_SET))
        with col5b:
            model5 = st.selectbox("Model", TABULAR_MODELS, key="opt_exp5_model")
        add_dyn5 = st.checkbox("Also add Dynamics features (Exp 4)", key="opt_exp5_dyn")
        if st.button("▶ Train with Physics Variables", key="opt_exp5_run", type="primary"):
            with st.spinner("Training with physics features…"):
                try:
                    run = train_research_model_exp(
                        "Bz", model5, granularity="Hourly", horizon=1,
                        feature_set=fs5, add_dynamics=add_dyn5, add_physics=True,
                        experiment_tag="exp5_physics",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} nT  ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        exp5_runs = [r for r in list_runs("Bz") if r.get("experiment_tag") == "exp5_physics"]
        if exp5_runs:
            rows5 = []
            for r in exp5_runs:
                m = r["metrics"]
                rows5.append({"Feature Set": r.get("feature_set", "—"), "Model": r["model_type"],
                               "+Dynamics": r.get("add_dynamics", False), "+Physics": r.get("add_physics", False),
                               "R²": round(m["r2"], 4), "MAE": round(m["mae"], 4),
                               "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")})
            st.dataframe(pd.DataFrame(rows5), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 5 runs yet.")

    # ── Exp 6 · ML Model Sweep ────────────────────────────────────────────────
    with exp_tabs[5]:
        st.markdown("##### Experiment 6 — ML Model Comparison Sweep")
        st.caption(
            "Run all available model types with the same feature set.  Use the best configuration "
            "found in Exp 3–5 to give each model an equal footing."
        )
        col6a, col6b = st.columns(2)
        with col6a:
            fs6 = st.selectbox("Feature Set", list(FEATURE_SET_OPTIONS), key="opt_exp6_fs",
                                index=list(FEATURE_SET_OPTIONS).index(DEFAULT_FEATURE_SET))
        with col6b:
            add_dyn6 = st.checkbox("Add Dynamics", key="opt_exp6_dyn")
        add_phys6 = st.checkbox("Add Physics", key="opt_exp6_phys")

        model6 = st.selectbox("Model to train", TABULAR_MODELS, key="opt_exp6_model")
        if st.button(f"▶ Train {model6}", key="opt_exp6_run", type="primary"):
            with st.spinner(f"Training {model6}…"):
                try:
                    run = train_research_model_exp(
                        "Bz", model6, granularity="Hourly", horizon=1,
                        feature_set=fs6, add_dynamics=add_dyn6, add_physics=add_phys6,
                        experiment_tag="exp6_model_sweep",
                    )
                    st.success(f"{model6} — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} nT")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        exp6_runs = [r for r in list_runs("Bz") if r.get("experiment_tag") == "exp6_model_sweep"]
        if exp6_runs:
            rows6 = []
            best6_r2 = max(r["metrics"]["r2"] for r in exp6_runs)
            for r in exp6_runs:
                m = r["metrics"]
                label = r["model_type"] + (" ⭐" if abs(m["r2"] - best6_r2) < 1e-6 else "")
                rows6.append({"Model": label, "Feature Set": r.get("feature_set", "—"),
                               "+Dyn": r.get("add_dynamics", False), "+Phys": r.get("add_physics", False),
                               "R²": round(m["r2"], 4), "MAE": round(m["mae"], 4),
                               "RMSE": round(m["rmse"], 4), "Bias": round(m["bias"], 4),
                               "Features": len(r["feature_columns"]),
                               "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")})
            rows6.sort(key=lambda x: -x["R²"])
            st.dataframe(pd.DataFrame(rows6), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 6 runs yet.")

    # ── Exp 7 · Feature Importance ────────────────────────────────────────────
    with exp_tabs[6]:
        st.markdown("##### Experiment 7 — Feature Importance")
        st.caption(
            "Select any tree-based run from the study to inspect feature importance.  "
            "Top features across Exps 3–6 answer 'what should we keep, what can we drop'."
        )
        study_runs = [r for r in list_runs("Bz")
                      if r.get("experiment_tag", "").startswith("exp") and r.get("feature_importance")]
        if not study_runs:
            st.info("No tree-based study runs with feature importance yet.  Train Random Forest or XGBoost in Exps 3–6.")
        else:
            def _run_lbl(r):
                tag = r.get("experiment_tag", "")
                return f"{tag} · {r['model_type']} · R²={r['metrics']['r2']:.4f} · {pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')}"
            run_options = {_run_lbl(r): r for r in study_runs}
            chosen_lbl = st.selectbox("Run", list(run_options), key="opt_exp7_run")
            chosen_run = run_options[chosen_lbl]
            fi_pairs = chosen_run["feature_importance"]
            fi_df = pd.DataFrame(fi_pairs, columns=["Feature", "Importance"]).head(25)
            fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                                   marker_color="steelblue"))
            fig.update_layout(title="Top 25 Features by Importance", height=560,
                              yaxis=dict(autorange="reversed"), xaxis_title="Importance")
            plot_retro(fig)
            st.markdown("**Full feature importance table**")
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

    # ── Exp 8 · Promote to Production ────────────────────────────────────────
    with exp_tabs[7]:
        st.markdown("##### Experiment 8 — Promote Best Model to Production")
        st.caption(
            "Archives the current production bz_gsm_1h.joblib, installs the selected research "
            "model, and updates models/imf/metrics.json.  "
            "**Irreversible without the archive** — read the steps below before promoting."
        )
        st.markdown("""
**What happens when you promote:**
1. Current `models/imf/bz_gsm_1h.joblib` → `models/imf/archive/bz_gsm_1h_<timestamp>.joblib`
2. Research model `.joblib` → `models/imf/bz_gsm_1h.joblib`
3. `models/imf/metrics.json["bz_gsm_1h"]` updated with new R²/MAE/feature_columns
4. Run marked `promoted_to_production=True` in registry
5. **Next prediction job** picks up the new model automatically (no restart needed)

**To roll back:** copy the archived `.joblib` back to `models/imf/bz_gsm_1h.joblib` and restore metrics.json.
        """)

        # Only show Hourly, Bz, 1h, non-sequence runs
        promotable = [
            r for r in list_runs("Bz")
            if r.get("granularity") == "Hourly"
            and r.get("horizon") == 1
            and r.get("model_type") not in SEQUENCE_MODELS
            and r.get("model_path") is not None
        ]
        if not promotable:
            st.info("No promotable runs yet — train a Hourly · Bz · 1h model in any Exp above.")
        else:
            def _promo_lbl(r):
                tag = r.get("experiment_tag", "")
                return (f"{tag} · {r['model_type']} · R²={r['metrics']['r2']:.4f} · "
                        f"MAE={r['metrics']['mae']:.4f} · {pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M UTC')}")
            promo_options = {_promo_lbl(r): r for r in sorted(promotable, key=lambda r: -r["metrics"]["r2"])}
            chosen_promo_lbl = st.selectbox("Select run to promote", list(promo_options), key="opt_exp8_run")
            chosen_promo = promo_options[chosen_promo_lbl]

            m_p = chosen_promo["metrics"]
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("R²", f"{m_p['r2']:.4f}")
            pc2.metric("MAE", f"{m_p['mae']:.4f} nT")
            pc3.metric("Features", len(chosen_promo["feature_columns"]))

            promo_notes = st.text_area("Promotion notes (optional)", key="opt_exp8_notes",
                                       placeholder="e.g. Exp 6 winner — XGBoost + All SW + Physics, R²=0.62")

            already_promoted = chosen_promo.get("promoted_to_production", False)
            if already_promoted:
                st.warning("This run has already been promoted to production.")

            confirm = st.checkbox("I understand this will overwrite the production model (a rollback archive will be created)", key="opt_exp8_confirm")
            if st.button("🚀 Promote to Production", key="opt_exp8_promote", type="primary", disabled=not confirm):
                with st.spinner("Promoting to production…"):
                    try:
                        result = promote_to_production(chosen_promo["run_id"], notes=promo_notes)
                        st.success(f"Promoted successfully!  Archive: `{result['archive_path']}`")
                        if result.get("old_metrics"):
                            old = result["old_metrics"]
                            new = result["new_metrics"]
                            delta_r2 = new["r2"] - old.get("r2", 0)
                            delta_mae = new["mae"] - old.get("mae", 0)
                            d1, d2 = st.columns(2)
                            d1.metric("R² change", f"{new['r2']:.4f}", delta=f"{delta_r2:+.4f}")
                            d2.metric("MAE change", f"{new['mae']:.4f} nT", delta=f"{delta_mae:+.4f} nT",
                                      delta_color="inverse")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Promotion failed: {exc}")

        # Study Summary — all study runs ranked
        st.markdown("---")
        st.markdown("##### All Study Runs — Ranked by R²")
        all_study = [r for r in list_runs("Bz")
                     if r.get("experiment_tag", "").startswith("exp") or r.get("experiment_tag") == "persistence_benchmark"]
        # Also include persistence benchmark
        bench = next((r for r in list_runs("Bz") if r.get("experiment_tag") == "persistence_benchmark"), None)
        if bench and bench not in all_study:
            all_study.append(bench)
        if all_study:
            rows8 = []
            best8_r2 = max(r["metrics"]["r2"] for r in all_study)
            for r in sorted(all_study, key=lambda x: -x["metrics"]["r2"]):
                m = r["metrics"]
                rows8.append({
                    "Experiment": r.get("experiment_tag", "—"),
                    "Model": r["model_type"] + (" ⭐" if abs(m["r2"] - best8_r2) < 1e-6 else ""),
                    "Feature Set": r.get("feature_set", "—"),
                    "+Dyn": r.get("add_dynamics", "—"),
                    "+Phys": r.get("add_physics", "—"),
                    "R²": round(m["r2"], 4),
                    "MAE (nT)": round(m["mae"], 4),
                    "RMSE (nT)": round(m["rmse"], 4),
                    "Bias (nT)": round(m["bias"], 4),
                    "Features": len(r.get("feature_columns", [])),
                    "Promoted": "✅" if r.get("promoted_to_production") else "",
                })
            st.dataframe(pd.DataFrame(rows8), use_container_width=True, hide_index=True)
        else:
            st.info("Run experiments above to populate this summary.")


def render_imf_research_laboratory() -> None:
    hdr_col, pause_col = st.columns([3, 1])
    with hdr_col:
        st.markdown("### 🧪 IMF Research Laboratory")
        st.caption(
            "A research environment for comparing Bz/Bt/Bx/By forecasting architectures — fully isolated "
            "from the Production Prediction tab, which continues using the current trained production "
            "model exactly as-is. Models trained here never overwrite it."
        )
    with pause_col:
        st.session_state.setdefault("pause_autorefresh", False)
        if st.toggle("⏸ Pause Live Refresh", value=st.session_state["pause_autorefresh"], key="imf_lab_pause_toggle"):
            st.session_state["pause_autorefresh"] = True
        else:
            st.session_state["pause_autorefresh"] = False
    _imf_research_notes()

    sub = st.tabs(
        [
            "🔬 Bz Optimization Study",
            "Training Runs",
            "Horizon Analysis",
            "Model Comparison",
            "Feature Engineering",
            "Physics Experiments",
            "Sequence Models",
            "Hyperparameter Tuning",
            "Hypothesis Testing",
        ]
    )
    with sub[0]:
        render_imf_bz_optimization_study()
    with sub[1]:
        render_imf_training_runs_tab()
    with sub[2]:
        render_imf_horizon_analysis_tab()
    with sub[3]:
        render_imf_model_comparison_tab()
    with sub[4]:
        render_imf_feature_engineering_tab()
    with sub[5]:
        render_imf_physics_experiments_tab()
    with sub[6]:
        render_imf_sequence_models_tab()
    with sub[7]:
        render_imf_hyperparameter_tuning_tab()
    with sub[8]:
        render_imf_hypothesis_testing_tab()
