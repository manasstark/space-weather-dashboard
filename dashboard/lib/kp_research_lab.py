"""Kp Research Laboratory — extracted verbatim from dashboard/home.py.
See dashboard/home.py's Analytics page for where
render_kp_research_laboratory() is called.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.lib.shared_ui import REFRESH_SECONDS, metric_card, plot_retro
from swdss.models import ae_research, kp_research
from swdss.models.registry import VARIABLE_LABELS

# ==================== Kp Research Laboratory ====================
# Fully isolated from the Production Prediction tab — see
# swdss.models.kp_research module docstring for the production-safety
# contract. Nothing below ever writes to models/analytics/ or its
# metrics.json. Answers "why does one model perform better than another?"
# and "which physics actually improves Kp prediction?" — never intended
# to replace the operational forecast.


def _kp_research_notes() -> None:
    st.info(
        "**What this lab is for.** Production answers *what is the operational forecast* — this lab "
        "answers *why* one model or feature set performs better than another. Every run trains "
        "against the exact same target (NOAA's next official 3-hour Kp interval) and the same "
        "3-year historical dataset production itself trains on, so results here are genuinely "
        "comparable to production's own R²≈0.68 — not a different, easier problem."
    )


def _kp_research_hyperparam_inputs(model_type: str, key_prefix: str) -> dict:
    schema = kp_research.HYPERPARAM_SCHEMA.get(model_type, {})
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


def _kp_research_model_selector(key_prefix: str) -> str:
    options = kp_research.ALL_TRAINABLE_MODELS + kp_research.FUTURE_MODELS

    def _fmt(name):
        return f"{name} (coming soon)" if name in kp_research.FUTURE_MODELS else name

    choice = st.selectbox("Model Architecture", options, format_func=_fmt, key=f"{key_prefix}_model_select")
    if choice in kp_research.FUTURE_MODELS:
        st.warning(f"{choice} is a registered placeholder for future work — not trainable yet.")
    return choice


def _kp_research_feature_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or kp_research.default_feature_toggles()
    toggles = {}
    cols = st.columns(len(kp_research.FEATURE_GROUP_COLUMNS))
    for i, (group, group_cols) in enumerate(kp_research.FEATURE_GROUP_COLUMNS.items()):
        with cols[i]:
            st.markdown(f"**{group}**")
            toggles[group] = {}
            for col in group_cols:
                label = VARIABLE_LABELS.get(col, col.replace("_", " ").title())
                toggles[group][col] = st.checkbox(
                    label, value=defaults.get(group, {}).get(col, True), key=f"{key_prefix}_feat_{group}_{col}"
                )
    return toggles


def _kp_research_engineered_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or kp_research.default_engineered_toggles()
    toggles = {}
    cols = st.columns(len(kp_research.ENGINEERED_GROUPS))
    for i, group in enumerate(kp_research.ENGINEERED_GROUPS):
        with cols[i]:
            toggles[group] = st.checkbox(group, value=defaults.get(group, True), key=f"{key_prefix}_eng_{group}")
    return toggles


def _kp_research_physics_toggle_form(key_prefix: str, defaults: dict = None) -> dict:
    defaults = defaults or {}
    toggles = {}
    cols = st.columns(3)
    for i, name in enumerate(kp_research.PHYSICS_FEATURE_OPTIONS):
        with cols[i % 3]:
            toggles[name] = st.checkbox(name, value=defaults.get(name, False), key=f"{key_prefix}_phys_{name}")
    return toggles


def _kp_research_run_row(run: dict, best_run_id: str = None, key_prefix: str = "kp_runs") -> None:
    """key_prefix disambiguates widget keys when the same run is rendered
    from more than one tab in the same script run — see the identical
    IMF lab pattern (_imf_research_run_row) this was copied from.
    """
    m = run["metrics"]
    is_best = run["run_id"] == best_run_id
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.22, 0.11, 0.11, 0.11, 0.11, 0.11, 0.23])
        with c1:
            star = "⭐ " if is_best else ""
            promoted_tag = " 🚀" if run.get("promoted") else ""
            st.markdown(f"**{star}{run['model_type']}**{promoted_tag}")
            seq_note = f" · seq={run['sequence_length']}h" if run.get("sequence_length") else ""
            st.caption(f"Kp{seq_note} · {pd.Timestamp(run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}")
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
                    kp_research.promote_run(run["run_id"])
                    st.toast("Marked as promoted (label only — production untouched).")
                    st.rerun()
            with b2:
                if st.button(
                    "Load",
                    key=f"{key_prefix}_load_{run['run_id']}",
                    use_container_width=True,
                    disabled=run["model_type"] in kp_research.SEQUENCE_MODELS,
                ):
                    try:
                        model = kp_research.load_trained_model(run["run_id"])
                        st.toast(f"Loaded {run['model_type']} model ({type(model).__name__}) into memory.")
                    except Exception as exc:
                        st.error(f"Load failed: {exc}")
            with b3:
                if st.button("Delete", key=f"{key_prefix}_delete_{run['run_id']}", use_container_width=True):
                    kp_research.delete_run(run["run_id"])
                    st.toast("Run deleted.")
                    st.rerun()


def render_kp_model_comparison_tab() -> None:
    st.caption(
        "Train, evaluate, and compare Kp models — every run auto-saves to its own registry (never "
        "overwrites production) and can be reloaded via each run card's Load button. Target is "
        "always NOAA's next official 3-hour Kp interval, identical to production's own definition."
    )
    model_type = _kp_research_model_selector("kp_compare")
    with st.expander("Feature Groups", expanded=True):
        feature_toggles = _kp_research_feature_toggle_form("kp_compare")
    with st.expander("Engineered Features", expanded=False):
        engineered_groups = _kp_research_engineered_toggle_form("kp_compare")
    with st.expander("Physics Experiment Features (optional)", expanded=False):
        physics_features = _kp_research_physics_toggle_form("kp_compare")

    sequence_length = None
    if model_type in kp_research.SEQUENCE_MODELS:
        sequence_length = st.selectbox(
            "Sequence Length (hours, look-back window)",
            kp_research.SEQUENCE_LENGTH_OPTIONS,
            index=kp_research.SEQUENCE_LENGTH_OPTIONS.index(kp_research.DEFAULT_SEQUENCE_LENGTH),
            key="kp_compare_seqlen",
        )

    st.markdown("**Hyperparameters**")
    hyperparams = (
        _kp_research_hyperparam_inputs(model_type, "kp_compare") if model_type in kp_research.ALL_TRAINABLE_MODELS else {}
    )
    notes = st.text_input("Notes (optional)", key="kp_compare_notes")

    if st.button(
        "🧪 Train Model", key="kp_compare_train_btn", type="primary",
        disabled=model_type not in kp_research.ALL_TRAINABLE_MODELS,
    ):
        with st.spinner(f"Training {model_type}..."):
            try:
                run = kp_research.train_kp_research_model(
                    model_type,
                    feature_toggles=feature_toggles,
                    engineered_groups=engineered_groups,
                    physics_features=physics_features,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes=notes,
                    run_cv=True,
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Run Log")
    runs = kp_research.list_runs()
    if not runs:
        st.info("No training runs yet. Train a model above.")
        return
    best_id = max(runs, key=lambda r: r["metrics"]["r2"])["run_id"]
    for run in runs[:10]:
        _kp_research_run_row(run, best_run_id=best_id, key_prefix="kp_compare")
    if len(runs) > 10:
        st.caption(f"Showing 10 most recent of {len(runs)} runs — see Experiment Tracking for the full history.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Compare Models")

    def _run_label(r):
        seq = f" seq={r['sequence_length']}h" if r.get("sequence_length") else ""
        return f"{r['model_type']}{seq} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    run_labels = [_run_label(r) for r in runs]
    default_n = min(4, len(run_labels))
    selected = st.multiselect("Models to compare", run_labels, default=run_labels[:default_n], key="kp_compare_select")
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
    detail_label = st.selectbox("Inspect model", selected, key="kp_compare_detail")
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
            title="Predicted vs. Actual (held-out test sample)", height=340, xaxis_title="Actual Kp", yaxis_title="Predicted Kp"
        )
        plot_retro(fig, key="kp_compare_pred_actual")
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=4), name="Residual"))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Residual Plot", height=340, yaxis_title="Predicted − Actual")
        plot_retro(fig2, key="kp_compare_residual")

    if detail_run.get("feature_importance"):
        st.markdown("##### Feature Importance (Top 20)")
        fi_df = pd.DataFrame(detail_run["feature_importance"], columns=["Feature", "Importance"])
        fig3 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig3.update_layout(title="Top Contributing Features", height=420, yaxis=dict(autorange="reversed"))
        plot_retro(fig3, key="kp_compare_feature_importance")
    else:
        st.caption("Feature importance not available for this model type (e.g. SVR, LSTM/GRU).")

    if detail_run.get("loss_history"):
        st.markdown("##### Training / Validation Loss")
        lh = detail_run["loss_history"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig4.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig4.update_layout(title="Training / Validation Loss", height=340, xaxis_title="Epoch", yaxis_title="MSE Loss")
        plot_retro(fig4, key="kp_compare_loss")


def render_kp_feature_ablation_tab() -> None:
    st.caption(
        "Trains a Full Model (every feature group + every engineered group enabled) then retrains "
        "once per unit with just that one disabled — ranked by how much R² drops when removed, i.e. "
        "how much the full model actually relies on it. This is leave-one-out, not the cumulative "
        "'enable Ey, then enable VBz, ...' style — a cumulative sweep's deltas depend on the order "
        "features are added (whichever goes first tends to look most important merely by going "
        "first), whereas leave-one-out is order-independent. Physics-experiment features are tested "
        "individually in Hypothesis Testing instead."
    )
    model_type = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kp_ablation_model")
    if st.button("🔬 Run Feature Ablation Sweep", key="kp_ablation_run_btn", type="primary"):
        n_units = len(kp_research.FEATURE_ABLATION_UNITS) + 1
        with st.spinner(f"Training {n_units} models ({model_type})..."):
            try:
                result = kp_research.run_feature_ablation_sweep(model_type)
                st.session_state["kp_ablation_result"] = result
                st.toast("Feature ablation sweep complete.")
            except Exception as exc:
                st.error(f"Ablation sweep failed: {exc}")

    result = st.session_state.get("kp_ablation_result")
    if not result:
        st.info("Pick a model above, then run the sweep.")
        return

    st.markdown(f"##### {result['model_type']} — Full Model R² = {result['full_r2']:.4f}")
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
    plot_retro(fig, key="kp_ablation_bar")

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
        "means removing that group made the model worse (it was contributing); negative means the "
        "model did slightly better without it on this particular held-out split."
    )


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _cached_kp_research_frame(feature_toggles: dict = None, engineered_groups: dict = None, physics_features: dict = None):
    """Read-only exploratory calls into kp_research.load_kp_research_frame()
    (e.g. the Physics Experiments tab) re-run on every script rerun — every
    15s via the app-wide auto_refresh(), plus once per Kp Research Lab
    sub-tab per rerun since st.tabs() executes every tab body regardless of
    which one is visually active — even though the underlying
    analytics_features.csv is a static historical file that never changes
    within a session. Caching here eliminates that redundant CSV read +
    physics-feature recomputation with zero staleness risk (the source
    data genuinely doesn't change). NEVER used by train_kp_research_model
    itself — training always calls the uncached engine function directly,
    since that call only happens once per explicit Train click, not on
    every idle rerun.
    """
    return kp_research.load_kp_research_frame(feature_toggles, engineered_groups, physics_features)


def render_kp_physics_experiments_tab() -> None:
    st.caption(
        "Exploratory analysis: how does each individual physics-derived feature relate to Kp's "
        "behavior historically? Enable one to inspect it, or quick-train a model with just that one "
        "feature added on top of the full default feature set."
    )
    label = st.selectbox("Physics feature", kp_research.PHYSICS_FEATURE_OPTIONS, key="kp_physics_feature_select")

    try:
        base_frame, base_cols = _cached_kp_research_frame()
        frame, cols_with = _cached_kp_research_frame(physics_features={label: True})
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

    recent = frame.tail(24 * 30).dropna(subset=[col, "kp"])
    if not recent.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=recent.index, y=recent["kp"], name="Kp", line=dict(color="#1f5a7a")), secondary_y=False)
        fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=label, line=dict(color="#7a1f5a")), secondary_y=True)
        fig.update_layout(title=f"Kp vs. {label} — last {len(recent)} hours", height=380)
        fig.update_yaxes(title_text="Kp", secondary_y=False)
        fig.update_yaxes(title_text=label, secondary_y=True)
        plot_retro(fig, key=f"kp_physics_timeseries_{col}")

    next_target = kp_research.build_kp_interval_target(frame)
    valid = frame[[col]].join(next_target.rename("next_kp")).dropna()
    if len(valid) > 10:
        corr = valid[col].corr(valid["next_kp"])
        st.caption(f"Correlation between {label} and the next official Kp interval, across the full history: **{corr:.3f}**")
        fig2 = go.Figure(go.Scattergl(x=valid[col], y=valid["next_kp"], mode="markers", marker=dict(size=3, opacity=0.4)))
        fig2.update_layout(title=f"{label} vs. Next Kp Interval", height=380, xaxis_title=label, yaxis_title="Next official Kp interval")
        plot_retro(fig2, key=f"kp_physics_corr_{col}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button(f"🧪 Quick-Train Linear Regression with {label} added", key="kp_physics_quick_train"):
        with st.spinner("Training..."):
            try:
                run = kp_research.train_kp_research_model(
                    "Linear Regression", physics_features={label: True}, notes=f"Physics Experiments — quick test of {label}"
                )
                st.success(
                    f"Trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.3f} "
                    "(compare against the Feature Ablation / Model Comparison full-model baseline)."
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def render_kp_sequence_models_tab() -> None:
    st.caption(
        "LSTM/GRU look back over a window of consecutive hours (Sequence Length) rather than a "
        "single row, letting the model learn how Kp's drivers have been evolving, not just their "
        "instantaneous values. Trains in an isolated subprocess (see swdss.models.kp_research) — "
        "TensorFlow cannot safely share a process with this lab's scikit-learn/XGBoost/LightGBM/"
        "CatBoost imports."
    )
    if not kp_research.SEQUENCE_MODELS:
        st.warning("TensorFlow/Keras is not installed — sequence models are unavailable in this environment.")
        return

    model_type = st.selectbox("Model", kp_research.SEQUENCE_MODELS, key="kp_seq_model")
    sequence_length = st.selectbox(
        "Sequence Length (hours)",
        kp_research.SEQUENCE_LENGTH_OPTIONS,
        index=kp_research.SEQUENCE_LENGTH_OPTIONS.index(kp_research.DEFAULT_SEQUENCE_LENGTH),
        key="kp_seq_seqlen",
    )
    with st.expander("Feature Groups", expanded=False):
        feature_toggles = _kp_research_feature_toggle_form("kp_seq")
    hyperparams = _kp_research_hyperparam_inputs(model_type, "kp_seq")

    if st.button("🧪 Train Sequence Model", key="kp_seq_train_btn", type="primary"):
        with st.spinner(f"Training {model_type} (seq={sequence_length}h) — isolated subprocess, ~10-30s..."):
            try:
                run = kp_research.train_kp_research_model(
                    model_type,
                    feature_toggles=feature_toggles,
                    sequence_length=sequence_length,
                    hyperparams=hyperparams,
                    notes="Sequence Models tab",
                )
                st.toast(f"Trained {model_type} — R²={run['metrics']['r2']:.4f}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    all_runs = kp_research.list_runs()
    seq_runs = [r for r in all_runs if r["model_type"] in kp_research.SEQUENCE_MODELS]
    tabular_runs = [r for r in all_runs if r["model_type"] not in kp_research.SEQUENCE_MODELS]
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
        return f"{r['model_type']} seq={r['sequence_length']}h ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in seq_runs]
    selected = st.selectbox("Inspect run", labels, key="kp_seq_inspect")
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
        plot_retro(fig, key="kp_seq_loss")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### All Sequence Model Runs")
    for r in seq_runs:
        _kp_research_run_row(r, key_prefix="kp_seq")


def render_kp_experiment_tracking_tab() -> None:
    st.caption(
        "Every experiment ever run in this lab — full configuration and results, for reproducibility. "
        "Filter, inspect, promote, or delete any run."
    )
    runs = kp_research.list_runs()
    if not runs:
        st.info("No experiments recorded yet — train a model in any other tab.")
        return

    model_filter = st.selectbox("Filter by model", ["All"] + kp_research.ALL_TRAINABLE_MODELS, key="kp_track_model_filter")
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
        _kp_research_run_row(r, key_prefix="kp_track")
    if len(filtered) > 15:
        st.caption(f"Showing 15 most recent of {len(filtered)}.")


def render_kp_hypothesis_testing_tab() -> None:
    st.caption(
        "Fixed, reproducible hypotheses: each trains a baseline WITHOUT the tested feature and an "
        "experimental run WITH it (everything else at defaults), then reports ΔR²/ΔMAE/ΔRMSE and an "
        "Accept/Reject verdict (Accept requires ΔR² ≥ "
        f"{kp_research.HYPOTHESIS_ACCEPT_THRESHOLD_R2})."
    )
    hypothesis_label = st.selectbox("Hypothesis", list(kp_research.HYPOTHESIS_DEFINITIONS), key="kp_hyp_select")
    model_type = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kp_hyp_model")

    if st.button("🔬 Run Hypothesis Test", key="kp_hyp_run_btn", type="primary"):
        with st.spinner(f"Testing: {hypothesis_label}..."):
            try:
                result = kp_research.run_hypothesis_test(hypothesis_label, model_type)
                st.session_state["kp_hyp_last_result"] = result
                st.toast(f"{result['verdict']} — ΔR²={result['delta_r2']:+.4f}")
            except Exception as exc:
                st.error(f"Hypothesis test failed: {exc}")

    result = st.session_state.get("kp_hyp_last_result")
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
    results = kp_research.list_hypothesis_results()
    if not results:
        st.info("No hypothesis tests run yet.")
        return
    rows = [
        {
            "Hypothesis": r["hypothesis"],
            "Model": r["model_type"],
            "Verdict": r["verdict"],
            "ΔR²": round(r["delta_r2"], 4),
            "ΔMAE": round(r["delta_mae"], 4),
            "ΔRMSE": round(r["delta_rmse"], 4),
            "Tested": pd.Timestamp(r["tested_at"]).strftime("%Y-%m-%d %H:%M UTC"),
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_kp_visualization_tab() -> None:
    st.caption(
        "Cross-experiment views: how R² has moved across every run in this lab, which model "
        "architecture performs best overall, and a full diagnostic breakdown for any single run."
    )
    runs = kp_research.list_runs()
    if not runs:
        st.info("No experiments yet.")
        return

    runs_sorted = sorted(runs, key=lambda r: r["trained_at"])
    fig = go.Figure(
        go.Scatter(
            x=[pd.Timestamp(r["trained_at"]) for r in runs_sorted],
            y=[r["metrics"]["r2"] for r in runs_sorted],
            mode="markers+lines",
            text=[r["model_type"] for r in runs_sorted],
            name="R² over time",
        )
    )
    fig.update_layout(title="Experiment History — R² Over Time", height=360, xaxis_title="Trained At", yaxis_title="R²")
    plot_retro(fig, key="kp_viz_history")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    by_model: dict = {}
    for r in runs:
        by_model.setdefault(r["model_type"], []).append(r["metrics"]["r2"])
    model_avg = {m: sum(v) / len(v) for m, v in by_model.items()}
    fig2 = go.Figure(go.Bar(x=list(model_avg.keys()), y=list(model_avg.values())))
    fig2.update_layout(title="Average R² by Model Architecture (across all runs)", height=360, yaxis_title="Mean R²")
    plot_retro(fig2, key="kp_viz_model_avg")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Inspect a Single Run")

    def _run_label(r):
        seq = f" seq={r['sequence_length']}h" if r.get("sequence_length") else ""
        return f"{r['model_type']}{seq} — R²={r['metrics']['r2']:.4f} ({pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')})"

    labels = [_run_label(r) for r in runs]
    selected = st.selectbox("Run", labels, key="kp_viz_run_select")
    run = runs[labels.index(selected)]
    sample = run["prediction_sample"]
    y_true, y_pred = sample["y_true"], sample["y_pred"]
    residuals = [p - t for p, t in zip(y_pred, y_true)]

    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=4), name="Predicted vs Actual"))
        lo, hi = min(y_true + y_pred), max(y_true + y_pred)
        fig3.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
        fig3.update_layout(title="Predicted vs. Actual", height=320)
        plot_retro(fig3, key="kp_viz_pred_actual")
    with col_b:
        fig4 = go.Figure(go.Scatter(y=residuals, mode="markers", marker=dict(size=4)))
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(title="Residual Plot", height=320)
        plot_retro(fig4, key="kp_viz_residual")

    if run.get("feature_importance"):
        fi_df = pd.DataFrame(run["feature_importance"][:15], columns=["Feature", "Importance"])
        fig5 = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h"))
        fig5.update_layout(title="Top 15 Feature Importance", height=380, yaxis=dict(autorange="reversed"))
        plot_retro(fig5, key="kp_viz_feature_importance")

    if run.get("loss_history"):
        lh = run["loss_history"]
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(y=lh["loss"], mode="lines", name="Training Loss"))
        if lh.get("val_loss"):
            fig6.add_trace(go.Scatter(y=lh["val_loss"], mode="lines", name="Validation Loss"))
        fig6.update_layout(title="Learning Curve", height=320)
        plot_retro(fig6, key="kp_viz_loss")


def _kp_opt_isolated_toggles(groups: dict) -> dict:
    """UI-local mirror of kp_research._build_isolated_toggles — builds a
    feature_toggles dict with everything off except the named columns, so
    Exp 3/4/5's manual tabs can isolate one input combination exactly like
    the automated pipeline does.
    """
    toggles = {group: {col: False for col in cols} for group, cols in kp_research.FEATURE_GROUP_COLUMNS.items()}
    for group, cols in groups.items():
        for col in cols:
            toggles[group][col] = True
    return toggles


def _kp_opt_latest_baseline_r2() -> float:
    runs = [r for r in kp_research.list_runs() if r.get("experiment_tag", "").endswith("exp1_baseline")]
    return runs[0]["metrics"]["r2"] if runs else None


def _render_kp_automl_section() -> None:
    """AutoML orchestration layer for the Kp Optimization Study — mirrors
    the Bz lab's automation section, adapted for Kp's 10-experiment
    magnetosphere-coupling-focused pipeline. The manual Exp 1-10 tabs
    below stay fully intact; this only sequences the same underlying
    kp_research functions those tabs call.
    """
    st.markdown("### 🤖 Automated Optimization (AutoML)")
    st.caption(
        "Runs all 10 experiments end-to-end — production baseline, persistence benchmark, Solar "
        "Wind / IMF / Geomagnetic History input search, the full Physics Optimization sweep (26 "
        "coupling-function groups), a structured combination of the strongest physics contributors, "
        "a full model sweep, feature importance, SHAP, and a production recommendation."
    )
    st.warning(
        "⚠️ This runs 45-55+ separate training runs, several on 100+ engineered features across "
        "~27,000 rows — expect this to take a long time. **Turn on ⏸ Pause Live Refresh above before "
        "clicking**, or the dashboard's 15-second auto-refresh can interrupt the run partway through."
    )

    if st.button("🚀 Run Complete Kp Optimization Study", key="kp_automl_run_study", type="primary"):
        status_box = st.status("Running complete Kp optimization study…", expanded=True)

        def _cb(step, total, msg):
            status_box.update(label=f"Step {step}/{total} — {msg}")
            status_box.write(f"**Step {step}/{total}:** {msg}")

        try:
            study = kp_research.run_complete_kp_optimization_study(progress_cb=_cb)
            status_box.update(label="Optimization study complete.", state="complete", expanded=False)
            st.session_state["kp_automl_last_study_id"] = study["study_id"]
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
    studies = kp_research.list_kp_studies()
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
    last_id = st.session_state.get("kp_automl_last_study_id")
    if last_id:
        for i, s in enumerate(studies):
            if s["study_id"] == last_id:
                default_idx = i
                break
    chosen_label = st.selectbox("Select a study to inspect", study_labels, index=default_idx, key="kp_automl_study_select")
    study = studies[study_labels.index(chosen_label)]
    _render_kp_study_detail(study)


def _render_kp_study_detail(study: dict) -> None:
    st.markdown(
        f"**Study ID:** `{study['study_id'][:8]}…`  ·  "
        f"**Started:** {pd.Timestamp(study['started_at']).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
        f"**Completed:** {pd.Timestamp(study['completed_at']).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    step_cols = st.columns(5)
    step_cols[0].metric("1. Baseline R²", f"{study['baseline_metrics']['r2']:.4f}", help=study["baseline_model_type"])
    step_cols[1].metric("2. Persistence R²", f"{study['persistence_metrics']['r2']:.4f}")
    step_cols[2].metric(
        "6. Top Physics", study["physics_results"][0]["name"] if study.get("physics_results") else "—",
        help=f"ΔR²={study['physics_results'][0]['delta_r2']:+.4f}" if study.get("physics_results") else "",
    )
    step_cols[3].metric("7. Winner", study["winner_model_type"], help=f"R²={study['winner_metrics']['r2']:.4f}")
    step_cols[4].metric("10. Recommendation", study["recommendation"])

    detail_tabs = st.tabs(
        ["Solar Wind / IMF / Geomag", "Physics Optimization", "Leaderboard", "Feature Importance",
         "SHAP", "Production Comparison", "Promotion"]
    )

    with detail_tabs[0]:
        st.markdown("**Experiment 3 — Solar Wind Inputs**")
        st.dataframe(
            pd.DataFrame([
                {"Combination": r["name"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4), "Features": r["feature_count"]}
                for r in study["solar_wind_results"]
            ]), use_container_width=True, hide_index=True,
        )
        st.markdown("**Experiment 4 — IMF Inputs**")
        st.dataframe(
            pd.DataFrame([
                {"Combination": r["name"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4), "Features": r["feature_count"]}
                for r in study["imf_results"]
            ]), use_container_width=True, hide_index=True,
        )
        st.markdown("**Experiment 5 — Geomagnetic History**")
        st.dataframe(
            pd.DataFrame([
                {"Combination": r["name"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4), "Features": r["feature_count"]}
                for r in study["geomagnetic_results"]
            ]), use_container_width=True, hide_index=True,
        )

    with detail_tabs[1]:
        st.markdown("**Experiment 6 — Physics Optimization (ranked by ΔR² vs. Baseline)**")
        st.caption(
            "Positive ΔR² always means the variable helps, whether tested by adding it (physics-"
            "experiment features) or removing it (Ey/VBz/Dynamic Pressure, already in the baseline)."
        )
        phys_rows = [
            {"Physics Variable": r["name"], "Test": "Add" if r["kind"] == "add_physics" else "Remove-from-baseline",
             "ΔR²": round(r["delta_r2"], 4), "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4)}
            for r in study["physics_results"]
        ]
        st.dataframe(pd.DataFrame(phys_rows), use_container_width=True, hide_index=True)
        st.markdown("**Best Combined Feature Set**")
        combo = study["best_feature_config"]
        st.write(f"Combined physics groups: {', '.join(combo['combined_physics_names']) or '(none had positive ΔR²)'}")
        st.caption(f"Combined-set R² = {combo['r2']:.4f} (confirms whether the top contributors combine well, not just individually).")

    with detail_tabs[2]:
        st.markdown("**Experiment 7 — Model Optimization Leaderboard**")
        lb_rows = [
            {
                "Rank": r["rank"], "Model": r["model_type"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4),
                "RMSE": round(r["rmse"], 4), "MAPE (%)": round(r["mape"], 2) if r["mape"] is not None else None,
                "Bias": round(r["bias"], 4),
                "Train Time (s)": round(r["training_time_sec"], 3) if r["training_time_sec"] is not None else None,
                "Inference (ms/sample)": round(r["inference_time_ms_per_sample"], 4) if r["inference_time_ms_per_sample"] is not None else None,
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

    with detail_tabs[3]:
        st.markdown("**Experiment 8 — Feature Importance (winning / best-available candidate)**")
        if study.get("feature_importance"):
            fi_df = pd.DataFrame(study["feature_importance"], columns=["Feature", "Importance"]).head(25)
            fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h", marker_color="steelblue"))
            fig.update_layout(title="Top 25 Features by Importance", height=520, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
        else:
            st.info("Feature importance not available for any candidate in this study.")

    with detail_tabs[4]:
        st.markdown("**Experiment 9 — SHAP Analysis**")
        shap_result = study.get("shap_result") or {}
        if shap_result.get("supported"):
            shap_df = pd.DataFrame(shap_result["shap_importance"], columns=["Feature", "Mean |SHAP|"]).head(25)
            fig = go.Figure(go.Bar(x=shap_df["Mean |SHAP|"], y=shap_df["Feature"], orientation="h", marker_color="indianred"))
            fig.update_layout(title="Top 25 Features by Mean |SHAP Value|", height=520, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
        else:
            st.info(shap_result.get("skipped_reason", "SHAP analysis not available for this study."))

    with detail_tabs[5]:
        st.markdown("**Production Comparison**")
        comp = study["production_comparison"]
        if comp["current"] is None:
            st.warning("No production Kp model currently on disk to compare against.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Current Production**")
                st.metric("Algorithm", comp["current"]["algorithm"])
                st.metric("R²", f"{comp['current']['r2']:.4f}")
                st.metric("MAE", f"{comp['current']['mae']:.4f}")
                st.metric("RMSE", f"{comp['current']['rmse']:.4f}")
            with c2:
                st.markdown("**Candidate**")
                st.metric("Algorithm", comp["candidate"]["algorithm"])
                st.metric("R²", f"{comp['candidate']['r2']:.4f}", delta=f"{comp['delta_r2']:+.4f}")
                st.metric("MAE", f"{comp['candidate']['mae']:.4f}", delta=f"{comp['delta_mae']:+.4f}", delta_color="inverse")
                st.metric("RMSE", f"{comp['candidate']['rmse']:.4f}", delta=f"{comp['delta_rmse']:+.4f}", delta_color="inverse")
            with c3:
                st.markdown("**Verdict**")
                if study["recommendation"] == "Promote":
                    st.success("✅ Promote")
                else:
                    st.warning("⏸ Keep Current Production")

    with detail_tabs[6]:
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
            notes = st.text_area("Promotion notes (optional)", key=f"kp_automl_promo_notes_{study['study_id']}")
            pc1, pc2 = st.columns(2)
            with pc1:
                confirm = st.checkbox(
                    "I understand this will overwrite the production Kp model (a rollback archive will be created)",
                    key=f"kp_automl_promo_confirm_{study['study_id']}",
                    disabled=not eligible,
                )
                if st.button(
                    "🚀 Promote Winning Candidate to Production", key=f"kp_automl_promote_{study['study_id']}",
                    type="primary", disabled=not (eligible and confirm),
                ):
                    with st.spinner("Promoting to production…"):
                        try:
                            result = kp_research.promote_kp_to_production(study["winner_run_id"], notes=notes)
                            kp_research.mark_kp_study_promoted(study["study_id"], study["winner_run_id"])
                            st.success(f"Promoted! Archive: `{result['archive_path']}`")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Promotion failed: {exc}")
                if not eligible:
                    st.caption("Promotion disabled — one or more criteria above failed.")
            with pc2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("⏸ Keep Current Production", key=f"kp_automl_reject_{study['study_id']}"):
                    kp_research.mark_kp_study_rejected(study["study_id"], notes=notes)
                    st.rerun()


def render_kp_optimization_study() -> None:
    """10-experiment structured study to discover the best scientifically
    defensible Production Kp forecasting model — NOT simply the highest
    R². Kp is an Earth-RESPONSE forecasting problem (Solar Wind -> IMF ->
    magnetosphere-ionosphere coupling -> geomagnetic activity), unlike
    Bz's upstream-IMF forecasting problem, so this study is structured
    around validating physical coupling mechanisms before model choice:

      1  Production Baseline — reproduce current production exactly
      2  Persistence Benchmark — naïve lower bound
      3  Solar Wind Inputs — Speed/Density/Temperature combinations
      4  IMF Inputs — Bt/Bx/By/Bz individually and combined
      5  Geomagnetic History — how much does magnetosphere memory help
      6  Physics Optimization — 26 coupling-function groups, individually
      7  Model Optimization — all 12 model types on the winning feature set
      8  Feature Importance — tree-based ranking
      9  SHAP Analysis — which variables consistently drive Kp
      10 Optimization Summary — leaderboard, ranking, recommendation
    """
    st.info(
        "**Kp Production Model Optimization Study** — 10 structured experiments to answer: can the "
        "current production Kp interval model (R²≈0.6715, XGBoost) be improved, and if so, what "
        "coupling physics/model should replace it? Unlike Bz (an upstream IMF-forecasting problem), "
        "Kp is an Earth-RESPONSE forecasting problem — Solar Wind → IMF → magnetosphere-ionosphere "
        "coupling → geomagnetic activity — so this study emphasizes physical coupling functions and "
        "geomagnetic memory over raw statistical performance."
    )

    _render_kp_automl_section()
    st.markdown("---")
    st.markdown(
        "### 🔬 Manual Experiments — Exp 1 through Exp 10\n"
        "Run any experiment individually below for hands-on investigation. This is exactly what "
        "**Run Complete Kp Optimization Study** above orchestrates automatically — nothing here changes."
    )

    exp_tabs = st.tabs([
        "Exp 1 · Baseline", "Exp 2 · Persistence", "Exp 3 · Solar Wind", "Exp 4 · IMF Inputs",
        "Exp 5 · Geomag History", "Exp 6 · Physics Optimization", "Exp 7 · Model Optimization",
        "Exp 8 · Feature Importance", "Exp 9 · SHAP", "Exp 10 · Summary & Promote",
    ])

    # ── Exp 1 · Production Baseline ─────────────────────────────────────────
    with exp_tabs[0]:
        st.markdown("##### Experiment 1 — Reproduce Production Baseline")
        st.caption(
            f"Train {kp_research.PRODUCTION_BASELINE_MODEL} with every base feature group and every "
            "engineered group enabled (no physics-experiment features) — the exact configuration "
            "production's own train_kp_interval_model uses. Should reproduce R²≈0.6715."
        )
        if st.button("▶ Run Baseline", key="kpopt_exp1_run", type="primary"):
            with st.spinner(f"Training {kp_research.PRODUCTION_BASELINE_MODEL} baseline…"):
                try:
                    run = kp_research.train_kp_research_model(
                        kp_research.PRODUCTION_BASELINE_MODEL,
                        feature_toggles=kp_research.default_feature_toggles(),
                        engineered_groups=kp_research.default_engineered_toggles(),
                        physics_features={},
                        experiment_tag="exp1_baseline",
                    )
                    st.success(f"Baseline trained — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        baseline_runs = [r for r in kp_research.list_runs() if r.get("experiment_tag") == "exp1_baseline"]
        if baseline_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "RMSE": round(r["metrics"]["rmse"], 4), "Features": len(r["feature_columns"]),
                 "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in baseline_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No baseline runs yet — click Run Baseline above.")

    # ── Exp 2 · Persistence Benchmark ───────────────────────────────────────
    with exp_tabs[1]:
        st.markdown("##### Experiment 2 — Persistence Benchmark")
        st.caption("Naïve forecast: next official Kp interval = current official Kp value. Stored permanently.")
        bench_run = next((r for r in kp_research.list_runs() if r.get("experiment_tag") == "persistence_benchmark"), None)
        if st.button("▶ Compute Persistence Benchmark", key="kpopt_exp2_run", type="primary"):
            with st.spinner("Computing persistence benchmark…"):
                try:
                    bench_run = kp_research.compute_kp_persistence_benchmark()
                    st.success(f"Persistence baseline — R²={bench_run['metrics']['r2']:.4f}, MAE={bench_run['metrics']['mae']:.4f}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")
        if bench_run:
            m = bench_run["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{m['r2']:.4f}")
            c2.metric("MAE", f"{m['mae']:.4f}")
            c3.metric("RMSE", f"{m['rmse']:.4f}")
            c4.metric("Bias", f"{m['bias']:.4f}")
            st.caption(f"Test samples: {bench_run['n_test_samples']:,} · Computed: {pd.Timestamp(bench_run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            st.info("Click above to compute and store the persistence benchmark.")

    # ── Exp 3 · Solar Wind Inputs ────────────────────────────────────────────
    with exp_tabs[2]:
        st.markdown("##### Experiment 3 — Solar Wind Inputs")
        st.caption("Systematically test Speed/Density/Temperature combinations, isolated against nothing else.")
        combo_names = [c["name"] for c in kp_research.SOLAR_WIND_INPUT_GRID]
        col3a, col3b = st.columns(2)
        with col3a:
            combo3 = st.selectbox("Combination", combo_names, key="kpopt_exp3_combo")
        with col3b:
            model3 = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kpopt_exp3_model")
        if st.button("▶ Train This Combination", key="kpopt_exp3_run", type="primary"):
            spec = next(c for c in kp_research.SOLAR_WIND_INPUT_GRID if c["name"] == combo3)
            with st.spinner(f"Training {model3} on '{combo3}'…"):
                try:
                    run = kp_research.train_kp_research_model(
                        model3, feature_toggles=_kp_opt_isolated_toggles(spec["groups"]),
                        engineered_groups=kp_research.default_engineered_toggles(), physics_features={},
                        experiment_tag="exp3_solar_wind_inputs",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp3_runs = [r for r in kp_research.list_runs() if r.get("experiment_tag") == "exp3_solar_wind_inputs"]
        if exp3_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp3_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 3 runs yet.")

    # ── Exp 4 · IMF Inputs ────────────────────────────────────────────────────
    with exp_tabs[3]:
        st.markdown("##### Experiment 4 — IMF Inputs")
        st.caption("Evaluate Bt/Bx/By/Bz individually and combined, with lags/rolling stats/rate of change.")
        combo_names4 = [c["name"] for c in kp_research.IMF_INPUT_GRID]
        col4a, col4b = st.columns(2)
        with col4a:
            combo4 = st.selectbox("Combination", combo_names4, key="kpopt_exp4_combo")
        with col4b:
            model4 = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kpopt_exp4_model")
        if st.button("▶ Train This Combination", key="kpopt_exp4_run", type="primary"):
            spec = next(c for c in kp_research.IMF_INPUT_GRID if c["name"] == combo4)
            with st.spinner(f"Training {model4} on '{combo4}'…"):
                try:
                    run = kp_research.train_kp_research_model(
                        model4, feature_toggles=_kp_opt_isolated_toggles(spec["groups"]),
                        engineered_groups=kp_research.default_engineered_toggles(), physics_features={},
                        experiment_tag="exp4_imf_inputs",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp4_runs = [r for r in kp_research.list_runs() if r.get("experiment_tag") == "exp4_imf_inputs"]
        if exp4_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp4_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 4 runs yet.")

    # ── Exp 5 · Geomagnetic History ───────────────────────────────────────────
    with exp_tabs[4]:
        st.markdown("##### Experiment 5 — Geomagnetic History")
        st.caption("How much does the magnetosphere's own recent memory (previous Kp/Dst/AE) contribute?")
        combo_names5 = [c["name"] for c in kp_research.GEOMAGNETIC_HISTORY_GRID]
        col5a, col5b = st.columns(2)
        with col5a:
            combo5 = st.selectbox("Combination", combo_names5, key="kpopt_exp5_combo")
        with col5b:
            model5 = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kpopt_exp5_model")
        if st.button("▶ Train This Combination", key="kpopt_exp5_run", type="primary"):
            spec = next(c for c in kp_research.GEOMAGNETIC_HISTORY_GRID if c["name"] == combo5)
            with st.spinner(f"Training {model5} on '{combo5}'…"):
                try:
                    run = kp_research.train_kp_research_model(
                        model5, feature_toggles=_kp_opt_isolated_toggles(spec["groups"]),
                        engineered_groups=kp_research.default_engineered_toggles(), physics_features={},
                        experiment_tag="exp5_geomagnetic_history",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp5_runs = [r for r in kp_research.list_runs() if r.get("experiment_tag") == "exp5_geomagnetic_history"]
        if exp5_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp5_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 5 runs yet.")

    # ── Exp 6 · Physics Optimization ──────────────────────────────────────────
    with exp_tabs[5]:
        st.markdown("##### Experiment 6 — Physics Optimization (the primary Kp experiment)")
        st.caption(
            "Evaluate each of 26 coupling-physics variables/functions individually against the full "
            "production baseline — never all at once. Positive ΔR² means the variable helps."
        )
        phys_names = [c["name"] for c in kp_research.PHYSICS_OPTIMIZATION_GRID]
        col6a, col6b = st.columns(2)
        with col6a:
            phys6 = st.selectbox("Physics Variable / Coupling Function", phys_names, key="kpopt_exp6_var")
        with col6b:
            model6 = st.selectbox("Model", kp_research.TABULAR_MODELS, key="kpopt_exp6_model")
        if st.button("▶ Train This Physics Group", key="kpopt_exp6_run", type="primary"):
            entry = next(c for c in kp_research.PHYSICS_OPTIMIZATION_GRID if c["name"] == phys6)
            if entry["kind"] == "remove_column":
                toggles = kp_research.default_feature_toggles()
                toggles[entry["group"]] = dict(toggles[entry["group"]])
                toggles[entry["group"]][entry["column"]] = False
                physics_features = {}
            else:
                toggles = kp_research.default_feature_toggles()
                physics_features = {entry.get("physics_name", entry["name"]): True}
            with st.spinner(f"Training {model6} — {phys6}…"):
                try:
                    run = kp_research.train_kp_research_model(
                        model6, feature_toggles=toggles, engineered_groups=kp_research.default_engineered_toggles(),
                        physics_features=physics_features, experiment_tag="exp6_physics_optimization",
                        notes=f"Physics Optimization — {entry['kind']}: {phys6}",
                    )
                    baseline_r2 = _kp_opt_latest_baseline_r2()
                    delta_txt = ""
                    if baseline_r2 is not None:
                        delta = (run["metrics"]["r2"] - baseline_r2) if entry["kind"] == "add_physics" else (baseline_r2 - run["metrics"]["r2"])
                        delta_txt = f" (ΔR² vs. Exp 1 baseline: {delta:+.4f})"
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f}{delta_txt}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        exp6_runs = [r for r in kp_research.list_runs() if r.get("experiment_tag") == "exp6_physics_optimization"]
        if exp6_runs:
            baseline_r2 = _kp_opt_latest_baseline_r2()
            rows6 = []
            for r in exp6_runs:
                notes = r.get("notes", "")
                kind = "add_physics" if "add_physics:" in notes else ("remove_column" if "remove_column:" in notes else "?")
                var_name = notes.split(": ", 1)[-1] if ": " in notes else "?"
                delta = None
                if baseline_r2 is not None:
                    delta = (r["metrics"]["r2"] - baseline_r2) if kind == "add_physics" else (baseline_r2 - r["metrics"]["r2"])
                rows6.append({
                    "Physics Variable": var_name, "Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4),
                    "ΔR² vs Baseline": round(delta, 4) if delta is not None else None,
                    "MAE": round(r["metrics"]["mae"], 4), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC"),
                })
            rows6.sort(key=lambda x: -(x["ΔR² vs Baseline"] or -999))
            st.dataframe(pd.DataFrame(rows6), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 6 runs yet.")

    # ── Exp 7 · Model Optimization ────────────────────────────────────────────
    with exp_tabs[6]:
        st.markdown("##### Experiment 7 — Model Optimization")
        st.caption(
            "Compare all model types on identical training data/forecast horizon/methodology — the "
            "full baseline feature set (same as Exp 1), so results are comparable across model types."
        )
        model7 = st.selectbox("Model to train", kp_research.TABULAR_MODELS + kp_research.SEQUENCE_MODELS, key="kpopt_exp7_model")
        seq_len7 = None
        if model7 in kp_research.SEQUENCE_MODELS:
            seq_len7 = st.selectbox("Sequence Length (hours)", kp_research.SEQUENCE_LENGTH_OPTIONS,
                                     index=kp_research.SEQUENCE_LENGTH_OPTIONS.index(kp_research.DEFAULT_SEQUENCE_LENGTH),
                                     key="kpopt_exp7_seqlen")
        if st.button(f"▶ Train {model7}", key="kpopt_exp7_run", type="primary"):
            with st.spinner(f"Training {model7}…"):
                try:
                    run = kp_research.train_kp_research_model(
                        model7, feature_toggles=kp_research.default_feature_toggles(),
                        engineered_groups=kp_research.default_engineered_toggles(), physics_features={},
                        sequence_length=seq_len7, experiment_tag="exp7_model_optimization",
                    )
                    st.success(f"{model7} — R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        exp7_runs = [r for r in kp_research.list_runs() if r.get("experiment_tag") == "exp7_model_optimization"]
        if exp7_runs:
            best7_r2 = max(r["metrics"]["r2"] for r in exp7_runs)
            rows7 = []
            for r in exp7_runs:
                m = r["metrics"]
                label = r["model_type"] + (" ⭐" if abs(m["r2"] - best7_r2) < 1e-6 else "")
                rows7.append({
                    "Model": label, "R²": round(m["r2"], 4), "MAE": round(m["mae"], 4), "RMSE": round(m["rmse"], 4),
                    "Bias": round(m["bias"], 4), "Features": len(r["feature_columns"]),
                    "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC"),
                })
            rows7.sort(key=lambda x: -x["R²"])
            st.dataframe(pd.DataFrame(rows7), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 7 runs yet.")

    # ── Exp 8 · Feature Importance ────────────────────────────────────────────
    with exp_tabs[7]:
        st.markdown("##### Experiment 8 — Feature Importance")
        study_runs = [
            r for r in kp_research.list_runs()
            if r.get("experiment_tag", "").startswith("exp") and r.get("feature_importance")
        ]
        if not study_runs:
            st.info("No tree-based Optimization Study runs with feature importance yet.")
        else:
            def _run_lbl(r):
                return f"{r.get('experiment_tag')} · {r['model_type']} · R²={r['metrics']['r2']:.4f} · {pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')}"
            run_options = {_run_lbl(r): r for r in study_runs}
            chosen_lbl = st.selectbox("Run", list(run_options), key="kpopt_exp8_run")
            chosen_run = run_options[chosen_lbl]
            fi_df = pd.DataFrame(chosen_run["feature_importance"], columns=["Feature", "Importance"]).head(25)
            fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h", marker_color="steelblue"))
            fig.update_layout(title="Top 25 Features by Importance", height=560, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

    # ── Exp 9 · SHAP Analysis ──────────────────────────────────────────────────
    with exp_tabs[8]:
        st.markdown("##### Experiment 9 — SHAP Analysis")
        shap_candidates = [
            r for r in kp_research.list_runs()
            if r.get("experiment_tag", "").startswith("exp") and r["model_type"] in kp_research.SHAP_SUPPORTED_MODELS
        ]
        if not shap_candidates:
            st.info("No SHAP-supported Optimization Study runs yet (linear or tree/boosting model types).")
        else:
            def _shap_lbl(r):
                return f"{r.get('experiment_tag')} · {r['model_type']} · R²={r['metrics']['r2']:.4f} · {pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M')}"
            shap_options = {_shap_lbl(r): r for r in shap_candidates}
            chosen_shap_lbl = st.selectbox("Run", list(shap_options), key="kpopt_exp9_run")
            chosen_shap_run = shap_options[chosen_shap_lbl]
            if st.button("▶ Compute SHAP", key="kpopt_exp9_compute", type="primary"):
                with st.spinner("Computing SHAP values…"):
                    try:
                        result = kp_research.compute_shap_importance_kp(chosen_shap_run["run_id"])
                        st.session_state["kpopt_exp9_result"] = result
                    except Exception as exc:
                        st.error(f"SHAP failed: {exc}")
            result = st.session_state.get("kpopt_exp9_result")
            if result and result.get("supported") and result.get("run_id") == chosen_shap_run["run_id"]:
                shap_df = pd.DataFrame(result["shap_importance"], columns=["Feature", "Mean |SHAP|"]).head(25)
                fig = go.Figure(go.Bar(x=shap_df["Mean |SHAP|"], y=shap_df["Feature"], orientation="h", marker_color="indianred"))
                fig.update_layout(title="Top 25 Features by Mean |SHAP Value|", height=560, yaxis=dict(autorange="reversed"))
                plot_retro(fig)
                st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # ── Exp 10 · Summary & Promote ────────────────────────────────────────────
    with exp_tabs[9]:
        st.markdown("##### Experiment 10 — Optimization Summary & Promotion")
        st.caption("All Exp 1-9 manual runs ranked by R², compared against current production.")

        all_study_runs = [r for r in kp_research.list_runs() if r.get("experiment_tag", "").startswith("exp") or r.get("experiment_tag") == "persistence_benchmark"]
        if all_study_runs:
            best_r2 = max(r["metrics"]["r2"] for r in all_study_runs)
            rows10 = []
            for r in sorted(all_study_runs, key=lambda x: -x["metrics"]["r2"]):
                m = r["metrics"]
                rows10.append({
                    "Experiment": r.get("experiment_tag", "—"), "Model": r["model_type"] + (" ⭐" if abs(m["r2"] - best_r2) < 1e-6 else ""),
                    "R²": round(m["r2"], 4), "MAE": round(m["mae"], 4), "RMSE": round(m["rmse"], 4),
                    "Features": len(r.get("feature_columns", [])), "Promoted": "✅" if r.get("promoted_to_production") else "",
                })
            st.dataframe(pd.DataFrame(rows10), use_container_width=True, hide_index=True)
        else:
            st.info("Run experiments above to populate this summary.")

        st.markdown("---")
        st.markdown("**Promote a manual run to production**")
        promotable = [
            r for r in kp_research.list_runs()
            if r.get("experiment_tag", "").startswith("exp") and r.get("model_type") not in kp_research.SEQUENCE_MODELS
            and r.get("model_path")
        ]
        if not promotable:
            st.info("No promotable manual runs yet.")
        else:
            def _promo_lbl(r):
                return f"{r.get('experiment_tag')} · {r['model_type']} · R²={r['metrics']['r2']:.4f} · MAE={r['metrics']['mae']:.4f} · {pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M UTC')}"
            promo_options = {_promo_lbl(r): r for r in sorted(promotable, key=lambda r: -r["metrics"]["r2"])}
            chosen_promo_lbl = st.selectbox("Select run to promote", list(promo_options), key="kpopt_exp10_run")
            chosen_promo = promo_options[chosen_promo_lbl]

            prod_metrics = kp_research.get_production_kp_metrics()
            check = kp_research.check_promotion_criteria_kp(chosen_promo, prod_metrics)
            for item in check["checklist"]:
                icon = "✅" if item["passed"] else "❌"
                st.markdown(f"{icon} {item['criterion']} — {item['detail']}")

            promo_notes = st.text_area("Promotion notes (optional)", key="kpopt_exp10_notes")
            confirm10 = st.checkbox(
                "I understand this will overwrite the production Kp model (a rollback archive will be created)",
                key="kpopt_exp10_confirm", disabled=not check["eligible"],
            )
            if st.button("🚀 Promote to Production", key="kpopt_exp10_promote", type="primary",
                         disabled=not (check["eligible"] and confirm10)):
                with st.spinner("Promoting to production…"):
                    try:
                        result = kp_research.promote_kp_to_production(chosen_promo["run_id"], notes=promo_notes)
                        st.success(f"Promoted! Archive: `{result['archive_path']}`")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Promotion failed: {exc}")
            if not check["eligible"]:
                st.caption("Promotion disabled — one or more criteria above failed.")


def render_kp_research_laboratory() -> None:
    st.markdown("### 🧪 Kp Research Laboratory")
    hdr_col, pause_col = st.columns([3, 1])
    with hdr_col:
        st.caption(
            "A scientific experimentation platform for Kp forecasting — fully isolated from the "
            "Production Prediction tab, which continues using the current trained production model "
            "exactly as-is. Nothing trained here overwrites it; Promote only labels a run for your own "
            "tracking, wiring a model into production is always a manual, deliberate step."
        )
    with pause_col:
        st.session_state.setdefault("pause_autorefresh", False)
        if st.toggle("⏸ Pause Live Refresh", value=st.session_state["pause_autorefresh"], key="kp_lab_pause_toggle"):
            st.session_state["pause_autorefresh"] = True
        else:
            st.session_state["pause_autorefresh"] = False
    _kp_research_notes()
    with st.expander("Future Research — planned architecture extensions"):
        st.write(", ".join(kp_research.FUTURE_MODELS))
        st.caption(
            "Registered as disabled entries in the model selector now, so adding a real "
            "implementation later never requires redesigning this interface."
        )

    sub = st.tabs(
        [
            "🔬 Kp Optimization Study",
            "Model Comparison",
            "Feature Ablation",
            "Physics Experiments",
            "Sequence Models",
            "Experiment Tracking",
            "Hypothesis Testing",
            "Visualization",
        ]
    )
    with sub[0]:
        render_kp_optimization_study()
    with sub[1]:
        render_kp_model_comparison_tab()
    with sub[2]:
        render_kp_feature_ablation_tab()
    with sub[3]:
        render_kp_physics_experiments_tab()
    with sub[4]:
        render_kp_sequence_models_tab()
    with sub[5]:
        render_kp_experiment_tracking_tab()
    with sub[6]:
        render_kp_hypothesis_testing_tab()
    with sub[7]:
        render_kp_visualization_tab()


def _ae_opt_isolated_toggles(groups: dict) -> dict:
    return ae_research._build_isolated_toggles(groups)


def _render_ae_automl_section() -> None:
    """AutoML orchestration layer for the AE Optimization Study — the
    flagship scientific component of this project. Unlike the Bz/Kp
    studies (a single target, single horizon), this runs the full
    10-experiment methodology independently at all 5 production horizons
    (1h/3h/6h/12h/24h), then performs Experiment 10's Cross-Horizon
    Scientific Synthesis. The manual Exp 1-10 tabs below stay fully
    intact for hands-on, single-horizon investigation; this only
    sequences the same underlying ae_research functions those tabs call.
    """
    st.markdown("### 🤖 Automated Optimization (AutoML)")
    st.caption(
        "Runs all 10 experiments end-to-end, independently at every one of AE's 5 production "
        "horizons — production baseline reproduction, persistence benchmark, the Solar Wind + IMF "
        "raw explanatory floor, 14 coupling-physics variables tested individually, a structured "
        "cumulative Physics Engine ablation, Geomagnetic Memory (Previous AE/Kp/Dst), the best "
        "combined feature set per horizon, a full 12-model sweep, feature importance/SHAP/"
        "permutation importance, and — the centerpiece — a Cross-Horizon Scientific Synthesis "
        "that checks whether AE prediction shifts from persistence-dominated to physics-dominated "
        "as the horizon grows."
    )
    st.warning(
        "⚠️ This runs roughly 45-50 training runs PER HORIZON (~230-250 total across all 5 "
        "horizons), including SVR/MLP and a permutation-importance pass at every horizon — measured "
        "on this project's own hardware, expect **an hour or more** for the complete study, not just "
        "a few minutes. **Turn on ⏸ Pause Live Refresh above before clicking**, or the dashboard's "
        "15-second auto-refresh can interrupt the run partway through."
    )
    st.caption(
        "The minute-resolution Kyoto AE archive plays NO role in this study — Production forecasts "
        "hourly AE, so every experiment here trains and evaluates against the same hourly "
        "ae_analytics_features.csv Production itself uses."
    )

    if st.button("🚀 Run Complete AE Optimization Study", key="ae_automl_run_study", type="primary"):
        status_box = st.status("Running complete AE optimization study across all 5 horizons…", expanded=True)

        def _cb(step, total, msg):
            status_box.update(label=f"Step {step}/{total} — {msg}")
            status_box.write(f"**Step {step}/{total}:** {msg}")

        try:
            study = ae_research.run_complete_ae_optimization_study(progress_cb=_cb)
            status_box.update(label="Optimization study complete.", state="complete", expanded=False)
            st.session_state["ae_automl_last_study_id"] = study["study_id"]
            rec_count = sum(1 for h in study["horizons"] if study["horizon_results"][str(h)]["recommendation"] == "Promote")
            st.success(f"Study complete across all 5 horizons — {rec_count}/{len(study['horizons'])} horizons recommended for promotion.")
            st.rerun()
        except Exception as exc:
            status_box.update(label="Optimization study failed.", state="error")
            st.error(f"Study failed: {exc}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Study History")
    studies = ae_research.list_ae_studies()
    if not studies:
        st.info("No optimization studies run yet — click the button above to run the first one.")
        return

    def _study_label(s):
        rec_count = sum(1 for h in s["horizons"] if s["horizon_results"][str(h)]["recommendation"] == "Promote")
        promoted_count = sum(1 for v in s.get("promotion_status_by_horizon", {}).values() if v == "promoted")
        return (
            f"{pd.Timestamp(s['started_at']).strftime('%Y-%m-%d %H:%M UTC')} · "
            f"{rec_count}/{len(s['horizons'])} horizons recommended · {promoted_count} promoted"
        )

    study_labels = [_study_label(s) for s in studies]
    default_idx = 0
    last_id = st.session_state.get("ae_automl_last_study_id")
    if last_id:
        for i, s in enumerate(studies):
            if s["study_id"] == last_id:
                default_idx = i
                break
    chosen_label = st.selectbox("Select a study to inspect", study_labels, index=default_idx, key="ae_automl_study_select")
    study = studies[study_labels.index(chosen_label)]
    _render_ae_study_detail(study)


def _render_ae_study_detail(study: dict) -> None:
    st.markdown(
        f"**Study ID:** `{study['study_id'][:8]}…`  ·  "
        f"**Started:** {pd.Timestamp(study['started_at']).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
        f"**Completed:** {pd.Timestamp(study['completed_at']).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    st.markdown("#### Experiment 10 — Cross-Horizon Scientific Synthesis")
    cross = study["cross_horizon_synthesis"]
    if cross["crossover_detected"]:
        st.success(f"✅ {cross['crossover_conclusion']}")
    else:
        st.warning(f"⚠️ {cross['crossover_conclusion']}")

    plot_cols = st.columns(2)
    with plot_cols[0]:
        pih = cross["persistence_importance_vs_horizon"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["horizon"] for r in pih], y=[r["persistence_r2"] for r in pih], mode="lines+markers", name="Persistence R²"))
        fig.add_trace(go.Scatter(x=[r["horizon"] for r in pih], y=[r["winner_r2"] for r in pih], mode="lines+markers", name="Best Model R²"))
        fig.add_trace(go.Scatter(x=[r["horizon"] for r in pih], y=[r["production_r2"] for r in pih], mode="lines+markers", name="Production R²"))
        fig.update_layout(title="Persistence Importance vs. Horizon", xaxis_title="Horizon (h)", yaxis_title="R²", height=380)
        plot_retro(fig)
    with plot_cols[1]:
        piv = cross["physics_importance_vs_horizon"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["horizon"] for r in piv], y=[r["raw_floor_r2"] for r in piv], mode="lines+markers", name="Raw Floor R²"))
        fig.add_trace(go.Scatter(x=[r["horizon"] for r in piv], y=[r["best_single_coupling_variable_r2"] for r in piv], mode="lines+markers", name="Best Single Coupling Var R²"))
        fig.add_trace(go.Scatter(x=[r["horizon"] for r in piv], y=[r["top_ablation_delta_r2"] for r in piv], mode="lines+markers", name="Top Ablation ΔR²"))
        fig.update_layout(title="Physics Importance vs. Horizon", xaxis_title="Horizon (h)", yaxis_title="R² / ΔR²", height=380)
        plot_retro(fig)

    plot_cols2 = st.columns(2)
    with plot_cols2[0]:
        msh = cross["model_skill_vs_horizon"]
        fig = go.Figure(go.Bar(
            x=[r["horizon"] for r in msh], y=[r["winner_r2"] for r in msh],
            text=[r["winner_model"] for r in msh], marker_color="steelblue",
        ))
        fig.update_layout(title="Model Skill vs. Horizon", xaxis_title="Horizon (h)", yaxis_title="Winner R²", height=380)
        plot_retro(fig)
    with plot_cols2[1]:
        fgih = cross["feature_group_importance_vs_horizon"]
        all_groups = sorted({g for r in fgih for g in r["group_shares"]})
        fig = go.Figure()
        for g in all_groups:
            fig.add_trace(go.Bar(name=g, x=[r["horizon"] for r in fgih], y=[r["group_shares"].get(g, 0) for r in fgih]))
        fig.update_layout(barmode="stack", title="Feature Group Importance vs. Horizon", xaxis_title="Horizon (h)", yaxis_title="Share of Importance", height=380)
        plot_retro(fig)

    st.markdown("#### Per-Horizon Detail (Experiments 1-9 + Promotion)")
    horizon_tabs = st.tabs([f"{h}h" for h in study["horizons"]])
    for tab, h in zip(horizon_tabs, study["horizons"]):
        with tab:
            _render_ae_study_horizon_detail(study, h)

    st.markdown("#### Scientific Report")
    report = study["report"]
    with st.expander("📄 Full Scientific Report", expanded=False):
        st.markdown("**Scientific Conclusions**")
        for c in report["scientific_conclusions"]:
            st.markdown(f"- {c}")
        st.markdown("**Future Work**")
        for c in report["future_work"]:
            st.markdown(f"- {c}")
        st.markdown("**Best Models by Horizon**")
        st.json(report["best_models"])
        st.markdown("**Best Feature Sets by Horizon**")
        st.json(report["best_feature_sets"])


def _render_ae_study_horizon_detail(study: dict, horizon: int) -> None:
    hr = study["horizon_results"][str(horizon)]

    step_cols = st.columns(5)
    step_cols[0].metric("1. Baseline R²", f"{hr['baseline']['metrics']['r2']:.4f}", help=hr["baseline"]["model_type"])
    step_cols[1].metric("2. Persistence R²", f"{hr['persistence']['metrics']['r2']:.4f}")
    step_cols[2].metric("3. Raw Floor R²", f"{hr['raw_floor']['metrics']['r2']:.4f}")
    step_cols[3].metric("8. Winner", hr["winner"]["model_type"], help=f"R²={hr['winner']['metrics']['r2']:.4f}")
    step_cols[4].metric("Recommendation", hr["recommendation"])

    detail_tabs = st.tabs([
        "Coupling Physics", "Physics Ablation", "Geomagnetic Memory", "Best Feature Sets",
        "Leaderboard", "Feature Importance", "SHAP / Permutation", "Production Comparison", "Promotion",
    ])

    with detail_tabs[0]:
        st.markdown("**Experiment 4 — Coupling Physics (individually, isolated)**")
        st.caption("Each variable tested ALONE — a pure marginal/standalone-information test, distinct from Experiment 5's baseline-relative ablation.")
        st.dataframe(pd.DataFrame([
            {"Variable": r["name"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4), "RMSE": round(r["rmse"], 4)}
            for r in hr["coupling_results"]
        ]), use_container_width=True, hide_index=True)

    with detail_tabs[1]:
        st.markdown("**Experiment 5 — Physics Engine Ablation (cumulative from Production)**")
        st.caption("Do not assume more variables are better — flat/zero ΔR² across a step means that variable was already fully captured by Production.")
        st.dataframe(pd.DataFrame([
            {
                "Step": r["name"], "R²": round(r["r2"], 4), "ΔR² vs Baseline": round(r["delta_r2_from_baseline"], 4),
                "Already in Baseline": ", ".join(r["already_in_baseline"]) or "—",
            }
            for r in hr["ablation_results"]
        ]), use_container_width=True, hide_index=True)

    with detail_tabs[2]:
        st.markdown("**Experiment 6 — Geomagnetic Memory**")
        st.caption("Does Previous Kp/Dst carry information about future AE beyond AE's own persistence?")
        st.dataframe(pd.DataFrame([
            {"Combination": r["name"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4), "RMSE": round(r["rmse"], 4)}
            for r in hr["geomag_results"]
        ]), use_container_width=True, hide_index=True)

    with detail_tabs[3]:
        st.markdown("**Experiment 7 — Best Combined Feature Sets**")
        st.dataframe(pd.DataFrame([
            {"Feature Set": r["name"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4), "RMSE": round(r["rmse"], 4)}
            for r in hr["combo_results"]
        ]), use_container_width=True, hide_index=True)
        st.success(f"Best feature set for {horizon}h: **{hr['best_combo']['name']}** (R²={hr['best_combo']['r2']:.4f}) — used for Experiment 8's model sweep below.")

    with detail_tabs[4]:
        st.markdown("**Experiment 8 — Model Comparison Leaderboard**")
        lb_rows = [
            {
                "Rank": r["rank"], "Model": r["model_type"], "R²": round(r["r2"], 4), "MAE": round(r["mae"], 4),
                "RMSE": round(r["rmse"], 4), "MAPE (%)": round(r["mape"], 2) if r["mape"] is not None else None,
                "Bias": round(r["bias"], 4),
                "Train Time (s)": round(r["training_time_sec"], 3) if r["training_time_sec"] is not None else None,
                "Inference (ms/sample)": round(r["inference_time_ms_per_sample"], 4) if r["inference_time_ms_per_sample"] is not None else None,
                "Model Size (KB)": round(r["model_size_kb"], 1) if r["model_size_kb"] is not None else None,
                "Features": r["feature_count"],
            }
            for r in hr["leaderboard"]
        ]
        st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)
        if hr.get("failed_candidates"):
            with st.expander(f"⚠️ {len(hr['failed_candidates'])} candidate(s) failed to train"):
                for f in hr["failed_candidates"]:
                    st.caption(f"**{f['model_type']}**: {f['error']}")

    with detail_tabs[5]:
        st.markdown("**Experiment 9 — Feature Importance (winning / best-available candidate)**")
        if hr.get("feature_importance"):
            fi_df = pd.DataFrame(hr["feature_importance"], columns=["Feature", "Importance"]).head(25)
            fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h", marker_color="steelblue"))
            fig.update_layout(title=f"Top 25 Features by Importance ({horizon}h)", height=520, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
        else:
            st.info("Feature importance not available for any candidate at this horizon.")

    with detail_tabs[6]:
        shap_result = hr.get("shap_result") or {}
        st.markdown("**SHAP Analysis**")
        if shap_result.get("supported"):
            shap_df = pd.DataFrame(shap_result["shap_importance"], columns=["Feature", "Mean |SHAP|"]).head(25)
            fig = go.Figure(go.Bar(x=shap_df["Mean |SHAP|"], y=shap_df["Feature"], orientation="h", marker_color="indianred"))
            fig.update_layout(title=f"Top 25 Features by Mean |SHAP| ({horizon}h)", height=520, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
        else:
            st.info(shap_result.get("skipped_reason", "SHAP analysis not available at this horizon."))

        st.markdown("**Permutation Importance (SVR/MLP fallback)**")
        perm_result = hr.get("permutation_result") or {}
        if perm_result.get("supported"):
            perm_df = pd.DataFrame(perm_result["permutation_importance"], columns=["Feature", "Importance (ΔR²)"]).head(25)
            fig = go.Figure(go.Bar(x=perm_df["Importance (ΔR²)"], y=perm_df["Feature"], orientation="h", marker_color="darkorange"))
            fig.update_layout(title=f"Top 25 Features by Permutation Importance ({horizon}h)", height=520, yaxis=dict(autorange="reversed"))
            plot_retro(fig)
        else:
            st.info(perm_result.get("skipped_reason", "Permutation importance not available at this horizon."))

    with detail_tabs[7]:
        st.markdown("**Production Comparison**")
        comp = hr["production_comparison"]
        if comp["current"] is None:
            st.warning(f"No production ae_{horizon}h model currently on disk to compare against.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Current Production**")
                st.metric("Algorithm", comp["current"]["algorithm"])
                st.metric("R²", f"{comp['current']['r2']:.4f}")
                st.metric("MAE", f"{comp['current']['mae']:.4f}")
                st.metric("RMSE", f"{comp['current']['rmse']:.4f}")
            with c2:
                st.markdown("**Candidate**")
                st.metric("Algorithm", comp["candidate"]["algorithm"])
                st.metric("R²", f"{comp['candidate']['r2']:.4f}", delta=f"{comp['delta_r2']:+.4f}")
                st.metric("MAE", f"{comp['candidate']['mae']:.4f}", delta=f"{comp['delta_mae']:+.4f}", delta_color="inverse")
                st.metric("RMSE", f"{comp['candidate']['rmse']:.4f}", delta=f"{comp['delta_rmse']:+.4f}", delta_color="inverse")
            with c3:
                st.markdown("**Verdict**")
                if hr["recommendation"] == "Promote":
                    st.success("✅ Promote")
                else:
                    st.warning("⏸ Keep Current Production")

    with detail_tabs[8]:
        st.markdown(f"**Promotion Criteria Checklist — {horizon}h**")
        for item in hr["promotion_check"]["checklist"]:
            icon = "✅" if item["passed"] else "❌"
            st.markdown(f"{icon} **{item['criterion']}** — {item['detail']}")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        status = study.get("promotion_status_by_horizon", {}).get(str(horizon), "pending")
        if status == "promoted":
            promoted_at = study.get("promoted_at_by_horizon", {}).get(str(horizon))
            st.success(
                "This horizon's winning candidate was already promoted to production"
                + (f" at {pd.Timestamp(promoted_at).strftime('%Y-%m-%d %H:%M UTC')}." if promoted_at else ".")
            )
        elif status == "rejected":
            st.info("This horizon was reviewed and Kept Current Production.")
        else:
            eligible = hr["promotion_check"]["eligible"]
            notes = st.text_area("Promotion notes (optional)", key=f"ae_automl_promo_notes_{study['study_id']}_{horizon}")
            pc1, pc2 = st.columns(2)
            with pc1:
                confirm = st.checkbox(
                    f"I understand this will overwrite the production ae_{horizon}h model (a rollback archive will be created)",
                    key=f"ae_automl_promo_confirm_{study['study_id']}_{horizon}",
                    disabled=not eligible,
                )
                if st.button(
                    f"🚀 Promote {horizon}h Candidate to Production", key=f"ae_automl_promote_{study['study_id']}_{horizon}",
                    type="primary", disabled=not (eligible and confirm),
                ):
                    with st.spinner(f"Promoting {horizon}h model to production…"):
                        try:
                            result = ae_research.promote_ae_to_production(hr["winner"]["run_id"], horizon, notes=notes)
                            ae_research.mark_ae_study_promoted(study["study_id"], horizon, hr["winner"]["run_id"])
                            st.success(f"Promoted! Archive: `{result['archive_path']}`")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Promotion failed: {exc}")
                if not eligible:
                    st.caption("Promotion disabled — one or more criteria above failed.")
            with pc2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button(f"⏸ Keep Current Production ({horizon}h)", key=f"ae_automl_reject_{study['study_id']}_{horizon}"):
                    ae_research.mark_ae_study_rejected(study["study_id"], horizon, notes=notes)
                    st.rerun()


def render_ae_optimization_study() -> None:
    """10-experiment structured study to discover the best scientifically
    defensible Production AE forecasting model, INDEPENDENTLY at all 5
    production horizons (1h/3h/6h/12h/24h) — the flagship scientific
    component of this project. The objective is NOT simply to maximize
    R² — it is to understand what fundamentally determines AE
    predictability across forecast horizons (persistence vs. raw solar
    wind vs. coupling physics vs. geomagnetic memory) while discovering
    the strongest defensible Production model per horizon along the way.

      1  Production Baseline — reproduce each horizon's production model exactly
      2  Persistence Benchmark — naïve lower bound, per horizon
      3  Solar Wind + IMF Raw Floor — raw explanatory floor, no coupling/memory
      4  Coupling Physics — 14 variables tested individually, isolated
      5  Physics Engine Ablation — structured cumulative addition from Production
      6  Geomagnetic Memory — Previous AE/Kp/Dst, individually and combined
      7  Best Combined Feature Sets — per horizon
      8  Model Comparison — all 12 model types on the winning feature set
      9  Feature Importance / SHAP / Permutation Importance
      10 Cross-Horizon Scientific Synthesis — persistence vs. physics crossover
    """
    st.info(
        "**AE Production Model Optimization Study** — 10 structured experiments, run independently "
        "at all 5 production horizons, to answer: how much of AE predictability comes from "
        "persistence, upstream solar wind, coupling physics, and geomagnetic memory — and does that "
        "balance shift as the forecast horizon grows? AE has 5 INDEPENDENTLY trained production "
        "models (ae_1h through ae_24h), unlike Bz/Kp's single target — so every experiment runs once "
        "per horizon with identical methodology, and Experiment 10 compares all 5 results together."
    )

    _render_ae_automl_section()
    st.markdown("---")
    st.markdown(
        "### 🔬 Manual Experiments — Exp 1 through Exp 10\n"
        "Run any experiment individually, at a horizon of your choosing, for hands-on investigation. "
        "This is exactly what **Run Complete AE Optimization Study** above orchestrates automatically "
        "at all 5 horizons — nothing here changes."
    )

    exp_tabs = st.tabs([
        "Exp 1 · Baseline", "Exp 2 · Persistence", "Exp 3 · Raw Floor", "Exp 4 · Coupling Physics",
        "Exp 5 · Physics Ablation", "Exp 6 · Geomag Memory", "Exp 7 · Best Feature Sets",
        "Exp 8 · Model Comparison", "Exp 9 · Importance & SHAP", "Exp 10 · Promote",
    ])

    # ── Exp 1 · Production Baseline ─────────────────────────────────────────
    with exp_tabs[0]:
        st.markdown("##### Experiment 1 — Reproduce Production Baseline")
        horizon1 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp1_horizon")
        baseline_model = ae_research.PRODUCTION_BASELINE_MODEL_BY_HORIZON[horizon1]
        st.caption(
            f"Train {baseline_model} on Solar Wind + IMF + Persistence + Ey/VBz/Dynamic Pressure "
            f"(verified directly against models/ae/metrics.json's stored feature_columns — NOT this "
            f"lab's broader default toggles, which also include Clock Angle/Southward Duration/"
            f"Integrated Southward Bz that production's ae_{horizon1}h model does not actually use) "
            f"— the exact configuration production trained on."
        )
        if st.button("▶ Run Baseline", key="aeopt_exp1_run", type="primary"):
            with st.spinner(f"Training {baseline_model} baseline at {horizon1}h…"):
                try:
                    run = ae_research.train_ae_research_model(
                        baseline_model, horizon=horizon1,
                        feature_toggles=ae_research._production_baseline_toggles(),
                        experiment_tag=f"exp1_baseline_h{horizon1}",
                        notes=f"Manual Exp 1 — Production Baseline reproduction at {horizon1}h",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp1_runs = [r for r in ae_research.list_runs() if r.get("experiment_tag") == f"exp1_baseline_h{horizon1}"]
        if exp1_runs:
            st.dataframe(pd.DataFrame([
                {"R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4), "RMSE": round(r["metrics"]["rmse"], 4),
                 "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp1_runs
            ]), use_container_width=True, hide_index=True)

        prod_metrics1 = ae_research.get_production_ae_metrics(horizon1)
        if prod_metrics1:
            st.caption(f"Current production ae_{horizon1}h: {prod_metrics1['algorithm']} · R²={prod_metrics1['r2']:.4f} · MAE={prod_metrics1['mae']:.4f}")

    # ── Exp 2 · Persistence Benchmark ───────────────────────────────────────
    with exp_tabs[1]:
        st.markdown("##### Experiment 2 — Persistence Benchmark")
        horizon2 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp2_horizon")
        st.caption(f"Naïve forecast: AE at t+{horizon2}h = AE now. Stored permanently.")
        bench_run = next((r for r in ae_research.list_runs() if r.get("run_id") == f"persistence_ae_{horizon2}h"), None)
        if st.button("▶ Compute Persistence Benchmark", key="aeopt_exp2_run", type="primary"):
            with st.spinner(f"Computing persistence benchmark at {horizon2}h…"):
                try:
                    bench_run = ae_research.compute_ae_persistence_benchmark(horizon2)
                    st.success(f"Persistence baseline — R²={bench_run['metrics']['r2']:.4f}, MAE={bench_run['metrics']['mae']:.4f}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")
        if bench_run:
            m = bench_run["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{m['r2']:.4f}")
            c2.metric("MAE", f"{m['mae']:.4f}")
            c3.metric("RMSE", f"{m['rmse']:.4f}")
            c4.metric("Bias", f"{m['bias']:.4f}")
            st.caption(f"Test samples: {bench_run['n_test_samples']:,} · Computed: {pd.Timestamp(bench_run['trained_at']).strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            st.info("Click above to compute and store the persistence benchmark for this horizon.")

    # ── Exp 3 · Solar Wind + IMF Raw Floor ───────────────────────────────────
    with exp_tabs[2]:
        st.markdown("##### Experiment 3 — Solar Wind + IMF Raw Floor")
        horizon3 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp3_horizon")
        model3 = st.selectbox("Model", ae_research.TABULAR_MODELS, key="aeopt_exp3_model")
        st.caption(
            "Only Speed/Density/Temperature/Bt/Bx/By/Bz (+ their lags/rolling/change) — no "
            "persistence, no Derived Physics, no coupling functions. Establishes the raw "
            "explanatory floor before any physics or memory is introduced."
        )
        if st.button("▶ Train Raw Floor", key="aeopt_exp3_run", type="primary"):
            with st.spinner(f"Training {model3} raw floor at {horizon3}h…"):
                try:
                    run = ae_research.train_ae_research_model(
                        model3, horizon=horizon3,
                        feature_toggles=_ae_opt_isolated_toggles(ae_research.SOLAR_WIND_IMF_RAW_FLOOR_GROUPS),
                        engineered_groups=ae_research.default_engineered_toggles(), physics_features={},
                        experiment_tag=f"exp3_raw_floor_h{horizon3}",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp3_runs = [r for r in ae_research.list_runs() if r.get("experiment_tag") == f"exp3_raw_floor_h{horizon3}"]
        if exp3_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp3_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 3 runs yet at this horizon.")

    # ── Exp 4 · Coupling Physics ─────────────────────────────────────────────
    with exp_tabs[3]:
        st.markdown("##### Experiment 4 — Coupling Physics (individually, isolated)")
        st.caption("Each of 14 coupling variables tested ALONE — a pure marginal/standalone-information test.")
        horizon4 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp4_horizon")
        combo_names4 = [c["name"] for c in ae_research.COUPLING_PHYSICS_GRID]
        col4a, col4b = st.columns(2)
        with col4a:
            combo4 = st.selectbox("Variable", combo_names4, key="aeopt_exp4_combo")
        with col4b:
            model4 = st.selectbox("Model", ae_research.TABULAR_MODELS, key="aeopt_exp4_model")
        if st.button("▶ Train This Variable", key="aeopt_exp4_run", type="primary"):
            entry = next(c for c in ae_research.COUPLING_PHYSICS_GRID if c["name"] == combo4)
            if entry["kind"] == "core_column":
                toggles4 = _ae_opt_isolated_toggles({"Derived Physics": [entry["column"]]})
                physics4 = {}
            else:
                toggles4 = _ae_opt_isolated_toggles({})
                physics4 = {entry["physics_name"]: True}
            with st.spinner(f"Training {model4} on '{combo4}' at {horizon4}h…"):
                try:
                    run = ae_research.train_ae_research_model(
                        model4, horizon=horizon4, feature_toggles=toggles4,
                        engineered_groups=ae_research.default_engineered_toggles(), physics_features=physics4,
                        experiment_tag=f"exp4_coupling_h{horizon4}",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp4_runs = [r for r in ae_research.list_runs() if r.get("experiment_tag") == f"exp4_coupling_h{horizon4}"]
        if exp4_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "Features": ", ".join(r["feature_columns"][:2]) + ("…" if len(r["feature_columns"]) > 2 else ""),
                 "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp4_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 4 runs yet at this horizon.")

    # ── Exp 5 · Physics Engine Ablation ──────────────────────────────────────
    with exp_tabs[4]:
        st.markdown("##### Experiment 5 — Physics Engine Ablation (cumulative from Production)")
        st.caption("Structured cumulative addition from the Production baseline — do not assume more variables are better.")
        horizon5 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp5_horizon")
        step_names5 = [s["name"] for s in ae_research.PHYSICS_ENGINE_ABLATION_STEPS]
        col5a, col5b = st.columns(2)
        with col5a:
            step5 = st.selectbox("Step", step_names5, key="aeopt_exp5_step")
        with col5b:
            model5 = st.selectbox("Model", ae_research.TABULAR_MODELS, key="aeopt_exp5_model")
        if st.button("▶ Train This Step", key="aeopt_exp5_run", type="primary"):
            spec = next(s for s in ae_research.PHYSICS_ENGINE_ABLATION_STEPS if s["name"] == step5)
            with st.spinner(f"Training {model5} on '{step5}' at {horizon5}h…"):
                try:
                    run = ae_research.train_ae_research_model(
                        model5, horizon=horizon5, feature_toggles=ae_research.default_feature_toggles(),
                        engineered_groups=ae_research.default_engineered_toggles(), physics_features=spec["physics_features"],
                        experiment_tag=f"exp5_ablation_h{horizon5}",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp5_runs = [r for r in ae_research.list_runs() if r.get("experiment_tag") == f"exp5_ablation_h{horizon5}"]
        if exp5_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp5_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 5 runs yet at this horizon.")

    # ── Exp 6 · Geomagnetic Memory ────────────────────────────────────────────
    with exp_tabs[5]:
        st.markdown("##### Experiment 6 — Geomagnetic Memory")
        st.caption("Does Previous Kp/Dst carry information about future AE beyond AE's own persistence?")
        horizon6 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp6_horizon")
        combo_names6 = [c["name"] for c in ae_research.GEOMAGNETIC_MEMORY_GRID]
        col6a, col6b = st.columns(2)
        with col6a:
            combo6 = st.selectbox("Combination", combo_names6, key="aeopt_exp6_combo")
        with col6b:
            model6 = st.selectbox("Model", ae_research.TABULAR_MODELS, key="aeopt_exp6_model")
        if st.button("▶ Train This Combination", key="aeopt_exp6_run", type="primary"):
            spec = next(c for c in ae_research.GEOMAGNETIC_MEMORY_GRID if c["name"] == combo6)
            with st.spinner(f"Training {model6} on '{combo6}' at {horizon6}h…"):
                try:
                    run = ae_research.train_ae_research_model(
                        model6, horizon=horizon6, feature_toggles=_ae_opt_isolated_toggles(spec["groups"]),
                        engineered_groups=ae_research.default_engineered_toggles(), physics_features={},
                        include_kp=spec["include_kp"], include_dst=spec["include_dst"],
                        experiment_tag=f"exp6_geomag_h{horizon6}",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp6_runs = [r for r in ae_research.list_runs() if r.get("experiment_tag") == f"exp6_geomag_h{horizon6}"]
        if exp6_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp6_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 6 runs yet at this horizon.")

    # ── Exp 7 · Best Combined Feature Sets ───────────────────────────────────
    with exp_tabs[6]:
        st.markdown("##### Experiment 7 — Best Combined Feature Sets")
        horizon7 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp7_horizon")
        combo_grid7 = ae_research._best_combo_grid()
        combo_names7 = [c["name"] for c in combo_grid7]
        col7a, col7b = st.columns(2)
        with col7a:
            combo7 = st.selectbox("Feature Set", combo_names7, key="aeopt_exp7_combo")
        with col7b:
            model7 = st.selectbox("Model", ae_research.TABULAR_MODELS, key="aeopt_exp7_model")
        if st.button("▶ Train This Feature Set", key="aeopt_exp7_run", type="primary"):
            spec = next(c for c in combo_grid7 if c["name"] == combo7)
            with st.spinner(f"Training {model7} on '{combo7}' at {horizon7}h…"):
                try:
                    run = ae_research.train_ae_research_model(
                        model7, horizon=horizon7, feature_toggles=_ae_opt_isolated_toggles(spec["groups"]),
                        engineered_groups=ae_research.default_engineered_toggles(), physics_features=spec["physics_features"],
                        include_kp=spec["include_kp"], include_dst=spec["include_dst"],
                        experiment_tag=f"exp7_bestcombo_h{horizon7}",
                    )
                    st.success(f"R²={run['metrics']['r2']:.4f}, MAE={run['metrics']['mae']:.4f} ({len(run['feature_columns'])} features)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        exp7_runs = [r for r in ae_research.list_runs() if r.get("experiment_tag") == f"exp7_bestcombo_h{horizon7}"]
        if exp7_runs:
            st.dataframe(pd.DataFrame([
                {"Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "Features": len(r["feature_columns"]), "Trained": pd.Timestamp(r["trained_at"]).strftime("%m-%d %H:%M UTC")}
                for r in exp7_runs
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 7 runs yet at this horizon.")

    # ── Exp 8 · Model Comparison ──────────────────────────────────────────────
    with exp_tabs[7]:
        st.markdown("##### Experiment 8 — Model Comparison")
        st.caption("Compare all 12 model types on one chosen feature set at one horizon.")
        horizon8 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp8_horizon")
        combo_grid8 = ae_research._best_combo_grid()
        combo8 = st.selectbox("Feature Set", [c["name"] for c in combo_grid8], key="aeopt_exp8_combo")
        if st.button("▶ Run Full Model Sweep", key="aeopt_exp8_run", type="primary"):
            spec = next(c for c in combo_grid8 if c["name"] == combo8)
            toggles8 = _ae_opt_isolated_toggles(spec["groups"])
            progress8 = st.progress(0.0, text="Starting model sweep…")
            models_to_run = ae_research.TABULAR_MODELS + (ae_research.SEQUENCE_MODELS if ae_research.KERAS_AVAILABLE else [])
            for i, model_type in enumerate(models_to_run):
                progress8.progress((i + 1) / len(models_to_run), text=f"Training {model_type}…")
                try:
                    ae_research.train_ae_research_model(
                        model_type, horizon=horizon8, feature_toggles=toggles8,
                        engineered_groups=ae_research.default_engineered_toggles(), physics_features=spec["physics_features"],
                        include_kp=spec["include_kp"], include_dst=spec["include_dst"],
                        experiment_tag=f"exp8_modelsweep_h{horizon8}",
                    )
                except Exception as exc:
                    st.warning(f"{model_type} failed: {exc}")
            progress8.empty()
            st.rerun()
        exp8_runs = [r for r in ae_research.list_runs() if r.get("experiment_tag") == f"exp8_modelsweep_h{horizon8}"]
        if exp8_runs:
            exp8_runs = sorted(exp8_runs, key=lambda r: -r["metrics"]["r2"])
            st.dataframe(pd.DataFrame([
                {"Rank": i + 1, "Model": r["model_type"], "R²": round(r["metrics"]["r2"], 4), "MAE": round(r["metrics"]["mae"], 4),
                 "RMSE": round(r["metrics"]["rmse"], 4), "Train Time (s)": round(r["training_time_sec"], 3) if r["training_time_sec"] else None}
                for i, r in enumerate(exp8_runs)
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No Exp 8 runs yet at this horizon.")

    # ── Exp 9 · Feature Importance / SHAP / Permutation ─────────────────────
    with exp_tabs[8]:
        st.markdown("##### Experiment 9 — Feature Importance, SHAP, and Permutation Importance")
        horizon9 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp9_horizon")
        candidates9 = [r for r in ae_research.list_runs() if r.get("horizon") == horizon9 and r.get("model_path")]
        if not candidates9:
            st.info("No trained runs with a saved model exist at this horizon yet — train something in Exp 1-8 first.")
        else:
            def _lbl9(r):
                return f"{r['model_type']} · R²={r['metrics']['r2']:.4f} · {r.get('experiment_tag', '—')} · {pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M UTC')}"
            options9 = {_lbl9(r): r for r in sorted(candidates9, key=lambda r: -r["metrics"]["r2"])}
            chosen_lbl9 = st.selectbox("Run", list(options9), key="aeopt_exp9_run_select")
            chosen9 = options9[chosen_lbl9]

            if chosen9.get("feature_importance"):
                st.markdown("**Native Feature Importance**")
                fi_df = pd.DataFrame(chosen9["feature_importance"], columns=["Feature", "Importance"]).head(25)
                fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h", marker_color="steelblue"))
                fig.update_layout(title="Top 25 Features by Importance", height=480, yaxis=dict(autorange="reversed"))
                plot_retro(fig)

            col9a, col9b = st.columns(2)
            with col9a:
                if st.button("▶ Compute SHAP", key="aeopt_exp9_shap_btn"):
                    with st.spinner("Computing SHAP values…"):
                        try:
                            st.session_state["aeopt_exp9_shap_result"] = ae_research.compute_shap_importance_ae(chosen9["run_id"])
                        except Exception as exc:
                            st.error(f"SHAP failed: {exc}")
                shap_res9 = st.session_state.get("aeopt_exp9_shap_result")
                if shap_res9 and shap_res9.get("run_id") == chosen9["run_id"]:
                    if shap_res9.get("supported"):
                        shap_df9 = pd.DataFrame(shap_res9["shap_importance"], columns=["Feature", "Mean |SHAP|"]).head(20)
                        fig = go.Figure(go.Bar(x=shap_df9["Mean |SHAP|"], y=shap_df9["Feature"], orientation="h", marker_color="indianred"))
                        fig.update_layout(title="Top 20 by Mean |SHAP|", height=440, yaxis=dict(autorange="reversed"))
                        plot_retro(fig)
                    else:
                        st.info(shap_res9["skipped_reason"])
            with col9b:
                if st.button("▶ Compute Permutation Importance", key="aeopt_exp9_perm_btn"):
                    with st.spinner("Computing permutation importance…"):
                        try:
                            st.session_state["aeopt_exp9_perm_result"] = ae_research.compute_permutation_importance_ae(chosen9["run_id"])
                        except Exception as exc:
                            st.error(f"Permutation importance failed: {exc}")
                perm_res9 = st.session_state.get("aeopt_exp9_perm_result")
                if perm_res9 and perm_res9.get("run_id") == chosen9["run_id"]:
                    if perm_res9.get("supported"):
                        perm_df9 = pd.DataFrame(perm_res9["permutation_importance"], columns=["Feature", "Importance (ΔR²)"]).head(20)
                        fig = go.Figure(go.Bar(x=perm_df9["Importance (ΔR²)"], y=perm_df9["Feature"], orientation="h", marker_color="darkorange"))
                        fig.update_layout(title="Top 20 by Permutation Importance", height=440, yaxis=dict(autorange="reversed"))
                        plot_retro(fig)
                    else:
                        st.info(perm_res9["skipped_reason"])

    # ── Exp 10 · Promote ──────────────────────────────────────────────────────
    with exp_tabs[9]:
        st.markdown("##### Experiment 10 — Promote a Manual Run to Production")
        st.caption(
            "For the full Cross-Horizon Scientific Synthesis, use the Automated Optimization section "
            "above — this tab only lets you promote an individual manual run at a chosen horizon."
        )
        horizon10 = st.selectbox("Horizon", ae_research.HORIZON_OPTIONS, key="aeopt_exp10_horizon")
        promotable10 = [
            r for r in ae_research.list_runs()
            if r.get("horizon") == horizon10 and r.get("model_type") not in ae_research.SEQUENCE_MODELS and r.get("model_path")
        ]
        if not promotable10:
            st.info("No promotable manual runs yet at this horizon.")
        else:
            def _promo_lbl10(r):
                return f"{r.get('experiment_tag') or '—'} · {r['model_type']} · R²={r['metrics']['r2']:.4f} · MAE={r['metrics']['mae']:.4f} · {pd.Timestamp(r['trained_at']).strftime('%m-%d %H:%M UTC')}"
            promo_options10 = {_promo_lbl10(r): r for r in sorted(promotable10, key=lambda r: -r["metrics"]["r2"])}
            chosen_promo_lbl10 = st.selectbox("Select run to promote", list(promo_options10), key="aeopt_exp10_run")
            chosen_promo10 = promo_options10[chosen_promo_lbl10]

            prod_metrics10 = ae_research.get_production_ae_metrics(horizon10)
            check10 = ae_research.check_promotion_criteria_ae(chosen_promo10, prod_metrics10, horizon10)
            for item in check10["checklist"]:
                icon = "✅" if item["passed"] else "❌"
                st.markdown(f"{icon} {item['criterion']} — {item['detail']}")

            promo_notes10 = st.text_area("Promotion notes (optional)", key="aeopt_exp10_notes")
            confirm10 = st.checkbox(
                f"I understand this will overwrite the production ae_{horizon10}h model (a rollback archive will be created)",
                key="aeopt_exp10_confirm", disabled=not check10["eligible"],
            )
            if st.button("🚀 Promote to Production", key="aeopt_exp10_promote", type="primary",
                         disabled=not (check10["eligible"] and confirm10)):
                with st.spinner(f"Promoting {horizon10}h model to production…"):
                    try:
                        result = ae_research.promote_ae_to_production(chosen_promo10["run_id"], horizon10, notes=promo_notes10)
                        st.success(f"Promoted! Archive: `{result['archive_path']}`")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Promotion failed: {exc}")
            if not check10["eligible"]:
                st.caption("Promotion disabled — one or more criteria above failed.")
