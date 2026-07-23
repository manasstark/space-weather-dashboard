"""The Operational Forecast Engine's three public entry points.

run_forecast_cycle(): ensures a job exists for every (dataset, variable,
horizon) in matrix.PRODUCTION_MATRIX — this is what removes the "must
click Start Prediction" requirement. Uses jobs.has_active_job() as a
cheap precheck so the expensive jobs.start_job() (which always calls
predict_live first) only runs when nothing is already tracking that
combination. Because a new job is only created once the previous one
completes, this already gives each (dataset, variable, horizon) a
natural forecast cycle paced by its own horizon — a 1h job cycles about
hourly, a 24h job about daily.

evaluate_due_forecasts(): a thin, documented wrapper around
jobs.tick_all_active_jobs() — kept separate purely for API symmetry
(run_forecast_cycle creates work, evaluate_due_forecasts advances it).

refresh_dashboard_products(): reads current job state + fresh live
readings, computes confidence/physics/outlook/alerts, and writes the
storage products in swdss.engine.storage — the ONLY thing dashboard/lib/
command_centre.py reads. Never starts or advances a job itself.

LOCKED FORECASTS — a forecast is displayed exactly as it was at
generation time, never as it currently stands. jobs.py's ticks[] array
is a continuous internal refinement (fed by every new NOAA minute while
a job is "in_progress" — see jobs._advance_predicting), which is exactly
right for research/drift-tracking, but wrong for an operational display:
an operator needs one stable number to act on, not one that silently
changes underneath them every cycle. So every forecast entry below uses
ticks[0] (the value computed at job/cycle creation — see jobs.start_job)
or, for Kp, job["production_prediction"] (the frozen Mode 1 value,
already never touched again after creation) — never ticks[-1]. Nothing
in jobs.py changes: its internal ticking still runs for the Research
Labs and quicklook tracking; only what this engine chooses to SHOW as
"the forecast" changes.

None of these three runs as a daemon on its own — see live_update.py for
how they're called once per its existing ~60s loop cycle. They remain
independently callable for manual/dev use.
"""

import json
import traceback

import pandas as pd

from swdss.engine import explanation, packages, storage
from swdss.engine.alerts import build_alerts
from swdss.engine.confidence import score_forecast_confidence
from swdss.engine.drift import detect_drift
from swdss.engine.labels import classify_current_reading
from swdss.engine.matrix import PRODUCTION_MATRIX
from swdss.engine.outlook import classify_activity_regime, classify_overall_outlook
from swdss.engine.physics_snapshot import build_physics_snapshot
from swdss.models import jobs as jobs_module
from swdss.models import predict
from swdss.models.jobs import (
    DB_PATH,
    compute_variable_metrics,
    get_running_jobs,
    has_active_job,
    prune_old_engine_jobs,
    stability_metric,
    start_job,
)
from swdss.models.registry import metrics_path
from swdss.paths import PROCESSED_DIR

ENGINE_VERSION = "1.0"

# Reference variable used to detect data-feed staleness per dataset, for
# the freshness section of the snapshot.
FRESHNESS_CHECKS = {
    "solar_wind": ("solar_wind", "speed", 90),
    "imf": ("imf", "bz_gsm", 90),
    "kp": ("kp", "kp", 360),
    "dst": ("dst", "dst", 120),
    "ae": ("ae", "ae", 1500),
}

# variable name (as shown in current_conditions) -> (dataset, raw column)
CURRENT_CONDITIONS_SOURCES = {
    "speed": ("solar_wind", "speed"),
    "density": ("solar_wind", "density"),
    "temperature": ("solar_wind", "temperature"),
    "bt": ("imf", "bt"),
    "bx_gsm": ("imf", "bx_gsm"),
    "by_gsm": ("imf", "by_gsm"),
    "bz_gsm": ("imf", "bz_gsm"),
    "kp": ("kp", "kp"),
    "dst": ("dst", "dst"),
    "ae": ("ae", "ae"),
}

# Variables shown as a range (predicted_value ± the model's own held-out
# MAE) rather than a single precise decimal — the two the brief singles
# out as higher-uncertainty. MAE is a real, already-computed number (the
# model's own typical error), not a fabricated interval.
RANGE_VARIABLES = {"bz_gsm", "kp"}

_metrics_cache = {}


def _load_dataset_metrics(dataset: str) -> dict:
    if dataset not in _metrics_cache:
        try:
            with open(metrics_path(dataset)) as f:
                _metrics_cache[dataset] = json.load(f)
        except Exception:
            _metrics_cache[dataset] = {}
    return _metrics_cache[dataset]


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _now_iso() -> str:
    return _now().isoformat()


def run_forecast_cycle() -> dict:
    started_at = _now_iso()
    started = []
    skipped = 0
    errors = []

    for dataset, variable, horizons in PRODUCTION_MATRIX:
        # Kp's "interval" horizon isn't one of HORIZONS — start_job/
        # prediction_panel both use horizon=1 as a placeholder for it
        # (predict_kp_interval ignores the argument and always targets
        # the next official NOAA interval); matching that exact
        # convention here keeps an engine-started Kp job indistinguishable
        # from a manually-started one.
        job_horizons = horizons if isinstance(horizons, list) else [1]

        for horizon in job_horizons:
            try:
                if has_active_job(dataset, variable, horizon, source="engine"):
                    skipped += 1
                    continue
                job, created = start_job(dataset, variable, horizon, source="engine")
                if created:
                    started.append(job["job_id"])
                else:
                    skipped += 1
            except Exception as exc:
                errors.append({"dataset": dataset, "variable": variable, "horizon": horizon, "error": str(exc)})
                traceback.print_exc()

    # Only 'engine'-sourced, terminal-state jobs older than the retention
    # window are ever removed — manual jobs are never touched, regardless
    # of age (see prune_old_engine_jobs' own docstring). Without this, an
    # engine that now creates jobs automatically forever would grow
    # predictions.db without bound.
    try:
        pruned = prune_old_engine_jobs()
    except Exception:
        pruned = 0
        traceback.print_exc()

    completed_at = _now_iso()
    result = {"ran_at": started_at, "completed_at": completed_at, "started": started, "skipped": skipped, "errors": errors, "pruned": pruned}
    storage.append_log_lines([{
        "ts": completed_at,
        "level": "ERROR" if errors else "INFO",
        "stage": "run_forecast_cycle",
        "message": f"Started {len(started)} job(s), skipped {skipped}, pruned {pruned} old engine job(s), {len(errors)} error(s).",
        "errors": errors,
    }])
    return result


def evaluate_due_forecasts() -> dict:
    ran_at = _now_iso()
    jobs_module.tick_all_active_jobs()
    storage.append_log_lines([{"ts": ran_at, "level": "INFO", "stage": "evaluate_due_forecasts", "message": "Ticked all active jobs."}])
    return {"ran_at": ran_at}


def _valid_window(dataset: str, variable: str, target_hour, horizon) -> tuple:
    """The window a forecast is FOR — target_hour is the instant the
    model predicts; the valid period is that instant through the next
    boundary (1h for every hourly-checkpoint job, 3h for Kp's official
    NOAA interval).
    """
    target_hour = pd.Timestamp(target_hour)
    if target_hour.tzinfo is None:
        target_hour = target_hour.tz_localize("UTC")
    span_hours = 3 if (dataset in ("analytics", "experimental") and variable == "kp") else 1
    return target_hour, target_hour + pd.Timedelta(hours=span_hours)


def _lifecycle_status(job_status: str, actual_value, valid_start: pd.Timestamp, now: pd.Timestamp) -> str:
    if job_status == "completed":
        return "VERIFIED" if actual_value is not None else "AWAITING EVALUATION"
    if job_status == "evaluating":
        return "ACTIVE"
    # in_progress
    return "LIVE" if now < valid_start else "ACTIVE"


def _trend_word(variable: str, current_value, predicted_value) -> str:
    if current_value is None or predicted_value is None:
        return "—"
    delta = predicted_value - current_value
    if abs(delta) < 1e-9:
        return "Stable"
    if variable == "bz_gsm":
        return "More Southward" if delta < 0 else "Less Southward"
    return "Increasing" if delta > 0 else "Decreasing"


def _build_forecast_entry(dataset: str, variable: str, horizon, job: dict, trend_history: list, current_value, current_observed_at) -> dict:
    if job is None:
        return None

    ticks = job.get("ticks") or []
    is_kp_interval = dataset in ("analytics", "experimental") and variable == "kp"

    if is_kp_interval:
        predicted_value = job.get("production_prediction")
        generated_at = job.get("production_observed_at") or job.get("created_at")
    else:
        first_tick = ticks[0] if ticks else None
        predicted_value = first_tick["predicted_value"] if first_tick else None
        generated_at = job.get("created_at")

    metrics = job.get("metrics") or {}
    stability_label, stability_std = stability_metric(job)

    score, category = score_forecast_confidence(
        r2=metrics.get("r2"),
        cv_r2_std=metrics.get("cv_r2_std"),
        horizon=horizon,
        stability_label=stability_label,
        trend=trend_history,
    )

    valid_start, valid_end = _valid_window(dataset, variable, job["target_hour"], horizon)
    now = _now()
    lead_time_minutes = max(0.0, (valid_start - now).total_seconds() / 60)
    actual_value = job.get("actual_value")
    status = _lifecycle_status(job["status"], actual_value, valid_start, now)
    abs_error = abs(predicted_value - actual_value) if predicted_value is not None and actual_value is not None else None
    delta = (predicted_value - current_value) if predicted_value is not None and current_value is not None else None

    range_low = range_high = None
    if variable in RANGE_VARIABLES and predicted_value is not None:
        mae = metrics.get("mae")
        if mae is not None:
            range_low, range_high = predicted_value - mae, predicted_value + mae

    ds_metrics = _load_dataset_metrics(dataset)
    metrics_key = "kp_interval" if is_kp_interval else f"{variable}_{horizon}h"
    trained_at = ds_metrics.get(metrics_key, {}).get("trained_at")

    entry = {
        "job_id": job["job_id"],
        "dataset": dataset,
        "variable": variable,
        "horizon": horizon,
        "status": status,
        "job_status": job["status"],
        "generated_at": generated_at,
        "valid_start": valid_start.isoformat(),
        "valid_end": valid_end.isoformat(),
        "lead_time_minutes": lead_time_minutes,
        "target_hour": job["target_hour"],
        "current_value": current_value,
        "current_observed_at": current_observed_at,
        "predicted_value": predicted_value,
        "range_low": range_low,
        "range_high": range_high,
        "delta": delta,
        "trend": _trend_word(variable, current_value, predicted_value),
        "model_name": job["model_name"],
        "model_trained_at": trained_at,
        "metrics": {k: metrics.get(k) for k in ("r2", "mae", "rmse", "cv_r2_mean", "cv_r2_std", "cv_n_folds")},
        "confidence": {"score": score, "category": category},
        "stability": {"label": stability_label, "std": stability_std},
        "actual_value": actual_value,
        "abs_error": abs_error,
    }

    if dataset == "ae":
        entry["quicklook_ae"] = job.get("quicklook_ae")
        entry["quicklook_confidence"] = job.get("quicklook_confidence")
        entry["quicklook_hour_coverage"] = job.get("quicklook_hour_coverage")

    return entry


def _freshness() -> dict:
    result = {}
    for name, (dataset, variable, max_age) in FRESHNESS_CHECKS.items():
        try:
            ts, _ = predict.latest_minute_observation(dataset, variable)
        except Exception:
            ts = None
        if ts is None:
            result[name] = {"status": "No data", "age": "N/A", "age_minutes": None}
            continue
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_minutes = (_now() - ts).total_seconds() / 60
        status = "Fresh" if age_minutes <= max_age else "Stale"
        age_text = f"{age_minutes:.1f} min old" if age_minutes < 60 else f"{age_minutes / 60:.1f} hr old"
        result[name] = {"status": status, "age": age_text, "age_minutes": age_minutes}
    return result


def _current_conditions() -> dict:
    conditions = {}
    anchor_candidates = []
    for variable, (dataset, raw_var) in CURRENT_CONDITIONS_SOURCES.items():
        try:
            ts, value = predict.latest_minute_observation(dataset, raw_var)
        except Exception:
            ts, value = None, None
        label_col = "bz" if variable == "bz_gsm" else variable
        meaning, risk = classify_current_reading(label_col, value)
        conditions[variable] = {
            "value": value,
            "observed_at": ts.isoformat() if ts is not None else None,
            "meaning": meaning,
            "risk": risk,
        }
        if variable in ("dst", "kp", "bz_gsm") and ts is not None:
            anchor_candidates.append(pd.Timestamp(ts))
    conditions["anchor_time"] = max(anchor_candidates).isoformat() if anchor_candidates else None
    return conditions


def _service_health(freshness: dict, run_forecast_cycle_errors: int, active_jobs_total: int) -> list:
    """Returns a flat list of {name, status, detail} rows for the System
    tab's service table — Forecast Engine / Physics Engine / NOAA /
    Kyoto / DONKI / Database / Scheduler.
    """
    rows = []

    engine_ok = run_forecast_cycle_errors == 0
    rows.append({
        "name": "Forecast Engine",
        "status": "RUNNING" if engine_ok else "DEGRADED",
        "detail": f"{run_forecast_cycle_errors} error(s) last cycle" if not engine_ok else f"{active_jobs_total} job(s) tracked",
    })
    rows.append({"name": "Physics Engine", "status": "RUNNING", "detail": "Descriptive only, no model dependency"})

    noaa_ok = freshness.get("solar_wind", {}).get("status") == "Fresh" and freshness.get("imf", {}).get("status") == "Fresh"
    rows.append({
        "name": "NOAA (Solar Wind / IMF / Kp / Dst)",
        "status": "ONLINE" if noaa_ok else "STALE",
        "detail": f"Solar Wind {freshness.get('solar_wind', {}).get('age', 'N/A')}",
    })

    ae_freshness = freshness.get("ae", {})
    rows.append({
        "name": "Kyoto WDC (AE)",
        "status": "ONLINE" if ae_freshness.get("status") == "Fresh" else "DELAYED",
        "detail": ae_freshness.get("age", "N/A"),
    })

    cme_path = PROCESSED_DIR / "cme" / "cme_processed.parquet"
    if cme_path.exists():
        try:
            cme_df = pd.read_parquet(cme_path, columns=["timestamp_utc"])
            cme_df["timestamp_utc"] = pd.to_datetime(cme_df["timestamp_utc"], utc=True, errors="coerce")
            latest_cme = cme_df["timestamp_utc"].max()
            age_hr = (_now() - latest_cme).total_seconds() / 3600 if pd.notna(latest_cme) else None
            donki_detail = f"{age_hr:.1f} hr since last detection" if age_hr is not None else "No data"
            donki_status = "ONLINE" if age_hr is not None and age_hr <= 48 else "STALE"
        except Exception:
            donki_detail, donki_status = "Read error", "UNKNOWN"
    else:
        donki_detail, donki_status = "No data", "UNKNOWN"
    rows.append({"name": "DONKI (CME)", "status": donki_status, "detail": donki_detail})

    if DB_PATH.exists():
        db_age_min = (_now() - pd.Timestamp(DB_PATH.stat().st_mtime, unit="s", tz="UTC")).total_seconds() / 60
        rows.append({"name": "Database (predictions.db)", "status": "ONLINE", "detail": f"Last write {db_age_min:.1f} min ago"})
    else:
        rows.append({"name": "Database (predictions.db)", "status": "OFFLINE", "detail": "Not yet created"})

    recent_logs = storage.load_recent_logs(3)
    scheduler_detail = recent_logs[0]["ts"] if recent_logs else "No cycles recorded yet"
    rows.append({"name": "Scheduler (live_update loop)", "status": "RUNNING" if recent_logs else "UNKNOWN", "detail": f"Last cycle: {scheduler_detail}"})

    return rows


def refresh_dashboard_products() -> dict:
    current_conditions = _current_conditions()

    forecasts = {"solar_wind": {}, "imf": {}, "analytics": {}, "ae": {}}
    history_rows = []
    evaluation_rows = []
    already_evaluated = storage.already_evaluated_job_ids()

    # source='engine' explicitly — the Forecast Package must only ever
    # reflect the engine's own automatic forecasts, never a job a user
    # happened to start manually for the same (dataset, variable, horizon)
    # from a Production tab at the same time.
    jobs_by_dataset = {}
    for dataset in {ds for ds, _, _ in PRODUCTION_MATRIX}:
        jobs_by_dataset[dataset] = get_running_jobs(dataset, limit=500, source="engine")

    trend_cache = {}
    predicted_kp = predicted_dst = predicted_ae = None
    generated_at = _now_iso()
    active_jobs_total = 0

    for dataset, variable, horizons in PRODUCTION_MATRIX:
        semantic_horizons = horizons if isinstance(horizons, list) else [horizons]
        job_horizons = horizons if isinstance(horizons, list) else [1]

        cache_key = (dataset, variable)
        if cache_key not in trend_cache:
            trend_cache[cache_key] = compute_variable_metrics(dataset, variable)["trend"]
        trend = trend_cache[cache_key]

        current_entry = current_conditions.get(variable, {})
        current_value = current_entry.get("value")
        current_observed_at = current_entry.get("observed_at")

        variable_block = forecasts.setdefault(dataset, {}).setdefault(variable, {})

        for semantic_horizon, job_horizon in zip(semantic_horizons, job_horizons):
            candidates = [j for j in jobs_by_dataset[dataset] if j["variable"] == variable and j["horizon"] == job_horizon]
            active = [j for j in candidates if j["status"] in ("in_progress", "evaluating")]
            if active:
                active_jobs_total += 1
            job = active[0] if active else (candidates[0] if candidates else None)

            horizon_label = f"{semantic_horizon}h" if isinstance(semantic_horizon, int) else semantic_horizon
            entry = _build_forecast_entry(dataset, variable, semantic_horizon, job, trend, current_value, current_observed_at)
            variable_block[horizon_label] = entry

            if entry is not None:
                history_rows.append({
                    "generated_at": generated_at,
                    "dataset": dataset,
                    "variable": variable,
                    "horizon": horizon_label,
                    "job_id": entry["job_id"],
                    "status": entry["status"],
                    "job_status": entry["job_status"],
                    "valid_start": entry["valid_start"],
                    "valid_end": entry["valid_end"],
                    "current_value": entry["current_value"],
                    "predicted_value": entry["predicted_value"],
                    "model_name": entry["model_name"],
                    "r2": entry["metrics"].get("r2"),
                    "mae": entry["metrics"].get("mae"),
                    "confidence_score": entry["confidence"]["score"],
                    "confidence_category": entry["confidence"]["category"],
                    "actual_value": entry["actual_value"],
                    "abs_error": entry["abs_error"],
                })

                if entry["job_status"] == "completed" and entry["actual_value"] is not None and entry["job_id"] not in already_evaluated:
                    signed_error = entry["predicted_value"] - entry["actual_value"]
                    evaluation_rows.append({
                        "job_id": entry["job_id"],
                        "dataset": dataset,
                        "variable": variable,
                        "horizon": horizon_label,
                        "valid_start": entry["valid_start"],
                        "valid_end": entry["valid_end"],
                        "completed_at": job.get("completed_at"),
                        "final_predicted_value": entry["predicted_value"],
                        "actual_value": entry["actual_value"],
                        "error": signed_error,
                        "abs_error": entry["abs_error"],
                        "squared_error": signed_error ** 2,
                        "model_name": entry["model_name"],
                        "confidence_score_at_issuance": entry["confidence"]["score"],
                        "confidence_category_at_issuance": entry["confidence"]["category"],
                        # The model's own held-out training MAE — drift.detect_drift
                        # compares recent live abs_error against this. Activity
                        # regime is filled in below, once this cycle's overall
                        # Kp/Dst/AE outlook is known — see the tagging pass after
                        # this loop.
                        "training_mae": entry["metrics"].get("mae"),
                        "activity_regime": None,
                    })

            is_headline = semantic_horizon == "interval" or semantic_horizon == 1
            if is_headline and entry is not None:
                if dataset == "analytics" and variable == "kp":
                    predicted_kp = entry["predicted_value"]
                elif dataset == "analytics" and variable == "dst":
                    predicted_dst = entry["predicted_value"]
                elif dataset == "ae" and variable == "ae":
                    predicted_ae = entry["predicted_value"]

    try:
        physics_summary = build_physics_snapshot()
    except Exception:
        physics_summary = {}
        traceback.print_exc()

    outlook_level, outlook_reasoning = classify_overall_outlook(
        predicted_kp=predicted_kp, predicted_dst=predicted_dst, predicted_ae=predicted_ae
    )

    # Coarse activity-regime tag for every evaluation logged THIS cycle —
    # an approximation (today's outlook, not the regime at the moment
    # each job was actually issued), collected now purely so a future
    # version has labeled history to compute quiet/active/storm-
    # conditional error bands from, per the engine's forward-compatibility
    # scope — no segmented calculation happens yet (see swdss.engine.
    # outlook.classify_activity_regime).
    activity_regime = classify_activity_regime(predicted_kp=predicted_kp, predicted_dst=predicted_dst, predicted_ae=predicted_ae)
    for row in evaluation_rows:
        row["activity_regime"] = activity_regime

    freshness = _freshness()
    alerts = build_alerts(outlook_level=outlook_level, physics=physics_summary, freshness=freshness)

    # Drift check against ALREADY-persisted history (this cycle's brand
    # new evaluation_rows haven't been written yet) — a one-cycle lag is
    # irrelevant for a signal that's inherently about a sustained trend.
    try:
        drift_alerts = detect_drift(storage.load_evaluation_history())
    except Exception:
        drift_alerts = []
        traceback.print_exc()
    alerts.extend(drift_alerts)

    recent_cycle_errors = 0
    recent_run_cycle_errors = []
    recent_logs = storage.load_recent_logs(5)
    for log in recent_logs:
        if log.get("stage") == "run_forecast_cycle" and log.get("level") == "ERROR":
            recent_cycle_errors += 1
            if not recent_run_cycle_errors:
                recent_run_cycle_errors = log.get("errors") or []
    system_health = _service_health(freshness, recent_cycle_errors, active_jobs_total)

    # Forecast Explanation Engine — connects the Physics Engine's live
    # readings to the Kp/Dst/AE forecasts they drive (see
    # swdss.engine.explanation). Attached directly onto each headline
    # entry so the Forecast tab can show "why" alongside "what", rather
    # than leaving Physics and Forecast as isolated views.
    headline_entries = {
        "kp": forecasts.get("analytics", {}).get("kp", {}).get("interval"),
        "dst": forecasts.get("analytics", {}).get("dst", {}).get("1h"),
        "ae": forecasts.get("ae", {}).get("ae", {}).get("1h"),
    }
    try:
        explanations = explanation.build_explanations(physics_summary, headline_entries)
        for variable, entry in headline_entries.items():
            if entry is not None:
                entry["explanation"] = explanations.get(variable)
    except Exception:
        explanations = {}
        traceback.print_exc()

    completed_at = _now_iso()
    snapshot = {
        "generated_at": completed_at,
        "engine_version": ENGINE_VERSION,
        "overall_outlook": {"level": outlook_level, "reasoning": outlook_reasoning},
        "physics_summary": physics_summary,
        "current_conditions": current_conditions,
        "forecasts": forecasts,
        "freshness": freshness,
        "alerts": alerts,
        "system_health": system_health,
        "active_jobs_total": active_jobs_total,
    }

    # Forecast Package — the synchronized operational product bundling
    # the 10 headline forecasts (see swdss.engine.packages). Pure
    # re-packaging of what was just computed above; starts no job, calls
    # no model.
    try:
        package = packages.build_current_package(
            forecasts, physics_summary, {"level": outlook_level, "reasoning": outlook_reasoning},
            explanations, recent_run_cycle_errors,
        )
        cycle = storage.append_package_history_row(package)
        package["forecast_cycle"] = cycle
        storage.write_current_package(package)
        snapshot["current_package_id"] = package["package_id"]

        verification_summary = packages.build_verification_summary(package)
        if verification_summary is not None and package["package_id"] not in storage.already_verified_package_ids():
            storage.append_package_verification_row(verification_summary)
    except Exception:
        traceback.print_exc()

    storage.write_snapshot(snapshot)
    storage.append_forecast_history_rows(history_rows)
    storage.append_evaluation_rows(evaluation_rows)
    storage.append_log_lines([{
        "ts": completed_at,
        "level": "INFO",
        "stage": "refresh_dashboard_products",
        "message": f"Snapshot refreshed — outlook={outlook_level}, {len(evaluation_rows)} new evaluation(s), {len(alerts)} alert(s).",
    }])

    return snapshot
