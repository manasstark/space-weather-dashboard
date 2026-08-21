"""The Operational Forecast Engine's three public entry points.

run_forecast_cycle(): ensures a job exists for every (dataset, variable,
horizon) in matrix.PRODUCTION_MATRIX — this is what removes the "must
click Start Prediction" requirement. For every horizon EXCEPT 1,
jobs.has_active_job() is used as a cheap precheck so the expensive
jobs.start_job() (which always calls predict_live first) only runs when
nothing is already tracking that combination — a new job there is only
created once the previous one completes, giving each of those a natural
forecast cycle paced by its own horizon (a 6h job cycles about every 6
hours, a 24h job about daily).

Horizon-1 jobs (which includes Kp's interval job, tracked at horizon=1
in jobs.py) are deliberately NOT gated this way: issuing the next hour's
forecast must never wait on the previous hour's still being evaluated,
or the wall-clock hour that gap falls in silently never gets its own
forecast. So the precheck is skipped for horizon==1 — start_job is
always attempted, and its own target-hour-keyed dedup (see
jobs.start_job's continuous_reissue branch) is what prevents duplicates,
while deliberately allowing a not-yet-evaluated job and a freshly issued
one to coexist briefly.

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

from swdss.engine import calibration, explanation, packages, skill, storage
from swdss.engine.alerts import ambient_solar_wind_speed, build_alerts
from swdss.engine.confidence import score_forecast_confidence
from swdss.engine.drift import detect_drift
from swdss.engine.labels import classify_current_reading
from swdss.engine.matrix import PRODUCTION_MATRIX
from swdss.engine.outlook import classify_activity_regime, classify_overall_outlook
from swdss.engine.physics_snapshot import PHYSICS_QUANTITY_KEYS, build_physics_snapshot, physics_completeness
from swdss.models import jobs as jobs_module
from swdss.models import predict
from swdss.models.f107_forecast import forecast_f107
from swdss.models.jobs import (
    DB_PATH,
    compute_variable_metrics,
    get_running_jobs,
    has_active_job,
    prune_old_engine_jobs,
    resolve_actual_value,
    stability_metric,
    start_job,
)
from swdss.models.registry import metrics_path
from swdss.paths import PROCESSED_DIR
from swdss.physics.cme_dbm import estimate_cme_arrival_dbm

SOLAR_FORECAST_CME_LOOKBACK_HOURS = 120
FLARE_CME_OUTLOOK_REFRESH_MINUTES = 20  # JSOC queries + classifier inference are far more expensive than the DBM/F10.7 math above and don't need 60s-cycle freshness

# The permanent forecast_snapshot_history.parquet log only ever needs the
# 10 headline (dataset, variable, horizon) forecasts — the ones the
# Forecast Package and Verification tab's "Operational" split already
# treat as the operationally meaningful set — not the extended horizons
# (3h/6h/12h/24h) on top of them. Extended horizons are still computed
# and shown live on the Forecast tab; they're just never logged, since
# logging every candidate at every horizon every cycle (continuous
# re-issue can mean several candidates per slot) is what grew this file
# to multiple millions of rows.
_HEADLINE_KEY_SET = set(packages.HEADLINE_KEYS)

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

# Every headline variable is shown as a range (predicted_value ± the
# model's own held-out MAE) rather than a single over-precise decimal —
# a point forecast with no uncertainty band is scientifically incomplete
# for an operational product, and the MAE is already a real, computed
# number (the model's own typical error) sitting in every model's
# metrics.json, not a fabricated interval invented for display.
RANGE_VARIABLES = {
    "speed", "density", "temperature", "bt", "bx_gsm", "by_gsm", "bz_gsm",
    "dst", "kp", "ae",
}

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
                # horizon==1 re-issues continuously (see module docstring
                # and jobs.start_job's continuous_reissue branch) — the
                # precheck would otherwise block a new hour's job from
                # starting just because the previous hour's is still
                # waiting to be evaluated, which is exactly the gap this
                # is meant to close.
                if horizon != 1 and has_active_job(dataset, variable, horizon, source="engine"):
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


def _resolve_persistence_anchor(dataset: str, variable: str, created_at):
    """The naive "persistence" forecast's input: the last known hourly-
    mean observation at/before the job's own creation time — i.e. "what
    if we'd just assumed nothing changes between now and the target
    hour." Reuses jobs.resolve_actual_value's existing hourly-mean
    lookup (identical mechanics — an observed value at a given hour) at
    the issuance timestamp instead of the target timestamp, rather than
    adding a second lookup path.

    AE always returns None: it has no live feed (swdss.models.registry's
    static_variables), so there is no real "value known at issuance" to
    persist from — a persistence baseline is scientifically meaningless
    for it, not merely unavailable.
    """
    if dataset == "ae" or created_at is None:
        return None
    try:
        return resolve_actual_value(dataset, variable, created_at)
    except Exception:
        return None


def _trend_word(variable: str, current_value, predicted_value) -> str:
    if current_value is None or predicted_value is None:
        return "—"
    delta = predicted_value - current_value
    if abs(delta) < 1e-9:
        return "Stable"
    if variable == "bz_gsm":
        return "More Southward" if delta < 0 else "Less Southward"
    return "Increasing" if delta > 0 else "Decreasing"


def _build_forecast_entry(
    dataset: str,
    variable: str,
    horizon,
    job: dict,
    trend_history: list,
    current_value,
    current_observed_at,
    current_regime=None,
    regime_mae_lookup: dict = None,
) -> dict:
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

    range_low = range_high = range_source = None
    if variable in RANGE_VARIABLES and predicted_value is not None:
        # Prefer the CURRENT activity regime's own historical MAE (Quiet/
        # Active/Storm — see swdss.engine.calibration.compute_regime_error_
        # bands) over the model's single all-time MAE, which is quiet-
        # dominated and understates the real error band whenever conditions
        # are actually active or stormy (see Storm Backtest findings in
        # README's Known Limitations). Falls back to the all-time MAE
        # whenever there isn't yet a MIN_SAMPLES-worthy regime-specific
        # history to draw from — regime_mae_lookup simply has no entry for
        # that case, so this never guesses off a handful of points.
        horizon_label = f"{horizon}h" if isinstance(horizon, int) else horizon
        mae = None
        if current_regime is not None and regime_mae_lookup is not None:
            mae = regime_mae_lookup.get((dataset, variable, horizon_label, current_regime))
            if mae is not None:
                range_source = current_regime
        if mae is None:
            mae = metrics.get("mae")
            range_source = "All-Time"
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
        "range_source": range_source,
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


def _format_age(age_minutes: float) -> str:
    """Minutes for anything under an hour, hours under a day, and days
    beyond that — AE's Kyoto WDC lag genuinely runs to months in this
    project, and reporting that as "4728.0 hr old" is unreadable and
    actively hides how stale the source is from an operator glancing at
    the System tab.
    """
    if age_minutes < 60:
        return f"{age_minutes:.1f} min old"
    if age_minutes < 1440:
        return f"{age_minutes / 60:.1f} hr old"
    return f"{age_minutes / 1440:.1f} days old"


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
        age_text = _format_age(age_minutes)
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


def _service_health(freshness: dict, run_forecast_cycle_errors: int, active_jobs_total: int, physics_summary: dict) -> list:
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

    # Reports on the HEALTH of the computed physics, not just whether the
    # engine process ran — a quantity can be silently None (e.g. no
    # dynamic pressure because density was unavailable) even while
    # build_physics_snapshot() itself completes without raising.
    completeness = physics_completeness(physics_summary) if physics_summary else {"available": 0, "total": len(PHYSICS_QUANTITY_KEYS), "missing": list(PHYSICS_QUANTITY_KEYS)}
    physics_ok = completeness["available"] == completeness["total"]
    if physics_ok:
        physics_detail = f"Complete — {completeness['available']}/{completeness['total']} variables"
    else:
        missing_preview = ", ".join(completeness["missing"][:3])
        if len(completeness["missing"]) > 3:
            missing_preview += f", +{len(completeness['missing']) - 3} more"
        physics_detail = f"{completeness['available']}/{completeness['total']} variables — missing {missing_preview}"
    rows.append({"name": "Physics Engine", "status": "RUNNING" if physics_ok else "DEGRADED", "detail": physics_detail})

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


def _solar_forecast_snapshot() -> dict:
    """The Solar Forecast tab's only data source (see storage.py's
    write_solar_forecast) — F10.7's harmonic-regression outlook
    (swdss.models.f107_forecast) and the Drag-Based Model arrival
    estimate for every recent CME (swdss.physics.cme_dbm), computed once
    here per engine cycle rather than on every dashboard render, the same
    "engine computes, dashboard only reads" separation every other
    Command Centre tab already follows. Reads processed parquet directly
    (never dashboard/home.py, which runs Streamlit script code at import
    time and must never be imported from engine/live_update code) —
    same reasoning swdss.engine.alerts already documents for its own
    direct CME/Solar Wind reads.
    """
    f107_path = PROCESSED_DIR / "f107" / "f107_processed.parquet"
    f107_result = None
    if f107_path.exists():
        try:
            f107_df = pd.read_parquet(f107_path)
            f107_df["timestamp_utc"] = pd.to_datetime(f107_df["timestamp_utc"], utc=True, errors="coerce")
            f107_result = forecast_f107(f107_df)
        except Exception:
            traceback.print_exc()

    cme_path = PROCESSED_DIR / "cme" / "cme_processed.parquet"
    cme_arrivals = []
    if cme_path.exists():
        try:
            cme_df = pd.read_parquet(cme_path)
            cme_df["timestamp_utc"] = pd.to_datetime(cme_df["timestamp_utc"], utc=True, errors="coerce")
            cme_df = cme_df.dropna(subset=["timestamp_utc"])
            cutoff = _now() - pd.Timedelta(hours=SOLAR_FORECAST_CME_LOOKBACK_HOURS)
            recent = cme_df[cme_df["timestamp_utc"] >= cutoff].sort_values("timestamp_utc", ascending=False)
            for _, row in recent.iterrows():
                speed = row.get("speed")
                if speed is None or pd.isna(speed) or float(speed) <= 0:
                    continue
                launch_time = row["timestamp_utc"]
                ambient = ambient_solar_wind_speed(launch_time)
                dbm = estimate_cme_arrival_dbm(launch_time, float(speed), ambient)
                if dbm is None:
                    continue
                cme_arrivals.append({
                    "launch_time": launch_time,
                    "launch_speed_km_s": float(speed),
                    "half_angle": row.get("half_angle"),
                    "cme_type": row.get("cme_type"),
                    "active_region": row.get("active_region"),
                    **dbm,
                })
        except Exception:
            traceback.print_exc()

    flare_outlook, cme_outlook, cme_arrivals, helicity_by_ar, flare_cme_computed_at = _flare_cme_outlook_and_lean(cme_arrivals)

    return {
        "generated_at": _now_iso(),
        "f107": f107_result,
        "cme_arrivals": cme_arrivals,
        "flare_outlook": flare_outlook,
        "cme_outlook": cme_outlook,
        "helicity_by_ar": helicity_by_ar,
        "flare_cme_computed_at": flare_cme_computed_at,
    }


def _previous_flare_cme_outlook() -> tuple[dict | None, dict | None, dict | None, str | None]:
    """The throttle in _flare_cme_outlook_and_lean must compare against
    when the flare/CME outlook itself was last actually (re)computed —
    NOT the enclosing solar_forecast.json snapshot's own `generated_at`,
    which is stamped fresh every ~60s engine cycle regardless of whether
    this specific, expensive computation ran that cycle. A real bug found
    here: this previously read `generated_at`, so `age_minutes` was
    always ~0 and the throttle silently never let a refresh through again
    after the first successful computation — caught when a newly
    retrained flare/CME model's improved metrics never appeared in the
    live snapshot no matter how long the engine kept running.
    """
    previous = storage.load_solar_forecast()
    if previous is None:
        return None, None, None, None
    return (
        previous.get("flare_outlook"),
        previous.get("cme_outlook"),
        previous.get("helicity_by_ar"),
        previous.get("flare_cme_computed_at"),
    )


def _attach_helicity(cme_arrivals: list, helicity_by_ar: dict) -> list:
    """Applied every cycle, regardless of the SHARP-fetch throttle below
    — cme_arrivals itself is rebuilt fresh every 60s cycle as CMEs enter
    and leave the lookback window, so annotating it must not be nested
    inside the same throttle as the expensive JSOC fetch (a real bug
    found and fixed here: the first version nested this loop inside the
    throttled branch, so a CME appearing mid-throttle-window would sit
    at "no matched active region" for up to FLARE_CME_OUTLOOK_REFRESH_MINUTES
    even when its region's helicity was already known).
    """
    for cme in cme_arrivals:
        ar = cme.get("active_region")
        if ar is None or pd.isna(ar):
            cme["source_region_helicity"] = None
            continue
        entry = helicity_by_ar.get(str(int(ar))) if helicity_by_ar else None
        cme["source_region_helicity"] = entry
    return cme_arrivals


def _flare_cme_outlook_and_lean(cme_arrivals: list) -> tuple[dict, dict, list, dict, str]:
    """Flare Outlook / CME Outlook (swdss.models.flare_cme_predict), plus
    a per-CME source-region magnetic-helicity note attached to every
    entry in cme_arrivals — deliberately NOT a Bz-sign prediction (see
    swdss.ingest.sharp's module docstring on the hemispheric helicity
    rule): just the region's measured helicity sign, a descriptive fact
    a researcher can weigh, not a confident forecast.

    Only the SHARP fetch + classifier inference is throttled to
    FLARE_CME_OUTLOOK_REFRESH_MINUTES (reusing the previous cycle's
    helicity_by_ar lookup when still fresh) — the per-CME attachment
    itself always runs against the current cycle's cme_arrivals, via
    _attach_helicity above. The returned `computed_at` is stamped fresh
    on every path that actually attempts this work (success or failure),
    and only ever carried over unchanged on the throttled-reuse path —
    see _previous_flare_cme_outlook for why this must be tracked
    separately from the enclosing snapshot's own `generated_at`.
    """
    prev_flare, prev_cme, prev_helicity_by_ar, prev_computed_at = _previous_flare_cme_outlook()
    if prev_computed_at is not None:
        age_minutes = (_now() - pd.Timestamp(prev_computed_at)).total_seconds() / 60
        if age_minutes < FLARE_CME_OUTLOOK_REFRESH_MINUTES:
            cme_arrivals = _attach_helicity(cme_arrivals, prev_helicity_by_ar or {})
            return (
                prev_flare or {"status": "not_trained"},
                prev_cme or {"status": "not_trained"},
                cme_arrivals,
                (prev_helicity_by_ar or {}),
                prev_computed_at,
            )

    from swdss.models.flare_cme_features import SHARP_FEATURE_COLUMNS, resample_sharp_hourly
    from swdss.models.flare_cme_predict import (
        FEATURE_COLUMNS,
        load_flare_cme_metrics,
        predict_cme_probability,
        predict_flare_probability,
    )

    metrics = load_flare_cme_metrics()
    flare_status = (metrics or {}).get("flare", {}).get("status", "not_trained")
    cme_status = (metrics or {}).get("cme", {}).get("status", "not_trained")
    flare_ready = flare_status == "trained"
    cme_ready = cme_status == "trained"

    try:
        from swdss.ingest.sharp import fetch_sharp_recent
        recent = fetch_sharp_recent(hours=30)
    except Exception:
        traceback.print_exc()
        cme_arrivals = _attach_helicity(cme_arrivals, prev_helicity_by_ar or {})
        return {"status": "fetch_failed"}, {"status": "fetch_failed"}, cme_arrivals, (prev_helicity_by_ar or {}), _now_iso()

    hourly = resample_sharp_hourly(recent)
    if hourly.empty:
        cme_arrivals = _attach_helicity(cme_arrivals, prev_helicity_by_ar or {})
        return {"status": "no_active_regions"}, {"status": "no_active_regions"}, cme_arrivals, (prev_helicity_by_ar or {}), _now_iso()

    hourly = hourly.sort_values(["NOAA_AR", "hour"])
    for col in SHARP_FEATURE_COLUMNS:
        hourly[f"{col}_24h_ago"] = hourly.groupby("NOAA_AR")[col].shift(24)
        hourly[f"{col}_24h_delta"] = hourly[col] - hourly[f"{col}_24h_ago"]
    latest_per_ar = hourly.sort_values("hour").groupby("NOAA_AR").tail(1).set_index("NOAA_AR")
    ready_rows = latest_per_ar.dropna(subset=FEATURE_COLUMNS)

    def _top_region_summary(proba: pd.Series | None) -> dict:
        if proba is None or proba.empty:
            return {"status": "no_scorable_regions"}
        ranked = proba.sort_values(ascending=False)
        return {
            "status": "ok",
            "regions": [{"noaa_ar": int(ar), "probability": float(p)} for ar, p in ranked.items()],
            "highest_risk_ar": int(ranked.index[0]),
            "highest_risk_probability": float(ranked.iloc[0]),
        }

    flare_outlook = _top_region_summary(predict_flare_probability(ready_rows)) if flare_ready else {"status": flare_status}
    cme_outlook = _top_region_summary(predict_cme_probability(ready_rows)) if cme_ready else {"status": cme_status}

    helicity_by_ar = {}
    for ar_int, row in latest_per_ar.iterrows():
        meanjzh = row["MEANJZH"]
        helicity_by_ar[str(ar_int)] = {
            "noaa_ar": int(ar_int),
            "mean_current_helicity": None if pd.isna(meanjzh) else float(meanjzh),
            "sign_label": None if pd.isna(meanjzh) else ("negative (left-handed)" if meanjzh < 0 else "positive (right-handed)"),
        }
    cme_arrivals = _attach_helicity(cme_arrivals, helicity_by_ar)

    return flare_outlook, cme_outlook, cme_arrivals, helicity_by_ar, _now_iso()


def refresh_dashboard_products() -> dict:
    # ==================== Stage 1: Per-Variable Forecast Entries ====================
    # Walks PRODUCTION_MATRIX, building the live `forecasts` tree (what
    # the Forecast tab displays), the permanent `history_rows` log
    # (Search/Timeline/Downloads), and `evaluation_rows` for any job that
    # completed this cycle (Verification/Skill/Calibration).
    current_conditions = _current_conditions()

    # Current activity regime, from OBSERVED conditions right now (not a
    # prediction) — used below to pick a storm/quiet-appropriate error band
    # per forecast, instead of one all-time MAE. classify_activity_regime
    # is purely a numeric bucketing function; feeding it live observations
    # rather than a forecast is a deliberate, simpler choice than waiting
    # for this cycle's own predicted_kp/dst/ae (only known once every
    # forecast entry below has already been built) — see Development
    # Roadmap for why this was chosen over a two-pass restructure.
    current_regime = classify_activity_regime(
        predicted_kp=current_conditions.get("kp", {}).get("value"),
        predicted_dst=current_conditions.get("dst", {}).get("value"),
        predicted_ae=current_conditions.get("ae", {}).get("value"),
    )

    # Pre-this-cycle evaluation history — same "one cycle stale is fine"
    # reasoning already used by the drift check below, which reuses this
    # exact same load rather than reading the file twice for the same data.
    evaluation_history_before_cycle = storage.load_evaluation_history()
    regime_mae_lookup = {
        (row["dataset"], row["variable"], row["horizon"], row["activity_regime"]): row["mean_abs_error"]
        for row in calibration.compute_regime_error_bands(evaluation_history_before_cycle)
    }

    forecasts = {"solar_wind": {}, "imf": {}, "analytics": {}, "ae": {}}
    history_rows = []
    evaluation_rows = []
    report_rows = []
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
            entry = _build_forecast_entry(
                dataset, variable, semantic_horizon, job, trend, current_value, current_observed_at,
                current_regime, regime_mae_lookup,
            )
            variable_block[horizon_label] = entry

            # Build (and log/evaluate) an entry for EVERY candidate this
            # cycle, not just the one selected above for the Forecast
            # tab's display — see jobs.start_job's continuous_reissue
            # note. Each candidate has its own distinct valid_start, so
            # logging all of them creates no ambiguity for readers like
            # the Search tab (already dedupes by (horizon, valid_start),
            # keeping the latest logged state for each) — it just means a
            # completed-but-superseded job actually gets a log entry
            # reflecting its TRUE final state and gets properly
            # evaluated, instead of being silently frozen forever at
            # whatever it looked like the cycle before a newer job took
            # over as the display representative.
            #
            # This was a real, confirmed regression, not a hypothetical:
            # once continuous re-issue guarantees there's always a newer
            # ACTIVE job for a 1h/interval slot, `job` above is always
            # bound to that active one — so checking only `job` meant a
            # completed candidate's `job_status == "completed"` could
            # never be seen here again. Confirmed live two ways: (1) the
            # Search tab showing a fully-elapsed 1h Speed forecast still
            # frozen at [ACTIVE] / actual=— even though its real
            # completed value already existed in the jobs table, and (2)
            # querying the jobs table directly showed ALL 10 of a single
            # day's completed 1h Speed jobs had a real actual_value with
            # zero of them ever evaluated.
            is_headline_horizon = (dataset, variable, horizon_label) in _HEADLINE_KEY_SET

            for candidate in candidates:
                candidate_entry = entry if candidate is job else _build_forecast_entry(
                    dataset, variable, semantic_horizon, candidate, trend, current_value, current_observed_at,
                    current_regime, regime_mae_lookup,
                )
                if candidate_entry is None:
                    continue

                # Once a candidate is completed AND already evaluated, its
                # history row can never change again — logging it again on
                # every subsequent cycle is pure waste, and with 1h-horizon
                # jobs re-issuing roughly hourly and staying in the jobs
                # table for PRUNE_RETENTION_DAYS (7 days) before pruning,
                # that waste compounds badly: a single finished job got
                # re-logged, unchanged, once per ~60s cycle for up to a
                # week. This was the actual cause of individual headline
                # slots accumulating 150K-225K history rows within a mere
                # 72h window — not the horizon count, not retention length.
                # already_evaluated is already computed above for exactly
                # this kind of check (see its use for evaluation_rows
                # below); reusing it here needs no new state.
                already_logged_terminal = candidate["status"] == "completed" and candidate["job_id"] in already_evaluated

                # Only the 10 headline (dataset, variable, horizon) slots
                # get a permanent history row — extended horizons are
                # still fully computed and evaluated below (skill score,
                # calibration, drift all keep working for every horizon),
                # they're just not logged to forecast_snapshot_history,
                # which is what actually grew unbounded.
                if is_headline_horizon and not already_logged_terminal:
                    history_rows.append({
                        "generated_at": generated_at,
                        "dataset": dataset,
                        "variable": variable,
                        "horizon": horizon_label,
                        "job_id": candidate_entry["job_id"],
                        "status": candidate_entry["status"],
                        "job_status": candidate_entry["job_status"],
                        "valid_start": candidate_entry["valid_start"],
                        "valid_end": candidate_entry["valid_end"],
                        "current_value": candidate_entry["current_value"],
                        "predicted_value": candidate_entry["predicted_value"],
                        "model_name": candidate_entry["model_name"],
                        "r2": candidate_entry["metrics"].get("r2"),
                        "mae": candidate_entry["metrics"].get("mae"),
                        "confidence_score": candidate_entry["confidence"]["score"],
                        "confidence_category": candidate_entry["confidence"]["category"],
                        "actual_value": candidate_entry["actual_value"],
                        "abs_error": candidate_entry["abs_error"],
                    })

                if candidate["status"] != "completed" or candidate_entry["actual_value"] is None:
                    continue
                if candidate["job_id"] in already_evaluated:
                    continue
                if candidate_entry["predicted_value"] is None:
                    continue

                signed_error = candidate_entry["predicted_value"] - candidate_entry["actual_value"]

                # Persistence skill score input — see swdss.engine.skill.
                # Resolved once, here, at evaluation time (not every
                # cycle) since it never changes once a job is created.
                persistence_value = _resolve_persistence_anchor(dataset, variable, candidate.get("created_at"))
                persistence_error = (
                    candidate_entry["actual_value"] - persistence_value if persistence_value is not None else None
                )

                evaluation_rows.append({
                    "job_id": candidate_entry["job_id"],
                    "dataset": dataset,
                    "variable": variable,
                    "horizon": horizon_label,
                    "valid_start": candidate_entry["valid_start"],
                    "valid_end": candidate_entry["valid_end"],
                    "completed_at": candidate.get("completed_at"),
                    "final_predicted_value": candidate_entry["predicted_value"],
                    "actual_value": candidate_entry["actual_value"],
                    "error": signed_error,
                    "abs_error": candidate_entry["abs_error"],
                    "squared_error": signed_error ** 2,
                    "model_name": candidate_entry["model_name"],
                    "confidence_score_at_issuance": candidate_entry["confidence"]["score"],
                    "confidence_category_at_issuance": candidate_entry["confidence"]["category"],
                    # The model's own held-out training MAE — drift.detect_drift
                    # compares recent live abs_error against this. Activity
                    # regime is filled in below, once this cycle's overall
                    # Kp/Dst/AE outlook is known — see the tagging pass after
                    # this loop.
                    "training_mae": candidate_entry["metrics"].get("mae"),
                    "activity_regime": None,
                    # Persistence baseline (see swdss.engine.skill) — the
                    # naive "assume no change" forecast's own error at
                    # this exact event. None for AE (no live feed) or
                    # if the issuance hour's actual isn't resolvable.
                    "persistence_value": persistence_value,
                    "persistence_abs_error": abs(persistence_error) if persistence_error is not None else None,
                    "persistence_squared_error": persistence_error ** 2 if persistence_error is not None else None,
                })

                # Reports tab (see storage.append_report_rows) — only the
                # 10 headline slots get a report row, and only the instant
                # a candidate is newly evaluated above (so this fires once
                # per (variable, horizon, hour), not every cycle). A gap
                # here for a given hour IS the "missing data" signal — no
                # candidate ever completed/evaluated for that slot that
                # hour — so there's no separate flag needed for it.
                if is_headline_horizon:
                    persistence_abs_error = abs(persistence_error) if persistence_error is not None else None
                    flags = []
                    if candidate_entry["confidence"]["category"] in ("low", "poor"):
                        flags.append("low_confidence")
                    if persistence_abs_error is not None and candidate_entry["abs_error"] > persistence_abs_error:
                        flags.append("underperformed_persistence")

                    report_rows.append({
                        "dataset": dataset,
                        "variable": variable,
                        "horizon": horizon_label,
                        "valid_start": candidate_entry["valid_start"],
                        "valid_end": candidate_entry["valid_end"],
                        "predicted_value": candidate_entry["predicted_value"],
                        "actual_value": candidate_entry["actual_value"],
                        "error": signed_error,
                        "abs_error": candidate_entry["abs_error"],
                        "model_name": candidate_entry["model_name"],
                        "confidence_category": candidate_entry["confidence"]["category"],
                        "persistence_abs_error": persistence_abs_error,
                        "flags": ",".join(flags),
                        "evaluated_at": _now_iso(),
                    })

            is_headline = semantic_horizon == "interval" or semantic_horizon == 1
            if is_headline and entry is not None:
                if dataset == "analytics" and variable == "kp":
                    predicted_kp = entry["predicted_value"]
                elif dataset == "analytics" and variable == "dst":
                    predicted_dst = entry["predicted_value"]
                elif dataset == "ae" and variable == "ae":
                    predicted_ae = entry["predicted_value"]

    # ==================== Stage 2: Physics & Overall Outlook ====================
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

    # ==================== Stage 3: Freshness, Alerts & Drift ====================
    freshness = _freshness()
    alerts = build_alerts(outlook_level=outlook_level, physics=physics_summary, freshness=freshness)

    # Drift check against ALREADY-persisted history (this cycle's brand
    # new evaluation_rows haven't been written yet) — a one-cycle lag is
    # irrelevant for a signal that's inherently about a sustained trend.
    # Reuses the same pre-cycle load already fetched above for the regime
    # error-band lookup, rather than reading the same file twice.
    try:
        drift_alerts = detect_drift(evaluation_history_before_cycle)
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
    system_health = _service_health(freshness, recent_cycle_errors, active_jobs_total, physics_summary)

    # ==================== Stage 4: Forecast Explanation Engine ====================
    # Connects the Physics Engine's live readings to the Kp/Dst/AE
    # forecasts they drive (see swdss.engine.explanation). Attached
    # directly onto each headline entry so the Forecast tab can show
    # "why" alongside "what", rather than leaving Physics and Forecast
    # as isolated views.
    headline_entries = {
        "kp": forecasts.get("analytics", {}).get("kp", {}).get("interval"),
        "dst": forecasts.get("analytics", {}).get("dst", {}).get("1h"),
        "ae": forecasts.get("ae", {}).get("ae", {}).get("1h"),
    }
    try:
        explanations = explanation.build_explanations(physics_summary, headline_entries)
    except Exception:
        explanations = {}
        traceback.print_exc()

    # Layers real per-forecast SHAP attribution against the deployed
    # model on top of the rule-based ranking above (see
    # swdss.engine.explanation.attach_shap_attribution) — isolated in
    # its own try/except so a SHAP failure never discards the
    # rule-based explanations that already succeeded.
    try:
        explanation.attach_shap_attribution(explanations, headline_entries)
    except Exception:
        traceback.print_exc()

    for variable, entry in headline_entries.items():
        if entry is not None:
            entry["explanation"] = explanations.get(variable)

    # ==================== Stage 5: Assemble the Snapshot ====================
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

    # ==================== Stage 6: Forecast Package ====================
    # The synchronized operational product bundling the 10 headline
    # forecasts (see swdss.engine.packages). Pure re-packaging of what
    # was just computed above; starts no job, calls no model.
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

    # ==================== Stage 7: Persist History & Verification Science ====================
    try:
        storage.write_snapshot(snapshot)
        storage.append_forecast_history_rows(history_rows)
        storage.append_evaluation_rows(evaluation_rows)
        storage.append_report_rows(report_rows)
    except Exception:
        traceback.print_exc()

    # Verification science — skill.py and calibration.py are pure
    # analysis over the FULL evaluation history (including the rows just
    # appended above, unlike drift's deliberate one-cycle lag), so both
    # are recomputed from scratch every cycle rather than maintained
    # incrementally. Cheap: evaluation_history is small (one row per
    # completed forecast) even after months of continuous operation.
    try:
        full_history = storage.load_evaluation_history()
        storage.write_skill_scores(skill.compute_skill_scores(full_history))
        storage.write_calibration_report(
            calibration.compute_confidence_calibration(full_history),
            calibration.compute_regime_error_bands(full_history),
        )
    except Exception:
        traceback.print_exc()

    # ==================== Stage 8: Solar Forecast (F10.7 + CME Arrival) ====================
    # Independent of the PRODUCTION_MATRIX loop above — F10.7 and CME
    # arrival aren't trained/persisted models, so they're computed fresh
    # here rather than through jobs.py's job-lifecycle machinery.
    try:
        storage.write_solar_forecast(_solar_forecast_snapshot())
    except Exception:
        traceback.print_exc()

    storage.append_log_lines([{
        "ts": completed_at,
        "level": "INFO",
        "stage": "refresh_dashboard_products",
        "message": f"Snapshot refreshed — outlook={outlook_level}, {len(evaluation_rows)} new evaluation(s), {len(alerts)} alert(s).",
    }])

    return snapshot
