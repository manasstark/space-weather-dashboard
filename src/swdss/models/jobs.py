"""Continuous "prediction job" tracking, backed by SQLite (not JSON — a
month of minute-level ticks would make a single JSON blob grow unbounded
and require a full rewrite on every tick).

A job is started for a fixed (dataset, variable, horizon) and represents:
"keep forecasting the value at a fixed target time, refining the estimate
every time a new NOAA minute-level reading arrives, for as long as it takes
for that target time's actual observation to be published."

Completion is tied to the forecast horizon, not an arbitrary 1-hour window:
a 24h job keeps running for roughly 24 hours, not 1. Since we only have
trained models at five discrete horizons (1, 3, 6, 12, 24h), a job refines
its estimate at "checkpoints" — the moments when the remaining time to the
fixed target exactly matches one of those trained horizons. Concretely, a
24h job anchored at 17:00 hits checkpoints at remaining=24 (immediately),
12 (12h later), 6, 3, and 1 hour before target, switching to the
corresponding model each time. Within a checkpoint hour, the model's input
bucket is still "live" (partial-hour average updating every NOAA minute),
so each checkpoint itself drifts the same way the old 1h-only design did —
no checkpoint, no fixed target, that's the whole point.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager

import numpy as np
import pandas as pd

from swdss.physics.core import dynamic_pressure_scalar, ey_scalar, vbz_scalar

from swdss.ingest.kyoto_ae import fetch_kyoto_ae_hour
from swdss.ingest.kyoto_ae_quicklook import estimate_kyoto_quicklook_ae_full
from swdss.models.predict import latest_minute_observation, load_live_features, resolve_static_actual
from swdss.models.predict import predict as predict_live
from swdss.models.predict import predict_kp_rolling
from swdss.models.registry import DATASETS, HORIZONS, IMF_VARIABLES, SOLAR_WIND_VARIABLES
from swdss.paths import DATA_DIR

DB_PATH = DATA_DIR / "predictions" / "predictions.db"

# How often the verification engine actually hits Kyoto WDC, independent
# of how often poll_jobs itself is called (which can be every page
# render/auto-refresh — far more often than this). Tracked per-job via
# the verification_checked_at column. Kyoto WDC's official digital AE
# values publish with a real lag of ~10-20 days, so there is no value in
# checking more than once a day — doing so would just be needless load
# against an upstream service that hasn't changed.
KYOTO_VERIFICATION_INTERVAL = pd.Timedelta(days=1)

# How often the Quicklook estimate is automatically re-captured for a
# completed AE job while its target hour's graph coverage isn't ~complete
# yet — independent of the manual "Refresh Quicklook Estimate" button, so
# the estimate keeps catching up as Kyoto draws more of the hour even if
# the user never clicks refresh. Much shorter than
# KYOTO_VERIFICATION_INTERVAL since this hits Kyoto's continuously-updating
# real-time graph, not the slow-to-publish official digital data.
QUICKLOOK_RECHECK_INTERVAL = pd.Timedelta(minutes=5)

# Below this fraction of the hour drawn, the estimate is not presented as
# representative of the target hour at all (see quicklook_label). Above
# roughly 1.0 (allowing a little slack for pixel-column rounding at the
# hour boundary), the hour is treated as fully drawn.
QUICKLOOK_PARTIAL_COVERAGE = 0.6
QUICKLOOK_COMPLETE_COVERAGE = 0.97

JOB_HISTORY_LIMIT = 20

# AE predictions are precious, low-frequency research artifacts, not
# disposable high-frequency jobs like Solar Wind/IMF — every one must
# stay visible until the user explicitly deletes it, never silently
# capped out of the "Running Predictions" view.
UNCAPPED_HISTORY_DATASETS = {"ae"}

# Kp and Dst are published by NOAA as a single, already-final value per
# native interval (confirmed against NOAA's own kyoto-dst.json: exactly
# one row per hour, never sub-hourly) — unlike Solar Wind/IMF, which are
# genuinely minute-resolution and need a real hourly average computed
# from live ticks. See _advance_evaluating.
INSTANT_FINALIZE_VARIABLES = ("kp", "dst")

STABILITY_WINDOW = 10

# The live Solar Wind + IMF + Geomagnetic readings shown per-tick in the
# Analytics page's terminal, since its predictions are driven by all of
# them together, not a single self-referential variable like the
# standalone tabs. Dst and Kp are included too — for the Kp interval
# forecast specifically, "Current Dst" and "Previous Kp" are explicit
# table columns, and Previous Kp is expected to stay fixed for the whole
# session (it only changes when NOAA publishes a new official value).
ANALYTICS_INPUT_VARIABLES = ["speed", "density", "temperature", "bt", "bx_gsm", "by_gsm", "bz_gsm", "dst", "kp", "ae"]

# The live Solar Wind + IMF readings shown per-tick for the AE predictor —
# deliberately excludes Kp/Dst (AE V1 is kept independent of them, see
# README's staged AE plan). Derived physics (Ey/VBz/Dynamic Pressure) and
# the latest known AE value are computed/looked up separately below, the
# same way they're computed for training and live inference.
AE_INPUT_VARIABLES = SOLAR_WIND_VARIABLES + IMF_VARIABLES


# ==================== Database Setup ====================

@contextmanager
def _connect():
    """Every call commits and immediately checkpoint-truncates the WAL back
    into the main .db file. WAL mode normally defers that, which is fine
    for throughput but means a non-graceful shutdown (process killed, or
    the -wal/-shm sidecar files getting cleaned up separately from the
    main file) can lose writes that never made it past the WAL. Saved
    predictions are meant to survive forever, so we trade a little
    per-write overhead for not risking that again.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                variable TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                start_hour TEXT NOT NULL,
                target_hour TEXT NOT NULL,
                model_name TEXT,
                metrics_json TEXT,
                status TEXT NOT NULL,
                saved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_minute_seen TEXT,
                actual_value REAL,
                completed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                minute_at TEXT,
                noaa_value REAL,
                predicted_value REAL NOT NULL,
                used_horizon INTEGER NOT NULL,
                logged_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_job ON ticks(job_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                minute_at TEXT,
                noaa_value REAL,
                running_avg REAL,
                logged_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_ticks_job ON eval_ticks(job_id)")
        _ensure_column(conn, "ticks", "inputs_json", "TEXT")
        # Only meaningfully used by static-target variables (e.g. AE) —
        # see _advance_predicting / _verify_static_jobs. NULL for every
        # other job, whose completion already depends on a real actual
        # value being resolved directly.
        _ensure_column(conn, "jobs", "verification_status", "TEXT")
        # Timestamp of the last actual Kyoto WDC check (not every poll —
        # see KYOTO_VERIFICATION_INTERVAL), so the verification engine's
        # own cadence is independent of how often the dashboard re-renders.
        _ensure_column(conn, "jobs", "verification_checked_at", "TEXT")
        # Timestamp of the moment verification actually SUCCEEDED — distinct
        # from verification_checked_at, which updates on every attempt
        # whether or not that attempt found new data. Nullable; set once,
        # permanently, the moment verification_status flips to 'verified'.
        _ensure_column(conn, "jobs", "verified_at", "TEXT")
        # Quicklook Verification (AE only) — an approximate, image-based
        # estimate read off Kyoto WDC's real-time graph immediately after
        # completion. Explicitly NOT the official verification source
        # (that's verification_status/verified_at above) — stored in its
        # own separate columns so it can never overwrite or be confused
        # with the official result.
        _ensure_column(conn, "jobs", "quicklook_ae", "REAL")
        _ensure_column(conn, "jobs", "quicklook_image_url", "TEXT")
        _ensure_column(conn, "jobs", "quicklook_checked_at", "TEXT")
        # Image-processing diagnostics captured alongside quicklook_ae:
        # confidence label + estimated range (uncertainty from graph
        # extraction), and when Kyoto's server last regenerated the graph
        # image itself (distinct from quicklook_checked_at, which is when
        # *we* last read it).
        _ensure_column(conn, "jobs", "quicklook_confidence", "TEXT")
        _ensure_column(conn, "jobs", "quicklook_range_low", "REAL")
        _ensure_column(conn, "jobs", "quicklook_range_high", "REAL")
        _ensure_column(conn, "jobs", "quicklook_graph_created", "TEXT")
        # Fraction (0-1) of the target hour's 60 minutes that had a drawn
        # curve at capture time. Distinct from quicklook_confidence (which
        # also factors in curve-height noise) — this is reported directly
        # so the UI can warn plainly when an estimate is really just a
        # partial-hour average, not the eventual full-hour figure.
        _ensure_column(conn, "jobs", "quicklook_hour_coverage", "REAL")
        # Kp interval jobs only: the FROZEN production forecast (Mode 1),
        # computed once at job creation from features strictly before the
        # target interval — see predict.predict_kp_interval — and never
        # touched again. Every tick logged after creation is instead the
        # separate, deliberately unrestricted "ticks" table, so the two
        # products never overwrite each other.
        _ensure_column(conn, "jobs", "production_prediction", "REAL")
        # The exact pre-interval feature timestamp the frozen forecast was
        # computed from (predict_kp_interval's own `observed_at`) — shown
        # on the dashboard so it's never ambiguous which data window
        # produced the Mode 1 number.
        _ensure_column(conn, "jobs", "production_observed_at", "TEXT")
        # 'manual' (default — a user's own "Start Prediction" click, meant
        # to persist and stay visible for study until explicitly deleted)
        # or 'engine' (created automatically by swdss.engine.orchestrator.
        # run_forecast_cycle, every ~hour, forever). Existing rows predate
        # this column and default to 'manual' via the DEFAULT clause,
        # which is the correct backfill — every job before the Operational
        # Forecast Engine existed WAS a manual one. get_running_jobs/
        # get_saved_jobs filter to 'manual' by default so the Production
        # tabs' study view is never crowded out by the engine's own
        # continuous job creation; the engine's own internal lookups pass
        # source='engine' explicitly. See prune_old_engine_jobs — only
        # 'engine' jobs are ever pruned; a manual job persists exactly as
        # it always has, forever, until the user deletes it.
        _ensure_column(conn, "jobs", "source", "TEXT NOT NULL DEFAULT 'manual'")


def _ensure_column(conn, table: str, column: str, decl: str) -> None:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


_init_db()


# ==================== Internal Row/Tick Helpers ====================

def _to_utc_iso(ts) -> str:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _row_to_job(row: sqlite3.Row) -> dict:
    job = dict(row)
    job["metrics"] = json.loads(job.pop("metrics_json") or "{}")
    job["saved"] = bool(job["saved"])
    return job


def _fetch_ticks(conn, job_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT minute_at, noaa_value, predicted_value, used_horizon, logged_at, inputs_json "
        "FROM ticks WHERE job_id = ? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    ticks = []
    for row in rows:
        tick = dict(row)
        raw_inputs = tick.pop("inputs_json", None)
        tick["inputs"] = json.loads(raw_inputs) if raw_inputs else None
        ticks.append(tick)
    return ticks


def _capture_live_inputs(dataset: str) -> dict:
    """Snapshots the current live Solar Wind + IMF readings, for the
    Analytics/AE pages' multi-input terminal display. Returns {} for
    single-source datasets (nothing extra to show beyond their own value).
    """
    if dataset == "analytics":
        inputs = {}
        for var in ANALYTICS_INPUT_VARIABLES:
            _, value = latest_minute_observation(dataset, var)
            inputs[var] = value
        return inputs

    if dataset == "ae":
        inputs = {}
        for var in AE_INPUT_VARIABLES:
            _, value = latest_minute_observation(dataset, var)
            inputs[var] = value

        # Scalar counterparts of swdss.models.features.add_derived_physics_features,
        # for a single live reading rather than a dataframe — see
        # swdss.physics.core, the Physics Engine's canonical implementation.
        speed, density, bz = inputs.get("speed"), inputs.get("density"), inputs.get("bz_gsm")
        if speed is not None and bz is not None:
            inputs["vbz"] = vbz_scalar(speed, bz)
            inputs["ey"] = ey_scalar(speed, bz)
        if speed is not None and density is not None:
            inputs["dynamic_pressure"] = dynamic_pressure_scalar(density, speed)

        _, inputs["ae"] = latest_minute_observation(dataset, "ae")
        return inputs

    if dataset == "experimental":
        inputs = {}
        for var in ANALYTICS_INPUT_VARIABLES:
            if var == "ae":
                continue  # observed AE must never appear in the experimental cascade
            _, value = latest_minute_observation("analytics", var)
            inputs[var] = value

        # predicted_ae has no stored source (see predict.generate_predicted_ae)
        # — recomputing the cascade's frame here is the only way to show the
        # live figure feeding this session's Kp/Dst models.
        frame = load_live_features("experimental")
        predicted_ae_series = frame["predicted_ae"].dropna() if "predicted_ae" in frame.columns else pd.Series(dtype=float)
        inputs["predicted_ae"] = float(predicted_ae_series.iloc[-1]) if not predicted_ae_series.empty else None
        return inputs

    return {}


def _append_tick(
    conn, job_id: str, minute_ts, minute_val, predicted_value: float, used_horizon, inputs: dict = None
) -> None:
    minute_iso = None if minute_ts is None else _to_utc_iso(minute_ts)
    inputs_json = json.dumps(inputs) if inputs else None
    conn.execute(
        "INSERT INTO ticks (job_id, minute_at, noaa_value, predicted_value, used_horizon, logged_at, inputs_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            job_id,
            minute_iso,
            minute_val,
            predicted_value,
            used_horizon,
            _to_utc_iso(pd.Timestamp.now(tz="UTC")),
            inputs_json,
        ),
    )
    conn.execute("UPDATE jobs SET last_minute_seen = ? WHERE job_id = ?", (minute_iso, job_id))


def _fetch_eval_ticks(conn, job_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT minute_at, noaa_value, running_avg, logged_at FROM eval_ticks WHERE job_id = ? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _append_eval_tick(conn, job_id: str, minute_ts, minute_val: float) -> None:
    """Logs one NOAA minute reading collected during the target hour, along
    with the running average of every eval tick logged so far — this is
    the live "how is the target hour's average shaping up" view.
    """
    minute_iso = _to_utc_iso(minute_ts)
    prior = conn.execute(
        "SELECT noaa_value FROM eval_ticks WHERE job_id=? ORDER BY id ASC", (job_id,)
    ).fetchall()
    values = [r["noaa_value"] for r in prior if r["noaa_value"] is not None] + [minute_val]
    running_avg = sum(values) / len(values)
    conn.execute(
        "INSERT INTO eval_ticks (job_id, minute_at, noaa_value, running_avg, logged_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, minute_iso, minute_val, running_avg, _to_utc_iso(pd.Timestamp.now(tz="UTC"))),
    )
    conn.execute("UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_id))


def _tick_reference_variable(dataset: str, variable: str) -> str:
    """Which variable's live cadence should drive tick timing/dedup. For
    single-source datasets that's the predicted variable itself (it IS the
    minute-level driver). For Analytics, the predicted variable is Kp or
    Dst — both update far slower (3h / 1h) than the Solar Wind/IMF inputs
    that actually feed the model every minute. Using Kp/Dst's own sparse
    timestamp here was a real bug: a job started at 10:42 would get its
    first tick stamped with whatever stale hour Kp last updated (e.g.
    06:00), instead of "now". Speed updates close to every NOAA minute, so
    it's used as the freshness reference for any Analytics job.

    AE has the same issue in an even sharper form: it has no live feed at
    all, so its own timestamp never advances — using it as the reference
    would log exactly one tick ever and then look permanently stuck.
    """
    return "speed" if dataset in ("analytics", "ae", "experimental") else variable


# ==================== Job Lifecycle Core ====================

def start_job(dataset: str, variable: str, horizon: int, source: str = "manual") -> tuple:
    """Starts a new continuous prediction job, or returns the existing one
    if a job for this exact slot is already tracked (see the dedup
    branches below — the definition of "already tracked" differs for
    continuously-reissued engine jobs vs. everything else).

    `source` is 'manual' (a user's own "Start Prediction" click — the
    default, so every existing call site needs no change) or 'engine'
    (swdss.engine.orchestrator.run_forecast_cycle, which always passes
    this explicitly) — see the `source` column's own comment in
    _init_db for why this distinction exists. The dedup check below is
    scoped to the SAME source: a manual job can never silently absorb
    an engine job's slot (or vice versa). This was previously source-
    agnostic on the theory that sharing one job was more sensible than
    running the prediction twice — in practice that let a stray manual
    job permanently block the engine from ever getting its own job for
    that (dataset, variable, horizon) until the manual one's target hour
    happened to resolve (up to 24h for the slowest horizon), which is
    a far worse outcome than the rare extra prediction call.

    CONTINUOUS RE-ISSUE (engine, horizon == 1 — this includes Kp's
    interval job, which is tracked at horizon=1 in this table): issuing
    the next forecast must never wait on the previous one being
    evaluated. Chaining "start a new job only once the old one
    completes" means the new job's target hour is decided by whenever
    evaluation happens to finish, not by the wall clock — which silently
    skips whichever hour falls in that gap. So for this case, dedup is
    keyed on the exact target_hour (regardless of the existing job's
    status) rather than on "is anything still in_progress": a job for
    12:00-13:00 that's still waiting to be evaluated and a freshly
    issued job for 13:00-14:00 are expected to coexist briefly, not be
    serialized. Every other case (manual jobs, and engine jobs at every
    other horizon) keeps the original in_progress-based dedup.
    """
    is_kp_interval_job = dataset in ("analytics", "experimental") and variable == "kp"
    # For Kp, predict_live/predict_kp_interval returns the FROZEN Mode 1
    # production forecast (training-consistent, capped to data strictly
    # before the target interval) — this is what gets stored permanently
    # on the job and never recomputed. Every other variable's `result` is
    # just its one and only prediction, same as always.
    result = predict_live(dataset, variable, horizon)
    start_hour_iso = _to_utc_iso(result["observed_at"])
    target_hour_iso = _to_utc_iso(result["predicted_for"])
    continuous_reissue = source == "engine" and horizon == 1

    with _connect() as conn:
        if continuous_reissue:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE dataset=? AND variable=? AND horizon=? AND target_hour=? AND source=?",
                (dataset, variable, horizon, target_hour_iso, source),
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE dataset=? AND variable=? AND horizon=? AND start_hour=? AND status='in_progress' AND source=?",
                (dataset, variable, horizon, start_hour_iso, source),
            ).fetchone()
        if existing is not None:
            job = _row_to_job(existing)
            job["ticks"] = _fetch_ticks(conn, job["job_id"])
            job["eval_ticks"] = _fetch_eval_ticks(conn, job["job_id"])
            return job, False

        minute_ts, minute_val = latest_minute_observation(dataset, _tick_reference_variable(dataset, variable))

        # The first entry in the Mode 2 rolling-estimate timeline — computed
        # separately from the frozen Mode 1 value above, since it
        # deliberately uses whatever data is freshest right now (which may
        # already be inside the target interval), never capped to the
        # pre-interval window. Falls back to the frozen value only if the
        # rolling call itself fails (e.g. a transient feature-availability
        # issue) so job creation never breaks over this secondary product.
        first_tick_value = result["predicted_value"]
        if is_kp_interval_job:
            try:
                first_tick_value = predict_kp_rolling(dataset)["predicted_value"]
            except Exception:
                pass

        job_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO jobs (job_id, dataset, variable, horizon, start_hour, target_hour, model_name, "
            "metrics_json, status, saved, created_at, last_minute_seen, production_prediction, "
            "production_observed_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, ?, ?, ?)",
            (
                job_id,
                dataset,
                variable,
                horizon,
                start_hour_iso,
                target_hour_iso,
                result["model_name"],
                json.dumps(result["metrics"]),
                "in_progress",
                _to_utc_iso(pd.Timestamp.now(tz="UTC")),
                result["predicted_value"] if is_kp_interval_job else None,
                _to_utc_iso(result["observed_at"]) if is_kp_interval_job else None,
                source,
            ),
        )
        used_horizon = "rolling" if is_kp_interval_job else horizon
        _append_tick(
            conn, job_id, minute_ts, minute_val, first_tick_value, used_horizon, _capture_live_inputs(dataset)
        )

        job = _row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        job["ticks"] = _fetch_ticks(conn, job_id)
        job["eval_ticks"] = _fetch_eval_ticks(conn, job_id)
        return job, True


def resolve_actual_value(dataset: str, variable: str, target_hour) -> float:
    """Looks up the hourly-mean actual value for a target hour from live
    data, once that hour has occurred. Returns None ("Pending") if the hour
    hasn't arrived yet or live data doesn't cover it.

    `static_variables` (e.g. AE) have no live feed at all — their entry in
    load_live_features is forward-filled and must never be reported as a
    real observation, so those are resolved separately and honestly.
    """
    config = DATASETS[dataset]
    if variable in (config.static_variables or []):
        return resolve_static_actual(dataset, variable, target_hour)

    frame = load_live_features(dataset)
    ts = pd.Timestamp(target_hour)
    ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    ts = ts.floor("h")
    if ts not in frame.index:
        return None
    value = frame.loc[ts, variable]
    return None if pd.isna(value) else float(value)


def _finalize_job(conn, job_row: sqlite3.Row, actual_value: float) -> None:
    conn.execute(
        "UPDATE jobs SET status='completed', actual_value=?, completed_at=? WHERE job_id=?",
        (actual_value, _to_utc_iso(pd.Timestamp.now(tz="UTC")), job_row["job_id"]),
    )


# ==================== AE Quicklook Verification ====================
# Approximate, image-based cross-check against Kyoto WDC's real-time
# graph — never the official verification source (see resolve_actual_value
# / the official Kyoto digital AE ingest for that). Purely additive.

def _capture_quicklook_estimate(conn, job_id: str, target_hour) -> None:
    """Quicklook Verification (AE only): an approximate, image-based
    estimate read off Kyoto WDC's continuously-updating real-time graph,
    captured immediately on completion and re-triable later via
    refresh_quicklook_estimate. This is explicitly NOT the official
    verification source — it never blocks or affects prediction
    completion (already finished by the time this runs) or the separate,
    official verification engine.

    A None result (network hiccup, or the target hour's column genuinely
    not drawn yet) is deliberately NEVER written over an already-captured
    real estimate — the marker Kyoto uses for its most-recent, still-
    provisional minutes is only a few pixels wide, so re-fetching the
    same moment can flip between finding it and not; once a real estimate
    is captured, a later flaky/failed attempt must not erase it. The
    checked_at timestamp still updates either way, so "last checked"
    stays accurate.
    """
    try:
        result = estimate_kyoto_quicklook_ae_full(target_hour)
    except Exception:
        return

    checked_at = _to_utc_iso(pd.Timestamp.now(tz="UTC"))
    graph_created = _to_utc_iso(result["graph_created"]) if result.get("graph_created") is not None else None

    if result["estimate"] is None:
        conn.execute(
            "UPDATE jobs SET quicklook_image_url=?, quicklook_checked_at=?, quicklook_graph_created=? "
            "WHERE job_id=?",
            (result["image_url"], checked_at, graph_created, job_id),
        )
        return

    conn.execute(
        "UPDATE jobs SET quicklook_ae=?, quicklook_image_url=?, quicklook_checked_at=?, "
        "quicklook_confidence=?, quicklook_range_low=?, quicklook_range_high=?, quicklook_graph_created=?, "
        "quicklook_hour_coverage=? WHERE job_id=?",
        (
            result["estimate"],
            result["image_url"],
            checked_at,
            result["confidence"],
            result["range_low"],
            result["range_high"],
            graph_created,
            result["hour_coverage"],
            job_id,
        ),
    )


def refresh_quicklook_estimate(job_id: str) -> dict:
    """Manually re-runs the Quicklook estimate for a completed AE job
    (e.g. if the first automatic attempt failed, or to re-read the graph
    now that more of the day has been drawn). Returns the updated job.
    """
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is not None:
            _capture_quicklook_estimate(conn, job_id, pd.Timestamp(row["target_hour"]))
    return get_job(job_id)


# ==================== Job Advancement (Ticking) ====================
# The state machine every active job moves through each cycle:
# in_progress (refining) -> evaluating (target hour reached, awaiting
# actual) -> completed. poll_jobs is the per-dataset entry point;
# tick_all_active_jobs (bottom of this file) is the public one called
# once per live_update.py loop.

def _advance_predicting(conn, job_row: sqlite3.Row) -> None:
    """Phase 1 (status='in_progress'): refine the forecast for the fixed
    target hour every time a new NOAA minute arrives, switching models at
    checkpoints as the remaining time to target narrows. Three variable
    classes are handled, each in its own branch below:

    1. Static targets (config.static_variables, e.g. AE) have NO live feed
       for the *predicted* variable itself, so completion can't be gated
       on it reaching the target hour. Their reference/input variable
       (_tick_reference_variable, e.g. "speed") does keep updating live
       though, so the same "wait for real data, not wall clock alone"
       principle still applies to it: keep ticking on every new reference
       minute, and only complete once that reference data reaches the
       target hour (or a 10-minute stall timeout, in case NOAA stops
       publishing entirely).

    2. Kp interval jobs run TWO independent products (see
       predict.predict_kp_interval / predict_kp_rolling): a frozen
       production_prediction set once at job creation (never touched
       here), and a separate, deliberately unrestricted "rolling"
       nowcast ticked here on every new minute. No stall timeout — Kp not
       being published yet is the normal state for the whole target
       interval, not a stuck feed.

    3. Everything else (Dst, Solar Wind, IMF) is checkpoint-based:
       transitions to evaluating once the target variable's own live data
       reaches the target hour (or a 10-minute stall timeout), then ticks
       at discrete trained horizons (1/3/6/12/24h) as the remaining time
       to target narrows.

    In all three cases, transitioning out of 'in_progress' is gated on the
    freshest *available* data reaching the target — never wall-clock time
    alone, which would silently skip whatever readings hadn't landed yet
    at the exact instant poll_jobs happened to run (live_update.py fetches
    on its own ~1-2 minute cadence, so there's normally a small lag).
    """
    dataset = job_row["dataset"]
    variable = job_row["variable"]
    target_hour = pd.Timestamp(job_row["target_hour"])
    now = pd.Timestamp.now(tz="UTC")
    is_kp_interval = dataset in ("analytics", "experimental") and variable == "kp"
    is_static_target = variable in (DATASETS[dataset].static_variables or [])

    if is_static_target:
        ref_minute_ts, ref_minute_val = latest_minute_observation(dataset, _tick_reference_variable(dataset, variable))
        data_has_reached_target = ref_minute_ts is not None and pd.Timestamp(ref_minute_ts) >= target_hour
        stalled_past_target = now >= target_hour + pd.Timedelta(minutes=10)

        # Always the job's OWN horizon model for its whole life — no
        # discrete-horizon checkpoint switching applies here (that's what
        # the shared tick logic below is for), so log every new reference
        # minute directly, right up through the one that first reaches the
        # target hour. Must use job_row["horizon"], not a hardcoded 1: a
        # job started at horizon=24 still needs its final recorded
        # prediction to come from the 24h model, not silently from the 1h
        # one.
        if ref_minute_ts is not None:
            minute_iso = _to_utc_iso(ref_minute_ts)
            if minute_iso != job_row["last_minute_seen"]:
                try:
                    result = predict_live(dataset, variable, job_row["horizon"])
                    _append_tick(
                        conn, job_row["job_id"], ref_minute_ts, ref_minute_val, result["predicted_value"],
                        job_row["horizon"], _capture_live_inputs(dataset),
                    )
                except Exception:
                    conn.execute(
                        "UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_row["job_id"])
                    )

        if data_has_reached_target or stalled_past_target:
            conn.execute(
                "UPDATE jobs SET status='completed', verification_status='pending', completed_at=? WHERE job_id=?",
                (_to_utc_iso(now), job_row["job_id"]),
            )
            if variable == "ae":
                _capture_quicklook_estimate(conn, job_row["job_id"], target_hour)
        return

    if is_kp_interval:
        # Mode 1 (production_prediction, frozen at job creation — see
        # start_job) never gets touched again here. Every tick logged in
        # this branch is Mode 2, the EXPERIMENTAL rolling nowcast
        # (predict_kp_rolling) — deliberately never predict_kp_interval,
        # which would just recompute the same frozen, training-consistent
        # value over and over (it's capped to data before the target
        # interval, which doesn't change while we wait) and would also
        # misuse the production model on fresher, in-interval data the
        # instant that cap were lifted. No stall-timeout fallback either:
        # unlike hourly variables, "NOAA hasn't published yet" is Kp's
        # entirely normal state for the whole 3h+ target interval, not a
        # sign of a stuck feed — transition to evaluating only once the
        # official value has actually arrived.
        target_minute_ts, _ = latest_minute_observation(dataset, variable)
        data_has_reached_target = target_minute_ts is not None and pd.Timestamp(target_minute_ts) >= target_hour
        if data_has_reached_target:
            conn.execute(
                "UPDATE jobs SET status='evaluating', last_minute_seen=NULL WHERE job_id=?",
                (job_row["job_id"],),
            )
            return

        minute_ts, minute_val = latest_minute_observation(dataset, _tick_reference_variable(dataset, variable))
        if minute_ts is None:
            return
        minute_iso = _to_utc_iso(minute_ts)
        if minute_iso == job_row["last_minute_seen"]:
            return
        try:
            rolling_result = predict_kp_rolling(dataset)
        except Exception:
            conn.execute("UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_row["job_id"]))
            return
        _append_tick(
            conn, job_row["job_id"], minute_ts, minute_val, rolling_result["predicted_value"], "rolling",
            _capture_live_inputs(dataset),
        )
        return

    # Has the TARGET variable's own data started arriving for the new
    # interval? Checked against the predicted variable itself (e.g. Dst),
    # since that's specifically asking "has NOAA started publishing for
    # the interval we're forecasting" — not a freshness/tick-timing check.
    target_minute_ts, _ = latest_minute_observation(dataset, variable)
    data_has_reached_target = target_minute_ts is not None and pd.Timestamp(target_minute_ts) >= target_hour
    stalled_past_target = now >= target_hour + pd.Timedelta(minutes=10)
    if data_has_reached_target or stalled_past_target:
        conn.execute(
            "UPDATE jobs SET status='evaluating', last_minute_seen=NULL WHERE job_id=?",
            (job_row["job_id"],),
        )
        return

    # Tick timing/dedup driven by the fastest-updating reference variable
    # instead — see _tick_reference_variable for why.
    minute_ts, minute_val = latest_minute_observation(dataset, _tick_reference_variable(dataset, variable))
    if minute_ts is None:
        return

    minute_iso = _to_utc_iso(minute_ts)
    if minute_iso == job_row["last_minute_seen"]:
        return

    try:
        probe = predict_live(dataset, variable, 1)
    except Exception:
        conn.execute("UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_row["job_id"]))
        return

    observed_at_utc = pd.Timestamp(probe["observed_at"]).tz_localize("UTC")
    remaining = round((target_hour - observed_at_utc).total_seconds() / 3600)

    if remaining <= 0 or remaining not in HORIZONS:
        conn.execute("UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_row["job_id"]))
        return

    try:
        result = predict_live(dataset, variable, remaining)
    except Exception:
        conn.execute("UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_row["job_id"]))
        return

    _append_tick(
        conn, job_row["job_id"], minute_ts, minute_val, result["predicted_value"], remaining,
        _capture_live_inputs(dataset),
    )


def _advance_evaluating(conn, job_row: sqlite3.Row) -> None:
    """Phase 2 (status='evaluating'): the target hour has started.

    Kp and Dst are published by NOAA as a single, already-final value for
    their own native interval — hourly for Dst, every 3 hours for Kp —
    confirmed directly against NOAA's own kyoto-dst.json: exactly one row
    per hour, never sub-hourly. There is no raw minute-level series to
    average for these two; the moment that published value is available,
    it already IS the ground truth the model was trained to predict, so
    the job finalizes immediately instead of waiting out an extra hour.

    Solar Wind and IMF are genuinely different: NOAA publishes them at
    true minute resolution, and the model was trained against the
    *hourly mean* of those minutes — so for those, this function logs
    every new NOAA minute landing inside the target hour (the running
    average "watch it converge" view) and only finalizes once the target
    hour fully closes, using the real hourly-resampled mean.
    """
    dataset = job_row["dataset"]
    variable = job_row["variable"]
    target_hour = pd.Timestamp(job_row["target_hour"])
    now = pd.Timestamp.now(tz="UTC")

    if variable in INSTANT_FINALIZE_VARIABLES:
        # Deliberately resolve_static_actual here, NOT resolve_actual_value:
        # the latter reads load_live_features' hourly-resampled +
        # time-interpolated frame, which was empirically confirmed to
        # silently extrapolate the last published value forward into
        # hours that haven't actually been published yet (e.g. holding
        # Kp's 15:00 reading through 18:00 before NOAA's next official
        # value exists). resolve_static_actual reads the raw source
        # directly with no interpolation and correctly returns None until
        # that exact hour has a genuine published reading.
        actual = resolve_static_actual(dataset, variable, target_hour)
        if actual is not None:
            _finalize_job(conn, job_row, actual)
        return

    if now >= target_hour + pd.Timedelta(hours=1):
        actual = resolve_actual_value(dataset, variable, target_hour)
        if actual is None:
            ticks = _fetch_eval_ticks(conn, job_row["job_id"])
            values = [t["noaa_value"] for t in ticks if t["noaa_value"] is not None]
            actual = (sum(values) / len(values)) if values else None
        if actual is not None:
            _finalize_job(conn, job_row, actual)
        return

    minute_ts, minute_val = latest_minute_observation(dataset, variable)
    if minute_ts is None or minute_val is None:
        return

    minute_iso = _to_utc_iso(minute_ts)
    if minute_iso == job_row["last_minute_seen"]:
        return

    if pd.Timestamp(minute_ts) < target_hour:
        # Stale reading from before the target hour started — nothing new yet.
        conn.execute("UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_row["job_id"]))
        return

    _append_eval_tick(conn, job_row["job_id"], minute_ts, minute_val)


def _verify_static_jobs(conn, dataset: str) -> None:
    """Phase 2 (verification) for variables with no live feed at all (e.g.
    AE) — completely independent of Phase 1 (prediction) above, both in
    when it runs and in its data source. A job reaching 'completed' here
    has already frozen its final prediction purely because the target
    time arrived, with no dependency on an observation ever showing up.

    AE is verified exclusively against Kyoto WDC's published real-time
    (quicklook) AE digital data (swdss.ingest.kyoto_ae) — never NOAA,
    which publishes no AE product at all, and never the local historical
    file predict.resolve_static_actual reads (that's a different, stale
    source used only as the forward-filled live *feature*, not as ground
    truth). Throttled to roughly once per KYOTO_VERIFICATION_INTERVAL per
    job, independent of how often poll_jobs itself is invoked, so a page
    left open with a fast auto-refresh doesn't hammer Kyoto WDC.
    """
    now = pd.Timestamp.now(tz="UTC")
    rows = conn.execute(
        "SELECT * FROM jobs WHERE dataset=? AND status='completed' AND verification_status='pending'",
        (dataset,),
    ).fetchall()
    for row in rows:
        if row["variable"] != "ae":
            continue  # only AE has a Kyoto-based verification source today

        last_checked = row["verification_checked_at"]
        if last_checked is not None and (now - pd.Timestamp(last_checked)) < KYOTO_VERIFICATION_INTERVAL:
            continue

        try:
            actual = fetch_kyoto_ae_hour(pd.Timestamp(row["target_hour"]))
        except Exception:
            continue  # transient network/parse issue — try again next interval

        conn.execute(
            "UPDATE jobs SET verification_checked_at=? WHERE job_id=?",
            (_to_utc_iso(now), row["job_id"]),
        )
        if actual is not None:
            # Persisted permanently: verification_status/verified_at/
            # actual_value never revert once set, regardless of anything
            # that happens to prediction jobs afterward.
            conn.execute(
                "UPDATE jobs SET actual_value=?, verification_status='verified', verified_at=? WHERE job_id=?",
                (actual, _to_utc_iso(now), row["job_id"]),
            )


def _auto_refresh_quicklook_jobs(conn, dataset: str) -> None:
    """Keeps completed AE jobs' Quicklook estimates catching up on their
    own, without the user needing to click "Refresh Quicklook Estimate":
    Kyoto's real-time graph keeps being redrawn throughout the target
    hour, so a job captured early (low hour coverage) should get a fresh,
    more-complete read automatically as the dashboard is left open.

    Stops re-checking once a job's coverage has reached
    QUICKLOOK_COMPLETE_COVERAGE — the hour is fully drawn and won't change
    further. Throttled to QUICKLOOK_RECHECK_INTERVAL per job so a fast
    dashboard auto-refresh doesn't hammer Kyoto's real-time image on every
    rerun.
    """
    if dataset != "ae":
        return

    now = pd.Timestamp.now(tz="UTC")
    rows = conn.execute(
        "SELECT * FROM jobs WHERE dataset=? AND variable='ae' AND status='completed'",
        (dataset,),
    ).fetchall()
    for row in rows:
        coverage = row["quicklook_hour_coverage"]
        if coverage is not None and coverage >= QUICKLOOK_COMPLETE_COVERAGE:
            continue

        last_checked = row["quicklook_checked_at"]
        if last_checked is not None and (now - pd.Timestamp(last_checked)) < QUICKLOOK_RECHECK_INTERVAL:
            continue

        _capture_quicklook_estimate(conn, row["job_id"], pd.Timestamp(row["target_hour"]))


def poll_jobs(dataset: str) -> None:
    """Advances every active job for this dataset (predicting or
    evaluating), then separately checks any completed-but-unverified
    static-target jobs (see _verify_static_jobs) and refreshes any
    not-yet-complete Quicklook estimates (see _auto_refresh_quicklook_jobs).
    Call this on every page render so jobs keep advancing as long as the
    dashboard is open.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND status IN ('in_progress', 'evaluating')",
            (dataset,),
        ).fetchall()
        for row in rows:
            if row["status"] == "in_progress":
                _advance_predicting(conn, row)
            else:
                _advance_evaluating(conn, row)
        _verify_static_jobs(conn, dataset)
        _auto_refresh_quicklook_jobs(conn, dataset)


# ==================== Job Querying ====================
# Read-only lookups the dashboard and the Operational Forecast Engine
# both call — the `source` parameter throughout this section is what
# keeps a user's manually-started jobs and the engine's own automatic
# ones from crowding each other out (see the `source` column's own note
# in _init_db).

def get_running_jobs(dataset: str, limit: int = JOB_HISTORY_LIMIT, source: str | None = "manual") -> list[dict]:
    """Returns this dataset's non-saved jobs (in-progress or completed but
    never saved), newest first, capped at `limit` — except for
    UNCAPPED_HISTORY_DATASETS (currently just "ae"), where every job stays
    visible indefinitely; only an explicit Delete ever removes one. Once a
    (capped-dataset) job is saved it moves out of this list entirely and
    into get_saved_jobs.

    `source` defaults to 'manual' — the Production tabs' "Running
    Predictions" list (every existing call site) should only ever show a
    user's own started jobs, never the Operational Forecast Engine's
    continuous automatic ones, which would otherwise crowd out (or, for
    capped datasets, eventually push out of the LIMIT window entirely) a
    job the user started specifically to study. Pass source='engine' or
    source=None (no filter — every job regardless of source) explicitly
    where that's genuinely what's wanted (see swdss.engine.orchestrator).
    """
    with _connect() as conn:
        source_clause = "" if source is None else "AND source=? "
        source_params = () if source is None else (source,)
        if dataset in UNCAPPED_HISTORY_DATASETS:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE dataset=? AND saved=0 {source_clause}ORDER BY created_at DESC",
                (dataset, *source_params),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE dataset=? AND saved=0 {source_clause}ORDER BY created_at DESC LIMIT ?",
                (dataset, *source_params, limit),
            ).fetchall()
        jobs = [_row_to_job(r) for r in rows]
        for job in jobs:
            job["ticks"] = _fetch_ticks(conn, job["job_id"])
            job["eval_ticks"] = _fetch_eval_ticks(conn, job["job_id"])
        return jobs


def get_saved_jobs(dataset: str, source: str | None = "manual") -> list[dict]:
    """Returns every saved job for this dataset, newest first, with no cap
    — saved predictions are meant to stick around indefinitely. Same
    source='manual' default as get_running_jobs, for the same reason —
    saving is a manual-workflow action a user takes from the Production
    tabs, and an engine job is never exposed to that UI to begin with.
    """
    with _connect() as conn:
        source_clause = "" if source is None else "AND source=? "
        source_params = () if source is None else (source,)
        rows = conn.execute(
            f"SELECT * FROM jobs WHERE dataset=? AND saved=1 {source_clause}ORDER BY created_at DESC",
            (dataset, *source_params),
        ).fetchall()
        jobs = [_row_to_job(r) for r in rows]
        for job in jobs:
            job["ticks"] = _fetch_ticks(conn, job["job_id"])
            job["eval_ticks"] = _fetch_eval_ticks(conn, job["job_id"])
        return jobs


def get_job(job_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        job = _row_to_job(row)
        job["ticks"] = _fetch_ticks(conn, job_id)
        job["eval_ticks"] = _fetch_eval_ticks(conn, job_id)
        return job


def find_matching_job(dataset: str, variable: str, target_hour) -> dict:
    """Finds the most recent job for `dataset`/`variable` whose target hour
    exactly matches `target_hour` — used to look up a production
    ("analytics") counterpart for an experimental job (or vice versa) so
    the dashboard can show them side by side. Returns None if no
    comparable job exists yet (e.g. no production job was ever started
    for that same target hour) — the comparison is "whenever possible",
    not guaranteed.
    """
    target_iso = _to_utc_iso(target_hour)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND variable=? AND target_hour=? ORDER BY created_at DESC LIMIT 1",
            (dataset, variable, target_iso),
        ).fetchone()
        if row is None:
            return None
        job = _row_to_job(row)
        job["ticks"] = _fetch_ticks(conn, job["job_id"])
        job["eval_ticks"] = _fetch_eval_ticks(conn, job["job_id"])
        return job


def has_active_job(dataset: str, variable: str, horizon: int, source: str = "engine") -> bool:
    """Cheap precheck for the Operational Forecast Engine's run_forecast_
    cycle(): is there already an in_progress/evaluating job for this exact
    (dataset, variable, horizon)? A single indexed-ish SQLite read, with no
    predict_live call — unlike start_job, which always pays the full
    feature-build + inference cost before it can even check for a
    duplicate. Steady-state behavior this enables: the engine skips
    calling start_job() for a combination that's still being tracked, and
    starts a fresh one within one cycle of the previous one completing —
    without needing to independently reproduce predict.py's hour/interval
    resolution logic just to decide whether to bother asking.

    Filtered to source='engine' by default so a user manually starting
    the identical (dataset, variable, horizon) from a Production tab can
    never block the engine's own automatic job creation for that slot.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE dataset=? AND variable=? AND horizon=? AND status IN ('in_progress', 'evaluating') AND source=? LIMIT 1",
            (dataset, variable, horizon, source),
        ).fetchone()
        return row is not None


# ==================== Job Management ====================

def save_job(job_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE jobs SET saved=1 WHERE job_id=?", (job_id,))
        return cur.rowcount > 0


def delete_job(job_id: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM ticks WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM eval_ticks WHERE job_id=?", (job_id,))
        cur = conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        return cur.rowcount > 0


PRUNE_RETENTION_DAYS = 7


def prune_old_engine_jobs(retention_days: int = PRUNE_RETENTION_DAYS) -> int:
    """Deletes engine-sourced jobs (and their ticks/eval_ticks) that
    finished more than `retention_days` ago — the only thing keeping
    predictions.db from growing forever now that run_forecast_cycle()
    creates jobs automatically, continuously, with no user ever clicking
    Delete. Manual jobs are NEVER touched here regardless of age — they
    persist exactly as they always have, until a user explicitly deletes
    one (see get_running_jobs/get_saved_jobs' docstrings). Only terminal-
    state jobs ('completed' or 'stopped') are eligible — an in_progress
    or evaluating job is never pruned, engine or not.

    Returns the number of jobs deleted, for logging.
    """
    cutoff_iso = _to_utc_iso(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=retention_days))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT job_id FROM jobs WHERE source='engine' AND status IN ('completed', 'stopped') "
            "AND completed_at IS NOT NULL AND completed_at < ?",
            (cutoff_iso,),
        ).fetchall()
        job_ids = [r["job_id"] for r in rows]
        for job_id in job_ids:
            conn.execute("DELETE FROM ticks WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM eval_ticks WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        return len(job_ids)


def stop_job(job_id: str) -> bool:
    """Manually halts an active job before its target ever arrives. Marked
    'stopped' (not 'completed') since there was never a real observation to
    evaluate against — distinct from a job that ran its full course.
    """
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='stopped', completed_at=? WHERE job_id=? AND status IN ('in_progress', 'evaluating')",
            (_to_utc_iso(pd.Timestamp.now(tz="UTC")), job_id),
        )
        return cur.rowcount > 0


# ==================== Statistics & Metrics ====================
# Read-only aggregation over already-stored job data — accuracy labels,
# stability/drift metrics, and dataset-wide statistics panels. Nothing
# below this point writes to the database.

def job_mae(job: dict) -> float:
    if job["actual_value"] is None or not job["ticks"]:
        return None
    errors = [abs(t["predicted_value"] - job["actual_value"]) for t in job["ticks"]]
    return sum(errors) / len(errors)


def final_percentage_error(job: dict) -> float:
    """Absolute Error as a percentage of the actual value, using the
    session's FINAL prediction (the operational forecast) — None if not
    yet verified, or if the actual value is exactly zero (AE is rarely
    but can be 0 at very quiet times, and a percentage of zero is
    undefined).
    """
    actual = job["actual_value"]
    if actual is None or not job["ticks"] or actual == 0:
        return None
    final_pred = job["ticks"][-1]["predicted_value"]
    return abs(final_pred - actual) / abs(actual) * 100


def production_error(job: dict) -> float:
    """Kp interval jobs only: absolute error of the FROZEN Mode 1
    production forecast against the official Kp value — this is the
    number that feeds official model evaluation (aggregated across many
    jobs into R²/MAE/RMSE elsewhere). None until production_prediction
    and actual_value are both set.
    """
    prod = job.get("production_prediction")
    actual = job.get("actual_value")
    if prod is None or actual is None:
        return None
    return abs(prod - actual)


def production_bias(job: dict) -> float:
    """Signed error (prediction - actual) of the frozen production
    forecast. Unlike production_error, the sign is the point here:
    persistently positive means the model runs high, persistently
    negative means it runs low — production_error alone can't show that.
    """
    prod = job.get("production_prediction")
    actual = job.get("actual_value")
    if prod is None or actual is None:
        return None
    return prod - actual


def rolling_final_error(job: dict) -> float:
    """Kp interval jobs only: absolute error of the FINAL Mode 2 rolling
    estimate (the last tick logged before the job resolved) against the
    official Kp value — kept strictly for research comparison against
    production_error, never as a substitute for it and never used for
    official model evaluation.
    """
    actual = job.get("actual_value")
    ticks = job.get("ticks")
    if actual is None or not ticks:
        return None
    return abs(ticks[-1]["predicted_value"] - actual)


def quicklook_error(job: dict) -> float:
    """Approximate error against the Quicklook estimate — NOT the official
    error (see final_percentage_error / job's actual_value). None if no
    Quicklook estimate has been captured yet.
    """
    quicklook_ae = job.get("quicklook_ae")
    if quicklook_ae is None or not job["ticks"]:
        return None
    final_pred = job["ticks"][-1]["predicted_value"]
    return abs(final_pred - quicklook_ae)


def quicklook_relative_error(job: dict) -> float:
    """quicklook_error expressed as a percentage of the Quicklook estimate,
    for context alongside the raw nT figure. None under the same
    conditions as quicklook_error, plus when the estimate is exactly zero.
    """
    quicklook_ae = job.get("quicklook_ae")
    error = quicklook_error(job)
    if error is None or not quicklook_ae:
        return None
    return error / abs(quicklook_ae) * 100


# Configurable thresholds (nT) for labeling how close a prediction landed
# to the Quicklook estimate. Approximate by construction — the Quicklook
# estimate itself carries several nT of extraction uncertainty — so these
# bands are deliberately coarse rather than precise error bars.
QUICKLOOK_ERROR_BANDS = (
    (25.0, "Excellent"),
    (75.0, "Good"),
    (150.0, "Moderate"),
)
QUICKLOOK_ERROR_FALLBACK_LABEL = "Large"


def classify_quicklook_error(error: float | None) -> str:
    """Maps a quicklook_error value to a coarse label via
    QUICKLOOK_ERROR_BANDS. None if no error is available yet."""
    if error is None:
        return None
    for threshold, label in QUICKLOOK_ERROR_BANDS:
        if error <= threshold:
            return label
    return QUICKLOOK_ERROR_FALLBACK_LABEL


def quicklook_label(job: dict) -> str:
    """Titles the Quicklook estimate card based on how much of the target
    hour Kyoto's graph had actually drawn at capture time — never presents
    a low-coverage reading under the same label as a fully-drawn one:

    - No estimate captured yet: "Quicklook Estimated AE" (the neutral
      default, paired with an N/A value).
    - coverage < QUICKLOOK_PARTIAL_COVERAGE: "Partial Quicklook Estimate" —
      explicitly not representative of the eventual full-hour value.
    - coverage >= QUICKLOOK_COMPLETE_COVERAGE: "Complete Quicklook
      Estimate" — the hour is (essentially) fully drawn.
    - Otherwise: "Quicklook Estimated AE" — meaningful coverage, but not
      quite complete.
    """
    coverage = job.get("quicklook_hour_coverage")
    if job.get("quicklook_ae") is None or coverage is None:
        return "Quicklook Estimated AE"
    if coverage < QUICKLOOK_PARTIAL_COVERAGE:
        return "Partial Quicklook Estimate"
    if coverage >= QUICKLOOK_COMPLETE_COVERAGE:
        return "Complete Quicklook Estimate"
    return "Quicklook Estimated AE"


def average_prediction(job: dict) -> float:
    """Mean of every prediction generated during the session — a stability
    indicator showing how much the forecast moved around, not the
    operational forecast itself (that's the final prediction).
    """
    if not job["ticks"]:
        return None
    values = [t["predicted_value"] for t in job["ticks"]]
    return sum(values) / len(values)


def forecast_drift(job: dict) -> float:
    """Final prediction minus the very first prediction — how much the
    forecast moved from start to end of the session.
    """
    if len(job["ticks"]) < 2:
        return None
    return job["ticks"][-1]["predicted_value"] - job["ticks"][0]["predicted_value"]


def stability_metric(job: dict) -> tuple:
    """Standard deviation of the most recent ticks' predicted values, as a
    rough signal of whether the forecast is converging or still bouncing
    around. Not a statistical confidence interval — just a relative
    dispersion measure over the latest window of ticks. Returns
    (label, delta) or (None, None) if there aren't enough ticks yet to say
    anything meaningful (fewer than 2).
    """
    recent = [t["predicted_value"] for t in job["ticks"][-STABILITY_WINDOW:]]
    if len(recent) < 2:
        return None, None
    mean_v = sum(recent) / len(recent)
    variance = sum((v - mean_v) ** 2 for v in recent) / len(recent)
    std = variance**0.5
    pct = (std / abs(mean_v) * 100) if mean_v else 0.0
    if pct < 2:
        label = "Stable"
    elif pct < 6:
        label = "Moderately Stable"
    else:
        label = "Unstable"
    return label, std


def model_quality_label(r2: float) -> str:
    """Categorizes a model's held-out R² into a plain-language quality
    tier. This describes the MODEL itself (how well it performed during
    training), not any individual forecast it produced.
    """
    if r2 is None:
        return "N/A"
    if r2 >= 0.95:
        return "Excellent"
    if r2 >= 0.85:
        return "Good"
    if r2 >= 0.70:
        return "Moderate"
    if r2 >= 0.50:
        return "Low"
    return "Poor"


def forecast_evaluation_label(abs_error: float, mae: float) -> str:
    """Categorizes how good THIS SPECIFIC forecast was, by comparing its
    absolute error against the model's own typical error (MAE) rather than
    against the actual value's magnitude — a percentage-of-actual measure
    is misleading for variables like Bz or Density that pass near zero.
    A ratio near or below 1.0 means the forecast performed about as well
    as (or better than) the model usually does.
    """
    if abs_error is None or mae is None or mae == 0:
        return "N/A"
    ratio = abs_error / mae
    if ratio <= 0.5:
        return "Excellent"
    if ratio <= 1.0:
        return "Good"
    if ratio <= 1.5:
        return "Moderate"
    if ratio <= 2.5:
        return "Low"
    return "Poor"


def get_job_stats(dataset: str) -> dict:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE dataset=?", (dataset,)).fetchall()
        jobs = [_row_to_job(r) for r in rows]

        running = sum(1 for j in jobs if j["status"] in ("in_progress", "evaluating"))
        today = pd.Timestamp.now(tz="UTC").date()
        completed_today = sum(
            1
            for j in jobs
            if j["status"] == "completed" and pd.Timestamp(j["created_at"]).date() == today
        )

        maes = []
        for j in jobs:
            if j["status"] == "completed" and j["actual_value"] is not None:
                j["ticks"] = _fetch_ticks(conn, j["job_id"])
                mae = job_mae(j)
                if mae is not None:
                    maes.append(mae)

        avg_mae = sum(maes) / len(maes) if maes else None
        return {"running": running, "completed_today": completed_today, "avg_mae": avg_mae}


def get_prediction_statistics(dataset: str) -> dict:
    """Aggregates every completed, evaluated forecast for this dataset into
    system-level performance metrics. This describes how well the
    forecasting system performs overall — not any single forecast.

    "Success" is defined as a forecast whose final absolute error came in
    at or below 1.5x the model's own typical error (MAE) — i.e. it
    performed about as well as, or not much worse than, the model usually
    does. This is a deliberate choice, not a NOAA/industry-standard
    definition.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND status='completed' AND actual_value IS NOT NULL",
            (dataset,),
        ).fetchall()
        jobs = [_row_to_job(r) for r in rows]
        for j in jobs:
            j["ticks"] = _fetch_ticks(conn, j["job_id"])

    records = []
    for j in jobs:
        if not j["ticks"]:
            continue
        final_pred = j["ticks"][-1]["predicted_value"]
        records.append(
            {
                "variable": j["variable"],
                "horizon": j["horizon"],
                "model_name": j["model_name"],
                "abs_error": abs(final_pred - j["actual_value"]),
                "mae": job_mae(j),
                "completed_at": j["completed_at"] or j["created_at"],
            }
        )

    if not records:
        return {
            "count": 0,
            "success_rate": None,
            "mae_by_variable": {},
            "mae_by_horizon": {},
            "model_usage": {},
            "trend": [],
        }

    def _avg(values):
        return sum(values) / len(values) if values else None

    variables = {r["variable"] for r in records}
    mae_by_variable = {v: _avg([r["abs_error"] for r in records if r["variable"] == v]) for v in variables}

    horizons = {r["horizon"] for r in records}
    mae_by_horizon = {h: _avg([r["abs_error"] for r in records if r["horizon"] == h]) for h in horizons}

    model_usage: dict = {}
    for r in records:
        model_usage[r["model_name"]] = model_usage.get(r["model_name"], 0) + 1

    model_names = {r["model_name"] for r in records}
    mae_by_model = {m: _avg([r["abs_error"] for r in records if r["model_name"] == m]) for m in model_names}
    best_model = min(mae_by_model, key=mae_by_model.get) if mae_by_model else None

    evaluable = [r for r in records if r["mae"]]
    successful = sum(1 for r in evaluable if r["abs_error"] <= 1.5 * r["mae"])
    success_rate = (successful / len(evaluable) * 100) if evaluable else None

    trend = sorted([(r["completed_at"], r["abs_error"]) for r in records], key=lambda t: t[0])

    return {
        "count": len(records),
        "success_rate": success_rate,
        "mae_by_variable": mae_by_variable,
        "mae_by_horizon": mae_by_horizon,
        "model_usage": model_usage,
        "mae_by_model": mae_by_model,
        "best_model": best_model,
        "trend": trend,
    }


# Simple, explicit thresholds for "storm" vs "quiet" segmentation — not
# NOAA/industry-official cutoffs, just the standard textbook ballpark
# (G1+ Kp, NOAA minor-storm Dst, active/storm-level AE) used consistently
# across this dashboard's own status labels (see variable_meaning_and_risk
# in dashboard/home.py).
_STORM_CHECKS = {
    "kp": lambda v: v >= 5,
    "dst": lambda v: v <= -50,
    "ae": lambda v: v >= 300,
}


def compute_variable_metrics(dataset: str, variable: str) -> dict:
    """Comprehensive research metrics (MAE, RMSE, R², MAPE, Bias, Max
    Error, Median Error, Average Drift, Forecast Stability, Storm vs.
    Quiet performance) across every completed & verified job for this
    dataset/variable. This is the measurement layer Hypothesis Testing
    and the Research Lab's metrics panels are built on — it never
    influences prediction, only reports on forecasts already made.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND variable=? AND status='completed' AND actual_value IS NOT NULL",
            (dataset, variable),
        ).fetchall()
        jobs = [_row_to_job(r) for r in rows]
        for j in jobs:
            j["ticks"] = _fetch_ticks(conn, j["job_id"])

    records = []
    for j in jobs:
        if not j["ticks"]:
            continue
        final_pred = j["ticks"][-1]["predicted_value"]
        actual = j["actual_value"]
        records.append(
            {
                "predicted": final_pred,
                "actual": actual,
                "drift": forecast_drift(j),
                "completed_at": j["completed_at"] or j["created_at"],
            }
        )

    empty = {
        "count": 0,
        "mae": None,
        "rmse": None,
        "r2": None,
        "mape": None,
        "bias": None,
        "max_error": None,
        "median_error": None,
        "avg_drift": None,
        "stability": None,
        "storm_mae": None,
        "storm_count": 0,
        "quiet_mae": None,
        "quiet_count": 0,
        "trend": [],
        "predicted_vs_actual": [],
        "errors": [],
    }
    if not records:
        return empty

    predicted = np.array([r["predicted"] for r in records], dtype=float)
    actual = np.array([r["actual"] for r in records], dtype=float)
    errors = predicted - actual
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    max_error = float(np.max(abs_errors))
    median_error = float(np.median(abs_errors))

    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    nonzero = actual != 0
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None

    drifts = [r["drift"] for r in records if r["drift"] is not None]
    avg_drift = float(np.mean(np.abs(drifts))) if drifts else None
    stability = float(np.std(drifts)) if len(drifts) > 1 else None

    is_storm = _STORM_CHECKS.get(variable)
    if is_storm:
        storm_errors = [abs(r["predicted"] - r["actual"]) for r in records if is_storm(r["actual"])]
        quiet_errors = [abs(r["predicted"] - r["actual"]) for r in records if not is_storm(r["actual"])]
    else:
        storm_errors, quiet_errors = [], []

    trend = sorted([(r["completed_at"], abs(r["predicted"] - r["actual"])) for r in records], key=lambda t: t[0])
    predicted_vs_actual = [(r["predicted"], r["actual"]) for r in records]

    return {
        "count": len(records),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "bias": bias,
        "max_error": max_error,
        "median_error": median_error,
        "avg_drift": avg_drift,
        "stability": stability,
        "storm_mae": float(np.mean(storm_errors)) if storm_errors else None,
        "storm_count": len(storm_errors),
        "quiet_mae": float(np.mean(quiet_errors)) if quiet_errors else None,
        "quiet_count": len(quiet_errors),
        "trend": trend,
        "predicted_vs_actual": predicted_vs_actual,
        "errors": errors.tolist(),
    }


# ==================== Public Entry Point ====================

def tick_all_active_jobs() -> None:
    """Advance every active prediction job across all datasets.

    Called by live_update.py at the end of each update cycle so jobs tick
    continuously in the background, completely independent of which dashboard
    page is open or whether the browser is open at all. The dashboard itself
    becomes a pure read/display layer — it never needs to call poll_jobs to
    advance a job, only to read results.

    poll_jobs is safe to call from any process: it reads from and writes to
    SQLite only, with no Streamlit dependency. Errors are swallowed per-dataset
    so one failing dataset never blocks the others.
    """
    all_datasets = list(DATASETS.keys())
    for dataset in all_datasets:
        try:
            poll_jobs(dataset)
        except Exception:
            import traceback
            print(f"[tick_all_active_jobs] Error ticking jobs for '{dataset}':")
            traceback.print_exc()
