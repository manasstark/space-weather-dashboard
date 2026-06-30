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

import pandas as pd

from swdss.models.predict import latest_minute_observation, load_live_features
from swdss.models.predict import predict as predict_live
from swdss.models.registry import HORIZONS
from swdss.paths import DATA_DIR

DB_PATH = DATA_DIR / "predictions" / "predictions.db"

JOB_HISTORY_LIMIT = 20
STABILITY_WINDOW = 10

# The live Solar Wind + IMF + Geomagnetic readings shown per-tick in the
# Analytics page's terminal, since its predictions are driven by all of
# them together, not a single self-referential variable like the
# standalone tabs. Dst and Kp are included too — for the Kp interval
# forecast specifically, "Current Dst" and "Previous Kp" are explicit
# table columns, and Previous Kp is expected to stay fixed for the whole
# session (it only changes when NOAA publishes a new official value).
ANALYTICS_INPUT_VARIABLES = ["speed", "density", "temperature", "bt", "bx_gsm", "by_gsm", "bz_gsm", "dst", "kp"]


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


def _ensure_column(conn, table: str, column: str, decl: str) -> None:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


_init_db()


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
    Analytics page's multi-input terminal display. Returns {} for
    single-source datasets (nothing extra to show beyond their own value).
    """
    if dataset != "analytics":
        return {}
    inputs = {}
    for var in ANALYTICS_INPUT_VARIABLES:
        _, value = latest_minute_observation(dataset, var)
        inputs[var] = value
    return inputs


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
    """
    return "speed" if dataset == "analytics" else variable


def start_job(dataset: str, variable: str, horizon: int) -> tuple:
    """Starts a new continuous prediction job, or returns the existing one
    if a job for this exact (dataset, variable, horizon, start_hour) is
    already in progress.

    Returns (job, created) where created=False signals a duplicate.
    """
    result = predict_live(dataset, variable, horizon)
    start_hour_iso = _to_utc_iso(result["observed_at"])

    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND variable=? AND horizon=? AND start_hour=? AND status='in_progress'",
            (dataset, variable, horizon, start_hour_iso),
        ).fetchone()
        if existing is not None:
            job = _row_to_job(existing)
            job["ticks"] = _fetch_ticks(conn, job["job_id"])
            job["eval_ticks"] = _fetch_eval_ticks(conn, job["job_id"])
            return job, False

        minute_ts, minute_val = latest_minute_observation(dataset, _tick_reference_variable(dataset, variable))

        job_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO jobs (job_id, dataset, variable, horizon, start_hour, target_hour, model_name, "
            "metrics_json, status, saved, created_at, last_minute_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL)",
            (
                job_id,
                dataset,
                variable,
                horizon,
                start_hour_iso,
                _to_utc_iso(result["predicted_for"]),
                result["model_name"],
                json.dumps(result["metrics"]),
                "in_progress",
                _to_utc_iso(pd.Timestamp.now(tz="UTC")),
            ),
        )
        used_horizon = "interval" if (dataset == "analytics" and variable == "kp") else horizon
        _append_tick(
            conn, job_id, minute_ts, minute_val, result["predicted_value"], used_horizon, _capture_live_inputs(dataset)
        )

        job = _row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        job["ticks"] = _fetch_ticks(conn, job_id)
        job["eval_ticks"] = _fetch_eval_ticks(conn, job_id)
        return job, True


def resolve_actual_value(dataset: str, variable: str, target_hour) -> float:
    """Looks up the hourly-mean actual value for a target hour from live
    data, once that hour has occurred. Returns None ("Pending") if the hour
    hasn't arrived yet or live data doesn't cover it.
    """
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


def _advance_predicting(conn, job_row: sqlite3.Row) -> None:
    """Phase 1 (status='in_progress'): refine the forecast for the fixed
    target hour every time a new NOAA minute arrives, switching models at
    checkpoints as the remaining time to target narrows.

    Transitioning to the evaluation phase is gated on the freshest
    *available* NOAA minute reaching the target hour — not on wall-clock
    time alone. live_update.py fetches on its own ~1-2 minute cadence, so
    at the instant the wall clock crosses into the target hour, the local
    data can still be a couple of minutes behind (e.g. the 07:57/07:59
    readings might not land until just after 08:00). Transitioning on wall
    clock alone would skip logging those last ticks entirely, since once a
    job leaves 'in_progress' nothing ever logs a prediction tick again.
    A timeout fallback (10 min past target) avoids getting stuck forever
    if NOAA stops publishing.
    """
    dataset = job_row["dataset"]
    variable = job_row["variable"]
    target_hour = pd.Timestamp(job_row["target_hour"])
    now = pd.Timestamp.now(tz="UTC")
    is_kp_interval = dataset == "analytics" and variable == "kp"

    # Has the TARGET variable's own data started arriving for the new
    # interval? Checked against the predicted variable itself (e.g. Kp),
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

    if is_kp_interval:
        # Only one model (not five discrete horizons) — it already learned
        # to handle a 1-3 hour lookahead directly from training, so every
        # new minute just ticks straight through, no checkpoint gating.
        try:
            result = predict_live(dataset, variable, 1)
        except Exception:
            conn.execute("UPDATE jobs SET last_minute_seen=? WHERE job_id=?", (minute_iso, job_row["job_id"]))
            return
        _append_tick(
            conn, job_row["job_id"], minute_ts, minute_val, result["predicted_value"], "interval",
            _capture_live_inputs(dataset),
        )
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
    """Phase 2 (status='evaluating'): the target hour has started. Log
    every new NOAA minute that lands inside it, plus the running average
    of everything collected so far — this is the "watch the actual average
    converge" view. Once the target hour fully closes, finalize with the
    real hourly-resampled mean (falling back to the collected running
    average if live data hasn't caught up to a full resample yet).
    """
    dataset = job_row["dataset"]
    variable = job_row["variable"]
    target_hour = pd.Timestamp(job_row["target_hour"])
    now = pd.Timestamp.now(tz="UTC")

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


def poll_jobs(dataset: str) -> None:
    """Advances every active job for this dataset (predicting or
    evaluating). Call this on every page render so jobs keep advancing as
    long as the dashboard is open.
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


def get_running_jobs(dataset: str, limit: int = JOB_HISTORY_LIMIT) -> list[dict]:
    """Returns this dataset's non-saved jobs (in-progress or completed but
    never saved), newest first, capped at `limit`. Once a job is saved it
    moves out of this list entirely and into get_saved_jobs.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND saved=0 ORDER BY created_at DESC LIMIT ?",
            (dataset, limit),
        ).fetchall()
        jobs = [_row_to_job(r) for r in rows]
        for job in jobs:
            job["ticks"] = _fetch_ticks(conn, job["job_id"])
            job["eval_ticks"] = _fetch_eval_ticks(conn, job["job_id"])
        return jobs


def get_saved_jobs(dataset: str) -> list[dict]:
    """Returns every saved job for this dataset, newest first, with no cap
    — saved predictions are meant to stick around indefinitely.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND saved=1 ORDER BY created_at DESC",
            (dataset,),
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


def job_mae(job: dict) -> float:
    if job["actual_value"] is None or not job["ticks"]:
        return None
    errors = [abs(t["predicted_value"] - job["actual_value"]) for t in job["ticks"]]
    return sum(errors) / len(errors)


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
