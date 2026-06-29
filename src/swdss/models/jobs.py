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


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
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
        "SELECT minute_at, noaa_value, predicted_value, used_horizon, logged_at "
        "FROM ticks WHERE job_id = ? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _append_tick(conn, job_id: str, minute_ts, minute_val, predicted_value: float, used_horizon: int) -> None:
    minute_iso = None if minute_ts is None else _to_utc_iso(minute_ts)
    conn.execute(
        "INSERT INTO ticks (job_id, minute_at, noaa_value, predicted_value, used_horizon, logged_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, minute_iso, minute_val, predicted_value, used_horizon, _to_utc_iso(pd.Timestamp.now(tz="UTC"))),
    )
    conn.execute("UPDATE jobs SET last_minute_seen = ? WHERE job_id = ?", (minute_iso, job_id))


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
            return job, False

        minute_ts, minute_val = latest_minute_observation(dataset, variable)

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
        _append_tick(conn, job_id, minute_ts, minute_val, result["predicted_value"], horizon)

        job = _row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        job["ticks"] = _fetch_ticks(conn, job_id)
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


def _advance_job(conn, job_row: sqlite3.Row) -> None:
    dataset = job_row["dataset"]
    variable = job_row["variable"]
    target_hour = pd.Timestamp(job_row["target_hour"])
    now = pd.Timestamp.now(tz="UTC")

    if now >= target_hour:
        actual = resolve_actual_value(dataset, variable, target_hour)
        if actual is not None:
            _finalize_job(conn, job_row, actual)
            return

    minute_ts, minute_val = latest_minute_observation(dataset, variable)
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

    _append_tick(conn, job_row["job_id"], minute_ts, minute_val, result["predicted_value"], remaining)


def poll_jobs(dataset: str) -> None:
    """Advances every in-progress job for this dataset. Call this on every
    page render so jobs keep advancing as long as the dashboard is open.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? AND status='in_progress'", (dataset,)
        ).fetchall()
        for row in rows:
            _advance_job(conn, row)


def get_jobs(dataset: str, limit: int = JOB_HISTORY_LIMIT) -> list[dict]:
    """Returns jobs for this dataset, newest first. Saved jobs are always
    included regardless of age; unsaved jobs are capped to the most recent
    slots remaining after saved ones, so the tile grid doesn't grow forever.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE dataset=? ORDER BY created_at DESC", (dataset,)
        ).fetchall()
        jobs = [_row_to_job(r) for r in rows]

        saved = [j for j in jobs if j["saved"]]
        unsaved = [j for j in jobs if not j["saved"]]
        remaining_slots = max(limit - len(saved), 0)
        combined = saved + unsaved[:remaining_slots]
        combined.sort(key=lambda j: j["created_at"], reverse=True)

        for job in combined:
            job["ticks"] = _fetch_ticks(conn, job["job_id"])
        return combined


def get_job(job_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        job = _row_to_job(row)
        job["ticks"] = _fetch_ticks(conn, job_id)
        return job


def save_job(job_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE jobs SET saved=1 WHERE job_id=?", (job_id,))
        return cur.rowcount > 0


def delete_job(job_id: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM ticks WHERE job_id=?", (job_id,))
        cur = conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        return cur.rowcount > 0


def job_mae(job: dict) -> float:
    if job["actual_value"] is None or not job["ticks"]:
        return None
    errors = [abs(t["predicted_value"] - job["actual_value"]) for t in job["ticks"]]
    return sum(errors) / len(errors)


def stability_metric(job: dict) -> tuple:
    """Standard deviation of the most recent ticks' predicted values, as a
    rough signal of whether the forecast is converging or still bouncing
    around. Not a statistical confidence interval — just a relative
    dispersion measure over the latest window of ticks.
    """
    recent = [t["predicted_value"] for t in job["ticks"][-STABILITY_WINDOW:]]
    if len(recent) < 2:
        return None, None
    mean_v = sum(recent) / len(recent)
    variance = sum((v - mean_v) ** 2 for v in recent) / len(recent)
    std = variance**0.5
    pct = (std / abs(mean_v) * 100) if mean_v else 0.0
    if pct < 1:
        label = "Very Stable"
    elif pct < 3:
        label = "Stable"
    elif pct < 8:
        label = "Fluctuating"
    else:
        label = "Unstable"
    return label, std


def confidence_pct(job: dict) -> float:
    """Confidence proxy derived from the originally-selected model's R² on
    its held-out test split. Not a calibrated prediction interval — these
    are point-estimate regressors, not probabilistic models — just a
    simple, clearly-labeled translation of "how much of the variance this
    model explained during training" into a 0-100 scale.
    """
    r2 = (job.get("metrics") or {}).get("r2")
    if r2 is None:
        return None
    return max(0.0, min(100.0, r2 * 100))


def accuracy_label(error_pct: float) -> str:
    if error_pct < 2:
        return "Excellent"
    if error_pct < 5:
        return "Good"
    if error_pct < 10:
        return "Fair"
    return "Poor"


def get_job_stats(dataset: str) -> dict:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE dataset=?", (dataset,)).fetchall()
        jobs = [_row_to_job(r) for r in rows]

        running = sum(1 for j in jobs if j["status"] == "in_progress")
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
