"""Read/write helpers for the Operational Forecast Engine's stored
products — the storage layer dashboard/lib/command_centre.py reads from
for every tab except Search. The dashboard never calls a prediction
model or touches jobs.py's SQLite DB directly; the one deliberate
exception is the Search tab's 7-Day Extremes table, which reads
data/processed/ parquet directly (via swdss.paths.PROCESSED_DIR) since
those are observed values, not a forecast product this module owns.

- current/forecast_snapshot.json: single, always-overwritten "latest
  state" — Forecasting/Current/Physics/Alerts/System tabs.
- history/forecast_snapshot_history.parquet: one row per (dataset,
  variable, horizon) appended for headline slots only — Timeline tab.
  Rolling window, see FORECAST_HISTORY_RETENTION_HOURS.
- history/evaluation_history.parquet: one row per job, appended once
  (deduped by job_id) the first cycle it's observed completed with a
  real actual_value — Verification tab.
- logs/engine_log.jsonl: structured per-stage log lines, trimmed to the
  most recent MAX_LOG_LINES — Logs tab.
- current/forecast_package.json: the current Forecast Package (see
  swdss.engine.packages) — the synchronized operational product bundling
  the 10 headline forecasts. The package banner reads only this file.
- history/package_history.parquet: one row per LIFECYCLE CHANGE of a
  package (not one row per cycle), keyed by package_id — see
  PACKAGE_HISTORY_RETENTION_DAYS.
- history/package_verification_history.parquet: one row per package,
  written once its core (non-AE) members are all verified — the
  permanent Package Verification Summary record.
- history/hourly_report.parquet: the Reports tab's single file — one row
  per (variable, horizon, valid_end) written the moment that headline
  forecast is evaluated against its real observed value: predicted vs.
  actual, error, confidence, and a plain-language problem flag. Rolling
  window by the day being reported on, see REPORT_HISTORY_RETENTION_DAYS.
  This is the one file meant to be read directly by a human — everything
  else above is the engine's own operational/live state.
- current/skill_scores.json: swdss.engine.skill's persistence-based
  skill score per (dataset, variable, horizon), recomputed every cycle
  from the full evaluation history — Verification tab's "Forecast Skill
  vs. Persistence" table.
- current/calibration_report.json: swdss.engine.calibration's
  confidence-reliability and activity-regime error-band tables,
  recomputed every cycle from the full evaluation history —
  Verification tab's calibration sections.
- current/solar_forecast.json: F10.7 harmonic-regression outlook
  (swdss.models.f107_forecast) and Drag-Based Model CME arrival
  estimates (swdss.physics.cme_dbm), recomputed every cycle — the Solar
  Forecast tab's only data source.
"""

import json
import os

import pandas as pd

from swdss.paths import FORECASTS_DIR

CURRENT_SNAPSHOT_PATH = FORECASTS_DIR / "current" / "forecast_snapshot.json"
CURRENT_SOLAR_FORECAST_PATH = FORECASTS_DIR / "current" / "solar_forecast.json"
FORECAST_HISTORY_PATH = FORECASTS_DIR / "history" / "forecast_snapshot_history.parquet"
EVALUATION_HISTORY_PATH = FORECASTS_DIR / "history" / "evaluation_history.parquet"
ENGINE_LOG_PATH = FORECASTS_DIR / "logs" / "engine_log.jsonl"
CURRENT_PACKAGE_PATH = FORECASTS_DIR / "current" / "forecast_package.json"
PACKAGE_HISTORY_PATH = FORECASTS_DIR / "history" / "package_history.parquet"
PACKAGE_VERIFICATION_HISTORY_PATH = FORECASTS_DIR / "history" / "package_verification_history.parquet"
SKILL_SCORES_PATH = FORECASTS_DIR / "current" / "skill_scores.json"
CALIBRATION_REPORT_PATH = FORECASTS_DIR / "current" / "calibration_report.json"
REPORT_HISTORY_PATH = FORECASTS_DIR / "history" / "hourly_report.parquet"

MAX_LOG_LINES = 5000

# forecast_snapshot_history.parquet used to grow forever by design — see
# the module docstring's old note. That's what let it reach 7M+ rows and
# start blocking dashboard renders outright, so it's now a rolling window
# instead of a permanent archive. Keeps every read on this file bounded
# and fast regardless of how long the engine has been running. 24h, not
# 72h — enough to demo "yesterday and today" continuity without carrying
# three days of points into a chart that only needs one.
FORECAST_HISTORY_RETENTION_HOURS = 24

# package_history.parquet has no cutoff of its own — a package's lifecycle
# span is usually well under this, so it exists only as a backstop against
# the same unbounded-growth failure mode as forecast_snapshot_history.
PACKAGE_HISTORY_RETENTION_DAYS = 7

# hourly_report.parquet — the Reports tab's one-file, human-readable
# per-hour scorecard (see append_report_rows). Rolling window keyed off
# the hour being reported on (valid_end), not wall-clock write time.
# 7 days, day-grouped in the UI — same "today + last N days" convention
# NOAA uses for its own space weather products.
REPORT_HISTORY_RETENTION_DAYS = 7


def _ensure_parent(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_to_parquet(df: pd.DataFrame, path) -> None:
    """Write via temp file + os.replace, same pattern as every JSON writer
    in this module. to_parquet() writes its footer last and is not atomic
    on its own — a reader (another process, or the dashboard) landing
    mid-write sees a truncated file with no footer, surfacing as
    ArrowInvalid: Parquet magic bytes not found in footer."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _safe_read_parquet(path) -> pd.DataFrame:
    """Reads a history parquet file, treating an unreadable/corrupted file
    (e.g. from a process suspended mid-write across a laptop sleep) as an
    empty table instead of raising. Without this, every writer that reads
    existing rows before appending (append_*_row/rows) would crash forever
    on the same corrupt file, since the corruption never gets a chance to
    be overwritten by a fresh write. The corrupt file is preserved next to
    a '_corrupt_<timestamp>' copy rather than silently discarded."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        backup_dir = path.parent / "corrupted_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{path.stem}_corrupt_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
        os.replace(path, backup_path)
        print(f"[storage] {path.name} unreadable ({exc}); backed up to {backup_path} and reset to empty.")
        return pd.DataFrame()


def write_snapshot(snapshot: dict) -> None:
    """Atomic write (temp file + os.replace) — a reader can never observe
    a partially-written snapshot, even if a cycle is interrupted mid-write.
    """
    _ensure_parent(CURRENT_SNAPSHOT_PATH)
    tmp_path = CURRENT_SNAPSHOT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    os.replace(tmp_path, CURRENT_SNAPSHOT_PATH)


def load_current_snapshot() -> dict | None:
    if not CURRENT_SNAPSHOT_PATH.exists():
        return None
    with open(CURRENT_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_solar_forecast(payload: dict) -> None:
    """Always-overwritten, like write_snapshot — the Solar Forecast tab's
    F10.7 outlook and CME arrival estimates (swdss.models.f107_forecast,
    swdss.physics.cme_dbm), computed once per engine cycle in
    orchestrator.refresh_dashboard_products and read from here rather
    than recomputed on every dashboard render, same as every other
    Command Centre tab.
    """
    _ensure_parent(CURRENT_SOLAR_FORECAST_PATH)
    tmp_path = CURRENT_SOLAR_FORECAST_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, CURRENT_SOLAR_FORECAST_PATH)


def load_solar_forecast() -> dict | None:
    if not CURRENT_SOLAR_FORECAST_PATH.exists():
        return None
    with open(CURRENT_SOLAR_FORECAST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def append_forecast_history_rows(rows: list) -> None:
    if not rows:
        return
    _ensure_parent(FORECAST_HISTORY_PATH)
    new_df = pd.DataFrame(rows)
    existing = _safe_read_parquet(FORECAST_HISTORY_PATH)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    # Rolling retention window, not a permanent archive — see
    # FORECAST_HISTORY_RETENTION_HOURS. Applied on every write so the
    # file's size is self-correcting: it converges to the window's true
    # size within one retention period even from an already-oversized
    # file, with no separate cleanup job needed.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=FORECAST_HISTORY_RETENTION_HOURS)
    generated_at = pd.to_datetime(combined["generated_at"], utc=True, errors="coerce")
    combined = combined[generated_at >= cutoff]

    _atomic_to_parquet(combined, FORECAST_HISTORY_PATH)


def load_forecast_history(days: int = None) -> pd.DataFrame:
    df = _safe_read_parquet(FORECAST_HISTORY_PATH)
    if days is not None and not df.empty and "generated_at" in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        ts = pd.to_datetime(df["generated_at"], utc=True, errors="coerce")
        df = df[ts >= cutoff]
    return df


def already_evaluated_job_ids() -> set:
    df = load_evaluation_history()
    if df.empty or "job_id" not in df.columns:
        return set()
    return set(df["job_id"].tolist())


def append_evaluation_rows(rows: list) -> None:
    if not rows:
        return
    _ensure_parent(EVALUATION_HISTORY_PATH)
    new_df = pd.DataFrame(rows)
    existing = _safe_read_parquet(EVALUATION_HISTORY_PATH)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["job_id"], keep="first")
    else:
        combined = new_df
    _atomic_to_parquet(combined, EVALUATION_HISTORY_PATH)


def load_evaluation_history() -> pd.DataFrame:
    return _safe_read_parquet(EVALUATION_HISTORY_PATH)


def append_log_lines(lines: list) -> None:
    if not lines:
        return
    _ensure_parent(ENGINE_LOG_PATH)
    with open(ENGINE_LOG_PATH, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, default=str) + "\n")

    # Trim to the most recent MAX_LOG_LINES — read tail, rewrite. Cheap at
    # this file's expected growth rate (a handful of lines per ~60s
    # cycle), and keeps it from growing unbounded over months of uptime.
    with open(ENGINE_LOG_PATH, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    if len(all_lines) > MAX_LOG_LINES:
        with open(ENGINE_LOG_PATH, "w", encoding="utf-8") as f:
            f.writelines(all_lines[-MAX_LOG_LINES:])


def load_recent_logs(n: int = 200) -> list:
    if not ENGINE_LOG_PATH.exists():
        return []
    with open(ENGINE_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    records.reverse()  # newest first for display
    return records


def write_current_package(package: dict) -> None:
    _ensure_parent(CURRENT_PACKAGE_PATH)
    tmp_path = CURRENT_PACKAGE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2, default=str)
    os.replace(tmp_path, CURRENT_PACKAGE_PATH)


def load_current_package() -> dict | None:
    if not CURRENT_PACKAGE_PATH.exists():
        return None
    with open(CURRENT_PACKAGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Fields that define a genuinely new lifecycle state for a package —
# see append_package_history_row's dedup check below.
_PACKAGE_HISTORY_STATE_FIELDS = (
    "status", "completeness", "variables_complete", "evaluation_status",
    "overall_confidence_category", "outlook_level", "missing_variables", "kp_carried_over",
)


def append_package_history_row(package: dict) -> int:
    """Appends one row per LIFECYCLE CHANGE of a package (not one row per
    cycle-observation) and returns the forecast_cycle number assigned to
    this package_id — assigned once the first time a package_id is seen,
    then reused for every later observation of that same package as its
    lifecycle status advances.

    A package stays the current one across many refresh cycles (roughly
    hourly) while nothing about it has changed — logging a fresh row
    every ~60s cycle regardless was the exact same unbounded-growth bug
    fixed for forecast_snapshot_history.parquet, just never applied here:
    one real package sat at status=LIVE for 9 hours and picked up 282
    identical rows. The fix is the same shape — only append when
    _PACKAGE_HISTORY_STATE_FIELDS actually differ from the last logged
    row for this package_id — plus a retention cutoff as a backstop.
    """
    _ensure_parent(PACKAGE_HISTORY_PATH)
    package_id = package["package_id"]

    existing = _safe_read_parquet(PACKAGE_HISTORY_PATH)
    prior = existing[existing["package_id"] == package_id] if not existing.empty else existing
    if not prior.empty:
        cycle = int(prior["forecast_cycle"].iloc[0])
    elif not existing.empty:
        cycle = int(existing["forecast_cycle"].max()) + 1
    else:
        cycle = 1

    row = {
        "package_id": package_id,
        "forecast_cycle": cycle,
        "generated_at": package["generated_at"],
        "valid_start": package["valid_start"],
        "valid_end": package["valid_end"],
        "status": package["status"],
        "completeness": package["completeness"],
        "variables_complete": package.get("variables_complete"),
        "variables_total": package.get("variables_total"),
        "observations_used_through": package.get("observations_used_through"),
        "evaluation_status": package["evaluation_status"],
        "overall_confidence_score": package["overall_confidence"]["score"],
        "overall_confidence_category": package["overall_confidence"]["category"],
        "outlook_level": package.get("outlook", {}).get("level"),
        "package_summary": package["package_summary"],
        "missing_variables": ",".join(package.get("missing_variables") or []),
        "kp_carried_over": package.get("kp_carried_over", False),
        "snapshot_json": json.dumps(package, default=str),
        "logged_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    if not prior.empty:
        last_logged = prior.sort_values("logged_at").iloc[-1]
        unchanged = all(last_logged.get(field) == row[field] for field in _PACKAGE_HISTORY_STATE_FIELDS)
        if unchanged:
            return cycle

    new_df = pd.DataFrame([row])
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    # Retention backstop — see PACKAGE_HISTORY_RETENTION_DAYS. The dedup
    # check above should keep this file small on its own, but a package
    # that genuinely changes state every cycle (or a future bug) should
    # still never grow this file unbounded.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=PACKAGE_HISTORY_RETENTION_DAYS)
    logged_at = pd.to_datetime(combined["logged_at"], utc=True, errors="coerce")
    combined = combined[logged_at >= cutoff]

    _atomic_to_parquet(combined, PACKAGE_HISTORY_PATH)
    return cycle


def load_package_history(days: int = None) -> pd.DataFrame:
    df = _safe_read_parquet(PACKAGE_HISTORY_PATH)
    if days is not None and not df.empty and "generated_at" in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        ts = pd.to_datetime(df["generated_at"], utc=True, errors="coerce")
        df = df[ts >= cutoff]
    return df


def append_report_rows(rows: list) -> None:
    """Appends to hourly_report.parquet — one row per (variable, horizon,
    valid_end) the moment a headline forecast is actually evaluated
    against its real observed value. This is the Reports tab's single
    file: everything needed to read "how did each variable do this hour"
    lives in this one table, rolled to the last REPORT_HISTORY_RETENTION_DAYS
    by valid_end (the hour being reported on, not when it was written)."""
    if not rows:
        return
    _ensure_parent(REPORT_HISTORY_PATH)
    new_df = pd.DataFrame(rows)
    existing = _safe_read_parquet(REPORT_HISTORY_PATH)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=["variable", "horizon", "valid_end"], keep="last")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=REPORT_HISTORY_RETENTION_DAYS)
    valid_end = pd.to_datetime(combined["valid_end"], utc=True, errors="coerce")
    combined = combined[valid_end >= cutoff]

    _atomic_to_parquet(combined, REPORT_HISTORY_PATH)


def load_report_history() -> pd.DataFrame:
    return _safe_read_parquet(REPORT_HISTORY_PATH)


def already_verified_package_ids() -> set:
    df = load_package_verification_history()
    if df.empty or "package_id" not in df.columns:
        return set()
    return set(df["package_id"].tolist())


def append_package_verification_row(row: dict) -> None:
    _ensure_parent(PACKAGE_VERIFICATION_HISTORY_PATH)
    new_df = pd.DataFrame([row])
    existing = _safe_read_parquet(PACKAGE_VERIFICATION_HISTORY_PATH)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["package_id"], keep="last")
    else:
        combined = new_df
    _atomic_to_parquet(combined, PACKAGE_VERIFICATION_HISTORY_PATH)


def write_skill_scores(rows: list) -> None:
    """Always-overwritten, like write_current_package — skill.py
    recomputes this from the FULL evaluation history every cycle, so
    there is no incremental state to preserve between writes."""
    _ensure_parent(SKILL_SCORES_PATH)
    tmp_path = SKILL_SCORES_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    os.replace(tmp_path, SKILL_SCORES_PATH)


def load_skill_scores() -> list:
    if not SKILL_SCORES_PATH.exists():
        return []
    with open(SKILL_SCORES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_calibration_report(confidence_calibration: list, regime_error_bands: list) -> None:
    """Always-overwritten — same recompute-from-full-history pattern as
    write_skill_scores."""
    _ensure_parent(CALIBRATION_REPORT_PATH)
    report = {"confidence_calibration": confidence_calibration, "regime_error_bands": regime_error_bands}
    tmp_path = CALIBRATION_REPORT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    os.replace(tmp_path, CALIBRATION_REPORT_PATH)


def load_calibration_report() -> dict:
    if not CALIBRATION_REPORT_PATH.exists():
        return {"confidence_calibration": [], "regime_error_bands": []}
    with open(CALIBRATION_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_package_verification_history() -> pd.DataFrame:
    return _safe_read_parquet(PACKAGE_VERIFICATION_HISTORY_PATH)
