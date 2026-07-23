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
  variable, horizon) appended every cycle — Timeline tab + downloads.
  Grows forever by design (permanent operational history, no pruning).
- history/evaluation_history.parquet: one row per job, appended once
  (deduped by job_id) the first cycle it's observed completed with a
  real actual_value — Verification tab + downloads.
- logs/engine_log.jsonl: structured per-stage log lines, trimmed to the
  most recent MAX_LOG_LINES — Logs tab.
- current/forecast_package.json: the current Forecast Package (see
  swdss.engine.packages) — the synchronized operational product bundling
  the 10 headline forecasts. The package banner reads only this file.
- history/package_history.parquet: one row per cycle-observation of a
  package (same append-every-cycle pattern as forecast history), keyed
  by package_id — the package's own Timeline/lifecycle log.
- history/package_verification_history.parquet: one row per package,
  written once its core (non-AE) members are all verified — the
  permanent Package Verification Summary record.
"""

import json
import os

import pandas as pd

from swdss.paths import FORECASTS_DIR

CURRENT_SNAPSHOT_PATH = FORECASTS_DIR / "current" / "forecast_snapshot.json"
FORECAST_HISTORY_PATH = FORECASTS_DIR / "history" / "forecast_snapshot_history.parquet"
EVALUATION_HISTORY_PATH = FORECASTS_DIR / "history" / "evaluation_history.parquet"
ENGINE_LOG_PATH = FORECASTS_DIR / "logs" / "engine_log.jsonl"
CURRENT_PACKAGE_PATH = FORECASTS_DIR / "current" / "forecast_package.json"
PACKAGE_HISTORY_PATH = FORECASTS_DIR / "history" / "package_history.parquet"
PACKAGE_VERIFICATION_HISTORY_PATH = FORECASTS_DIR / "history" / "package_verification_history.parquet"

MAX_LOG_LINES = 5000


def _ensure_parent(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def append_forecast_history_rows(rows: list) -> None:
    if not rows:
        return
    _ensure_parent(FORECAST_HISTORY_PATH)
    new_df = pd.DataFrame(rows)
    if FORECAST_HISTORY_PATH.exists():
        existing = pd.read_parquet(FORECAST_HISTORY_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(FORECAST_HISTORY_PATH, index=False)


def load_forecast_history(days: int = None) -> pd.DataFrame:
    if not FORECAST_HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FORECAST_HISTORY_PATH)
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
    if EVALUATION_HISTORY_PATH.exists():
        existing = pd.read_parquet(EVALUATION_HISTORY_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["job_id"], keep="first")
    else:
        combined = new_df
    combined.to_parquet(EVALUATION_HISTORY_PATH, index=False)


def load_evaluation_history() -> pd.DataFrame:
    if not EVALUATION_HISTORY_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(EVALUATION_HISTORY_PATH)


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


def append_package_history_row(package: dict) -> int:
    """Appends one row per cycle-observation of a package (same pattern
    as append_forecast_history_rows) and returns the forecast_cycle
    number assigned to this package_id — assigned once the first time a
    package_id is seen, then reused for every later observation of that
    same package as its lifecycle status advances.
    """
    _ensure_parent(PACKAGE_HISTORY_PATH)
    package_id = package["package_id"]

    existing = pd.read_parquet(PACKAGE_HISTORY_PATH) if PACKAGE_HISTORY_PATH.exists() else None
    if existing is not None and not existing.empty:
        prior = existing[existing["package_id"] == package_id]
        if not prior.empty:
            cycle = int(prior["forecast_cycle"].iloc[0])
        else:
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
    new_df = pd.DataFrame([row])
    combined = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
    combined.to_parquet(PACKAGE_HISTORY_PATH, index=False)
    return cycle


def load_package_history(days: int = None) -> pd.DataFrame:
    if not PACKAGE_HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PACKAGE_HISTORY_PATH)
    if days is not None and not df.empty and "generated_at" in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        ts = pd.to_datetime(df["generated_at"], utc=True, errors="coerce")
        df = df[ts >= cutoff]
    return df


def already_verified_package_ids() -> set:
    df = load_package_verification_history()
    if df.empty or "package_id" not in df.columns:
        return set()
    return set(df["package_id"].tolist())


def append_package_verification_row(row: dict) -> None:
    _ensure_parent(PACKAGE_VERIFICATION_HISTORY_PATH)
    new_df = pd.DataFrame([row])
    if PACKAGE_VERIFICATION_HISTORY_PATH.exists():
        existing = pd.read_parquet(PACKAGE_VERIFICATION_HISTORY_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["package_id"], keep="last")
    else:
        combined = new_df
    combined.to_parquet(PACKAGE_VERIFICATION_HISTORY_PATH, index=False)


def load_package_verification_history() -> pd.DataFrame:
    if not PACKAGE_VERIFICATION_HISTORY_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(PACKAGE_VERIFICATION_HISTORY_PATH)
