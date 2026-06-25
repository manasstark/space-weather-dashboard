import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from swdss.features.build_master import (
    clean_dst,
    clean_imf,
    clean_kp,
    clean_solar_wind,
    fetch_json,
    save_processed,
    save_raw_json,
    to_hourly_for_master,
)
from swdss.paths import MASTER_V1_PATH, PROCESSED_DIR, ensure_data_dirs


@dataclass(frozen=True)
class DatasetJob:
    name: str
    cadence_seconds: int
    cleaner: callable


DATASET_JOBS = {
    "solar_wind": DatasetJob(
        name="solar_wind",
        cadence_seconds=60,
        cleaner=clean_solar_wind,
    ),
    "imf": DatasetJob(
        name="imf",
        cadence_seconds=60,
        cleaner=clean_imf,
    ),
    "dst": DatasetJob(
        name="dst",
        cadence_seconds=5 * 60,
        cleaner=clean_dst,
    ),
    "kp": DatasetJob(
        name="kp",
        cadence_seconds=3 * 60 * 60,
        cleaner=clean_kp,
    ),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def processed_path(name: str) -> Path:
    return PROCESSED_DIR / name / f"{name}_processed.parquet"


def update_dataset(job: DatasetJob) -> bool:
    print(f"[{utc_now().isoformat()}] Updating {job.name}...")

    payload = fetch_json(job.name)
    save_raw_json(job.name, payload)

    cleaned = job.cleaner(payload)
    save_processed(job.name, cleaned)

    latest_time = cleaned["timestamp_utc"].max() if not cleaned.empty else None
    print(f"[{utc_now().isoformat()}] Finished {job.name}. Latest timestamp: {latest_time}")
    return True


def read_processed_frames() -> dict[str, pd.DataFrame]:
    frames = {}

    for name in DATASET_JOBS:
        path = processed_path(name)
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        if df.empty or "timestamp_utc" not in df.columns:
            continue

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

        if not df.empty:
            frames[name] = df

    return frames


def rebuild_master_from_processed() -> pd.DataFrame | None:
    frames = read_processed_frames()

    frames = {
    name: to_hourly_for_master(name, df)
    for name, df in frames.items()
}

    if not frames:
        print(f"[{utc_now().isoformat()}] No processed data found. Master file not rebuilt.")
        return None

    min_time = min(df["timestamp_utc"].min() for df in frames.values())
    max_time = max(df["timestamp_utc"].max() for df in frames.values())

    master = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                start=min_time.floor("h"),
                end=max_time.ceil("h"),
                freq="1h",
                tz="UTC",
            )
        }
    )

    for name, df in frames.items():
        master = master.merge(df, on="timestamp_utc", how="left")

    master = master.sort_values("timestamp_utc").reset_index(drop=True)

    MASTER_V1_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(MASTER_V1_PATH, index=False)

    print(
        f"[{utc_now().isoformat()}] Rebuilt {MASTER_V1_PATH}. "
        f"Rows: {len(master)}. Latest: {master['timestamp_utc'].max()}"
    )

    return master


def run_live_update_loop() -> None:
    ensure_data_dirs()

    last_run = {name: 0.0 for name in DATASET_JOBS}
    first_run = True

    print("Starting SW-DSS live data updater.")
    print("Cadence:")
    print("- Solar Wind: 60 seconds")
    print("- IMF: 60 seconds")
    print("- Dst: 1 hour")
    print("- Kp: 3 hours")
    print("Press Ctrl+C to stop.")

    while True:
        loop_started = time.time()
        updated_any = False

        for name, job in DATASET_JOBS.items():
            due = first_run or (loop_started - last_run[name] >= job.cadence_seconds)

            if not due:
                continue

            try:
                update_dataset(job)
                last_run[name] = time.time()
                updated_any = True
            except Exception:
                print(f"[{utc_now().isoformat()}] Failed to update {name}:")
                traceback.print_exc()

        if updated_any:
            try:
                rebuild_master_from_processed()
            except Exception:
                print(f"[{utc_now().isoformat()}] Failed to rebuild master dataset:")
                traceback.print_exc()

        first_run = False

        elapsed = time.time() - loop_started
        sleep_seconds = max(10, 60 - elapsed)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    run_live_update_loop()
