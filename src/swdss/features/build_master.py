import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from swdss.paths import (
    MASTER_V1_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_data_dirs,
)

# NOAA retired the old separate /products/solar-wind/plasma-7-day.json and
# mag-7-day.json endpoints (both now 404 — the whole solar-wind/ directory
# is gone). Their replacement, propagated-solar-wind.json, combines plasma
# AND IMF into one minute-cadence, ~7-day rolling feed. solar_wind and imf
# are fetched from it independently (matching the existing one-job-per-
# dataset architecture) and each cleaner just selects its own columns.
_PROPAGATED_SOLAR_WIND_URL = "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind.json"

NOAA_URLS = {
    "solar_wind": _PROPAGATED_SOLAR_WIND_URL,
    "imf": _PROPAGATED_SOLAR_WIND_URL,
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "dst": "https://services.swpc.noaa.gov/products/kyoto-dst.json",
    "solar_events": "https://services.swpc.noaa.gov/json/edited_events.json",
    "f107": "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
}

DONKI_CME_URL = "https://api.nasa.gov/DONKI/CME"


def fetch_json(name: str) -> list:
    response = requests.get(
        NOAA_URLS[name],
        timeout=30,
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        },
    )
    response.raise_for_status()

    text = response.text.strip()

    try:
        return response.json()
    except ValueError as error:
        decoder = json.JSONDecoder()

        try:
            payload, end_index = decoder.raw_decode(text)
        except ValueError:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            bad_path = RAW_DIR / name / f"{name}_bad_response_{timestamp}.txt"
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            bad_path.write_text(text, encoding="utf-8")

            raise ValueError(
                f"Could not parse NOAA JSON for {name}. "
                f"Saved bad response to {bad_path}"
            ) from error

        trailing_text = text[end_index:].strip()
        if trailing_text:
            print(
                f"Warning: NOAA response for {name} had extra trailing text. "
                f"Ignored {len(trailing_text)} characters."
            )

        return payload


def save_raw_json(name: str, payload: list) -> None:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = RAW_DIR / name / f"{name}_{today}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def noaa_table_to_df(payload: list) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()

    if isinstance(payload[0], list):
        return pd.DataFrame(payload[1:], columns=payload[0])

    return pd.DataFrame(payload)


def clean_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    time_col = None
    for candidate in ["time_tag", "time", "timestamp", "datetime", "date"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")

    df["timestamp_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    return df


# NOAA's real-time feeds (confirmed here in Solar Wind plasma data —
# speed/density/temperature all sentineled together on the same row,
# consistent with a momentary DSCOVR instrument outage) use -9999 as a
# "no valid measurement" fill value rather than omitting the row
# entirely. Left unconverted, that literal -9999 gets averaged into
# hourly means like any other real reading — a proton density of -165
# p/cm3 (physically impossible) is exactly what that produces, which is
# also what was surfacing as "invalid value encountered in sqrt"
# warnings in the Physics Engine's Alfven speed calculation. -9990 is
# used as the cutoff (not -9999 exactly) purely as a tolerance margin;
# it is nowhere near any real value for any variable this pipeline
# handles — Dst's most extreme recorded value is still nowhere close to
# -9990, so this can never mistake a genuine (and often legitimately
# negative) reading for a fill value.
NOAA_FILL_THRESHOLD = -9990


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].where(df[column] > NOAA_FILL_THRESHOLD)
    return df


def hourly_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df[["timestamp_utc", *columns]].copy()
    df = to_numeric(df, columns)
    df = df.set_index("timestamp_utc").sort_index()
    df = df.resample("1h").mean()
    return df.reset_index()

def minute_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df[["timestamp_utc", *columns]].copy()
    df = to_numeric(df, columns)
    df = df.sort_values("timestamp_utc")
    df = df.dropna(how="all", subset=columns)
    return df.reset_index(drop=True)


def to_hourly_for_master(name: str, df: pd.DataFrame) -> pd.DataFrame:
    value_columns = [col for col in df.columns if col != "timestamp_utc"]

    hourly = df.set_index("timestamp_utc").sort_index()

    if name == "kp":
        hourly = hourly[value_columns].resample("1h").ffill()
    else:
        hourly = hourly[value_columns].resample("1h").mean()

    return hourly.reset_index()


def clean_solar_wind(payload: list) -> pd.DataFrame:
    df = noaa_table_to_df(payload)
    df = clean_time_column(df)

    rename_map = {
        "speed": "solar_wind_speed",
        "density": "proton_density",
        "temperature": "temperature",
    }

    df = df.rename(columns=rename_map)
    columns = ["solar_wind_speed", "proton_density", "temperature"]
    return minute_values(df, columns)


def clean_imf(payload: list) -> pd.DataFrame:
    df = noaa_table_to_df(payload)
    df = clean_time_column(df)

    # The old mag-7-day.json used "bx_gsm"/"by_gsm"/"bz_gsm"; the current
    # propagated-solar-wind.json already uses bare "bx"/"by"/"bz". Renaming
    # only when the _gsm-suffixed names are present keeps this working
    # against either shape.
    rename_map = {
        "bx_gsm": "bx",
        "by_gsm": "by",
        "bz_gsm": "bz",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    columns = ["bt", "bx", "by", "bz"]
    return minute_values(df, columns)


def clean_kp(payload: list) -> pd.DataFrame:
    df = noaa_table_to_df(payload)
    df = clean_time_column(df)

    kp_col = None
    for candidate in ["Kp", "kp", "estimated_kp"]:
        if candidate in df.columns:
            kp_col = candidate
            break

    if kp_col is None:
        raise ValueError(f"No Kp column found. Columns: {list(df.columns)}")

    df = df.rename(columns={kp_col: "kp"})
    df = to_numeric(df, ["kp"])
    df = df[["timestamp_utc", "kp"]].set_index("timestamp_utc").sort_index()

    # Kp is usually a 3-hour index, so forward-fill into hourly rows.
    df = df.resample("1h").ffill()
    return df.reset_index()


def clean_dst(payload: list) -> pd.DataFrame:
    df = noaa_table_to_df(payload)
    df = clean_time_column(df)

    dst_col = None
    for candidate in ["dst", "Dst", "DST"]:
        if candidate in df.columns:
            dst_col = candidate
            break

    if dst_col is None:
        raise ValueError(f"No Dst column found. Columns: {list(df.columns)}")

    df = df.rename(columns={dst_col: "dst"})
    return hourly_mean(df, ["dst"])


def clean_solar_events(payload) -> pd.DataFrame:
    """Solar Events is an event catalog, not a continuous series.

    NOAA's edited_events.json shape can vary (list-of-dicts or
    dict-of-date -> list-of-dicts), so this stays defensive and raises a
    clear error listing the actual columns if the schema drifts.
    """
    records: list[dict] = []

    if isinstance(payload, dict):
        for date_key, events in payload.items():
            if not isinstance(events, list):
                continue
            for event in events:
                if isinstance(event, dict):
                    event = dict(event)
                    event.setdefault("date_key", date_key)
                    records.append(event)
    elif isinstance(payload, list):
        records = [event for event in payload if isinstance(event, dict)]

    if not records:
        return pd.DataFrame(columns=["timestamp_utc"])

    df = pd.DataFrame(records)
    df.columns = [str(col).strip() for col in df.columns]

    time_col = None
    for candidate in [
        "time_tag",
        "begin_datetime",
        "max_datetime",
        "end_datetime",
        "begin",
        "max",
        "start_time",
        "date_key",
    ]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        raise ValueError(f"No usable time column for solar events. Columns: {list(df.columns)}")

    df["timestamp_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])

    rename_map = {
        "type": "event_type",
        "reg#": "active_region",
        "region": "active_region",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    particulars_cols = [col for col in df.columns if col.startswith("particulars")]
    if particulars_cols:
        combined = df[particulars_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        df["particulars"] = combined.replace(r"\s+", " ", regex=True)
        df["flare_class"] = df["particulars"].str.extract(r"([ABCMX]\d+(?:\.\d+)?)", expand=False)
        df["radio_burst_type"] = df["particulars"].str.extract(r"^([A-Za-z]+)/", expand=False)

    if "end_datetime" in df.columns:
        end_time = pd.to_datetime(df["end_datetime"], utc=True, errors="coerce")
        df["duration_minutes"] = (end_time - df["timestamp_utc"]).dt.total_seconds() / 60
        df.loc[df["duration_minutes"] < 0, "duration_minutes"] = pd.NA

    if "location" in df.columns:
        loc = df["location"].astype(str).str.strip()
        parsed = loc.str.extract(r"^([NS])(\d{1,2})([EW])(\d{1,3})$")
        lat_sign = parsed[0].map({"N": 1, "S": -1})
        lon_sign = parsed[2].map({"E": 1, "W": -1})
        df["heliographic_lat"] = pd.to_numeric(parsed[1], errors="coerce") * lat_sign
        df["heliographic_lon"] = pd.to_numeric(parsed[3], errors="coerce") * lon_sign

    return df.sort_values("timestamp_utc").reset_index(drop=True)


def clean_f107(payload: list) -> pd.DataFrame:
    df = noaa_table_to_df(payload)
    df = clean_time_column(df)

    flux_col = None
    for candidate in ["flux", "f10.7", "f107", "observed_flux", "adjusted_flux"]:
        if candidate in df.columns:
            flux_col = candidate
            break

    if flux_col is None:
        raise ValueError(f"No F10.7 flux column found. Columns: {list(df.columns)}")

    df = df.rename(columns={flux_col: "f107_flux"})
    df = to_numeric(df, ["f107_flux"])
    df = df.dropna(subset=["f107_flux"])

    return df[["timestamp_utc", "f107_flux"]].sort_values("timestamp_utc").reset_index(drop=True)


def fetch_cme(start_date: str, end_date: str, api_key: str | None = None) -> list:
    api_key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")

    response = requests.get(
        DONKI_CME_URL,
        params={"startDate": start_date, "endDate": end_date, "api_key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def clean_cme(payload: list) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=["timestamp_utc"])

    records = []
    for event in payload:
        if not isinstance(event, dict):
            continue

        analyses = event.get("cmeAnalyses") or []
        analysis = next((a for a in analyses if a.get("isMostAccurate")), None)
        if analysis is None and analyses:
            analysis = analyses[0]
        analysis = analysis or {}

        records.append(
            {
                "activity_id": event.get("activityID"),
                "start_time": event.get("startTime"),
                "active_region": event.get("activeRegionNum"),
                "source_location": event.get("sourceLocation"),
                "speed": analysis.get("speed"),
                "latitude": analysis.get("latitude"),
                "longitude": analysis.get("longitude"),
                "half_angle": analysis.get("halfAngle"),
                "cme_type": analysis.get("type"),
                "note": event.get("note"),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc"])

    df["timestamp_utc"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = to_numeric(df, ["speed", "latitude", "longitude", "half_angle"])

    return df.sort_values("timestamp_utc").reset_index(drop=True)


DONKI_FLR_URL = "https://api.nasa.gov/DONKI/FLR"


def fetch_flr(start_date: str, end_date: str, api_key: str | None = None) -> list:
    """DONKI's Flare catalog — added for historical flare-label backfill
    (swdss.models.flare_cme_features), NOT used by the live production
    pipeline, which still gets its Solar Events/flare data from NOAA's
    edited_events.json rolling feed (NOAA_URLS["solar_events"] above,
    unchanged). Kept here alongside fetch_cme/clean_cme rather than in
    a research-only module since it's the same general-purpose,
    date-range-queryable DONKI pattern already established by fetch_cme
    — a reusable utility, not a production behavior change.

    activeRegionNum here is already the full 5-digit NOAA active-region
    number, matching DONKI's own CME catalog and SHARP's NOAA_AR
    directly — unlike NOAA's edited_events.json feed (solar_events_
    processed.parquet's active_region), which uses the older truncated
    4-digit convention (see flare_cme_features.py's AR_NUMBER_OFFSET
    docstring for the confirmed, empirical mismatch between those two).
    Using DONKI FLR uniformly for training labels avoids reconciling two
    different AR conventions across the backfill.
    """
    api_key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")

    response = requests.get(
        DONKI_FLR_URL,
        params={"startDate": start_date, "endDate": end_date, "api_key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def clean_flr(payload: list) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=["timestamp_utc", "active_region", "flare_class"])

    records = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        records.append({
            "timestamp_utc": event.get("beginTime"),
            "active_region": event.get("activeRegionNum"),
            "flare_class": event.get("classType"),
            "source_location": event.get("sourceLocation"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "active_region", "flare_class"])

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "active_region"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def atomic_to_parquet(df: pd.DataFrame, path) -> None:
    """Write via temp file + os.replace. to_parquet() writes its footer
    last and is not atomic on its own — a reader (the dashboard, or
    another process) landing mid-write sees a truncated file with no
    footer, surfacing as ArrowInvalid: Parquet magic bytes not found in
    footer. These processed/ files are polled by the dashboard on every
    page load, so this window gets hit in practice, not just in theory.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def save_processed(name: str, df: pd.DataFrame) -> None:
    path = PROCESSED_DIR / name / f"{name}_processed.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(df, path)


def save_processed_append(
    name: str,
    new_df: pd.DataFrame,
    dedupe_subset: list[str] | None = None,
) -> pd.DataFrame:
    """For sources that only return a recent/today snapshot per request
    (e.g. NOAA's edited_events.json), append to the existing processed
    history instead of overwriting it, then drop duplicate rows.
    """
    path = PROCESSED_DIR / name / f"{name}_processed.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            existing = pd.read_parquet(path)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=dedupe_subset, keep="last")
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

    atomic_to_parquet(combined, path)
    return combined


def build_master() -> pd.DataFrame:
    ensure_data_dirs()

    cleaners = {
        "solar_wind": clean_solar_wind,
        "imf": clean_imf,
        "kp": clean_kp,
        "dst": clean_dst,
    }

    cleaned_frames = {}

    for name, cleaner in cleaners.items():
        print(f"Fetching {name}...")
        payload = fetch_json(name)
        save_raw_json(name, payload)

        print(f"Cleaning {name}...")
        cleaned = cleaner(payload)
        save_processed(name, cleaned)
        cleaned_frames[name] = to_hourly_for_master(name, cleaned)

    min_time = min(df["timestamp_utc"].min() for df in cleaned_frames.values())
    max_time = max(df["timestamp_utc"].max() for df in cleaned_frames.values())

    master = pd.DataFrame({
        "timestamp_utc": pd.date_range(
            start=min_time.floor("h"),
            end=max_time.ceil("h"),
            freq="1h",
            tz="UTC",
        )
    })

    for df in cleaned_frames.values():
        master = master.merge(df, on="timestamp_utc", how="left")

    master = master.sort_values("timestamp_utc").reset_index(drop=True)

    MASTER_V1_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(master, MASTER_V1_PATH)

    print(f"Saved master dataset: {MASTER_V1_PATH}")
    print(master.tail())

    return master


if __name__ == "__main__":
    build_master()