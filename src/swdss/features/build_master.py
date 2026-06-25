import json
from datetime import datetime, timezone

import pandas as pd
import requests

from swdss.paths import (
    RAW_DIR,
    PROCESSED_DIR,
    MASTER_V1_PATH,
    ensure_data_dirs,
)


NOAA_URLS = {
    "solar_wind": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
    "imf": "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "dst": "https://services.swpc.noaa.gov/products/kyoto-dst.json",
}


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


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
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

    rename_map = {
        "bt": "bt",
        "bx_gsm": "bx",
        "by_gsm": "by",
        "bz_gsm": "bz",
    }

    df = df.rename(columns=rename_map)
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


def save_processed(name: str, df: pd.DataFrame) -> None:
    path = PROCESSED_DIR / name / f"{name}_processed.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


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
    master.to_parquet(MASTER_V1_PATH, index=False)

    print(f"Saved master dataset: {MASTER_V1_PATH}")
    print(master.tail())

    return master


if __name__ == "__main__":
    build_master()