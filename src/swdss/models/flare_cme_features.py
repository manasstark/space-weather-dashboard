"""Joins SHARP magnetic-complexity data (swdss.ingest.sharp) to real
flare/CME outcomes (DONKI, already flowing through build_master.py) to
build a training set for the Flare Outlook / CME Outlook models — the
missing piece flagged repeatedly in the Solar Forecast discussion.

Real data quirk found and confirmed here (not assumed): NOAA active
region numbers appear in TWO different conventions across this
project's existing data sources. `cme_processed.parquet`'s
`active_region` (from DONKI's activeRegionNum) and SHARP's `NOAA_AR`
both use the full modern 5-digit number (e.g. 14498). But
`solar_events_processed.parquet`'s `active_region` (from GOES/SWPC
flare records) uses the older truncated 4-digit convention (4498) —
confirmed empirically: zero direct overlap between the two ID sets,
but `solar_events.active_region + 10000` overlaps `cme.active_region`
for 11 real active regions. AR_NUMBER_OFFSET below is that fix, applied
once here rather than silently getting either join wrong.
"""

from __future__ import annotations

import glob

import pandas as pd

from swdss.ingest.sharp import SHARP_CACHE_DIR

AR_NUMBER_OFFSET = 10000  # solar_events.active_region + this == cme.active_region == SHARP NOAA_AR
FLARE_LABEL_HORIZON_HOURS = 24
CME_LABEL_HORIZON_HOURS = 24
CME_EARTH_DIRECTED_LONGITUDE_DEG = 45  # same generous threshold used in the DBM storm-backtest script

SHARP_FEATURE_COLUMNS = [
    "USFLUX", "MEANGBZ", "R_VALUE", "TOTPOT",
    "MEANJZH", "MEANALP", "SAVNCPP", "MEANSHR", "ABSNJZH",
]


def load_sharp_history() -> pd.DataFrame:
    """Every cached SHARP day (swdss.ingest.sharp.fetch_sharp_day),
    concatenated. NOAA_AR == 0 rows (a patch not yet assigned an
    official NOAA number) are dropped — they can never be joined to a
    DONKI flare/CME event, so keeping them would just be noise.
    """
    paths = sorted(glob.glob(str(SHARP_CACHE_DIR / "sharp_*.csv")))
    if not paths:
        return pd.DataFrame(columns=["timestamp_utc", "HARPNUM", "NOAA_AR", *SHARP_FEATURE_COLUMNS])

    frames = [pd.read_csv(p, parse_dates=["timestamp_utc"]) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["NOAA_AR"] > 0].copy()
    df["NOAA_AR"] = df["NOAA_AR"].astype(int)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def resample_sharp_hourly(sharp_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (NOAA_AR, hour) — the last SHARP reading observed in
    that hour, same "collapse to latest" convention as
    f107_forecast.daily_series. An active region can have multiple
    HARPNUMs over its lifetime in principle, but in practice each AR
    maps to one HARPNUM at a time, so grouping by NOAA_AR (not HARPNUM)
    is what we actually want for joining to flare/CME events, which are
    tagged by NOAA_AR alone.
    """
    if sharp_df.empty:
        return pd.DataFrame(columns=["NOAA_AR", "hour", *SHARP_FEATURE_COLUMNS])

    df = sharp_df.copy()
    df["hour"] = df["timestamp_utc"].dt.floor("h")
    hourly = (
        df.sort_values("timestamp_utc")
        .groupby(["NOAA_AR", "hour"])[SHARP_FEATURE_COLUMNS]
        .last()
        .reset_index()
        .sort_values(["NOAA_AR", "hour"])
    )
    return hourly


def _flare_events_by_ar(solar_events_df: pd.DataFrame) -> pd.DataFrame:
    flares = solar_events_df.dropna(subset=["flare_class", "active_region"]).copy()
    flares["class_letter"] = flares["flare_class"].str[0]
    flares = flares[flares["class_letter"].isin(["M", "X"])]
    flares["NOAA_AR"] = (flares["active_region"] + AR_NUMBER_OFFSET).astype(int)
    return flares[["NOAA_AR", "timestamp_utc"]].drop_duplicates()


def _earth_directed_cmes_by_ar(cme_df: pd.DataFrame) -> pd.DataFrame:
    cmes = cme_df.dropna(subset=["active_region"]).copy()
    cmes = cmes[cmes["longitude"].isna() | (cmes["longitude"].abs() <= CME_EARTH_DIRECTED_LONGITUDE_DEG)]
    cmes["NOAA_AR"] = cmes["active_region"].astype(int)
    return cmes[["NOAA_AR", "timestamp_utc"]].drop_duplicates()


def _label_next_n_hours(hourly_sharp: pd.DataFrame, events_by_ar: pd.DataFrame, horizon_hours: int) -> pd.Series:
    """For each (NOAA_AR, hour) row, 1 if that AR has an event timestamp
    in (hour, hour + horizon_hours], else 0. A simple per-AR merge_asof
    is not quite right here since we need "any event in the window," not
    "nearest event" — done via an explicit interval check per AR group,
    which is fine at this project's data volume (tens of ARs, thousands
    of hourly rows, not millions).
    """
    labels = pd.Series(0, index=hourly_sharp.index, dtype=int)
    if events_by_ar.empty:
        return labels

    for ar, group in hourly_sharp.groupby("NOAA_AR"):
        ar_events = events_by_ar[events_by_ar["NOAA_AR"] == ar]["timestamp_utc"]
        if ar_events.empty:
            continue
        for idx, hour in group["hour"].items():
            window_end = hour + pd.Timedelta(hours=horizon_hours)
            if ((ar_events > hour) & (ar_events <= window_end)).any():
                labels.loc[idx] = 1
    return labels


def build_feature_label_matrix(sharp_df: pd.DataFrame, solar_events_df: pd.DataFrame, cme_df: pd.DataFrame) -> pd.DataFrame:
    """The full training table: one row per (NOAA_AR, hour) with SHARP
    features (current value + 24h-ago value + 24h delta, all strictly
    causal — only using data at-or-before `hour`) and both labels
    (flare_label_24h, cme_label_24h).
    """
    hourly = resample_sharp_hourly(sharp_df)
    if hourly.empty:
        return pd.DataFrame()

    hourly = hourly.sort_values(["NOAA_AR", "hour"]).reset_index(drop=True)
    for col in SHARP_FEATURE_COLUMNS:
        hourly[f"{col}_24h_ago"] = hourly.groupby("NOAA_AR")[col].shift(24)
        hourly[f"{col}_24h_delta"] = hourly[col] - hourly[f"{col}_24h_ago"]

    flare_events = _flare_events_by_ar(solar_events_df)
    cme_events = _earth_directed_cmes_by_ar(cme_df)
    hourly["flare_label_24h"] = _label_next_n_hours(hourly, flare_events, FLARE_LABEL_HORIZON_HOURS)
    hourly["cme_label_24h"] = _label_next_n_hours(hourly, cme_events, CME_LABEL_HORIZON_HOURS)

    # Right-censoring: a row whose 24h-ahead window extends past the
    # last hour we actually have SHARP data for cannot be trusted as a
    # true negative — the event window simply hasn't finished yet. Drop
    # those rows rather than silently mislabeling them 0, since with
    # only ~2 months of history this tail would otherwise be a
    # meaningful fraction of the training set, not a rounding error.
    max_hour = hourly["hour"].max()
    hourly = hourly[hourly["hour"] + pd.Timedelta(hours=FLARE_LABEL_HORIZON_HOURS) <= max_hour].copy()

    return hourly
