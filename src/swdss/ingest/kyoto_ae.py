"""Kyoto World Data Center (WDC) for Geomagnetism — real-time (quicklook)
AE index digital data client.

This is the ONLY source ever used to verify AE predictions. NOAA/DONKI
publish no AE product at all, so this is not a choice between sources —
it's the sole way to get an official AE observation. Keeping it in its
own module (never imported by build_master.py/live_update.py, which stay
NOAA/DONKI-only) keeps the prediction engine (driven by live NOAA Solar
Wind/IMF) and the verification engine (driven by Kyoto WDC) fully
independent — neither can influence the other's data source. (This
module DOES import one small persistence helper FROM build_master.py —
see _archive_minute_data below — but that's a one-directional
dependency; build_master.py itself still never imports Kyoto data or is
influenced by it, so the stated independence still holds.)

Data source: https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir/
One fixed-width WDC-format text file per UT day (e.g. "ae260621" for
2026-06-21), each of its 24 lines holding one hour's 60 one-minute AE
values plus a trailing hourly mean.

Two things are preserved from every fetch:

1. The hourly mean, returned by fetch_kyoto_ae_hour — unchanged
   behavior, matching the hourly resolution used everywhere else in
   this project's AE handling (production training, evaluation,
   feature engineering). This is the ONLY thing any other module in
   this codebase consumes, and that contract has not changed.

2. NEW: the complete 60 one-minute AE values per hour, permanently
   archived (both raw response text and a parsed, deduplicated
   minute-resolution parquet) purely for future research — substorm
   onset timing, minute-scale AE dynamics, event detection — none of
   which this project currently does anything with. Before this change,
   these 60 values were parsed into memory and discarded the instant
   _parse_day_file's loop moved to the next line; nothing anywhere in
   this codebase ever stored them. Every day that passed before this
   change is lost permanently (Kyoto's real-time quicklook page is
   continuously overwritten, not a permanent archive), which is why this
   was worth fixing even though nothing currently reads the new archive.

Why production keeps using the hourly value: every trained AE model,
every feature (lags, rolling stats, changes), and the evaluation target
itself are all defined at hourly cadence. Switching production to
minute resolution would be a real, separate research question (does
minute-scale AE forecasting even make sense given how the rest of this
pipeline works?) — not something this change decides. This change only
ensures that question is *answerable* later, by making sure the data
to answer it with actually exists.

Important: "real-time" here means "the real-time/quicklook index
product," not "available within minutes of the event." In practice Kyoto
WDC batches and quality-checks before publishing — observed lag has been
on the order of 1-3 weeks. A prediction can stay "Awaiting Official AE"
for a long time; that's expected, not a bug.
"""

import re

import pandas as pd
import requests

from swdss.features.build_master import save_processed_append
from swdss.paths import RAW_DIR

KYOTO_AE_BASE_URL = "https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir"

# Dataset name for the new minute-resolution archive — deliberately
# distinct from the existing "ae" dataset (data/processed/ae/), which is
# production's own separately-built historical hourly archive and is
# never touched by this module. "kyoto_ae_minute" makes the source and
# resolution unambiguous, and avoids any risk of the two ever colliding.
MINUTE_DATASET_NAME = "kyoto_ae_minute"

# e.g. "AEALAOAU    260621E00AE QUICKLK      70    67 ..."
# group(1) = YYMMDD, group(2) = hour (00-23); everything after the match
# is 61 whitespace-separated integers — 60 one-minute values, then the
# hourly mean as the last field.
_LINE_HEADER_RE = re.compile(r"^AEALAOAU\s+(\d{6})E(\d{2})AE")

# Per-day cache, populated only on a SUCCESSFUL parse. A 404 (not yet
# published) is deliberately never cached, so the next check naturally
# retries — caching a negative result would mean a day that gets
# published later would never be picked up without a process restart.
_day_cache: dict[str, dict[pd.Timestamp, float]] = {}


def _day_file_url(date: pd.Timestamp) -> str:
    return f"{KYOTO_AE_BASE_URL}/{date.strftime('%Y')}/{date.strftime('%m')}/{date.strftime('%d')}/ae{date.strftime('%y%m%d')}"


def _parse_day_file(text: str, date: pd.Timestamp) -> dict[pd.Timestamp, float]:
    day_start = date.normalize()
    hourly = {}
    for line in text.splitlines():
        match = _LINE_HEADER_RE.match(line)
        if not match:
            continue
        hour = int(match.group(2))
        values = line[match.end():].split()
        if not values:
            continue
        hourly[day_start + pd.Timedelta(hours=hour)] = float(values[-1])
    return hourly


def _parse_day_file_minutes(text: str, date: pd.Timestamp) -> pd.DataFrame:
    """Parses the SAME raw day-file text _parse_day_file reads, but keeps
    all 60 one-minute AE values per hour instead of only the trailing
    hourly mean. Entirely independent of _parse_day_file — a separate
    pass over the same text, so nothing about the existing hourly parse
    changes by this function existing.

    Returns a DataFrame with columns [timestamp_utc, ae], one row per
    minute, empty if the day has no parseable lines yet (e.g. not
    published, or published with fewer than 60 values on a given line —
    skipped rather than guessed at).
    """
    day_start = date.normalize()
    rows = []
    for line in text.splitlines():
        match = _LINE_HEADER_RE.match(line)
        if not match:
            continue
        hour = int(match.group(2))
        values = line[match.end():].split()
        if len(values) < 60:
            continue
        for minute_idx, raw_value in enumerate(values[:60]):
            try:
                value = float(raw_value)
            except ValueError:
                continue
            timestamp = day_start + pd.Timedelta(hours=hour, minutes=minute_idx)
            rows.append({"timestamp_utc": timestamp, "ae": value})
    return pd.DataFrame(rows)


def _save_raw_minute_day(date: pd.Timestamp, text: str) -> None:
    """Archives Kyoto's raw day-file text verbatim, one file per UT day —
    the same raw-then-processed archival pattern build_master.py uses for
    every NOAA-sourced dataset, adapted here for Kyoto's fixed-width text
    format (not JSON). Nothing in this codebase reads this file back; it
    exists purely as a safety net so the minute parser can be revisited
    or corrected later without having lost the original source text.
    """
    path = RAW_DIR / MINUTE_DATASET_NAME / f"{MINUTE_DATASET_NAME}_{date.strftime('%Y%m%d')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _archive_minute_data(date: pd.Timestamp, text: str) -> None:
    """Persists both the raw day-file text and the parsed minute-level AE
    series permanently. Purely additive: never called by, and has no
    effect on, fetch_kyoto_ae_hour's return value or the existing hourly
    _day_cache. Uses build_master.save_processed_append — the same
    append-and-deduplicate-by-key convention already used for every
    other incrementally-updated dataset in this project — so re-fetching
    a day already archived (which happens routinely, since _day_cache is
    only in-memory and resets on every process restart) safely overwrites
    that day's rows rather than duplicating them.

    Any failure here (disk full, permission error, malformed line, ...)
    is deliberately swallowed rather than raised: archiving minute data
    for future research must never be able to break the hourly
    verification path this module exists to serve today.
    """
    try:
        _save_raw_minute_day(date, text)
        minute_df = _parse_day_file_minutes(text, date)
        if not minute_df.empty:
            save_processed_append(MINUTE_DATASET_NAME, minute_df, dedupe_subset=["timestamp_utc"])
    except Exception:
        pass


def _fetch_kyoto_ae_day(date: pd.Timestamp) -> dict[pd.Timestamp, float]:
    date_key = date.strftime("%Y-%m-%d")
    if date_key in _day_cache:
        return _day_cache[date_key]

    response = requests.get(_day_file_url(date), timeout=30)
    if response.status_code == 404:
        return {}
    response.raise_for_status()

    _archive_minute_data(date, response.text)

    hourly = _parse_day_file(response.text, date)
    if hourly:
        _day_cache[date_key] = hourly
    return hourly


def fetch_kyoto_ae_hour(target_hour) -> float | None:
    """Returns the Kyoto WDC published hourly-mean AE value for
    `target_hour`'s UT hour bucket, or None if that day hasn't been
    published yet. This is the ONLY function the verification engine
    calls — never NOAA, never the prediction pipeline's own frozen AE
    model or local historical file.
    """
    ts = pd.Timestamp(target_hour)
    ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    ts = ts.floor("h")
    return _fetch_kyoto_ae_day(ts.normalize()).get(ts)
