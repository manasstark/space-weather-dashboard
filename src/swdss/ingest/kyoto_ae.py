"""Kyoto World Data Center (WDC) for Geomagnetism — real-time (quicklook)
AE index digital data client.

This is the ONLY source ever used to verify AE predictions. NOAA/DONKI
publish no AE product at all, so this is not a choice between sources —
it's the sole way to get an official AE observation. Keeping it in its
own module (never imported by build_master.py/live_update.py, which stay
NOAA/DONKI-only) keeps the prediction engine (driven by live NOAA Solar
Wind/IMF) and the verification engine (driven by Kyoto WDC) fully
independent — neither can influence the other's data source.

Data source: https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir/
One fixed-width WDC-format text file per UT day (e.g. "ae260621" for
2026-06-21), each of its 24 lines holding one hour's 60 one-minute AE
values plus a trailing hourly mean — that hourly mean is what this module
returns, matching the hourly resolution used everywhere else in this
project's AE handling.

Important: "real-time" here means "the real-time/quicklook index
product," not "available within minutes of the event." In practice Kyoto
WDC batches and quality-checks before publishing — observed lag has been
on the order of 1-3 weeks. A prediction can stay "Awaiting Official AE"
for a long time; that's expected, not a bug.
"""

import re

import pandas as pd
import requests

KYOTO_AE_BASE_URL = "https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir"

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


def _fetch_kyoto_ae_day(date: pd.Timestamp) -> dict[pd.Timestamp, float]:
    date_key = date.strftime("%Y-%m-%d")
    if date_key in _day_cache:
        return _day_cache[date_key]

    response = requests.get(_day_file_url(date), timeout=30)
    if response.status_code == 404:
        return {}
    response.raise_for_status()

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
