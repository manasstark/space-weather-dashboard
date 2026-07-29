"""SDO/HMI SHARP (Space-weather HMI Active Region Patches) magnetic-
complexity data — the missing ingredient for real flare/CME prediction
this project has been explicitly missing until now (see the Flare
Outlook / CME Outlook discussion). Fetched from JSOC (Stanford) via the
`drms` client, isolated in its own module exactly like Kyoto WDC (AE
verification) and storm_data.py (historical OMNI2/DONKI storm data) are
— never imported by build_master.py/live_update.py's own NOAA/DONKI
pipeline, so nothing about the existing production data path changes.

SHARP already computes the physics-meaningful summary parameters per
active region per 12-minute cadence — this module does not derive them
from raw magnetogram images, the same way build_master.py doesn't
re-derive VBz/Ey from raw fields; NASA's own pipeline already did that
work. Core keywords pulled here (a deliberately small, physically
load-bearing subset, not the full ~30+ SHARP keyword list):

- USFLUX   — total unsigned magnetic flux (Mx): how much flux this AR carries
- MEANGBZ  — mean vertical field gradient (G/Mx): a flare-risk proxy
- R_VALUE  — Schrijver's R: flux near strong polarity-inversion lines,
             the classic operational flare-risk indicator
- TOTPOT   — total photospheric magnetic free-energy proxy (erg/cm)
- MEANJZH  — mean current helicity (G^2/m): sign carries chirality info
- MEANALP  — mean force-free twist parameter (1/Mm): also chirality-related
- SAVNCPP  — sum of absolute net current per polarity (A): another free-energy proxy
- MEANSHR  — mean shear angle (deg): field departure from potential
- ABSNJZH  — absolute net current helicity (G^2/m)

NOAA_AR is returned directly by JSOC alongside HARPNUM — this is the
same active-region number DONKI's CME/flare events already carry
(`active_region` in cme_processed.parquet / solar_events_processed.parquet),
so joining SHARP to flare/CME events needs no separate lookup table.

T_REC is in TAI (International Atomic Time), not UTC — TAI currently
runs ~37 seconds ahead of UTC. That offset is irrelevant at this
project's hourly analysis cadence, so T_REC is treated as UTC here,
same simplification build_master.py already makes for other near-UTC
timestamps; the difference would only matter for sub-minute analysis
this project doesn't do.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from swdss.paths import RAW_DIR

SHARP_SERIES_DEFINITIVE = "hmi.sharp_cea_720s"
SHARP_SERIES_NRT = "hmi.sharp_cea_720s_nrt"
SHARP_CACHE_DIR = RAW_DIR / "sharp"

SHARP_KEYWORDS = [
    "T_REC", "HARPNUM", "NOAA_AR",
    "USFLUX", "MEANGBZ", "R_VALUE", "TOTPOT",
    "MEANJZH", "MEANALP", "SAVNCPP", "MEANSHR", "ABSNJZH",
]


def _tai_to_datetime(t_rec: str) -> pd.Timestamp:
    # "2026.07.28_00:12:00_TAI" -> Timestamp (treated as UTC, see module docstring)
    return pd.to_datetime(t_rec.replace("_TAI", ""), format="%Y.%m.%d_%H:%M:%S", utc=True, errors="coerce")


def _jsoc_time_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    def fmt(ts):
        return ts.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    return f"[][{fmt(start)}-{fmt(end)}]"


def fetch_sharp_day(day: dt.date, use_nrt: bool = True, timeout_retries: int = 2) -> pd.DataFrame:
    """One UT day of SHARP keyword data, cached to disk so re-running a
    backfill (or a live cycle) never re-queries JSOC for a day already
    fetched. Day-sized chunks (not a wider single query) since a single
    multi-day query was observed to time out against JSOC — see the
    module's own storm-backtest validation notes.

    Defaults to the NRT series: JSOC's definitive series
    (hmi.sharp_cea_720s) was found empty for anything from the last
    several months (its processing pipeline lags well behind real time —
    confirmed by direct testing, not assumed), while the NRT series
    covers exactly this recent window. Use use_nrt=False explicitly only
    for older historical dates (e.g. backtesting against a
    NAMED_STORMS-era event), where the definitive series is what's
    actually populated.

    Empty results are never cached — an empty response can mean "no
    data yet" (still processing) rather than "genuinely no active
    regions that day," and caching it would permanently block a later,
    successful retry.
    """
    import drms

    cache_path = SHARP_CACHE_DIR / f"sharp_{day.isoformat()}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["timestamp_utc"])

    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(hours=23, minutes=59, seconds=59)
    series = SHARP_SERIES_NRT if use_nrt else SHARP_SERIES_DEFINITIVE
    query = f"{series}{_jsoc_time_range(start, end)}"

    client = drms.Client()
    last_error = None
    for _ in range(timeout_retries + 1):
        try:
            df = client.query(query, key=SHARP_KEYWORDS)
            break
        except Exception as e:
            last_error = e
            df = None
    if df is None:
        raise RuntimeError(f"SHARP fetch failed for {day} after {timeout_retries + 1} attempts: {last_error}")

    if df.empty:
        return pd.DataFrame(columns=SHARP_KEYWORDS + ["timestamp_utc"])

    df["timestamp_utc"] = df["T_REC"].apply(_tai_to_datetime)
    df = df.dropna(subset=["timestamp_utc"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_sharp_range(start_date: dt.date, end_date: dt.date, use_nrt: bool = True) -> pd.DataFrame:
    """Day-by-day backfill over [start_date, end_date] inclusive, each
    day cached independently — a partial backfill that fails partway
    through (JSOC hiccup, rate limit) keeps every day already fetched,
    and re-running only fetches what's still missing.
    """
    frames = []
    day = start_date
    while day <= end_date:
        try:
            frames.append(fetch_sharp_day(day, use_nrt=use_nrt))
        except Exception as e:
            print(f"[sharp ingest] skipping {day}: {e}")
        day += dt.timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=SHARP_KEYWORDS + ["timestamp_utc"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["HARPNUM", "T_REC"]).sort_values("timestamp_utc").reset_index(drop=True)


def fetch_sharp_recent(hours: int = 6) -> pd.DataFrame:
    """Live/near-real-time SHARP snapshot for the engine's own periodic
    inference — uses the NRT series (available within ~1h of
    observation, unlike the definitive series' longer latency) and is
    NOT cached to disk, since it's meant to reflect "right now," not a
    permanent historical record.
    """
    import drms

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=hours)
    query = f"{SHARP_SERIES_NRT}{_jsoc_time_range(start, end)}"

    client = drms.Client()
    df = client.query(query, key=SHARP_KEYWORDS)
    if df.empty:
        return pd.DataFrame(columns=SHARP_KEYWORDS + ["timestamp_utc"])

    df["timestamp_utc"] = df["T_REC"].apply(_tai_to_datetime)
    return df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
