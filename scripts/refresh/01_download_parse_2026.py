"""Step 1 — Download and parse OMNI2 2026 data (Jan 1 – Jun 30).

Downloads the official NASA SPDF OMNI2 hourly file for 2026, extracts
exactly the same 12 variables used in master_omni.csv, replaces fill
values with NaN, and saves to data/raw/omni_refresh_2026.csv.

This file is a versioned input artifact — it is never overwritten by
later pipeline steps.

Run from project root:
    PYTHONPATH=src venv/bin/python3 scripts/refresh/01_download_parse_2026.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_PATH = PROJECT_ROOT / "data" / "raw" / "omni_refresh_2026.csv"
OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_2026.dat"

# 0-indexed column positions in the space-split OMNI2 hourly record.
# Verified against https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
COL = {
    "bt":          8,   # |B| scalar magnitude (nT)
    "bx_gsm":     12,   # Bx GSE/GSM (nT)
    "by_gsm":     15,   # By GSM (nT)
    "bz_gsm":     16,   # Bz GSM (nT)
    "temperature": 22,  # Proton temperature (K)
    "density":    23,   # Proton density (N/cm³)
    "speed":      24,   # Plasma flow speed (km/s)
    "ey":         35,   # Electric field Ey = -V·Bz·1e-3 (mV/m)
    "kp":         38,   # Kp index × 10 (e.g. 23 → Kp 2.3)
    "dst":        40,   # Dst index (nT)
    "ae":         41,   # AE index (nT)
    "f107":       50,   # F10.7 solar flux (sfu)
}

# Fill values used by OMNI for each variable (row is all-fill when missing).
FILL = {
    "bt":          999.9,
    "bx_gsm":     999.9,
    "by_gsm":     999.9,
    "bz_gsm":     999.9,
    "temperature": 9999999.0,
    "density":    999.9,
    "speed":      9999.0,
    "ey":         999.99,
    "kp":         99,
    "dst":        99999,
    "ae":         9999,
    "f107":       999.9,
}

# Restrict to Jan 1 – Jun 30 (DOY 1–181; 2026 is not a leap year)
DOY_MAX = 181


def download_raw() -> str:
    print(f"Downloading {OMNI_URL} ...")
    r = requests.get(OMNI_URL, timeout=120)
    r.raise_for_status()
    print(f"  Downloaded {len(r.content):,} bytes")
    return r.text


def parse_omni(text: str) -> pd.DataFrame:
    records = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 51:
            continue
        year, doy, hour = int(parts[0]), int(parts[1]), int(parts[2])
        if year != 2026 or doy > DOY_MAX:
            continue

        dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour)
        row: dict = {"datetime": dt}

        for var, col in COL.items():
            try:
                val = float(parts[col])
            except (ValueError, IndexError):
                val = float("nan")

            if val == FILL[var]:
                val = float("nan")
            row[var] = val

        records.append(row)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def print_coverage(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\nParsed {total} hourly rows (Jan 1 – Jun 30 2026)")
    print(f"\n{'Variable':<14} {'Good rows':<12} {'Coverage %':<12} {'Last valid'}")
    print("-" * 58)
    for var in COL:
        good = df[var].notna().sum()
        pct = good / total * 100
        last = df.loc[df[var].notna(), "datetime"].max()
        last_str = last.strftime("%Y-%m-%d %H:00") if pd.notna(last) else "N/A"
        print(f"{var:<14} {good:<12} {pct:<12.1f} {last_str}")


def main() -> None:
    text = download_raw()
    df = parse_omni(text)
    print_coverage(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_PATH}")
    print("Step 1 complete. Run 02_build_v2_datasets.py next.")


if __name__ == "__main__":
    main()
