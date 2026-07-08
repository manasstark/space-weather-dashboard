"""Step 2 — Merge 2026 data and build all v2 training CSVs.

Appends the parsed 2026 observations to the existing master_omni.csv,
deduplicates, validates timestamps, and writes:

  data/processed/master_omni_v2.csv          ← complete archive 2023–Jun 2026
  data/features/training_v2/solar_wind_features.csv
  data/features/training_v2/imf_features.csv
  data/features/training_v2/kp_features.csv
  data/features/training_v2/dst_features.csv
  data/features/training_v2/ae_analytics_features.csv
  data/features/training_v2/analytics_features.csv
  data/features/training_v2/experimental_features.csv   ← uses v1 AE model

Run from project root:
    PYTHONPATH=src venv/bin/python3 scripts/refresh/02_build_v2_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swdss.models.features import add_derived_physics_features, build_feature_frame
from swdss.models.predict import generate_predicted_ae
from swdss.models.registry import AE_FEATURE_VARIABLES

# Paths
RAW_2026      = PROJECT_ROOT / "data" / "raw" / "omni_refresh_2026.csv"
MASTER_V1     = PROJECT_ROOT / "data" / "processed" / "master_omni.csv"
MASTER_V2     = PROJECT_ROOT / "data" / "processed" / "master_omni_v2.csv"
TRAIN_V1_DIR  = PROJECT_ROOT / "data" / "features" / "training"
TRAIN_V2_DIR  = PROJECT_ROOT / "data" / "features" / "training_v2"


def load_master_v1() -> pd.DataFrame:
    df = pd.read_csv(MASTER_V1, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    print(f"v1 archive: {len(df)} rows  [{df.index.min()} → {df.index.max()}]")
    return df


def load_2026() -> pd.DataFrame:
    df = pd.read_csv(RAW_2026, parse_dates=["datetime"])
    df = df.set_index("datetime")
    df.index.name = "datetime"
    # Drop year/doy/hour if present (not in master_omni.csv either)
    for col in ["year", "doy", "hour"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    print(f"2026 data:  {len(df)} rows  [{df.index.min()} → {df.index.max()}]")
    return df


def merge(v1: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns present in master_omni.csv
    keep = [c for c in v1.columns if c in new.columns]
    new = new[keep]

    combined = pd.concat([v1, new])
    before = len(combined)
    combined = combined[~combined.index.duplicated(keep="first")]
    dupes = before - len(combined)
    if dupes:
        print(f"  Removed {dupes} duplicate timestamps")

    combined = combined.sort_index()

    # Validate monotonicity
    assert combined.index.is_monotonic_increasing, "Timestamp ordering broken after merge"
    print(f"Merged archive: {len(combined)} rows  [{combined.index.min()} → {combined.index.max()}]")
    return combined


def save_csv(df: pd.DataFrame, path: Path, columns: list[str], label: str) -> pd.DataFrame:
    out = df[columns].dropna(how="all").copy()
    out = out.reset_index()  # bring datetime back as a column
    out.to_csv(path, index=False)
    print(f"  {label}: {len(out)} rows → {path.name}")
    return out


def build_experimental(analytics_v2_path: Path, ae_analytics_v2_path: Path) -> None:
    """Generate predicted_ae from the frozen v1 AE 1h model, then build
    experimental_features_v2.csv.  Uses the v1 AE model intentionally —
    the v2 AE model isn't trained yet and the experimental cascade must
    replicate the exact inference-time AE generation logic."""

    ae_raw = pd.read_csv(ae_analytics_v2_path, parse_dates=["datetime"])
    ae_raw = ae_raw.sort_values("datetime").set_index("datetime")

    base_df = ae_raw[AE_FEATURE_VARIABLES].copy()
    derived_cols = add_derived_physics_features(base_df)
    feature_vars = AE_FEATURE_VARIABLES + derived_cols
    frame, feature_columns = build_feature_frame(base_df, feature_vars)

    predicted_ae = generate_predicted_ae(frame)

    analytics = pd.read_csv(analytics_v2_path, parse_dates=["datetime"])
    analytics = analytics.sort_values("datetime").set_index("datetime")
    analytics = analytics.drop(columns=["ae"], errors="ignore")

    experimental = analytics.copy()
    experimental["predicted_ae"] = predicted_ae.reindex(experimental.index)
    experimental = experimental.dropna(subset=["predicted_ae"])
    experimental = experimental.reset_index()

    out_path = TRAIN_V2_DIR / "experimental_features.csv"
    experimental.to_csv(out_path, index=False)
    print(f"  experimental: {len(experimental)} rows → {out_path.name}")


def print_summary(v1: pd.DataFrame, v2: pd.DataFrame) -> None:
    added = len(v2) - len(v1)
    v1_end = v1.index.max()
    v2_end = v2.index.max()
    print(f"\n=== Dataset Summary ===")
    print(f"v1 rows:  {len(v1):>7,}   ends: {v1_end}")
    print(f"v2 rows:  {len(v2):>7,}   ends: {v2_end}")
    print(f"Added:    {added:>7,} rows ({added/24:.0f} days of hourly data)")
    print()
    print("Variable coverage in new 2026 rows:")
    new_rows = v2[v2.index > v1_end]
    for col in ["speed", "density", "temperature", "bt", "bx_gsm", "bz_gsm", "kp", "dst", "ae", "f107"]:
        if col in new_rows.columns:
            good = new_rows[col].notna().sum()
            pct = good / len(new_rows) * 100 if len(new_rows) else 0
            print(f"  {col:<14} {good:>5} / {len(new_rows)} rows  ({pct:.1f}%)")


def main() -> None:
    if not RAW_2026.exists():
        print(f"ERROR: {RAW_2026} not found. Run 01_download_parse_2026.py first.")
        sys.exit(1)

    TRAIN_V2_DIR.mkdir(parents=True, exist_ok=True)

    v1 = load_master_v1()
    new = load_2026()
    v2 = merge(v1, new)

    # Save master v2
    v2_out = v2.reset_index()
    v2_out.to_csv(MASTER_V2, index=False)
    print(f"Saved master_omni_v2.csv ({len(v2_out)} rows)")

    print("\nBuilding v2 training CSVs...")

    save_csv(v2, TRAIN_V2_DIR / "solar_wind_features.csv",
             ["speed", "density", "temperature"], "solar_wind")

    save_csv(v2, TRAIN_V2_DIR / "imf_features.csv",
             ["bt", "bx_gsm", "by_gsm", "bz_gsm"], "imf")

    save_csv(v2, TRAIN_V2_DIR / "kp_features.csv",
             ["kp"], "kp")

    save_csv(v2, TRAIN_V2_DIR / "dst_features.csv",
             ["dst"], "dst")

    ae_analytics_path = TRAIN_V2_DIR / "ae_analytics_features.csv"
    save_csv(v2, ae_analytics_path,
             ["speed", "density", "temperature", "bt", "bx_gsm", "by_gsm", "bz_gsm", "ae"],
             "ae_analytics")

    analytics_path = TRAIN_V2_DIR / "analytics_features.csv"
    save_csv(v2, analytics_path,
             ["speed", "density", "temperature", "bt", "bx_gsm", "by_gsm", "bz_gsm", "kp", "dst", "ae"],
             "analytics")

    print("  Building experimental (predicted_ae via v1 AE model)...")
    build_experimental(analytics_path, ae_analytics_path)

    print_summary(v1, v2)
    print("\nStep 2 complete. Run 03_train_v2.py next.")


if __name__ == "__main__":
    main()
