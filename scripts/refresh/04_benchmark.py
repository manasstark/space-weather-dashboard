"""Step 4 — Benchmark v1 vs v2 models and produce promotion recommendations.

Reads metrics.json from v1 and v2 model directories, computes deltas for
every (dataset, variable, horizon) combination, and writes:

  reports/refresh_v2/benchmark.json   ← machine-readable full results
  reports/refresh_v2/report.txt       ← human-readable summary

Promotion rule:
  Recommend PROMOTE if v2 R² > v1 R² (strictly better on held-out test set).
  Recommend KEEP CURRENT if v2 R² ≤ v1 R².
  Flag REVIEW if |R² delta| > 0.05 regardless of direction (large swing).

Run from project root:
    PYTHONPATH=src venv/bin/python3 scripts/refresh/04_benchmark.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "refresh_v2"

DATASETS_ORDER = ["solar_wind", "imf", "kp", "dst", "ae", "analytics", "experimental"]

LARGE_SWING_THRESHOLD = 0.05


def load_metrics(dataset: str, version: str) -> dict:
    if version == "v1":
        path = MODELS_DIR / dataset / "metrics.json"
    else:
        path = MODELS_DIR / f"{dataset}_v2" / "metrics.json"

    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def recommendation(r2_v1: float, r2_v2: float) -> str:
    delta = r2_v2 - r2_v1
    if r2_v2 > r2_v1:
        tag = "PROMOTE"
    else:
        tag = "KEEP CURRENT"
    if abs(delta) > LARGE_SWING_THRESHOLD:
        tag += " ⚠ LARGE SWING"
    return tag


def build_comparison() -> list[dict]:
    rows = []
    for dataset in DATASETS_ORDER:
        v1 = load_metrics(dataset, "v1")
        v2 = load_metrics(dataset, "v2")
        if not v1 and not v2:
            continue

        all_keys = sorted(set(v1) | set(v2))
        for key in all_keys:
            e1 = v1.get(key, {})
            e2 = v2.get(key, {})

            variable  = e1.get("variable") or e2.get("variable") or key.split("_")[0]
            horizon   = e1.get("horizon")  or e2.get("horizon")  or key
            algo_v1   = e1.get("algorithm", "N/A")
            algo_v2   = e2.get("algorithm", "N/A")

            r2_v1  = e1.get("r2",   None)
            r2_v2  = e2.get("r2",   None)
            mae_v1 = e1.get("mae",  None)
            mae_v2 = e2.get("mae",  None)
            rmse_v1= e1.get("rmse", None)
            rmse_v2= e2.get("rmse", None)
            bias_v2= e2.get("bias", None)

            n_train_v1 = e1.get("n_samples", None)
            n_train_v2 = e2.get("n_samples", None)

            rec = "N/A"
            delta_r2 = None
            if r2_v1 is not None and r2_v2 is not None:
                delta_r2 = r2_v2 - r2_v1
                rec = recommendation(r2_v1, r2_v2)
            elif r2_v2 is not None:
                rec = "NEW (no v1 baseline)"

            rows.append({
                "dataset":    dataset,
                "variable":   variable,
                "horizon":    horizon,
                "algo_v1":    algo_v1,
                "algo_v2":    algo_v2,
                "r2_v1":      r2_v1,
                "r2_v2":      r2_v2,
                "delta_r2":   delta_r2,
                "mae_v1":     mae_v1,
                "mae_v2":     mae_v2,
                "rmse_v1":    rmse_v1,
                "rmse_v2":    rmse_v2,
                "bias_v2":    bias_v2,
                "n_v1":       n_train_v1,
                "n_v2":       n_train_v2,
                "recommendation": rec,
            })
    return rows


def fmt(val, fmt_str=".4f", fallback="N/A") -> str:
    if val is None:
        return fallback
    try:
        return format(val, fmt_str)
    except Exception:
        return str(val)


def write_report(rows: list[dict]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("  SW-DSS PRODUCTION MODEL REFRESH — v1 vs v2 BENCHMARK REPORT")
    lines.append(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 80)
    lines.append("")

    # Aggregate summary
    promote  = [r for r in rows if "PROMOTE" in r["recommendation"]]
    keep     = [r for r in rows if "KEEP CURRENT" in r["recommendation"]]
    swing    = [r for r in rows if "LARGE SWING" in r["recommendation"]]
    new_only = [r for r in rows if "NEW" in r["recommendation"]]

    lines.append(f"SUMMARY")
    lines.append(f"  Models assessed:          {len(rows)}")
    lines.append(f"  Recommend PROMOTE:        {len(promote)}")
    lines.append(f"  Recommend KEEP CURRENT:   {len(keep)}")
    lines.append(f"  Large swing (|ΔR²|>0.05): {len(swing)}")
    lines.append("")

    # Per-dataset sections
    for dataset in DATASETS_ORDER:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue

        lines.append("")
        lines.append(f"{'─'*80}")
        lines.append(f"  Dataset: {dataset.upper()}")
        lines.append(f"{'─'*80}")
        lines.append(
            f"  {'Variable':<14} {'Horizon':<10} "
            f"{'R² v1':<9} {'R² v2':<9} {'ΔR²':>8} "
            f"{'MAE v1':<10} {'MAE v2':<10} "
            f"{'RMSE v1':<10} {'RMSE v2':<10} "
            f"{'Bias v2':>9}  {'N v1':>7} {'N v2':>7}  Recommendation"
        )
        lines.append(
            f"  {'-'*14} {'-'*10} "
            f"{'-'*8} {'-'*8} {'-'*8} "
            f"{'-'*9} {'-'*9} "
            f"{'-'*9} {'-'*9} "
            f"{'-'*9}  {'-'*7} {'-'*7}  {'-'*30}"
        )

        for r in ds_rows:
            delta_str = (f"{r['delta_r2']:+.4f}" if r["delta_r2"] is not None else "   N/A")
            lines.append(
                f"  {str(r['variable']):<14} {str(r['horizon']):<10} "
                f"{fmt(r['r2_v1']):<9} {fmt(r['r2_v2']):<9} {delta_str:>8} "
                f"{fmt(r['mae_v1'], '.3f'):<10} {fmt(r['mae_v2'], '.3f'):<10} "
                f"{fmt(r['rmse_v1'], '.3f'):<10} {fmt(r['rmse_v2'], '.3f'):<10} "
                f"{fmt(r['bias_v2'], '+.3f'):>9}  {fmt(r['n_v1'], 'd', 'N/A'):>7} {fmt(r['n_v2'], 'd', 'N/A'):>7}  "
                f"{r['recommendation']}"
            )

    # Promotion list
    lines.append("")
    lines.append("=" * 80)
    lines.append("  PROMOTE LIST (copy v2 → production)")
    lines.append("=" * 80)
    if promote:
        for r in promote:
            lines.append(
                f"  models/{r['dataset']}_v2/{r['variable']}_{r['horizon']}h.joblib"
                f"  → models/{r['dataset']}/{r['variable']}_{r['horizon']}h.joblib"
                f"   (ΔR² {r['delta_r2']:+.4f})"
            )
    else:
        lines.append("  No models recommended for promotion.")

    # Keep list
    lines.append("")
    lines.append("=" * 80)
    lines.append("  KEEP CURRENT (v2 did not improve)")
    lines.append("=" * 80)
    if keep:
        for r in keep:
            lines.append(
                f"  models/{r['dataset']}/{r['variable']}_{r['horizon']}h.joblib   "
                f"(v1 R²={fmt(r['r2_v1'])}  v2 R²={fmt(r['r2_v2'])}  "
                f"ΔR²={r['delta_r2']:+.4f})"
            )
    else:
        lines.append("  All models improved.")

    # Large swings — flag for manual review
    if swing:
        lines.append("")
        lines.append("=" * 80)
        lines.append("  ⚠  LARGE SWINGS — REVIEW BEFORE ACTING")
        lines.append("=" * 80)
        for r in swing:
            lines.append(
                f"  {r['dataset']} / {r['variable']} / {r['horizon']}h  "
                f"ΔR²={r['delta_r2']:+.4f}"
            )

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v1 and v2 metrics...")
    rows = build_comparison()

    if not rows:
        print("No metrics found. Make sure both 01–03 steps have been run.")
        sys.exit(1)

    # Save machine-readable JSON
    json_path = REPORTS_DIR / "benchmark.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Saved {json_path}")

    # Generate and save human-readable report
    report = write_report(rows)
    txt_path = REPORTS_DIR / "report.txt"
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"Saved {txt_path}")

    # Print to terminal
    print()
    print(report)


if __name__ == "__main__":
    main()
