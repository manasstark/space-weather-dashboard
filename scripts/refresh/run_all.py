"""Master runner — execute the full Production Model Refresh pipeline.

Steps:
  01  Download and parse OMNI2 2026 data (Jan 1 – Jun 30)
  02  Merge with v1 archive, build v2 training CSVs
  03  Retrain all production models on v2 dataset
  04  Benchmark v1 vs v2 and write promotion report

Run from project root:
    PYTHONPATH=src venv/bin/python3 scripts/refresh/run_all.py

Individual steps can be re-run independently:
    PYTHONPATH=src venv/bin/python3 scripts/refresh/03_train_v2.py solar_wind imf
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(PROJECT_ROOT / "venv" / "bin" / "python3")
SCRIPTS = Path(__file__).parent

STEPS = [
    (SCRIPTS / "01_download_parse_2026.py",  "Download & parse OMNI 2026"),
    (SCRIPTS / "02_build_v2_datasets.py",    "Build v2 training datasets"),
    (SCRIPTS / "03_train_v2.py",             "Retrain all v2 models"),
    (SCRIPTS / "04_benchmark.py",            "Benchmark & generate report"),
]

def run_step(script: Path, label: str) -> None:
    print(f"\n{'#'*70}")
    print(f"#  STEP: {label}")
    print(f"{'#'*70}\n")
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    import os
    env.update(os.environ)
    result = subprocess.run(
        [PYTHON, str(script)],
        env=env,
    )
    if result.returncode != 0:
        print(f"\n✗ Step failed: {label}")
        print(f"  Fix the error above and re-run:  PYTHONPATH=src venv/bin/python3 {script}")
        sys.exit(result.returncode)
    print(f"\n✓ Step complete: {label}")


def main() -> None:
    for script, label in STEPS:
        run_step(script, label)

    print(f"\n{'='*70}")
    print("  ALL STEPS COMPLETE")
    print(f"  Report: {PROJECT_ROOT}/reports/refresh_v2/report.txt")
    print(f"  v2 models: {PROJECT_ROOT}/models/{{dataset}}_v2/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
