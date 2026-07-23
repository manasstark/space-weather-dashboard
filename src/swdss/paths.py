from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
TRAINING_FEATURES_DIR = FEATURES_DIR / "training"

MASTER_V1_PATH = FEATURES_DIR / "master_df_v1.parquet"

MODELS_DIR = PROJECT_ROOT / "models"

# Operational Forecast Engine's storage root — see swdss.engine.storage.
# Kept separate from FEATURES_DIR/MODELS_DIR since these are read/written
# by the engine's orchestration layer (run_forecast_cycle/evaluate_due_
# forecasts/refresh_dashboard_products), not by training or live-inference
# code directly.
FORECASTS_DIR = DATA_DIR / "forecasts"


def ensure_data_dirs() -> None:
    for folder in [
        RAW_DIR / "solar_wind",
        RAW_DIR / "imf",
        RAW_DIR / "kp",
        RAW_DIR / "dst",
        RAW_DIR / "solar_events",
        RAW_DIR / "cme",
        RAW_DIR / "f107",
        RAW_DIR / "kyoto_ae_minute",
        PROCESSED_DIR / "solar_wind",
        PROCESSED_DIR / "imf",
        PROCESSED_DIR / "kp",
        PROCESSED_DIR / "dst",
        PROCESSED_DIR / "ae",
        PROCESSED_DIR / "solar_events",
        PROCESSED_DIR / "cme",
        PROCESSED_DIR / "f107",
        PROCESSED_DIR / "kyoto_ae_minute",
        FEATURES_DIR,
        TRAINING_FEATURES_DIR,
        MODELS_DIR,
        FORECASTS_DIR / "current",
        FORECASTS_DIR / "history",
        FORECASTS_DIR / "logs",
    ]:
        folder.mkdir(parents=True, exist_ok=True)