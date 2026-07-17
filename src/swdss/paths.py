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
    ]:
        folder.mkdir(parents=True, exist_ok=True)