"""Single source of truth for the prediction module.

Both train.py and predict.py import dataset configuration from here so the
training pipeline and the live inference pipeline can never drift apart.
"""

from dataclasses import dataclass
from pathlib import Path

from swdss.paths import MODELS_DIR, PROCESSED_DIR, TRAINING_FEATURES_DIR

HORIZONS = [1, 3, 6, 12, 24]

SOLAR_WIND_VARIABLES = ["speed", "density", "temperature"]
IMF_VARIABLES = ["bt", "bx_gsm", "by_gsm", "bz_gsm"]

VARIABLE_LABELS = {
    "speed": "Speed",
    "density": "Density",
    "temperature": "Temperature",
    "bt": "Bt",
    "bx_gsm": "Bx",
    "by_gsm": "By",
    "bz_gsm": "Bz",
}

VARIABLE_UNITS = {
    "speed": "km/s",
    "density": "p/cm3",
    "temperature": "K",
    "bt": "nT",
    "bx_gsm": "nT",
    "by_gsm": "nT",
    "bz_gsm": "nT",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    training_csv: str
    variables: list
    processed_parquet: str
    raw_column_map: dict  # processed-parquet column -> training-feature column


DATASETS = {
    "solar_wind": DatasetConfig(
        name="solar_wind",
        training_csv=str(TRAINING_FEATURES_DIR / "solar_wind_features.csv"),
        variables=SOLAR_WIND_VARIABLES,
        processed_parquet=str(PROCESSED_DIR / "solar_wind" / "solar_wind_processed.parquet"),
        raw_column_map={
            "solar_wind_speed": "speed",
            "proton_density": "density",
            "temperature": "temperature",
        },
    ),
    "imf": DatasetConfig(
        name="imf",
        training_csv=str(TRAINING_FEATURES_DIR / "imf_features.csv"),
        variables=IMF_VARIABLES,
        processed_parquet=str(PROCESSED_DIR / "imf" / "imf_processed.parquet"),
        raw_column_map={
            "bt": "bt",
            "bx": "bx_gsm",
            "by": "by_gsm",
            "bz": "bz_gsm",
        },
    ),
}


def raw_column_for(dataset: str, variable: str) -> str:
    """Inverse of raw_column_map: given a training-feature variable name
    (e.g. "speed"), returns the matching live processed-parquet column
    (e.g. "solar_wind_speed").
    """
    config = DATASETS[dataset]
    for raw_col, training_col in config.raw_column_map.items():
        if training_col == variable:
            return raw_col
    raise KeyError(f"No raw column mapped to variable '{variable}' in dataset '{dataset}'")


def model_dir(dataset: str) -> Path:
    path = Path(MODELS_DIR) / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(dataset: str, variable: str, horizon: int) -> Path:
    return model_dir(dataset) / f"{variable}_{horizon}h.joblib"


def metrics_path(dataset: str) -> Path:
    return model_dir(dataset) / "metrics.json"
