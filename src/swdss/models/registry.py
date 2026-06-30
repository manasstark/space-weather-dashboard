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
KP_VARIABLES = ["kp"]
DST_VARIABLES = ["dst"]

# The Analytics "Combined Earth Analysis" predictor: forecasts Kp and Dst
# from the full Sun-Earth feature set (Solar Wind + IMF + Kp/Dst lags),
# rather than each dataset's own narrower self-referential feature set.
ANALYTICS_VARIABLES = ["kp", "dst"]
ANALYTICS_FEATURE_VARIABLES = [
    "speed",
    "density",
    "temperature",
    "bt",
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "kp",
    "dst",
]

VARIABLE_LABELS = {
    "speed": "Speed",
    "density": "Density",
    "temperature": "Temperature",
    "bt": "Bt",
    "bx_gsm": "Bx",
    "by_gsm": "By",
    "bz_gsm": "Bz",
    "kp": "Kp",
    "dst": "Dst",
}

VARIABLE_UNITS = {
    "speed": "km/s",
    "density": "p/cm3",
    "temperature": "K",
    "bt": "nT",
    "bx_gsm": "nT",
    "by_gsm": "nT",
    "bz_gsm": "nT",
    "kp": "",
    "dst": "nT",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    training_csv: str
    variables: list
    processed_parquet: str = None
    raw_column_map: dict = None  # processed-parquet column -> training-feature column

    # Only needed when the engineered feature set spans more variables than
    # are actually predicted (e.g. Analytics predicts Kp/Dst but engineers
    # features from Solar Wind + IMF + Kp + Dst together). Defaults to
    # `variables` when unset.
    feature_variables: list = None

    # Only needed for multi-source datasets (no single processed_parquet of
    # their own): names of other DATASETS entries to merge live data from,
    # each contributing via its own raw_column_map.
    source_datasets: list = None

    # Per-variable divisor applied to the TRAINING data only, to correct a
    # unit mismatch against the live processed data (e.g. Kp is stored as
    # tenths in the OMNI-derived training CSVs but as natural 0-9 values in
    # the live NOAA feed).
    scale_factors: dict = None


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
    "kp": DatasetConfig(
        name="kp",
        training_csv=str(TRAINING_FEATURES_DIR / "kp_features.csv"),
        variables=KP_VARIABLES,
        processed_parquet=str(PROCESSED_DIR / "kp" / "kp_processed.parquet"),
        raw_column_map={"kp": "kp"},
        scale_factors={"kp": 10.0},
    ),
    "dst": DatasetConfig(
        name="dst",
        training_csv=str(TRAINING_FEATURES_DIR / "dst_features.csv"),
        variables=DST_VARIABLES,
        processed_parquet=str(PROCESSED_DIR / "dst" / "dst_processed.parquet"),
        raw_column_map={"dst": "dst"},
    ),
    "analytics": DatasetConfig(
        name="analytics",
        training_csv=str(TRAINING_FEATURES_DIR / "analytics_features.csv"),
        variables=ANALYTICS_VARIABLES,
        feature_variables=ANALYTICS_FEATURE_VARIABLES,
        source_datasets=["solar_wind", "imf", "kp", "dst"],
        scale_factors={"kp": 10.0},
    ),
}


def owning_source_config(dataset: str, variable: str) -> DatasetConfig:
    """For multi-source datasets (e.g. "analytics"), returns the actual
    DATASETS entry that owns live data for `variable` (e.g. the "kp"
    config for variable="kp"), so live lookups can use its own
    processed_parquet + raw_column_map. For single-source datasets,
    just returns their own config.
    """
    config = DATASETS[dataset]
    if not config.source_datasets:
        return config
    for source_name in config.source_datasets:
        source_config = DATASETS[source_name]
        if variable in (source_config.raw_column_map or {}).values():
            return source_config
    raise KeyError(f"No source dataset owns variable '{variable}' for '{dataset}'")


def raw_column_for(dataset: str, variable: str) -> str:
    """Inverse of raw_column_map: given a training-feature variable name
    (e.g. "speed"), returns the matching live processed-parquet column
    (e.g. "solar_wind_speed"). Resolves through the owning source dataset
    for multi-source datasets.
    """
    config = owning_source_config(dataset, variable)
    for raw_col, training_col in (config.raw_column_map or {}).items():
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


def kp_interval_model_path(dataset: str = "analytics") -> Path:
    """Kp on the Analytics page follows NOAA's real 3-hour publishing
    cadence instead of an arbitrary hourly horizon, so it gets one model
    (not five) stored outside the normal {variable}_{horizon}h.joblib
    convention.
    """
    return model_dir(dataset) / "kp_interval.joblib"
