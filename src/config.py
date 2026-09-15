"""Central configuration shared by notebooks and command-line modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

OPSD_RELEASE = "2020-10-06"
OPSD_DATASET_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
)
TARGET_COLUMN = "DE_load_actual_entsoe_transparency"
TIME_COLUMN = "utc_timestamp"

INPUT_WINDOW_HOURS = 168
FORECAST_HORIZON_HOURS = 24
RANDOM_SEED = 42


def ensure_output_directories() -> None:
    """Create generated-data directories if they do not yet exist."""

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
