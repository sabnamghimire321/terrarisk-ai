"""
Central configuration for the TerraRisk AI data pipeline. Every
other module imports paths and constants from here rather than
hardcoding them, so the project can be moved, rescoped (new bounding
box, new resolution), or pointed at a different data directory by
editing this one file.
"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

NEPAL_BOUNDS = {
    "min_lat": 26.3,
    "max_lat": 30.5,
    "min_lon": 80.0,
    "max_lon": 88.3,
}

GRID_RESOLUTION_DEG = 0.25
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_MAX_RETRIES = 3
FLOOD_HISTORY_YEARS = 15
OPEN_METEO_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
FLOOD_RAW_DIR_NAME = "flood_raw"
FLOOD_FLUSH_EVERY_N_CELLS = 50