"""
Reads NASA's Global Landslide Catalog from a manually-downloaded CSV
rather than a live API. Three separate live-endpoint attempts against
COOLR's ArcGIS services failed during development (one returned 404,
one was blocked by robots.txt, one referenced a resource that had
since been deprecated) - this is a deliberate fallback to a
browser-driven export, not a workaround for a coding bug. Landslide
inventories also update irregularly, so a manual download-once
approach is a reasonable fit for this data source regardless.
"""

import numpy as np
import pandas as pd

from src.terrarisk.config import DATA_PROCESSED, DATA_RAW, NEPAL_BOUNDS
from src.terrarisk.grid.build_grid import build_grid

RAW_CSV_PATH = DATA_RAW / "global_landslide_catalog_raw.csv"


def _nearest_cell_id(lat: float, lon: float, grid: pd.DataFrame) -> str:
    distances = np.sqrt((grid["lat"] - lat) ** 2 + (grid["lon"] - lon) ** 2)
    nearest_idx = distances.idxmin()
    return grid.loc[nearest_idx, "cell_id"]


def load_and_filter_to_nepal(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Load the manually-downloaded global catalog, filter to events
    within NEPAL_BOUNDS, and join each event to its nearest grid cell.
    """
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} not found. Download the Global Landslide "
            "Catalog CSV from https://landslides.nasa.gov (Landslide "
            "Viewer -> Download the full Landslide Catalog) and save "
            f"it at exactly this path before running this script."
        )

    df = pd.read_csv(RAW_CSV_PATH)

    print(f"Loaded {len(df)} total global landslide events.")
    print("Columns found:")
    for col in df.columns:
        print(f"  - {col}")

    within_bounds = (
        (df["latitude"] >= NEPAL_BOUNDS["min_lat"])
        & (df["latitude"] <= NEPAL_BOUNDS["max_lat"])
        & (df["longitude"] >= NEPAL_BOUNDS["min_lon"])
        & (df["longitude"] <= NEPAL_BOUNDS["max_lon"])
    )
    nepal_df = df[within_bounds].copy()
    print(f"\n{len(nepal_df)} events fall within Nepal's bounding box.")

    if len(nepal_df) == 0:
        print(
            "WARNING: zero events in bounds. This likely means the "
            "latitude/longitude column names above don't match what "
            "this code expects — check the printed column list against "
            "the 'latitude'/'longitude' names used in the filter."
        )
        return nepal_df

    nepal_df["cell_id"] = nepal_df.apply(
        lambda row: _nearest_cell_id(row["latitude"], row["longitude"], grid),
        axis=1,
    )

    keep_columns = [
        "cell_id",
        "latitude",
        "longitude",
        "event_date",
        "landslide_category",
        "landslide_trigger",
        "source_name",
    ]
    available_columns = [c for c in keep_columns if c in nepal_df.columns]

    return nepal_df[available_columns]


if __name__ == "__main__":
    grid = build_grid()
    nepal_events = load_and_filter_to_nepal(grid)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    processed_path = DATA_PROCESSED / "landslide_events.csv"
    nepal_events.to_csv(processed_path, index=False)

    print(f"\nProcessed Nepal landslide events saved to {processed_path}")
    print(nepal_events.head())
    print(f"Total events: {len(nepal_events)}")