"""
Fetches static elevation for every grid cell from Open-Meteo's
Elevation API. Elevation has no time dimension, so this is the
simplest ingestion module in the project - batched requests (the API
accepts comma-separated coordinate lists), no retry/rate-limit
complexity needed at this scale.
"""

import json
import pandas as pd
from src.terrarisk.config import DATA_PROCESSED, DATA_RAW, OPEN_METEO_ELEVATION_URL
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.utils.http import get_json

BATCH_SIZE = 100  

def _batch(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]

def fetch_elevation_for_grid(grid: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    raw_responses = []

    row_batches = _batch(grid.to_dict("records"), BATCH_SIZE)

    for batch_num, batch in enumerate(row_batches, start=1):
        lats = [str(row["lat"]) for row in batch]
        lons = [str(row["lon"]) for row in batch]

        params = {
            "latitude": ",".join(lats),
            "longitude": ",".join(lons),
        }

        response = get_json(OPEN_METEO_ELEVATION_URL, params=params)
        raw_responses.append(response)

        elevations = response["elevation"]

        for row, elevation_m in zip(batch, elevations):
            all_rows.append(
                {
                    "cell_id": row["cell_id"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "elevation_m": elevation_m,
                }
            )

        print(f"Batch {batch_num}/{len(row_batches)} fetched "
              f"({len(batch)} cells).")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW / "open_meteo_elevation_raw.json"
    raw_path.write_text(json.dumps(raw_responses, indent=2))
    print(f"Raw responses saved to {raw_path}")

    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    grid = build_grid()
    elevation_df = fetch_elevation_for_grid(grid)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    processed_path = DATA_PROCESSED / "grid_elevation.csv"
    elevation_df.to_csv(processed_path, index=False)

    print(f"Processed elevation data saved to {processed_path}")
    print(elevation_df.head())
    print(f"Total cells: {len(elevation_df)}")