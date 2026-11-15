"""
Builds the fixed lat/lon grid the pipeline operates on (research
plan Section 7: ~0.25-degree cells, aggregated to region/country
only at the presentation layer). Pure geometry, no network calls.

Cell IDs are derived from rounded lat/lon (e.g. "grid_27.25_85.50"),
not an auto-increment integer: if the grid is regenerated later with
wider bounds or finer resolution, cells that still exist keep the
same ID.
"""

import numpy as np
import pandas as pd

from src.terrarisk.config import GRID_RESOLUTION_DEG, NEPAL_BOUNDS

def _make_cell_id(lat: float, lon: float) -> str:
    return f"grid_{lat:.2f}_{lon:.2f}"


def build_grid(
    bounds: dict = NEPAL_BOUNDS,
    resolution_deg: float = GRID_RESOLUTION_DEG,
) -> pd.DataFrame:
    lats = np.arange(bounds["min_lat"], bounds["max_lat"], resolution_deg)
    lons = np.arange(bounds["min_lon"], bounds["max_lon"], resolution_deg)
    lats = np.round(lats, 2)
    lons = np.round(lons, 2)

    lat_grid, lon_grid = np.meshgrid(lats, lons)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    cell_ids = [_make_cell_id(lat, lon) for lat, lon in zip(lat_flat, lon_flat)]

    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "lat": lat_flat,
            "lon": lon_flat,
        }
    )

if __name__ == "__main__":
    grid = build_grid()
    print(f"Generated {len(grid)} grid cells covering Nepal bounds.")
    print(grid.head())