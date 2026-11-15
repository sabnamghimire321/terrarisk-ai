"""
Climatological (frequency-based) baseline for landslide probability.
Deliberately the simplest possible baseline - no feature engineering,
just historical event frequency - so it can serve as a floor the ML
model has to beat.

With only 623 events across 578 cells, most cells have 0-1 historical
events, far too sparse to estimate a per-cell rate directly. Each
cell's rate is instead estimated from its N nearest neighbor cells
combined (spatial pooling), which assumes locally uniform risk within
a pooled cluster - a real approximation, not a claim that risk is
actually uniform across a region.

The observation window starts 2007-01-01, NASA's own stated start
date for the Global Landslide Catalog's compilation, not the date of
the first recorded event in this project's study area - using the
true monitoring-start date avoids understating quiet periods before
the first logged local event.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.grid.build_grid import build_grid

CATALOG_START = datetime(2007, 1, 1)
POOL_SIZE = 15
HORIZONS_DAYS = [1, 3, 7]

LANDSLIDE_EVENTS_PATH = DATA_PROCESSED / "landslide_events.csv"


def _build_distance_matrix(grid: pd.DataFrame) -> np.ndarray:
    lat = grid["lat"].to_numpy()
    lon = grid["lon"].to_numpy()
    lat_diff = lat[:, None] - lat[None, :]
    lon_diff = lon[:, None] - lon[None, :]
    return np.sqrt(lat_diff**2 + lon_diff**2)

def _nearest_n_pools(grid: pd.DataFrame, pool_size: int) -> dict:
    distances = _build_distance_matrix(grid)
    cell_ids = grid["cell_id"].to_numpy()

    pools = {}
    for i, cell_id in enumerate(cell_ids):
        nearest_indices = np.argsort(distances[i])[:pool_size]
        pools[cell_id] = cell_ids[nearest_indices].tolist()

    return pools


def compute_baseline(
    grid: pd.DataFrame,
    horizon_days: int,
    events: pd.DataFrame,
    pool_size: int = POOL_SIZE,
) -> pd.DataFrame:
    event_counts_by_cell = events["cell_id"].value_counts().to_dict()

    observation_days = (datetime.now() - CATALOG_START).days
    total_windows = observation_days / horizon_days

    pools = _nearest_n_pools(grid, pool_size)

    rows = []
    for cell_id, pooled_cell_ids in pools.items():
        pool_event_count = sum(
            event_counts_by_cell.get(cid, 0) for cid in pooled_cell_ids
        )
        pool_size_actual = len(pooled_cell_ids)

        probability = pool_event_count / (pool_size_actual * total_windows)

        rows.append(
            {
                "cell_id": cell_id,
                "horizon_days": horizon_days,
                "pool_event_count": pool_event_count,
                "pool_size": pool_size_actual,
                "baseline_probability": probability,
            }
        )

    return pd.DataFrame(rows)

if __name__ == "__main__":
    grid = build_grid()
    events = pd.read_csv(LANDSLIDE_EVENTS_PATH)

    all_horizons = []
    for horizon in HORIZONS_DAYS:
        print(f"Computing baseline for {horizon}-day horizon...")
        baseline_df = compute_baseline(grid, horizon, events)
        all_horizons.append(baseline_df)
        print(
            f"  Mean probability: {baseline_df['baseline_probability'].mean():.6f}, "
            f"Max: {baseline_df['baseline_probability'].max():.6f}"
        )

    combined = pd.concat(all_horizons, ignore_index=True)

    output_path = DATA_PROCESSED / "landslide_baseline.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nBaseline saved to {output_path}")
    print(combined.head(10))
    print(f"Total rows: {len(combined)} (should be {len(grid)} cells x {len(HORIZONS_DAYS)} horizons)")