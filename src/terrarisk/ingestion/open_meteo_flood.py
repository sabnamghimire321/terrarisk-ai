"""
Fetches historical daily river discharge from Open-Meteo's Global
Flood API (GloFAS-based), over a fixed FLOOD_HISTORY_YEARS window.

SCOPE REDUCTION: this does not fetch flood data for the full
578-cell Nepal grid. Two independent full-grid runs during
development showed a hard, request-count-based throttle (roughly
20-25 successful requests, then a wall, regardless of request
spacing) - much stricter than Open-Meteo's documented general
limits, which apply to lighter endpoints, not heavy 15-year
historical pulls on this one. Rather than keep tuning delays against
a wall that did not move, the grid is deliberately subsampled to 72
cells before fetching. This is also the more defensible scientific
choice, not just a rate-limit workaround: GloFAS discharge is only
meaningful near actual rivers, and most of the full grid (mountain
ridges, non-river terrain) does not have a strongly meaningful
"nearest river" signal anyway.

RESUMABILITY: each cell's raw response is written to its own file
and flushed to the output CSV in batches, and any cell already
present is skipped on a re-run. This makes the fetch safe to
interrupt and restart, which mattered in practice given the
throttling above.
"""

import json
import time
from datetime import date, timedelta

import pandas as pd

from src.terrarisk.config import (
    DATA_PROCESSED,
    DATA_RAW,
    FLOOD_FLUSH_EVERY_N_CELLS,
    FLOOD_HISTORY_YEARS,
    FLOOD_RAW_DIR_NAME,
    OPEN_METEO_FLOOD_URL,
)
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.utils.http import get_json

SECONDS_BETWEEN_REQUESTS = 3.0
PROGRESS_EVERY = 25
FLOOD_MAX_RETRIES = 6
COOLDOWN_EVERY_N_CELLS = 20
COOLDOWN_SECONDS = 90

OUTPUT_PATH = DATA_PROCESSED / "flood_discharge_history.csv"

def _date_range_for_history() -> tuple[str, str]:
    """Fixed FLOOD_HISTORY_YEARS window ending today, as explicit ISO dates."""
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * FLOOD_HISTORY_YEARS)
    return start_date.isoformat(), end_date.isoformat()

def _already_processed_cell_ids() -> set:
    """Cell IDs that already have rows in the output CSV from a prior run."""
    if not OUTPUT_PATH.exists():
        return set()
    existing = pd.read_csv(OUTPUT_PATH, usecols=["cell_id"])
    return set(existing["cell_id"].unique())

def subsample_grid_for_flood(grid: pd.DataFrame, every_n: int = 3) -> pd.DataFrame:
    """
    Takes every `every_n`-th cell in both lat and lon, cutting request
    count roughly (every_n^2)-fold. A deliberate scope reduction (see
    module docstring), not just a rate-limit workaround. A future
    version selecting cells by real river-network proximity (e.g.
    HydroSHEDS) instead of a uniform grid subsample would be more
    physically grounded, but is not implemented here.
    """
    unique_lats = sorted(grid["lat"].unique())[::every_n]
    unique_lons = sorted(grid["lon"].unique())[::every_n]
    subsampled = grid[grid["lat"].isin(unique_lats) & grid["lon"].isin(unique_lons)]
    print(f"Subsampled grid for flood: {len(subsampled)} of {len(grid)} "
          f"cells (every {every_n} in each dimension).")
    return subsampled


def fetch_flood_history_for_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches discharge history for every cell in `grid`, resuming from
    disk where possible. Each successful request's raw response is
    written to its own file before being merged into the output CSV
    in batches (FLOOD_FLUSH_EVERY_N_CELLS), and a cooldown is applied
    periodically to work around the throttling described in the
    module docstring.
    """
    start_date, end_date = _date_range_for_history()
    print(f"Fetching river discharge from {start_date} to {end_date} "
          f"({FLOOD_HISTORY_YEARS} years).")

    flood_raw_dir = DATA_RAW / FLOOD_RAW_DIR_NAME
    flood_raw_dir.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    done_cell_ids = _already_processed_cell_ids()
    if done_cell_ids:
        print(f"Resuming: {len(done_cell_ids)} cells already in the "
              "output CSV, skipping them.")

    cells = [
        c for c in grid.to_dict("records") if c["cell_id"] not in done_cell_ids
    ]
    total_remaining = len(cells)
    print(f"{total_remaining} of {len(grid)} cells remaining to fetch.")

    if total_remaining == 0:
        print("Nothing left to do — all cells already processed.")
        return pd.DataFrame(columns=["cell_id", "date", "river_discharge_m3s"])

    pending_frames = []
    all_frames_for_return = []

    for i, cell in enumerate(cells, start=1):
        raw_path = flood_raw_dir / f"{cell['cell_id']}.json"

        if raw_path.exists():
            response = json.loads(raw_path.read_text())
        else:
            params = {
                "latitude": cell["lat"],
                "longitude": cell["lon"],
                "daily": "river_discharge",
                "start_date": start_date,
                "end_date": end_date,
            }
            response = get_json(
                OPEN_METEO_FLOOD_URL, params=params, max_retries=FLOOD_MAX_RETRIES
            )
            raw_path.write_text(json.dumps(response))
            time.sleep(SECONDS_BETWEEN_REQUESTS)

            if i % COOLDOWN_EVERY_N_CELLS == 0:
                print(f"  Cooling down for {COOLDOWN_SECONDS}s after "
                      f"{COOLDOWN_EVERY_N_CELLS} requests...")
                time.sleep(COOLDOWN_SECONDS)

        daily = response["daily"]
        cell_df = pd.DataFrame(
            {
                "cell_id": cell["cell_id"],
                "date": daily["time"],
                "river_discharge_m3s": daily["river_discharge"],
            }
        )
        pending_frames.append(cell_df)

        if i % FLOOD_FLUSH_EVERY_N_CELLS == 0 or i == total_remaining:
            batch_df = pd.concat(pending_frames, ignore_index=True)
            write_header = (
                not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0
            )
            batch_df.to_csv(
                OUTPUT_PATH, mode="a", header=write_header, index=False
            )
            all_frames_for_return.append(batch_df)
            pending_frames = []

        if i % PROGRESS_EVERY == 0 or i == total_remaining:
            print(f"Processed {i}/{total_remaining} remaining cells "
                  f"({len(done_cell_ids) + i}/{len(grid)} total in file).")

    print(f"Processed discharge history saved to {OUTPUT_PATH}")
    if all_frames_for_return:
        return pd.concat(all_frames_for_return, ignore_index=True)
    return pd.DataFrame(columns=["cell_id", "date", "river_discharge_m3s"])


if __name__ == "__main__":
    full_grid = build_grid()
    grid = subsample_grid_for_flood(full_grid, every_n=3)
    discharge_df = fetch_flood_history_for_grid(grid)

    print(discharge_df.head())
    print(f"Total rows fetched this run: {len(discharge_df)}")

    if OUTPUT_PATH.exists():
        total_rows = sum(1 for _ in open(OUTPUT_PATH, encoding="utf-8")) - 1
        print(f"Total rows in output file so far: {total_rows}")