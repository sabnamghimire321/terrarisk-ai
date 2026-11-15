"""
Fetches historical daily precipitation from Open-Meteo's Archive API,
spanning the same window as the flood baseline's discharge history.
Unlike the live forecast pull in open_meteo_weather.py, this covers
years of history, which is what the model actually needs for training
features that line up with historical event labels.

Uses the same resumable, memory-bounded, cooldown-aware pattern as
open_meteo_flood.py - this endpoint was found to throttle in a
similar way during development.
"""

import json
import time
from datetime import date, timedelta
import pandas as pd

from src.terrarisk.config import (
    DATA_PROCESSED,
    DATA_RAW,
    FLOOD_HISTORY_YEARS,
    OPEN_METEO_ARCHIVE_URL,
)
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.utils.http import get_json

SECONDS_BETWEEN_REQUESTS = 3.0
PROGRESS_EVERY = 10
ARCHIVE_MAX_RETRIES = 6
COOLDOWN_EVERY_N_CELLS = 20
COOLDOWN_SECONDS = 90
FLUSH_EVERY_N_CELLS = 10

RAW_DIR_NAME = "weather_archive_raw"
OUTPUT_PATH = DATA_PROCESSED / "precipitation_history.csv"

def _date_range_for_history() -> tuple[str, str]:
    """Same window length as the flood baseline, for feature alignment."""
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * FLOOD_HISTORY_YEARS)
    return start_date.isoformat(), end_date.isoformat()

def _already_processed_cell_ids() -> set:
    if not OUTPUT_PATH.exists():
        return set()
    existing = pd.read_csv(OUTPUT_PATH, usecols=["cell_id"])
    return set(existing["cell_id"].unique())

def fetch_precipitation_history(grid: pd.DataFrame) -> pd.DataFrame:
    start_date, end_date = _date_range_for_history()
    print(f"Fetching precipitation history from {start_date} to {end_date}.")

    raw_dir = DATA_RAW / RAW_DIR_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)
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
        return pd.DataFrame(columns=["cell_id", "date", "precipitation_sum_mm", "rain_sum_mm"])

    pending_frames = []
    all_frames_for_return = []

    for i, cell in enumerate(cells, start=1):
        raw_path = raw_dir / f"{cell['cell_id']}.json"

        if raw_path.exists():
            response = json.loads(raw_path.read_text())
        else:
            params = {
                "latitude": cell["lat"],
                "longitude": cell["lon"],
                "daily": "precipitation_sum,rain_sum",
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "auto",
            }
            response = get_json(
                OPEN_METEO_ARCHIVE_URL, params=params, max_retries=ARCHIVE_MAX_RETRIES
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
                "precipitation_sum_mm": daily["precipitation_sum"],
                "rain_sum_mm": daily["rain_sum"],
            }
        )
        pending_frames.append(cell_df)

        if i % FLUSH_EVERY_N_CELLS == 0 or i == total_remaining:
            batch_df = pd.concat(pending_frames, ignore_index=True)
            write_header = (
                not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0
            )
            batch_df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
            all_frames_for_return.append(batch_df)
            pending_frames = []

        if i % PROGRESS_EVERY == 0 or i == total_remaining:
            print(f"Processed {i}/{total_remaining} remaining cells "
                  f"({len(done_cell_ids) + i}/{len(grid)} total in file).")

    print(f"Processed precipitation history saved to {OUTPUT_PATH}")
    if all_frames_for_return:
        return pd.concat(all_frames_for_return, ignore_index=True)
    return pd.DataFrame(columns=["cell_id", "date", "precipitation_sum_mm", "rain_sum_mm"])

if __name__ == "__main__":
    full_grid = build_grid()
    grid = subsample_grid_for_flood(full_grid, every_n=3)
    precip_df = fetch_precipitation_history(grid)

    print(precip_df.head())
    print(f"Total rows fetched this run: {len(precip_df)}")

    if OUTPUT_PATH.exists():
        total_rows = sum(1 for _ in open(OUTPUT_PATH, encoding="utf-8")) - 1
        print(f"Total rows in output file so far: {total_rows}")