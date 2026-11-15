"""Runs the core ingestion pipeline (grid, elevation, weather) end to end and checks that the outputs join cleanly on cell_id."""

import pandas as pd

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_elevation import fetch_elevation_for_grid
from src.terrarisk.ingestion.open_meteo_weather import fetch_precipitation_for_grid

def run_pipeline() -> None:
    print("=== Step 1: build grid ===")
    grid = build_grid()
    print(f"Grid built: {len(grid)} cells.\n")

    print("=== Step 2: fetch elevation ===")
    elevation_df = fetch_elevation_for_grid(grid)
    print(f"Elevation fetched: {len(elevation_df)} rows.\n")

    print("=== Step 3: fetch weather ===")
    weather_df = fetch_precipitation_for_grid(grid)
    print(f"Weather fetched: {len(weather_df)} rows.\n")

    print("=== Step 4: verify join integrity ===")
    weather_cell_ids = set(weather_df["cell_id"])
    elevation_cell_ids = set(elevation_df["cell_id"])

    missing_elevation = weather_cell_ids - elevation_cell_ids
    if missing_elevation:
        raise AssertionError(
            f"{len(missing_elevation)} cell_ids have weather data but "
            f"no matching elevation data: {sorted(missing_elevation)[:5]}"
            " (showing up to 5). This means the two datasets were built "
            "from different grids — re-run both against the same grid."
        )
    print("All weather cell_ids have matching elevation data. Join is clean.\n")

    print("=== Step 5: save merged dataset ===")
    merged = weather_df.merge(
        elevation_df[["cell_id", "elevation_m"]], on="cell_id", how="left"
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    merged_path = DATA_PROCESSED / "grid_weather_with_elevation.csv"
    merged.to_csv(merged_path, index=False)

    print(f"Merged dataset saved to {merged_path}")
    print(merged.head())
    print(f"Total merged rows: {len(merged)}")

if __name__ == "__main__":
    run_pipeline()