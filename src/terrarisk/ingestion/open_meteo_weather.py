"""
Fetches a short live window (past 7 days plus a 7-day forecast) of
daily precipitation per grid cell from Open-Meteo's forecast API.
This module exists mainly as a Phase 1 pipeline proof - see
open_meteo_weather_archive.py for the actual multi-year historical
pull used in model training.
"""

import json
import time
import pandas as pd
from src.terrarisk.config import DATA_PROCESSED, DATA_RAW, OPEN_METEO_FORECAST_URL
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.utils.http import get_json

PAST_DAYS = 7
FORECAST_DAYS = 7
SECONDS_BETWEEN_REQUESTS = 0.2  
PROGRESS_EVERY = 50

def fetch_precipitation_for_grid(grid: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    raw_responses = []

    cells = grid.to_dict("records")
    total = len(cells)

    for i, cell in enumerate(cells, start=1):
        params = {
            "latitude": cell["lat"],
            "longitude": cell["lon"],
            "daily": "precipitation_sum,rain_sum",
            "past_days": PAST_DAYS,
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
        }

        response = get_json(OPEN_METEO_FORECAST_URL, params=params)
        raw_responses.append({"cell_id": cell["cell_id"], "response": response})

        daily = response["daily"]
        dates = daily["time"]
        precipitation = daily["precipitation_sum"]
        rain = daily["rain_sum"]

        for date, precip_mm, rain_mm in zip(dates, precipitation, rain):
            all_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "date": date,
                    "precipitation_sum_mm": precip_mm,
                    "rain_sum_mm": rain_mm,
                }
            )

        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"Fetched {i}/{total} cells.")

        time.sleep(SECONDS_BETWEEN_REQUESTS)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW / "open_meteo_weather_raw.json"
    raw_path.write_text(json.dumps(raw_responses, indent=2))
    print(f"Raw responses saved to {raw_path}")

    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    grid = build_grid()
    weather_df = fetch_precipitation_for_grid(grid)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    processed_path = DATA_PROCESSED / "grid_weather.csv"
    weather_df.to_csv(processed_path, index=False)

    print(f"Processed weather data saved to {processed_path}")
    print(weather_df.head())
    print(f"Total rows: {len(weather_df)} "
          f"({weather_df['cell_id'].nunique()} cells x "
          f"{weather_df['date'].nunique()} dates)")