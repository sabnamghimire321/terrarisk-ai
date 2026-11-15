"""
Builds the feature table the ML models train on: one row per
(cell_id, date), with rolling rainfall features, discharge, elevation,
seasonality, and binary labels for both hazards.

Landslide events were originally assigned to their nearest cell
against the full 578-cell grid. Flood/precipitation features only
exist for the 72-cell subsample, so events are re-assigned here to
their nearest cell within that smaller grid specifically, so no event
is dropped just because its original full-grid assignment falls
outside the subsample.

Landslide events before the start of the precipitation history have
no matching weather features and are excluded - the model can only
learn from events where the weather beforehand is actually known.
"""

import numpy as np
import pandas as pd

from src.terrarisk.baseline.flood_baseline import compute_cell_thresholds
from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood

PRECIPITATION_PATH = DATA_PROCESSED / "precipitation_history.csv"
DISCHARGE_PATH = DATA_PROCESSED / "flood_discharge_history.csv"
ELEVATION_PATH = DATA_PROCESSED / "grid_elevation.csv"
LANDSLIDE_EVENTS_PATH = DATA_PROCESSED / "landslide_events.csv"

ROLLING_WINDOWS_DAYS = [1, 3, 7]
FLOOD_THRESHOLD_PERCENTILE = 90


def _nearest_cell_id(lat: float, lon: float, grid: pd.DataFrame) -> str:
    distances = np.sqrt((grid["lat"] - lat) ** 2 + (grid["lon"] - lon) ** 2)
    nearest_idx = distances.idxmin()
    return grid.loc[nearest_idx, "cell_id"]

def add_rolling_precipitation_features(
    precip_df: pd.DataFrame, windows: list = ROLLING_WINDOWS_DAYS
) -> pd.DataFrame:
    df = precip_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cell_id", "date"])

    for window in windows:
        col_name = f"precip_{window}day_sum_mm"
        df[col_name] = (
            df.groupby("cell_id")["precipitation_sum_mm"]
            .transform(lambda s: s.rolling(window, min_periods=1).sum())
        )

    return df

def add_flood_labels(
    df: pd.DataFrame, discharge_df: pd.DataFrame, percentile: int = FLOOD_THRESHOLD_PERCENTILE
) -> pd.DataFrame:
    discharge = discharge_df.copy()
    discharge["date"] = pd.to_datetime(discharge["date"])

    thresholds = compute_cell_thresholds(discharge_df, percentile)

    merged = df.merge(discharge, on=["cell_id", "date"], how="left")
    merged["flood_threshold_m3s"] = merged["cell_id"].map(thresholds)
    merged["flood_event"] = (
        merged["river_discharge_m3s"] > merged["flood_threshold_m3s"]
    ).astype(int)

    return merged

def add_landslide_labels(
    df: pd.DataFrame, landslide_events: pd.DataFrame, grid: pd.DataFrame
) -> pd.DataFrame:
    events = landslide_events.copy()
    events["event_date"] = pd.to_datetime(
    events["event_date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    events = events.dropna(subset=["event_date", "latitude", "longitude"])

    events["cell_id"] = events.apply(
        lambda row: _nearest_cell_id(row["latitude"], row["longitude"], grid),
        axis=1,
    )

    landslide_flags = (
        events[["cell_id", "event_date"]]
        .drop_duplicates()
        .rename(columns={"event_date": "date"})
    )
    landslide_flags["landslide_event"] = 1

    merged = df.merge(landslide_flags, on=["cell_id", "date"], how="left")
    merged["landslide_event"] = merged["landslide_event"].fillna(0).astype(int)

    return merged

def build_feature_table() -> pd.DataFrame:
    print("Loading source data...")
    precip_df = pd.read_csv(PRECIPITATION_PATH)
    discharge_df = pd.read_csv(DISCHARGE_PATH)
    elevation_df = pd.read_csv(ELEVATION_PATH)
    landslide_events = pd.read_csv(LANDSLIDE_EVENTS_PATH)

    full_grid = build_grid()
    grid_72 = subsample_grid_for_flood(full_grid, every_n=3)
    valid_cell_ids = set(grid_72["cell_id"])

    precip_df = precip_df[precip_df["cell_id"].isin(valid_cell_ids)]
    discharge_df = discharge_df[discharge_df["cell_id"].isin(valid_cell_ids)]
    elevation_df = elevation_df[elevation_df["cell_id"].isin(valid_cell_ids)]

    print("Computing rolling precipitation features...")
    df = add_rolling_precipitation_features(precip_df)

    print("Adding elevation (static per cell)...")
    df = df.merge(elevation_df[["cell_id", "elevation_m"]], on="cell_id", how="left")

    print("Adding day-of-year seasonality feature...")
    df["day_of_year"] = df["date"].dt.dayofyear

    print("Adding flood labels (per-cell threshold exceedance)...")
    df = add_flood_labels(df, discharge_df)

    print("Re-assigning landslide events to the 72-cell grid and adding labels...")
    df = add_landslide_labels(df, landslide_events, grid_72)

    return df

if __name__ == "__main__":
    feature_table = build_feature_table()

    output_path = DATA_PROCESSED / "feature_table.csv"
    feature_table.to_csv(output_path, index=False)

    print(f"\nFeature table saved to {output_path}")
    print(f"Shape: {feature_table.shape}")
    print(f"Columns: {list(feature_table.columns)}")
    print(f"\nFlood events: {feature_table['flood_event'].sum()} "
          f"of {len(feature_table)} rows ({feature_table['flood_event'].mean():.4%})")
    print(f"Landslide events: {feature_table['landslide_event'].sum()} "
          f"of {len(feature_table)} rows")
    print(feature_table.head())