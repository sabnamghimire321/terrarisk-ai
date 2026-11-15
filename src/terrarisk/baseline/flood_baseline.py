"""
Threshold-exceedance climatological baseline for flood probability.

With roughly 5,500 daily discharge observations per cell (15 years),
there is enough data to compute a per-cell threshold directly, unlike
the landslide baseline's sparse catalog. Each cell's own 90th
percentile of daily discharge is used as its threshold, since "high"
flow on a large river and a small stream are different absolute
magnitudes - comparing every cell to one global number would not be
meaningful. History is split into non-overlapping horizon-length
windows, and a window counts as an event if discharge exceeded the
threshold on any day within it.

A percentile-of-all-days threshold is simpler than proper
hydrological return-period analysis, which uses annual maxima rather
than all daily values - a documented simplification, not a claim of
hydrological rigor.
"""

import pandas as pd

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood

THRESHOLD_PERCENTILE = 90
HORIZONS_DAYS = [1, 3, 7]

FLOOD_HISTORY_PATH = DATA_PROCESSED / "flood_discharge_history.csv"

def compute_cell_thresholds(
    flood_df: pd.DataFrame, percentile: int = THRESHOLD_PERCENTILE
) -> pd.Series:
    """
    Per-cell discharge threshold: the given percentile of that cell's
    own daily discharge history. Returns a Series indexed by cell_id.
    """
    return flood_df.groupby("cell_id")["river_discharge_m3s"].quantile(
        percentile / 100
    )

def compute_flood_baseline(
    flood_df: pd.DataFrame,
    horizon_days: int,
    percentile: int = THRESHOLD_PERCENTILE,
) -> pd.DataFrame:
    """
    Compute the threshold-exceedance baseline probability for every
    cell in `flood_df`, at a given horizon.
    """
    df = flood_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cell_id", "date"])

    thresholds = compute_cell_thresholds(df, percentile)
    df["threshold_m3s"] = df["cell_id"].map(thresholds)
    df["exceeds"] = df["river_discharge_m3s"] > df["threshold_m3s"]

    df["row_within_cell"] = df.groupby("cell_id").cumcount()
    df["window_id"] = df["row_within_cell"] // horizon_days

    window_has_event = (
        df.groupby(["cell_id", "window_id"])["exceeds"].any().reset_index()
    )

    result = (
        window_has_event.groupby("cell_id")["exceeds"]
        .agg(baseline_probability="mean", window_count="count")
        .reset_index()
    )
    result["horizon_days"] = horizon_days
    result["threshold_m3s"] = result["cell_id"].map(thresholds)

    return result[
        ["cell_id", "horizon_days", "threshold_m3s", "window_count", "baseline_probability"]
    ]

if __name__ == "__main__":
    full_grid = build_grid()
    subsampled_grid = subsample_grid_for_flood(full_grid, every_n=3)
    valid_cell_ids = set(subsampled_grid["cell_id"])

    raw_flood_df = pd.read_csv(FLOOD_HISTORY_PATH)
    flood_df = raw_flood_df[raw_flood_df["cell_id"].isin(valid_cell_ids)]

    print(f"Filtered flood history from {raw_flood_df['cell_id'].nunique()} "
          f"cells (on disk) to {flood_df['cell_id'].nunique()} cells "
          f"(canonical subsampled grid).")

    all_horizons = []
    for horizon in HORIZONS_DAYS:
        print(f"Computing flood baseline for {horizon}-day horizon...")
        baseline_df = compute_flood_baseline(flood_df, horizon)
        all_horizons.append(baseline_df)
        print(
            f"  Mean probability: {baseline_df['baseline_probability'].mean():.4f}, "
            f"Max: {baseline_df['baseline_probability'].max():.4f}"
        )

    combined = pd.concat(all_horizons, ignore_index=True)

    output_path = DATA_PROCESSED / "flood_baseline.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nBaseline saved to {output_path}")
    print(combined.head(10))
    print(f"Total rows: {len(combined)} "
          f"(should be {len(valid_cell_ids)} cells x {len(HORIZONS_DAYS)} horizons)")