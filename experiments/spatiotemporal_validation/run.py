"""
Tests the same held-out spatial block as spatial_validation, but
restricts the test set to the block's most recent year only - unseen
geography AND unseen time at once. This is the cleanest test of
generalization to new places in this project: in a purely spatial holdout
the training cells see the same calendar dates as the held-out ones, so one
storm can reach both sides.

Split (src/terrarisk/models/spatial_holdout.spatiotemporal_holdout_split):
  TEST       held-out cells, the final 365 days
  VALIDATION the other cells, the 365 days before the test period (purged),
             so validation shares no date with test
  TRAIN      the other cells, earlier than validation (purged)
The earlier version used the SAME final 60 days for validation (other cells)
and test (held-out cells), so same-day events in neighbouring cells reached
early stopping and calibration; that is fixed here.

Baselines come from training cells only: pooled and nearest-neighbour cell
rates, plain and by calendar month. The nearest-neighbour seasonal baseline
is the stricter opponent (it carries geography).

Flood only: landslide's usable events within that spatial block, checked
beforehand, span only 23
raw event-days across 7 years, with some years at zero - not enough for a
trustworthy temporal split within an already-small spatial subset.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.spatial_validation.run import HOLDOUT_LATS, evaluate_holdout_split
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.models.paired_bootstrap import DEFAULT_N_BOOTSTRAP, DEFAULT_SEED
from src.terrarisk.models.spatial_holdout import cells_in_bands, spatiotemporal_holdout_split
from src.terrarisk.models.train_model import FEATURE_TABLE_PATH

RESULTS_DIR = Path(__file__).parent / "results"

SPATIOTEMPORAL_TEST_DAYS = 365


def run_spatiotemporal_for_flood(
    df: pd.DataFrame,
    holdout_cell_ids,
    grid: pd.DataFrame = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
    test_days: int = SPATIOTEMPORAL_TEST_DAYS,
) -> dict | None:
    split = spatiotemporal_holdout_split(
        df, "flood_event", holdout_cell_ids, test_days=test_days, grid=grid
    )
    return evaluate_holdout_split(
        split, "FLOOD", n_bootstrap=n_bootstrap, seed=seed, verbose=verbose,
        design_label="spatiotemporal: unseen geography AND unseen time",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("Loading feature table and grid...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    grid_72 = subsample_grid_for_flood(build_grid(), every_n=3)
    df = df[df["cell_id"].isin(set(grid_72["cell_id"]))]
    holdout_cells = cells_in_bands(grid_72, "lat", HOLDOUT_LATS)

    result = run_spatiotemporal_for_flood(
        df, holdout_cells, grid=grid_72,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "spatiotemporal_results.json"
    results_path.write_text(json.dumps({"flood": result}, indent=2))
    print(f"\nResults saved to {results_path}")
    print(f"\nNote: LANDSLIDE spatiotemporal validation was investigated and "
          f"found infeasible — only 23 raw event-days across 7 years within "
          f"the held-out spatial block (multiple years at 0 events).")
