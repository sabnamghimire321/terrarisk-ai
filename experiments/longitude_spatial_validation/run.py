"""
Same block-holdout logic as spatial_validation, applied to longitude
instead of latitude, to check whether the model's spatial
generalization holds on a second, independent geographic axis. The
holdout block (the two easternmost longitude bands) was chosen as a
geographic edge before looking at model performance, not because it
looked unusual in the distribution inspection.

Everything except the choice of held-out cells is shared with
experiments/spatial_validation/run.py (run_holdout_experiment): the
split, the training-cells-only baselines and the model recipe.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.spatial_validation.run import run_holdout_experiment
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.models.paired_bootstrap import DEFAULT_N_BOOTSTRAP, DEFAULT_SEED
from src.terrarisk.models.spatial_holdout import cells_in_bands
from src.terrarisk.models.train_model import FEATURE_TABLE_PATH

RESULTS_DIR = Path(__file__).parent / "results"

HOLDOUT_LONS = [87.50, 88.25]


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
    holdout_cells = cells_in_bands(grid_72, "lon", HOLDOUT_LONS)

    all_results = {}

    flood_result = run_holdout_experiment(
        df, "flood_event", "FLOOD", holdout_cells,
        n_bootstrap=args.n_bootstrap, seed=args.seed, grid=grid_72,
    )
    if flood_result:
        all_results["flood"] = flood_result

    landslide_result = run_holdout_experiment(
        df, "landslide_event", "LANDSLIDE", holdout_cells,
        max_usable_date=pd.Timestamp("2017-07-12"),
        n_bootstrap=args.n_bootstrap, seed=args.seed, grid=grid_72,
    )
    if landslide_result:
        all_results["landslide"] = landslide_result

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "longitude_spatial_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")
