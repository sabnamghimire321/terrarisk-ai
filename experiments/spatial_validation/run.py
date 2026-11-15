"""
Tests whether the model generalizes to geography never seen in
training, using a block holdout by latitude band rather than random
per-cell dropout - nearby cells are spatially correlated, so random
dropout would mostly test interpolation between seen neighbors, not
generalization to new geography. The held-out block (the two
southernmost bands, the Tarai lowlands) was chosen as a physically
distinct terrain zone before looking at model performance.

The split, the baselines and the model recipe are the shared ones:
src/terrarisk/models/spatial_holdout.py (training cells only, one-year
validation window, purge gap, strict labels, train-only baselines) and
src/terrarisk/models/recipe.py. The held-out cells are compared with the
POOLED training-cell rate and the pooled-by-month rate: the old
full-history baseline for those cells used their own outcomes.

run_holdout_experiment() is reused by the longitude holdout experiment, and
evaluate_holdout_split() by the spatiotemporal one.

When a grid is supplied, a NEIGHBOUR baseline is also built from the nearest
training cells. It carries geography, so it is the stricter opponent: the
pooled baseline gives every held-out cell the same value and cannot rank
cells at all, which flatters any model that sees elevation.
Brier mixes all years of the held-out cells, while the calibrator was
fitted on one validation year, so read the Brier results with that in
mind; the ranking metrics (ROC-AUC, PR-AUC) are not affected by it.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from experiments.uncertainty.run import paired_comparison_block
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.models.paired_bootstrap import DEFAULT_N_BOOTSTRAP, DEFAULT_SEED
from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.models.spatial_holdout import (
    NEIGHBOUR_BASELINE_COL,
    NEIGHBOUR_SEASONAL_BASELINE_COL,
    cells_in_bands,
    spatial_holdout_split,
)
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS,
    FEATURE_TABLE_PATH,
    TRAIN_ONLY_BASELINE_COL,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
)

RESULTS_DIR = Path(__file__).parent / "results"
HOLDOUT_LATS = [26.30, 27.05]

MIN_EXPECTED_TEST_POSITIVES = 15
PAIRED_METRIC_KEYS = ("pr_auc", "roc_auc", "brier")


def _point_metrics(y, scores) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
    }


def evaluate_holdout_split(
    split,
    hazard_name: str,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
    design_label: str = "spatial holdout",
) -> dict | None:
    """Trains the shared recipe on `split` and compares it with every baseline attached to test."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"HAZARD: {hazard_name} ({design_label})")
        print(f"{'=' * 60}")

    if split is None:
        if verbose:
            print("The data cannot support this split here. Skipping.")
        return None
    train, val, test, label_col = split

    y = test[label_col].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        if verbose:
            print("The held-out cells have only one class. Skipping.")
        return None

    if verbose:
        print(f"Train: {len(train)} rows ({train['cell_id'].nunique()} cells)")
        print(f"Val: {len(val)} rows ({val['cell_id'].nunique()} cells)")
        print(f"Test (held-out geography): {len(test)} rows "
              f"({test['cell_id'].nunique()} cells)")
        print(f"Positive counts - train: {int(train[label_col].sum())}, "
              f"val: {int(val[label_col].sum())}, test: {int(y.sum())}")
        if y.sum() < MIN_EXPECTED_TEST_POSITIVES:
            print(f"WARNING: only {int(y.sum())} positive events in the held-out "
                  f"geography; treat the result as suggestive, not conclusive.")

    model = fit_lightgbm(
        train[FEATURE_COLUMNS], train[label_col], val[FEATURE_COLUMNS], val[label_col]
    )
    calibrator = calibrate_on_validation(model, val[FEATURE_COLUMNS], val[label_col])
    model_scores = calibrator.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    baseline_columns = {
        "pooled_cell_baseline": TRAIN_ONLY_BASELINE_COL,
        "seasonal_baseline": TRAIN_ONLY_SEASONAL_BASELINE_COL,
    }
    if NEIGHBOUR_BASELINE_COL in test.columns:
        baseline_columns["neighbour_cell_baseline"] = NEIGHBOUR_BASELINE_COL
        baseline_columns["neighbour_seasonal_baseline"] = NEIGHBOUR_SEASONAL_BASELINE_COL

    results = {
        "design": design_label,
        "test_rows": int(len(test)),
        "test_cells": int(test["cell_id"].nunique()),
        "test_positives": int(y.sum()),
        "test_prevalence": float(y.mean()),
        "train_rows": int(len(train)),
        "model": _point_metrics(y, model_scores),
        "paired_vs_baseline": {},
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }
    for name, column in baseline_columns.items():
        baseline_scores = test[column].to_numpy()
        results[name] = _point_metrics(y, baseline_scores)
        if name in ("seasonal_baseline", "neighbour_seasonal_baseline"):
            results["paired_vs_baseline"][name] = paired_comparison_block(
                y, model_scores, baseline_scores,
                test["cell_id"].to_numpy(), test["date"],
                n_bootstrap=n_bootstrap, seed=seed, metric_keys=PAIRED_METRIC_KEYS,
            )

    if verbose:
        print(f"\n--- Results ({hazard_name}, {design_label}) ---")
        print(f"Test prevalence (random-classifier PR-AUC): {results['test_prevalence']:.4f}")
        print(f"{'':<30}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>9}")
        for name in ["model"] + list(baseline_columns):
            m = results[name]
            print(f"{name:<30}{m['pr_auc']:>9.4f}{m['roc_auc']:>9.4f}{m['brier']:>9.4f}")
        for baseline_name, by_clustering in results["paired_vs_baseline"].items():
            print(f"\nPaired difference, model minus {baseline_name} "
                  f"({n_bootstrap} replicates, seed {seed}; Brier: negative = model better)")
            for clustering, per_metric in by_clustering.items():
                for metric_key, r in per_metric.items():
                    ci = r["difference_ci"]
                    ci_text = "undefined" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    print(f"{clustering:<16}{metric_key:<9}diff {r['point_difference']:+.4f}  "
                          f"95% CI {ci_text}  valid {r['n_valid']}/{r['n_bootstrap']}")
    return results


def run_holdout_experiment(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    holdout_cell_ids,
    max_usable_date=None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
    grid: pd.DataFrame = None,
) -> dict | None:
    split = spatial_holdout_split(
        df, event_col, holdout_cell_ids, max_usable_date=max_usable_date, grid=grid
    )
    return evaluate_holdout_split(
        split, hazard_name, n_bootstrap=n_bootstrap, seed=seed, verbose=verbose
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
    results_path = RESULTS_DIR / "spatial_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")
