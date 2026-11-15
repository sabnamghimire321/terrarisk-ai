"""
Tests whether extending the precipitation feature set - 14/30-day
rolling windows plus a climatological anomaly feature - improves
either hazard. The anomaly feature is a genuine leakage risk if
"expected rainfall" is computed from the full history: climatology
here is fit on training data only and applied unchanged to
validation/test, the same discipline used for feature scaling
elsewhere in this project. Builds its own extended feature table
rather than modifying the canonical feature_table.csv that other
experiments depend on.

For each hazard it also runs the paired cluster bootstrap of each
feature set against the train-only SEASONAL baseline (the fair
opponent), using experiments.uncertainty.run.paired_comparison_block.
The primary comparison, fixed before running, is the extended feature
set against the seasonal baseline on PR-AUC; the other rows are reported
for context.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.features.build_features import (
    DISCHARGE_PATH,
    ELEVATION_PATH,
    LANDSLIDE_EVENTS_PATH,
    PRECIPITATION_PATH,
    add_flood_labels,
    add_landslide_labels,
    add_rolling_precipitation_features,
)
from src.terrarisk.models.paired_bootstrap import DEFAULT_N_BOOTSTRAP, DEFAULT_SEED
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS as ORIGINAL_FEATURE_COLUMNS,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
    prepare_hazard_data,
)
from experiments.calibration.run import evaluate_variant
from experiments.uncertainty.run import paired_comparison_block

PAIRED_METRIC_KEYS = ("pr_auc", "roc_auc", "brier")

RESULTS_DIR = Path(__file__).parent / "results"

EXTENDED_WINDOWS = [1, 3, 7, 14, 30]
ANOMALY_BASE_COLUMN = "precip_7day_sum_mm"
NEW_FEATURE_COLUMNS = ["precip_14day_sum_mm", "precip_30day_sum_mm", "precip_7day_anomaly_mm"]
WINDOWS_ONLY_COLUMNS = ORIGINAL_FEATURE_COLUMNS + ["precip_14day_sum_mm", "precip_30day_sum_mm"]
EXTENDED_FEATURE_COLUMNS = ORIGINAL_FEATURE_COLUMNS + NEW_FEATURE_COLUMNS

def build_extended_feature_table(grid_72: pd.DataFrame) -> pd.DataFrame:
    precip_df = pd.read_csv(PRECIPITATION_PATH)
    discharge_df = pd.read_csv(DISCHARGE_PATH)
    elevation_df = pd.read_csv(ELEVATION_PATH)
    landslide_events = pd.read_csv(LANDSLIDE_EVENTS_PATH)

    valid_cell_ids = set(grid_72["cell_id"])
    precip_df = precip_df[precip_df["cell_id"].isin(valid_cell_ids)]
    discharge_df = discharge_df[discharge_df["cell_id"].isin(valid_cell_ids)]
    elevation_df = elevation_df[elevation_df["cell_id"].isin(valid_cell_ids)]

    df = add_rolling_precipitation_features(precip_df, windows=EXTENDED_WINDOWS)
    df = df.merge(elevation_df[["cell_id", "elevation_m"]], on="cell_id", how="left")
    df["day_of_year"] = df["date"].dt.dayofyear
    df = add_flood_labels(df, discharge_df)
    df = add_landslide_labels(df, landslide_events, grid_72)

    return df

def compute_train_climatology(train: pd.DataFrame, value_col: str = ANOMALY_BASE_COLUMN) -> pd.Series:
    train = train.copy()
    train["month"] = train["date"].dt.month
    return train.groupby(["cell_id", "month"])[value_col].mean()

def apply_precip_anomaly(
    df: pd.DataFrame, climatology: pd.Series, value_col: str = ANOMALY_BASE_COLUMN
) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    fallback_mean = climatology.mean()

    lookup_values = [
        climatology.get((row.cell_id, row.month), fallback_mean)
        for row in df.itertuples()
    ]
    df["precip_7day_anomaly_mm"] = df[value_col] - lookup_values
    return df.drop(columns=["month"])

def score_feature_set(train, val, test, label_col, feature_columns):
    """Calibrated test-set probabilities from the shared recipe."""
    X_train, y_train = train[feature_columns], train[label_col]
    X_val, y_val = val[feature_columns], val[label_col]

    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    calibrator = calibrate_on_validation(model, X_val, y_val)
    return calibrator.predict_proba(test[feature_columns])[:, 1]


def train_and_evaluate_feature_set(train, val, test, label_col, feature_columns) -> dict:
    predictions = score_feature_set(train, val, test, label_col, feature_columns)
    return evaluate_variant(test[label_col], predictions)

def run_for_hazard(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    max_usable_date=None,
    min_test_days: int = 365,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"HAZARD: {hazard_name}")
    print(f"{'=' * 60}")

    prepared = prepare_hazard_data(
        df, event_col, hazard_name, max_usable_date, min_test_days, verbose=True
    )
    if prepared is None:
        return None
    train, val, test, label_col = prepared

    climatology = compute_train_climatology(train)
    train = apply_precip_anomaly(train, climatology)
    val = apply_precip_anomaly(val, climatology)
    test = apply_precip_anomaly(test, climatology)

    feature_sets = {
        "original": ORIGINAL_FEATURE_COLUMNS,
        "windows_only": WINDOWS_ONLY_COLUMNS,
        "extended": EXTENDED_FEATURE_COLUMNS,
    }
    scores = {
        name: score_feature_set(train, val, test, label_col, columns)
        for name, columns in feature_sets.items()
    }
    original_result = evaluate_variant(test[label_col], scores["original"])
    windows_only_result = evaluate_variant(test[label_col], scores["windows_only"])
    extended_result = evaluate_variant(test[label_col], scores["extended"])

    print(f"\n--- Original vs. Windows-only (+14d/30d) vs. Extended (+anomaly too) ({hazard_name}) ---")
    print(f"{'Feature set':<16}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>9}{'ECE':>9}")
    print(f"{'Original':<16}{original_result['pr_auc']:>9.4f}{original_result['roc_auc']:>9.4f}"
          f"{original_result['brier']:>9.4f}{original_result['ece']:>9.4f}")
    print(f"{'Windows-only':<16}{windows_only_result['pr_auc']:>9.4f}{windows_only_result['roc_auc']:>9.4f}"
          f"{windows_only_result['brier']:>9.4f}{windows_only_result['ece']:>9.4f}")
    print(f"{'+Anomaly':<16}{extended_result['pr_auc']:>9.4f}{extended_result['roc_auc']:>9.4f}"
          f"{extended_result['brier']:>9.4f}{extended_result['ece']:>9.4f}")

    improved = extended_result["pr_auc"] > original_result["pr_auc"]
    print(f"\nExtended feature set improves PR-AUC: {improved}")

    y = test[label_col].to_numpy()
    seasonal_scores = test[TRAIN_ONLY_SEASONAL_BASELINE_COL].to_numpy()
    paired_vs_seasonal = {
        name: paired_comparison_block(
            y, scores[name], seasonal_scores,
            test["cell_id"].to_numpy(), test["date"],
            n_bootstrap=n_bootstrap, seed=seed, metric_keys=PAIRED_METRIC_KEYS,
        )
        for name in feature_sets
    }

    print(f"\n--- Paired bootstrap, feature set minus seasonal baseline "
          f"({hazard_name}, {n_bootstrap} replicates, seed {seed}) ---")
    print("Brier: negative = model better; other metrics: positive = model better.")
    for name, by_clustering in paired_vs_seasonal.items():
        for clustering_name, per_metric in by_clustering.items():
            for metric_key, r in per_metric.items():
                ci = r["difference_ci"]
                ci_text = "undefined" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                print(f"{name:<14}{clustering_name:<16}{metric_key:<9}"
                      f"diff {r['point_difference']:+.4f}  95% CI {ci_text}  "
                      f"valid {r['n_valid']}/{r['n_bootstrap']}")

    return {
        "original": original_result,
        "windows_only": windows_only_result,
        "extended": extended_result,
        "pr_auc_improved": improved,
        "test_prevalence": float(y.mean()),
        "seasonal_baseline_pr_auc": float(
            evaluate_variant(test[label_col], seasonal_scores)["pr_auc"]
        ),
        "paired_vs_seasonal_baseline": paired_vs_seasonal,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("Building extended feature table (14/30-day windows)...")
    full_grid = build_grid()
    grid_72 = subsample_grid_for_flood(full_grid, every_n=3)
    df = build_extended_feature_table(grid_72)

    all_results = {}

    flood_results = run_for_hazard(
        df, "flood_event", "FLOOD", n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    if flood_results:
        all_results["flood"] = flood_results

    landslide_anchor = pd.Timestamp("2017-07-12")
    landslide_results = run_for_hazard(
        df, "landslide_event", "LANDSLIDE",
        max_usable_date=landslide_anchor, min_test_days=365,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "richer_precip_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    print(f"\nNote: LANDSLIDE numbers here should be read against "
          f"Step 6's uncertainty study — 132 test positives means "
          f"point-estimate swings of this size are encouraging but "
          f"not yet statistically confirmed.")