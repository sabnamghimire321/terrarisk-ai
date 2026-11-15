"""
Feature-group ablation study. Confirms the actual feature set in
train_model.py before designing combinations (weather: 3 precip
columns; terrain: elevation; seasonal: day_of_year) rather than
assuming a feature schema. Seven combinations: all features, each
group alone, and each leave-one-out. Uses the imbalance-corrected
and Platt-calibrated recipe throughout, since the calibration study
found that choice matters far more than the weighting decision.
"""

import json
from pathlib import Path

import pandas as pd

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.models.train_model import FEATURE_TABLE_PATH, prepare_hazard_data

from experiments.calibration.run import evaluate_variant

RESULTS_DIR = Path(__file__).parent / "results"

FEATURE_GROUPS = {
    "weather": ["precip_1day_sum_mm", "precip_3day_sum_mm", "precip_7day_sum_mm"],
    "terrain": ["elevation_m"],
    "seasonal": ["day_of_year"],
}

ALL_GROUP_NAMES = list(FEATURE_GROUPS.keys())

def build_feature_combinations() -> dict:
    combos = {}

    combos["all_features"] = [c for group in FEATURE_GROUPS.values() for c in group]

    for group_name, columns in FEATURE_GROUPS.items():
        combos[f"{group_name}_only"] = list(columns)

    for excluded_group in ALL_GROUP_NAMES:
        included = [
            c for name, group in FEATURE_GROUPS.items()
            if name != excluded_group
            for c in group
        ]
        combos[f"without_{excluded_group}"] = included

    return combos

def train_and_evaluate_combo(train, val, test, label_col, feature_columns) -> dict:
    X_train, y_train = train[feature_columns], train[label_col]
    X_val, y_val = val[feature_columns], val[label_col]
    X_test, y_test = test[feature_columns], test[label_col]

    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    calibrator = calibrate_on_validation(model, X_val, y_val)
    predictions = calibrator.predict_proba(X_test)[:, 1]

    return evaluate_variant(y_test, predictions)

def run_ablation_for_hazard(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    max_usable_date=None,
    min_test_days: int = 365,
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

    combos = build_feature_combinations()
    results = {}

    for combo_name, feature_columns in combos.items():
        results[combo_name] = train_and_evaluate_combo(
            train, val, test, label_col, feature_columns
        )
        results[combo_name]["features_used"] = feature_columns

    print(f"\n--- Ablation results ({hazard_name}) ---")
    print(f"{'Combination':<20}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>9}{'ECE':>9}")
    for combo_name, metrics in results.items():
        print(f"{combo_name:<20}{metrics['pr_auc']:>9.4f}{metrics['roc_auc']:>9.4f}"
              f"{metrics['brier']:>9.4f}{metrics['ece']:>9.4f}")

    return results


if __name__ == "__main__":
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    all_results = {}

    flood_results = run_ablation_for_hazard(df, "flood_event", "FLOOD")
    if flood_results:
        all_results["flood"] = flood_results

    landslide_anchor = pd.Timestamp("2017-07-12")
    landslide_results = run_ablation_for_hazard(
        df, "landslide_event", "LANDSLIDE",
        max_usable_date=landslide_anchor, min_test_days=365,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "ablation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")