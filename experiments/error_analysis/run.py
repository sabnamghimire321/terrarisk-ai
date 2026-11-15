"""
Breaks down false positives/negatives by latitude, season, and
7-day precipitation, using a fixed "flagged" definition: top 10% of
model scores, the same budget where the operational top-k experiment
found the landslide model losing to the climatological baseline.
The selection rule (top-10%) was fixed before looking at the
breakdown, not chosen afterward to make a pattern look cleaner.
"""

import json
from pathlib import Path

import pandas as pd

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS,
    FEATURE_TABLE_PATH,
    HORIZON_DAYS,
    prepare_hazard_data,
)

from experiments.operational_topk.run import get_top_k_indices

RESULTS_DIR = Path(__file__).parent / "results"

ERROR_ANALYSIS_K_PERCENT = 10

def assign_error_category(y_true: pd.Series, flagged_index: pd.Index) -> pd.Series:
    is_flagged = y_true.index.isin(flagged_index)
    is_event = y_true.astype(bool)

    category = pd.Series(index=y_true.index, dtype="object")
    category[is_flagged & is_event] = "true_positive"
    category[is_flagged & ~is_event] = "false_positive"
    category[~is_flagged & is_event] = "false_negative"
    category[~is_flagged & ~is_event] = "true_negative"
    return category


def summarize_by_group(test_with_categories: pd.DataFrame, group_col: str) -> pd.DataFrame:
    counts = pd.crosstab(test_with_categories[group_col], test_with_categories["error_category"])
    for col in ["true_positive", "false_positive", "false_negative", "true_negative"]:
        if col not in counts.columns:
            counts[col] = 0

    total_events = counts["true_positive"] + counts["false_negative"]
    counts["fn_rate_of_events"] = (
        counts["false_negative"] / total_events.replace(0, pd.NA)
    )

    return counts[["true_positive", "false_positive", "false_negative",
                    "true_negative", "fn_rate_of_events"]]


def run_error_analysis_for_hazard(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    grid_with_lat: pd.DataFrame,
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

    X_train, y_train = train[FEATURE_COLUMNS], train[label_col]
    X_val, y_val = val[FEATURE_COLUMNS], val[label_col]
    X_test, y_test = test[FEATURE_COLUMNS], test[label_col]

    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    calibrator = calibrate_on_validation(model, X_val, y_val)
    model_scores = pd.Series(
        calibrator.predict_proba(X_test)[:, 1], index=test.index
    )

    flagged_index = get_top_k_indices(model_scores, ERROR_ANALYSIS_K_PERCENT)

    test_with_categories = test.copy()
    test_with_categories["error_category"] = assign_error_category(y_test, flagged_index)

    test_with_categories = test_with_categories.merge(
        grid_with_lat[["cell_id", "lat"]], on="cell_id", how="left"
    )
    test_with_categories["season"] = test_with_categories["date"].dt.month.map(
        lambda m: (
            "winter" if m in (12, 1, 2) else
            "pre_monsoon" if m in (3, 4, 5) else
            "monsoon" if m in (6, 7, 8, 9) else
            "post_monsoon"
        )
    )
    test_with_categories["precip_7day_band"] = pd.cut(
        test_with_categories["precip_7day_sum_mm"],
        bins=[-0.01, 10, 50, 100, float("inf")],
        labels=["0-10mm", "10-50mm", "50-100mm", "100mm+"],
    )

    by_lat = summarize_by_group(test_with_categories, "lat")
    by_season = summarize_by_group(test_with_categories, "season")
    by_precip = summarize_by_group(test_with_categories, "precip_7day_band")

    print(f"\n--- Error breakdown by latitude ({hazard_name}) ---")
    print(by_lat.to_string())
    print(f"\n--- Error breakdown by season ({hazard_name}) ---")
    print(by_season.to_string())
    print(f"\n--- Error breakdown by 7-day precipitation ({hazard_name}) ---")
    print(by_precip.to_string())

    overall_counts = test_with_categories["error_category"].value_counts().to_dict()
    print(f"\nOverall: {overall_counts}")

    return {
        "overall": {k: int(v) for k, v in overall_counts.items()},
        "by_latitude": json.loads(by_lat.to_json(orient="index")),
        "by_season": json.loads(by_season.to_json(orient="index")),
        "by_precip_band": json.loads(by_precip.to_json(orient="index")),
    }


if __name__ == "__main__":
    print("Loading feature table and grid...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    full_grid = build_grid()
    grid_72 = subsample_grid_for_flood(full_grid, every_n=3)

    all_results = {}

    flood_results = run_error_analysis_for_hazard(
        df, "flood_event", "FLOOD", grid_72
    )
    if flood_results:
        all_results["flood"] = flood_results

    landslide_anchor = pd.Timestamp("2017-07-12")
    landslide_results = run_error_analysis_for_hazard(
        df, "landslide_event", "LANDSLIDE", grid_72,
        max_usable_date=landslide_anchor, min_test_days=365,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "error_analysis_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {results_path}")