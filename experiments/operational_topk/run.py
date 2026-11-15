"""
Answers a threshold-independent operational question: if only the
top k percent of predictions can be acted on, how many real events
does that slice actually contain? Directly motivated by the
calibration study, where every calibrated landslide variant
predicted zero positives at a fixed 0.5 threshold - top-k avoids
that failure mode by construction. get_top_k_indices() is reused by
the error-analysis experiment rather than reimplemented there.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS,
    FEATURE_TABLE_PATH,
    HORIZON_DAYS,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
    get_baseline_predictions,
    prepare_hazard_data,
)

RESULTS_DIR = Path(__file__).parent / "results"

TOP_K_PERCENTAGES = [1, 5, 10]

def get_top_k_indices(scores: pd.Series, k_percent: float) -> pd.Index:
    n_total = len(scores)
    n_selected = max(1, int(round(n_total * k_percent / 100)))
    return scores.sort_values(ascending=False).index[:n_selected]

def evaluate_top_k(y_true: pd.Series, scores: pd.Series, k_percent: float) -> dict:
    selected_indices = get_top_k_indices(scores, k_percent)
    n_selected = len(selected_indices)

    n_positive_total = int(y_true.sum())
    n_positive_captured = int(y_true.loc[selected_indices].sum())

    capture_rate = n_positive_captured / n_positive_total if n_positive_total > 0 else float("nan")
    precision = n_positive_captured / n_selected if n_selected > 0 else float("nan")
    false_alarms = n_selected - n_positive_captured

    return {
        "k_percent": k_percent,
        "n_selected": n_selected,
        "n_total": len(scores),
        "n_positive_total": n_positive_total,
        "n_positive_captured": n_positive_captured,
        "capture_rate": capture_rate,
        "precision": precision,
        "false_alarms": false_alarms,
    }


def run_topk_for_hazard(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    baseline_df: pd.DataFrame,
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

    baseline_scores = get_baseline_predictions(test, baseline_df)
    baseline_scores.index = test.index

    seasonal_scores = pd.Series(
        test[TRAIN_ONLY_SEASONAL_BASELINE_COL].to_numpy(), index=test.index
    )

    y_test_indexed = test[label_col]

    results = {"model": {}, "baseline": {}, "seasonal_baseline": {}}

    print(f"\n--- Top-k results ({hazard_name}) ---")
    print(f"{'Source':<10}{'k%':>5}{'#Selected':>11}{'#Captured':>11}"
          f"{'Capture':>10}{'Precision':>11}{'FalseAlarms':>13}")

    for k in TOP_K_PERCENTAGES:
        model_result = evaluate_top_k(y_test_indexed, model_scores, k)
        baseline_result = evaluate_top_k(y_test_indexed, baseline_scores, k)

        seasonal_result = evaluate_top_k(y_test_indexed, seasonal_scores, k)

        results["model"][f"top_{k}pct"] = model_result
        results["baseline"][f"top_{k}pct"] = baseline_result
        results["seasonal_baseline"][f"top_{k}pct"] = seasonal_result

        for source_name, r in [("model", model_result),
                               ("baseline", baseline_result),
                               ("seasonal", seasonal_result)]:
            print(f"{source_name:<10}{r['k_percent']:>4}%{r['n_selected']:>11}"
                  f"{r['n_positive_captured']:>11}{r['capture_rate']:>10.4f}"
                  f"{r['precision']:>11.4f}{r['false_alarms']:>13}")

    return results


if __name__ == "__main__":
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    flood_baseline = pd.read_csv(DATA_PROCESSED / "flood_baseline.csv")
    flood_baseline_7d = flood_baseline[flood_baseline["horizon_days"] == HORIZON_DAYS]

    landslide_baseline = pd.read_csv(DATA_PROCESSED / "landslide_baseline.csv")
    landslide_baseline_7d = landslide_baseline[
        landslide_baseline["horizon_days"] == HORIZON_DAYS
    ]

    all_results = {}

    flood_results = run_topk_for_hazard(
        df, "flood_event", "FLOOD", flood_baseline_7d
    )
    if flood_results:
        all_results["flood"] = flood_results

    landslide_anchor = pd.Timestamp("2017-07-12")
    landslide_results = run_topk_for_hazard(
        df, "landslide_event", "LANDSLIDE", landslide_baseline_7d,
        max_usable_date=landslide_anchor, min_test_days=365,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "topk_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")