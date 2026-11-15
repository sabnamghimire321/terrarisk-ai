"""
Computes exact TreeSHAP values via LightGBM's native pred_contrib,
not the external shap library. Importing shap on the development
machine triggered a security policy that blocked a transitive scipy
dependency it pulls in but that no other part of this project
needed - pred_contrib produces mathematically identical values
(verified by the additivity test in this experiment's test file)
with no extra dependency. Local explanations are selected using
error_analysis's own true/false positive/negative classification,
targeting the failure modes that experiment already documented
rather than arbitrary examples.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.models.recipe import fit_lightgbm
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS,
    FEATURE_TABLE_PATH,
    HORIZON_DAYS,
    prepare_hazard_data,
)
from experiments.operational_topk.run import get_top_k_indices
from experiments.error_analysis.run import assign_error_category

RESULTS_DIR = Path(__file__).parent / "results"
ERROR_ANALYSIS_K_PERCENT = 10


def get_shap_values(model, X: pd.DataFrame):

    n_features = X.shape[1]
    contributions = model.predict(X, pred_contrib=True)

    assert contributions.shape[1] == n_features + 1, (
        f"Unexpected pred_contrib shape {contributions.shape} — expected "
        f"(n_samples, {n_features + 1}). LightGBM's pred_contrib output "
        f"shape can vary by version; adjust get_shap_values if this fires."
    )

    shap_values = contributions[:, :n_features]
    base_value = float(contributions[:, n_features].mean())
    return shap_values, base_value

def global_shap_summary(shap_values: np.ndarray, feature_columns: list) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    summary = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    return summary

def explain_single_row(
    row_position: int, X: pd.DataFrame, shap_values: np.ndarray,
    base_value: float, feature_columns: list,
) -> dict:

    contributions = []
    for i, feature in enumerate(feature_columns):
        contributions.append({
            "feature": feature,
            "value": float(X.iloc[row_position][feature]),
            "shap_contribution": float(shap_values[row_position, i]),
        })
    contributions.sort(key=lambda c: abs(c["shap_contribution"]), reverse=True)

    return {
        "base_value": base_value,
        "raw_model_output": base_value + float(shap_values[row_position].sum()),
        "contributions": contributions,
    }


def print_explanation(label: str, explanation: dict) -> None:
    print(f"\n  {label}:")
    print(f"    Base value (average model output): {explanation['base_value']:.4f}")
    print(f"    Raw model output for this row: {explanation['raw_model_output']:.4f}")
    for c in explanation["contributions"]:
        sign = "+" if c["shap_contribution"] >= 0 else ""
        print(f"      {c['feature']:<22} value={c['value']:>10.2f}  "
              f"contribution={sign}{c['shap_contribution']:.4f}")


def run_shap_for_hazard(
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
    X_test = test[FEATURE_COLUMNS]

    model = fit_lightgbm(X_train, y_train, X_val, y_val)

    print("Computing SHAP values (LightGBM native pred_contrib)...")
    shap_values, base_value = get_shap_values(model, X_test)

    global_summary = global_shap_summary(shap_values, FEATURE_COLUMNS)
    native_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "native_importance": model.feature_importances_,
    }).sort_values("native_importance", ascending=False)

    print(f"\n--- Global SHAP importance ({hazard_name}) ---")
    print(global_summary.to_string(index=False))
    print(f"\n--- Native LightGBM feature importance (for comparison) ---")
    print(native_importance.to_string(index=False))

    shap_ranking = list(global_summary["feature"])
    native_ranking = list(native_importance["feature"])
    rankings_match = shap_ranking == native_ranking
    print(f"\nSHAP and native importance rankings identical: {rankings_match}")

    raw_scores = model.predict_proba(X_test)[:, 1]

    test_with_meta = test.copy().reset_index(drop=True)
    test_with_meta["raw_score"] = raw_scores
    flagged = get_top_k_indices(pd.Series(raw_scores, index=test_with_meta.index), ERROR_ANALYSIS_K_PERCENT)
    test_with_meta["error_category"] = assign_error_category(
        test_with_meta[label_col], flagged
    )
    test_with_meta = test_with_meta.merge(
        grid_with_lat[["cell_id", "lat"]], on="cell_id", how="left"
    )
    test_with_meta["season"] = test_with_meta["date"].dt.month.map(
        lambda m: (
            "winter" if m in (12, 1, 2) else
            "pre_monsoon" if m in (3, 4, 5) else
            "monsoon" if m in (6, 7, 8, 9) else
            "post_monsoon"
        )
    )

    print(f"\n--- Local explanations ({hazard_name}) ---")

    local_explanations = {}

    tp_rows = test_with_meta[test_with_meta["error_category"] == "true_positive"]
    if len(tp_rows) > 0:
        best_tp_position = tp_rows["raw_score"].idxmax()
        explanation = explain_single_row(best_tp_position, X_test, shap_values, base_value, FEATURE_COLUMNS)
        print_explanation(f"Best true positive (row {best_tp_position}, "
                           f"score={raw_scores[best_tp_position]:.4f})", explanation)
        local_explanations["best_true_positive"] = explanation

    if hazard_name == "LANDSLIDE":
        fn_rows = test_with_meta[
            (test_with_meta["error_category"] == "false_negative")
            & (test_with_meta["season"] == "post_monsoon")
        ]
        label = "Representative false negative (post-monsoon, known complete-miss failure mode from error analysis)"
    else:
        fn_rows = test_with_meta[
            (test_with_meta["error_category"] == "false_negative")
            & (test_with_meta["precip_7day_sum_mm"] < 50)
        ]
        label = "Representative false negative (low 7-day precipitation, known weak-performance band from error analysis)"

    if len(fn_rows) > 0:
        fn_position = fn_rows.index[0]
        explanation = explain_single_row(fn_position, X_test, shap_values, base_value, FEATURE_COLUMNS)
        print_explanation(label, explanation)
        local_explanations["representative_false_negative"] = explanation
    else:
        print(f"\n  No matching false negative found for the targeted failure mode "
              f"in this test set.")

    fp_rows = test_with_meta[test_with_meta["error_category"] == "false_positive"]
    if len(fp_rows) > 0:
        worst_fp_position = fp_rows["raw_score"].idxmax()
        explanation = explain_single_row(worst_fp_position, X_test, shap_values, base_value, FEATURE_COLUMNS)
        print_explanation(f"Highest-confidence false positive (row {worst_fp_position}, "
                           f"score={raw_scores[worst_fp_position]:.4f})", explanation)
        local_explanations["highest_confidence_false_positive"] = explanation

    return {
        "global_shap_ranking": shap_ranking,
        "native_importance_ranking": native_ranking,
        "rankings_match": rankings_match,
        "local_explanations": local_explanations,
    }


if __name__ == "__main__":
    print("Loading feature table and grid...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    full_grid = build_grid()
    grid_72 = subsample_grid_for_flood(full_grid, every_n=3)

    all_results = {}

    flood_results = run_shap_for_hazard(df, "flood_event", "FLOOD", grid_72)
    if flood_results:
        all_results["flood"] = flood_results

    landslide_anchor = pd.Timestamp("2017-07-12")
    landslide_results = run_shap_for_hazard(
        df, "landslide_event", "LANDSLIDE", grid_72,
        max_usable_date=landslide_anchor, min_test_days=365,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "shap_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {results_path}")