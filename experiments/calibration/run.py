"""
Formal 2x2 calibration study: {unweighted, imbalance-corrected}
training x {raw, Platt-calibrated} probabilities, LightGBM only,
both hazards. Implements Expected Calibration Error (ECE) from
scratch, since no prior experiment in this project had needed it.
Separates discrimination (PR-AUC, ROC-AUC) from calibration (Brier,
ECE) deliberately - a model can rank events well while producing
badly miscalibrated probabilities, and this project's earlier
results showed exactly that pattern for an uncalibrated,
imbalance-corrected model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS,
    FEATURE_TABLE_PATH,
    HORIZON_DAYS,
    prepare_hazard_data,
)

RESULTS_DIR = Path(__file__).parent / "results"

PRECISION_RECALL_THRESHOLD = 0.5
ECE_N_BINS = 10

def compute_reliability_table(y_true, y_prob, n_bins: int = ECE_N_BINS) -> pd.DataFrame:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    rows = []
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        rows.append({
            "bin_lower": round(bin_edges[b], 2),
            "bin_upper": round(bin_edges[b + 1], 2),
            "count": count,
            "mean_predicted": float(np.mean(y_prob[mask])) if count > 0 else None,
            "mean_actual": float(np.mean(np.asarray(y_true)[mask])) if count > 0 else None,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins: int = ECE_N_BINS) -> float:
    table = compute_reliability_table(y_true, y_prob, n_bins)
    populated = table[table["count"] > 0]
    total = populated["count"].sum()
    if total == 0:
        return float("nan")

    weighted_error = (
        populated["count"] / total
        * (populated["mean_predicted"] - populated["mean_actual"]).abs()
    ).sum()
    return float(weighted_error)


def evaluate_variant(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_pred_binary = (y_prob >= PRECISION_RECALL_THRESHOLD).astype(int)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    precision = (
        precision_score(y_true, y_pred_binary, zero_division=0)
    )
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "precision_at_0.5": float(precision),
        "recall_at_0.5": float(recall),
        "n_predicted_positive_at_0.5": int(y_pred_binary.sum()),
    }


def train_lightgbm(X_train, y_train, X_val, y_val, use_weighting: bool):
    """The shared recipe (early stopping on validation); only class weighting varies."""
    return fit_lightgbm(X_train, y_train, X_val, y_val, use_class_weight=use_weighting)


def run_calibration_study_for_hazard(
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

    X_train, y_train = train[FEATURE_COLUMNS], train[label_col]
    X_val, y_val = val[FEATURE_COLUMNS], val[label_col]
    X_test, y_test = test[FEATURE_COLUMNS], test[label_col]

    results = {}

    for weighting_label, use_weighting in [("unweighted", False), ("imbalance_corrected", True)]:
        model = train_lightgbm(X_train, y_train, X_val, y_val, use_weighting)

        raw_predictions = model.predict_proba(X_test)[:, 1]
        raw_key = f"{weighting_label}_raw"
        results[raw_key] = evaluate_variant(y_test, raw_predictions)

        calibrator = calibrate_on_validation(model, X_val, y_val)
        calibrated_predictions = calibrator.predict_proba(X_test)[:, 1]
        calibrated_key = f"{weighting_label}_calibrated"
        results[calibrated_key] = evaluate_variant(y_test, calibrated_predictions)

    print(f"\n--- Calibration study results ({hazard_name}) ---")
    print(f"{'Variant':<24}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>9}{'ECE':>9}"
          f"{'Prec@.5':>9}{'Rec@.5':>9}{'#Pos@.5':>9}")
    for variant_name, metrics in results.items():
        print(f"{variant_name:<24}{metrics['pr_auc']:>9.4f}{metrics['roc_auc']:>9.4f}"
              f"{metrics['brier']:>9.4f}{metrics['ece']:>9.4f}"
              f"{metrics['precision_at_0.5']:>9.4f}{metrics['recall_at_0.5']:>9.4f}"
              f"{metrics['n_predicted_positive_at_0.5']:>9d}")

    return results


if __name__ == "__main__":
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    all_results = {}

    flood_results = run_calibration_study_for_hazard(df, "flood_event", "FLOOD")
    if flood_results:
        all_results["flood"] = flood_results

    landslide_anchor = pd.Timestamp("2017-07-12")
    landslide_results = run_calibration_study_for_hazard(
        df, "landslide_event", "LANDSLIDE",
        max_usable_date=landslide_anchor, min_test_days=365,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "calibration_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")