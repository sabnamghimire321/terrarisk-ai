"""
Compares model families on identical, leakage-safe splits:
climatological baselines (per-cell rate and the seasonal cell x month
rate, both from training rows only), a single global threshold rule,
logistic regression, random forest, and LightGBM. Logistic regression,
random forest and LightGBM are Platt-calibrated on the validation split
the same way, so Brier-score comparisons across the ladder are fair.

An earlier version of this script never trained LightGBM at all: the
saved ladder results contained only four rows, so the LightGBM row in
the paper's ladder table could not be traced to this experiment. It is
now computed here, with the shared recipe in
src/terrarisk/models/recipe.py.

"_test_prevalence" is the share of test rows with label 1; a random
classifier scores a PR-AUC equal to it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

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

THRESHOLD_RULE_FEATURE = "precip_7day_sum_mm"
THRESHOLD_RULE_PERCENTILE = 90


def evaluate_predictions(y_true, y_pred_proba) -> dict:
    roc_auc = (
        float(roc_auc_score(y_true, y_pred_proba))
        if len(np.unique(np.asarray(y_true))) > 1 else float("nan")
    )
    return {
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "roc_auc": roc_auc,
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
    }

def run_threshold_rule(train, test, label_col) -> dict:
    threshold = np.percentile(train[THRESHOLD_RULE_FEATURE], THRESHOLD_RULE_PERCENTILE)
    predictions = (test[THRESHOLD_RULE_FEATURE] > threshold).astype(float)
    result = evaluate_predictions(test[label_col], predictions)
    result["note"] = "hard 0/1 output, not a calibrated probability"
    return result


def run_logistic_regression(train, val, test, label_col) -> dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[FEATURE_COLUMNS])
    X_val_scaled = scaler.transform(val[FEATURE_COLUMNS])
    X_test_scaled = scaler.transform(test[FEATURE_COLUMNS])

    n_pos = train[label_col].sum()
    n_neg = len(train) - n_pos
    class_weight = {0: 1.0, 1: n_neg / n_pos} if n_pos > 0 else None

    model = LogisticRegression(
        class_weight=class_weight, max_iter=1000, random_state=42
    )
    model.fit(X_train_scaled, train[label_col])

    raw_predictions = model.predict_proba(X_test_scaled)[:, 1]
    raw_result = evaluate_predictions(test[label_col], raw_predictions)

    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val_scaled, val[label_col])
    calibrated_predictions = calibrator.predict_proba(X_test_scaled)[:, 1]
    calibrated_result = evaluate_predictions(test[label_col], calibrated_predictions)

    return {
        "pr_auc": calibrated_result["pr_auc"],
        "roc_auc": calibrated_result["roc_auc"],
        "brier": calibrated_result["brier"],
        "pr_auc_raw": raw_result["pr_auc"],
        "brier_raw": raw_result["brier"],
    }

def run_random_forest(train, val, test, label_col) -> dict:
    n_pos = train[label_col].sum()
    n_neg = len(train) - n_pos
    class_weight = {0: 1.0, 1: n_neg / n_pos} if n_pos > 0 else None

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train[label_col])

    raw_predictions = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    raw_result = evaluate_predictions(test[label_col], raw_predictions)

    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(val[FEATURE_COLUMNS], val[label_col])
    calibrated_predictions = calibrator.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    calibrated_result = evaluate_predictions(test[label_col], calibrated_predictions)

    return {
        "pr_auc": calibrated_result["pr_auc"],
        "roc_auc": calibrated_result["roc_auc"],
        "brier": calibrated_result["brier"],
        "pr_auc_raw": raw_result["pr_auc"],
        "brier_raw": raw_result["brier"],
    }

def run_lightgbm(train, val, test, label_col) -> dict:
    """The shared recipe: early stopping and Platt scaling on validation only."""
    features = FEATURE_COLUMNS
    model = fit_lightgbm(
        train[features], train[label_col], val[features], val[label_col]
    )
    raw_predictions = model.predict_proba(test[features])[:, 1]
    raw_result = evaluate_predictions(test[label_col], raw_predictions)

    calibrator = calibrate_on_validation(model, val[features], val[label_col])
    calibrated_predictions = calibrator.predict_proba(test[features])[:, 1]
    calibrated_result = evaluate_predictions(test[label_col], calibrated_predictions)

    return {
        "pr_auc": calibrated_result["pr_auc"],
        "roc_auc": calibrated_result["roc_auc"],
        "brier": calibrated_result["brier"],
        "pr_auc_raw": raw_result["pr_auc"],
        "brier_raw": raw_result["brier"],
    }


def run_ladder_for_hazard(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    baseline_df: pd.DataFrame = None,
    max_usable_date=None,
    min_test_days: int = 365,
) -> dict | None:
    prepared = prepare_hazard_data(
        df, event_col, hazard_name, max_usable_date, min_test_days
    )
    if prepared is None:
        return None
    train, val, test, label_col = prepared

    results = {}

    baseline_predictions = get_baseline_predictions(test, baseline_df)
    results["climatological_baseline"] = evaluate_predictions(
        test[label_col], baseline_predictions
    )
    if TRAIN_ONLY_SEASONAL_BASELINE_COL in test.columns:
        results["seasonal_baseline"] = evaluate_predictions(
            test[label_col], test[TRAIN_ONLY_SEASONAL_BASELINE_COL]
        )
    results["threshold_rule"] = run_threshold_rule(train, test, label_col)
    results["logistic_regression"] = run_logistic_regression(train, val, test, label_col)
    results["random_forest"] = run_random_forest(train, val, test, label_col)
    results["lightgbm"] = run_lightgbm(train, val, test, label_col)

    results["_test_set_size"] = len(test)
    results["_test_positive_count"] = int(test[label_col].sum())
    results["_test_prevalence"] = float(test[label_col].mean())

    return results

def print_results(results: dict) -> None:
    print(f"Test rows: {results['_test_set_size']}  positives: "
          f"{results['_test_positive_count']}  prevalence "
          f"(random-classifier PR-AUC): {results['_test_prevalence']:.4f}")
    for model_name, metrics in results.items():
        if model_name.startswith("_"):
            continue
        line = (f"{model_name:<25} PR-AUC: {metrics['pr_auc']:.4f}  "
                f"ROC-AUC: {metrics['roc_auc']:.4f}  Brier: {metrics['brier']:.4f}")
        if "brier_raw" in metrics:
            line += f"  (raw Brier: {metrics['brier_raw']:.4f})"
        if "note" in metrics:
            line += f"  [{metrics['note']}]"
        print(line)


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

    print("\n" + "=" * 60)
    print("FLOOD")
    print("=" * 60)
    flood_results = run_ladder_for_hazard(
        df, "flood_event", "FLOOD", flood_baseline_7d
    )
    if flood_results:
        all_results["flood"] = flood_results
        print_results(flood_results)

    print("\n" + "=" * 60)
    print("LANDSLIDE")
    print("=" * 60)
    landslide_anchor = pd.Timestamp("2017-07-12")
    landslide_results = run_ladder_for_hazard(
        df, "landslide_event", "LANDSLIDE", landslide_baseline_7d,
        max_usable_date=landslide_anchor, min_test_days=365,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results
        print_results(landslide_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "ladder_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")