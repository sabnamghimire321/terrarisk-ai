"""
Three individually-justified perturbations, not a generic stress
test grid: missing data (a plausible real deployment gap), forecast
horizon sensitivity (reusing the existing horizon_days parameter),
and precipitation noise (measurement noise is plausible for a
physically measured quantity; elevation and day_of_year are never
actually uncertain in reality and are not perturbed). Landslide is
limited to the missing-data test - its already-thin positive count
makes the other two perturbations too unstable to interpret.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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

MISSING_DATA_FRACTIONS = [0.05, 0.10, 0.20]
HORIZONS_TO_TEST = [1, 3, 7]
NOISE_RELATIVE_FRACTIONS = [0.10, 0.20]
PRECIP_COLUMNS = ["precip_1day_sum_mm", "precip_3day_sum_mm", "precip_7day_sum_mm"]


def inject_missingness(X: pd.DataFrame, missing_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X_perturbed = X.copy()
    mask = rng.uniform(0, 1, X.shape) < missing_frac
    X_perturbed = X_perturbed.mask(mask)
    return X_perturbed

def inject_precip_noise(X: pd.DataFrame, relative_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X_perturbed = X.copy()
    for col in PRECIP_COLUMNS:
        noise = rng.normal(0, relative_frac * X_perturbed[col].abs(), size=len(X_perturbed))
        X_perturbed[col] = (X_perturbed[col] + noise).clip(lower=0)
    return X_perturbed


def train_default_model(train: pd.DataFrame, val: pd.DataFrame, label_col: str):
    X_train, y_train = train[FEATURE_COLUMNS], train[label_col]
    X_val, y_val = val[FEATURE_COLUMNS], val[label_col]

    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    calibrator = calibrate_on_validation(model, X_val, y_val)
    return model, calibrator


def run_missing_data_test(
    hazard_name: str, train, val, test, label_col
) -> dict:
    model, calibrator = train_default_model(train, val, label_col)
    y_test = test[label_col]

    clean_scores = calibrator.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    clean_pr_auc = average_precision_score(y_test, clean_scores)

    results = {"clean_pr_auc": float(clean_pr_auc), "missing_data": {}}

    print(f"\n--- Missing-data robustness ({hazard_name}) ---")
    print(f"Clean (0% missing) PR-AUC: {clean_pr_auc:.4f}")

    for frac in MISSING_DATA_FRACTIONS:
        X_perturbed = inject_missingness(test[FEATURE_COLUMNS], frac, seed=42)
        scores = calibrator.predict_proba(X_perturbed)[:, 1]
        pr_auc = average_precision_score(y_test, scores)
        degradation = clean_pr_auc - pr_auc
        results["missing_data"][f"{int(frac * 100)}pct"] = {
            "pr_auc": float(pr_auc), "degradation_from_clean": float(degradation),
        }
        print(f"{int(frac * 100)}% missing: PR-AUC={pr_auc:.4f} "
              f"(degradation: {degradation:+.4f})")

    return results


def run_horizon_sensitivity_test(df: pd.DataFrame, baseline_by_horizon: dict) -> dict:
    print(f"\n--- Horizon sensitivity (FLOOD ONLY) ---")
    results = {}

    for horizon in HORIZONS_TO_TEST:
        prepared = prepare_hazard_data(
            df, "flood_event", f"FLOOD (horizon={horizon})",
            horizon_days=horizon, verbose=False,
        )
        if prepared is None:
            print(f"Horizon {horizon}: could not be evaluated, skipping.")
            continue
        train, val, test, label_col = prepared
        model, calibrator = train_default_model(train, val, label_col)

        scores = calibrator.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        model_pr_auc = average_precision_score(test[label_col], scores)

        baseline_df_h = baseline_by_horizon.get(horizon)
        baseline_scores = get_baseline_predictions(test, baseline_df_h)
        baseline_pr_auc = average_precision_score(test[label_col], baseline_scores)
        seasonal_pr_auc = average_precision_score(
            test[label_col], test[TRAIN_ONLY_SEASONAL_BASELINE_COL]
        )
        prevalence = float(test[label_col].mean())

        results[f"horizon_{horizon}d"] = {
            "model_pr_auc": float(model_pr_auc),
            "baseline_pr_auc": float(baseline_pr_auc),
            "seasonal_baseline_pr_auc": float(seasonal_pr_auc),
            "model_advantage": float(model_pr_auc - baseline_pr_auc),
            "model_advantage_vs_seasonal": float(model_pr_auc - seasonal_pr_auc),
            "test_prevalence": prevalence,
            "test_positives": int(test[label_col].sum()),
        }
        print(f"Horizon {horizon}d: model={model_pr_auc:.4f}  "
              f"cell-rate baseline={baseline_pr_auc:.4f}  "
              f"seasonal baseline={seasonal_pr_auc:.4f}  "
              f"advantage vs seasonal={model_pr_auc - seasonal_pr_auc:+.4f}  "
              f"(prevalence={prevalence:.4f}, n_positive={int(test[label_col].sum())})")

    return results


def run_precip_noise_test(train, val, test, label_col) -> dict:
    print(f"\n--- Precipitation-noise robustness (FLOOD ONLY) ---")
    model, calibrator = train_default_model(train, val, label_col)
    y_test = test[label_col]

    clean_scores = calibrator.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    clean_pr_auc = average_precision_score(y_test, clean_scores)
    print(f"Clean (0% noise) PR-AUC: {clean_pr_auc:.4f}")

    results = {"clean_pr_auc": float(clean_pr_auc), "noise": {}}

    for frac in NOISE_RELATIVE_FRACTIONS:
        X_perturbed = inject_precip_noise(test[FEATURE_COLUMNS], frac, seed=42)
        scores = calibrator.predict_proba(X_perturbed)[:, 1]
        pr_auc = average_precision_score(y_test, scores)
        degradation = clean_pr_auc - pr_auc
        results["noise"][f"{int(frac * 100)}pct_relative"] = {
            "pr_auc": float(pr_auc), "degradation_from_clean": float(degradation),
        }
        print(f"{int(frac * 100)}% relative precip noise: PR-AUC={pr_auc:.4f} "
              f"(degradation: {degradation:+.4f})")

    return results


if __name__ == "__main__":
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    flood_baseline_all = pd.read_csv(DATA_PROCESSED / "flood_baseline.csv")
    landslide_baseline_all = pd.read_csv(DATA_PROCESSED / "landslide_baseline.csv")

    all_results = {}

    print(f"\n{'=' * 60}\nHAZARD: FLOOD\n{'=' * 60}")
    flood_baseline_7d = flood_baseline_all[flood_baseline_all["horizon_days"] == HORIZON_DAYS]
    prepared = prepare_hazard_data(df, "flood_event", "FLOOD", verbose=True)
    flood_results = {}
    if prepared:
        train, val, test, label_col = prepared
        flood_results["missing_data"] = run_missing_data_test(
            "FLOOD", train, val, test, label_col
        )
        flood_results["precip_noise"] = run_precip_noise_test(train, val, test, label_col)

    baseline_by_horizon = {
        h: flood_baseline_all[flood_baseline_all["horizon_days"] == h]
        for h in HORIZONS_TO_TEST
    }
    flood_results["horizon_sensitivity"] = run_horizon_sensitivity_test(df, baseline_by_horizon)
    all_results["flood"] = flood_results

    print(f"\n{'=' * 60}\nHAZARD: LANDSLIDE (missing-data test only)\n{'=' * 60}")
    landslide_anchor = pd.Timestamp("2017-07-12")
    prepared = prepare_hazard_data(
        df, "landslide_event", "LANDSLIDE",
        max_usable_date=landslide_anchor, min_test_days=365, verbose=True,
    )
    if prepared:
        train, val, test, label_col = prepared
        all_results["landslide"] = {
            "missing_data": run_missing_data_test("LANDSLIDE", train, val, test, label_col)
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "robustness_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")