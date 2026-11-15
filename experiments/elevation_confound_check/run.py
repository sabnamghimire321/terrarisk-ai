"""
Elevation confound check - see this module's docstring.

Investigates a specific concern raised during peer review: because
elevation_m is STATIC per grid cell (confirmed in
build_features.py), a model trained on elevation alone can only emit
ONE constant predicted score per cell, identical across every row of
that cell's test window. This makes "elevation-only" structurally
closer to a per-cell ranking exercise than a genuine day-to-day
classifier - the same shape as the climatological baseline, which
also assigns one constant score per cell.

That raises a real question: if both reduce to "one score per cell,"
why did elevation-only (the original ablation: PR-AUC 0.748) so
dramatically outperform the explicit climatological baseline
(PR-AUC 0.489)? Two competing explanations:

H1 (genuine physical signal): elevation captures real terrain-driven
   flood dynamics the baseline's frequency calculation misses.
H2 (disguised per-cell rate estimator): elevation functions as a
   near-unique cell identifier, and a supervised model trained on it
   is simply a MORE PRECISELY FITTED per-cell rate estimator than the
   baseline's own frequency calculation - not evidence of a terrain
   effect at all.

This experiment cannot conclusively prove one hypothesis from a
single test window, but it runs the most direct available test: for
each held-out cell, compare how closely (a) elevation-only's constant
score and (b) the baseline's constant score each correlate with (c)
that cell's ACTUAL realized positive rate during the test window. If
elevation-only correlates more closely with (c) than the baseline
does, that is direct, quantifiable evidence favoring H2 over H1.

STATUS ON THE CORRECTED PIPELINE. The premise above (elevation-only far
ahead of the baseline) came from the original 60-day single-season test
window. On the corrected pipeline (365-day windows, strict labels,
train-only baselines, shared model recipe) it is checked first by
premise_check(): if elevation-only is not clearly better than the
baseline, or is near the random-classifier level, the confound question
no longer has anything to explain, and this script says so instead of
interpreting the correlations as if the original result still stood.
The model here uses the shared recipe in src/terrarisk/models/recipe.py;
the baseline is the train-only per-cell rate attached to the test frame.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.models.train_model import (
    FEATURE_TABLE_PATH,
    get_baseline_predictions,
    prepare_hazard_data,
)

RESULTS_DIR = Path(__file__).parent / "results"

TERRAIN_FEATURE = ["elevation_m"]


def train_terrain_only_model(train, val, label_col):
    """Elevation-only model via the shared recipe (early stopping and Platt on validation)."""
    X_train, y_train = train[TERRAIN_FEATURE], train[label_col]
    X_val, y_val = val[TERRAIN_FEATURE], val[label_col]

    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    return calibrate_on_validation(model, X_val, y_val)


def check_constant_per_cell(test: pd.DataFrame, predictions: np.ndarray) -> dict:
    df = test.copy()
    df["prediction"] = predictions

    per_cell_std = df.groupby("cell_id")["prediction"].std()
    max_std = float(per_cell_std.max())
    is_constant = bool(max_std < 1e-6)

    return {
        "is_constant_per_cell": is_constant,
        "max_within_cell_std": max_std,
        "n_cells": int(df["cell_id"].nunique()),
        "n_distinct_prediction_values": int(df["prediction"].round(8).nunique()),
    }


def compute_realized_test_rate(test: pd.DataFrame, label_col: str) -> pd.Series:
    return test.groupby("cell_id")[label_col].mean()


def premise_check(y_test, elevation_predictions, baseline_predictions) -> dict:
    """
    Does the puzzle exist on this pipeline? Elevation-only versus the
    baseline, each next to the prevalence (the PR-AUC of a random
    classifier). The premise holds only if elevation-only is clearly
    above BOTH the baseline and the prevalence.
    """
    y = np.asarray(y_test)
    prevalence = float(y.mean())
    elevation_pr = float(average_precision_score(y, elevation_predictions))
    baseline_pr = float(average_precision_score(y, np.asarray(baseline_predictions)))
    elevation_roc = float(roc_auc_score(y, elevation_predictions))
    baseline_roc = float(roc_auc_score(y, np.asarray(baseline_predictions)))
    margin = 0.05
    premise_holds = bool(
        elevation_pr > baseline_pr + margin and elevation_pr > prevalence + margin
    )
    return {
        "test_prevalence": prevalence,
        "elevation_only_pr_auc": elevation_pr,
        "baseline_pr_auc": baseline_pr,
        "elevation_only_roc_auc": elevation_roc,
        "baseline_roc_auc": baseline_roc,
        "premise_margin": margin,
        "premise_holds": premise_holds,
    }


def compare_correlation_with_realized_rate(
    test: pd.DataFrame,
    label_col: str,
    elevation_predictions: np.ndarray,
    baseline_predictions: pd.Series,
) -> dict:
    df = test.copy()
    df["elevation_score"] = elevation_predictions
    df["baseline_score"] = baseline_predictions.values

    per_cell = df.groupby("cell_id").agg(
        elevation_score=("elevation_score", "first"),
        baseline_score=("baseline_score", "first"),
        realized_rate=(label_col, "mean"),
    )

    elevation_corr, elevation_p = spearmanr(per_cell["elevation_score"], per_cell["realized_rate"])
    baseline_corr, baseline_p = spearmanr(per_cell["baseline_score"], per_cell["realized_rate"])

    return {
        "n_cells": int(len(per_cell)),
        "elevation_vs_realized_spearman": float(elevation_corr),
        "elevation_vs_realized_pvalue": float(elevation_p),
        "baseline_vs_realized_spearman": float(baseline_corr),
        "baseline_vs_realized_pvalue": float(baseline_p),
        "elevation_correlates_more_closely": bool(abs(elevation_corr) > abs(baseline_corr)),
    }


if __name__ == "__main__":
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    prepared = prepare_hazard_data(df, "flood_event", "FLOOD", verbose=True)
    train, val, test, label_col = prepared

    print("\nTraining elevation-only model (shared recipe, same as ablation's terrain_only)...")
    model = train_terrain_only_model(train, val, label_col)
    predictions = model.predict_proba(test[TERRAIN_FEATURE])[:, 1]

    baseline_scores = get_baseline_predictions(test)

    print("\n--- Step 0: does the original puzzle exist on the corrected pipeline? ---")
    premise_result = premise_check(test[label_col], predictions, baseline_scores)
    for k, v in premise_result.items():
        print(f"  {k}: {v}")

    print("\n--- Step 1: is elevation-only structurally constant per cell? ---")
    constancy_result = check_constant_per_cell(test, predictions)
    for k, v in constancy_result.items():
        print(f"  {k}: {v}")

    print("\n--- Step 2: which score correlates more closely with REALIZED test-window rate? ---")
    correlation_result = compare_correlation_with_realized_rate(
        test, label_col, predictions, baseline_scores
    )
    for k, v in correlation_result.items():
        print(f"  {k}: {v}")

    if not premise_result["premise_holds"]:
        print("\nFINDING: on the corrected pipeline elevation-only is NOT clearly "
              "better than the baseline and the random-classifier level, so the "
              "original puzzle (elevation-only far ahead of the baseline) does "
              "not reproduce. The step 2 correlations are reported for "
              "completeness but are not needed to explain anything.")
    elif correlation_result["elevation_correlates_more_closely"]:
        print("\nFINDING: elevation-only's score correlates MORE closely with the "
              "realized test-window rate than the baseline's does - evidence "
              "favoring H2 (disguised per-cell rate estimator) over H1 (genuine "
              "terrain effect).")
    else:
        print("\nFINDING: elevation-only's score does NOT correlate more closely "
              "with realized test-window rate than the baseline - H2 is not "
              "supported by this specific test; H1 remains a live possibility.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "elevation_confound_results.json"
    results_path.write_text(json.dumps(
        {
            "premise_check": premise_result,
            "constancy_check": constancy_result,
            "correlation_check": correlation_result,
        },
        indent=2,
    ))
    print(f"\nResults saved to {results_path}")
