"""
Tests whether the flood comparison between the model and its baselines
depends on the specific 90th-percentile discharge threshold used
everywhere else in this project, by recomputing labels at the 85th and
95th percentile too. Flood only - landslide's label sensitivity would
require re-deriving the event-to-cell spatial join itself, a
materially larger piece of work left for future work.

Baselines are the train-only ones attached by prepare_hazard_data()
from the RELABELED data, so each threshold gets its own matching
baseline: the per-cell rate and the seasonal (cell x month) rate. The
seasonal baseline is the fair opponent for a model that knows the day
of year.

Disclosure: the per-cell discharge threshold itself is the chosen
percentile of the cell's FULL discharge history (compute_cell_thresholds),
so the definition of "a flood" uses climatology that includes the
validation and test years. This is a labelling convention, not a model
input, and it is reported as a limitation.

"same_day_positive_rate" is the share of days above the threshold.
"test_prevalence" is the share of test rows whose 7-day label is 1; this
is the rate that PR-AUC must be compared with (a random classifier
scores PR-AUC equal to it).
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.terrarisk.baseline.flood_baseline import compute_cell_thresholds
from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS,
    FEATURE_TABLE_PATH,
    TRAIN_ONLY_BASELINE_COL,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
    prepare_hazard_data,
)

RESULTS_DIR = Path(__file__).parent / "results"
PERCENTILES_TO_TEST = [85, 90, 95]

FLOOD_DISCHARGE_PATH = DATA_PROCESSED / "flood_discharge_history.csv"


def recompute_flood_labels(
    feature_df: pd.DataFrame, discharge_df: pd.DataFrame, percentile: int
) -> pd.DataFrame:

    df = feature_df.copy()
    thresholds = compute_cell_thresholds(discharge_df, percentile)
    df["threshold_m3s_alt"] = df["cell_id"].map(thresholds)
    df["flood_event"] = (df["river_discharge_m3s"] > df["threshold_m3s_alt"]).astype(int)
    return df.drop(columns=["threshold_m3s_alt"])


def _scores(y_test, predictions) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, predictions)),
        "brier": float(brier_score_loss(y_test, predictions)),
    }


def run_for_percentile(
    feature_df: pd.DataFrame, discharge_df: pd.DataFrame, percentile: int
) -> dict:
    print(f"\n--- Percentile threshold: {percentile} ---")

    relabeled = recompute_flood_labels(feature_df, discharge_df, percentile)
    positive_rate = relabeled["flood_event"].mean()
    print(f"Same-day positive rate at this threshold: {positive_rate:.4%}")

    prepared = prepare_hazard_data(
        relabeled, "flood_event", f"FLOOD (percentile={percentile})", verbose=False
    )
    if prepared is None:
        print("Could not be evaluated at this threshold — skipping.")
        return {"percentile": percentile, "evaluable": False}

    train, val, test, label_col = prepared

    X_train, y_train = train[FEATURE_COLUMNS], train[label_col]
    X_val, y_val = val[FEATURE_COLUMNS], val[label_col]
    X_test, y_test = test[FEATURE_COLUMNS], test[label_col]

    model = fit_lightgbm(X_train, y_train, X_val, y_val)
    calibrator = calibrate_on_validation(model, X_val, y_val)
    model_predictions = calibrator.predict_proba(X_test)[:, 1]

    model_scores = _scores(y_test, model_predictions)
    cell_scores = _scores(y_test, test[TRAIN_ONLY_BASELINE_COL])
    seasonal_scores = _scores(y_test, test[TRAIN_ONLY_SEASONAL_BASELINE_COL])
    prevalence = float(y_test.mean())

    beats_cell = model_scores["pr_auc"] > cell_scores["pr_auc"]
    beats_seasonal = model_scores["pr_auc"] > seasonal_scores["pr_auc"]

    print(f"Test prevalence (7-day label): {prevalence:.4f}")
    print(f"PR-AUC  model {model_scores['pr_auc']:.4f}  "
          f"cell-rate baseline {cell_scores['pr_auc']:.4f}  "
          f"seasonal baseline {seasonal_scores['pr_auc']:.4f}")
    print(f"Model beats cell-rate baseline: {beats_cell}   "
          f"Model beats seasonal baseline: {beats_seasonal}")

    return {
        "percentile": percentile,
        "evaluable": True,
        "same_day_positive_rate": float(positive_rate),
        "test_prevalence": prevalence,
        "test_positives": int(y_test.sum()),
        "model": model_scores,
        "cell_rate_baseline": cell_scores,
        "seasonal_baseline": seasonal_scores,
        "model_beats_cell_rate_baseline_pr_auc": bool(beats_cell),
        "model_beats_seasonal_baseline_pr_auc": bool(beats_seasonal),
    }


if __name__ == "__main__":
    print("Loading feature table and discharge history...")
    feature_df = pd.read_csv(FEATURE_TABLE_PATH)
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    discharge_df = pd.read_csv(FLOOD_DISCHARGE_PATH)

    print(f"\n{'=' * 60}")
    print("LABEL SENSITIVITY: FLOOD (discharge threshold percentile)")
    print(f"{'=' * 60}")

    all_results = []
    for percentile in PERCENTILES_TO_TEST:
        result = run_for_percentile(feature_df, discharge_df, percentile)
        all_results.append(result)

    evaluable = [r for r in all_results if r["evaluable"]]
    all_beat_cell = all(r["model_beats_cell_rate_baseline_pr_auc"] for r in evaluable)
    all_beat_seasonal = all(r["model_beats_seasonal_baseline_pr_auc"] for r in evaluable)

    print(f"\n--- Summary (PR-AUC) ---")
    print(f"{'Percentile':<12}{'SameDay':>9}{'Prevalence':>12}{'Model':>9}"
          f"{'CellRate':>10}{'Seasonal':>10}")
    for r in evaluable:
        print(f"{r['percentile']:<12}{r['same_day_positive_rate']:>9.2%}"
              f"{r['test_prevalence']:>12.4f}{r['model']['pr_auc']:>9.4f}"
              f"{r['cell_rate_baseline']['pr_auc']:>10.4f}"
              f"{r['seasonal_baseline']['pr_auc']:>10.4f}")

    print(f"\nModel beats the cell-rate baseline at ALL tested thresholds: {all_beat_cell}")
    print(f"Model beats the seasonal baseline at ALL tested thresholds: {all_beat_seasonal}")
    print("These are point estimates; see the uncertainty experiment for paired intervals.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "label_sensitivity_results.json"
    results_path.write_text(json.dumps(
        {
            "flood": all_results,
            "model_beats_cell_rate_baseline_at_all_thresholds": all_beat_cell,
            "model_beats_seasonal_baseline_at_all_thresholds": all_beat_seasonal,
        },
        indent=2,
    ))
    print(f"\nResults saved to {results_path}")

    print(f"\nNote: LANDSLIDE label sensitivity (nearest-cell matching "
          f"radius, event-date tolerance) is a genuinely different, "
          f"harder problem requiring re-derivation of the event-to-cell "
          f"spatial join — documented as future scope, not tested here.")
