"""
Uncertainty for the model-versus-climatology comparison.

The question is not "how wide is each method's own interval?" but
"is the DIFFERENCE between the model and a baseline distinguishable
from zero on this test sample?". Every replicate scores the model and
the baseline on the same resampled rows (paired bootstrap) and keeps
the difference. Two resampling units are used and both are reported:

  - by_cell:        whole grid cells (rows of a cell are correlated).
  - by_14day_block: whole 14-day blocks of dates, all cells together
                    (one storm hits many cells on the same days).

The model and the baselines are fixed: trained once with the shared
recipe in src/terrarisk/models/recipe.py and resampled, never retrained
per replicate. Baselines are computed from the training split only
(train_model.compute_train_only_baselines): the per-cell rate and the
seasonal (cell x month) rate. The seasonal baseline is the fair
opponent for a model that knows the day of year.

Metrics: PR-AUC, ROC-AUC, Brier score, and top-10% event capture.
A negative Brier difference means the model is better (lower is better).

paired_comparison_block() is reused by other experiments (for example
richer_precip_features) so every paired comparison in the project is
computed the same way. run_uncertainty_for_hazard() accepts a forecast
horizon and a feature list so the same analysis can be run for the
1-day and 3-day horizons.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.terrarisk.models.paired_bootstrap import (
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    paired_bootstrap_difference,
    time_block_labels,
)
from src.terrarisk.models.recipe import calibrate_on_validation, fit_lightgbm
from src.terrarisk.models.train_model import (
    FEATURE_COLUMNS,
    FEATURE_TABLE_PATH,
    HORIZON_DAYS,
    MIN_TEST_DAYS_DEFAULT,
    TRAIN_ONLY_BASELINE_COL,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
    prepare_hazard_data,
)

RESULTS_DIR = Path(__file__).parent / "results"

CONFIDENCE_LEVEL = 0.95
TOP_K_PERCENT_FOR_CI = 10
TIME_BLOCK_DAYS = 14

BASELINE_COLUMNS = {
    "cell_rate_baseline": TRAIN_ONLY_BASELINE_COL,
    "seasonal_baseline": TRAIN_ONLY_SEASONAL_BASELINE_COL,
}


def block_bootstrap_indices(cell_ids: pd.Series, seed: int) -> np.ndarray:
    """Row positions of one cell-level bootstrap sample (whole cells, with replacement)."""
    rng = np.random.RandomState(seed)
    unique_cells = cell_ids.unique()
    sampled_cells = rng.choice(unique_cells, size=len(unique_cells), replace=True)
    positions_by_cell = {
        cell: np.where(cell_ids.values == cell)[0] for cell in unique_cells
    }
    return np.concatenate([positions_by_cell[c] for c in sampled_cells])


def percentile_ci(values: np.ndarray, confidence: float = CONFIDENCE_LEVEL) -> tuple:
    clean = values[~np.isnan(values)]
    alpha = 1 - confidence
    lower = np.percentile(clean, 100 * alpha / 2)
    upper = np.percentile(clean, 100 * (1 - alpha / 2))
    return float(lower), float(upper), int(len(clean))


def top_k_capture_rate(y_true, scores, k_percent: float) -> float:
    """
    Share of all positives found in the top k% of rows by score.
    Ties are broken by row order (stable sort) so the result is
    deterministic; with heavily tied scores (a baseline that is constant
    within a cell and month) which tied rows are picked is arbitrary,
    which is a real limitation of this metric for such baselines.
    """
    y_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores, dtype=float)
    n_positive = y_arr.sum()
    if n_positive == 0:
        return float("nan")
    n_selected = max(1, int(round(len(y_arr) * k_percent / 100)))
    order = np.argsort(-scores_arr, kind="stable")[:n_selected]
    return float(y_arr[order].sum() / n_positive)


def top10_capture(y_true, scores) -> float:
    return top_k_capture_rate(y_true, scores, TOP_K_PERCENT_FOR_CI)


def _point_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "top10pct_capture": float(top10_capture(y, scores)),
    }


ALL_METRIC_KEYS = ("pr_auc", "roc_auc", "brier", "top10pct_capture")


def paired_comparison_block(
    y,
    model_scores,
    baseline_scores,
    cell_ids,
    dates,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    metric_keys=ALL_METRIC_KEYS,
) -> dict:
    """
    Paired bootstrap of (model - baseline) for each metric, under both
    resampling units. Returns {clustering_name: {metric_key: result}}.
    """
    metric_functions = {
        "pr_auc": "pr_auc",
        "roc_auc": "roc_auc",
        "brier": "brier",
        "top10pct_capture": top10_capture,
    }
    clusterings = {
        "by_cell": np.asarray(cell_ids),
        f"by_{TIME_BLOCK_DAYS}day_block": time_block_labels(dates, TIME_BLOCK_DAYS),
    }
    block = {}
    for clustering_name, groups in clusterings.items():
        per_metric = {}
        for metric_key in metric_keys:
            result = paired_bootstrap_difference(
                y, model_scores, baseline_scores, groups,
                metric=metric_functions[metric_key], n_bootstrap=n_bootstrap,
                seed=seed, confidence=CONFIDENCE_LEVEL,
            )
            result["metric"] = metric_key
            per_metric[metric_key] = result
        block[clustering_name] = per_metric
    return block


def run_uncertainty_for_hazard(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    max_usable_date=None,
    min_test_days: int = MIN_TEST_DAYS_DEFAULT,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
    horizon_days: int = HORIZON_DAYS,
    feature_columns=None,
) -> dict | None:
    features = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    prepared = prepare_hazard_data(
        df, event_col, hazard_name, max_usable_date, min_test_days,
        horizon_days=horizon_days, verbose=verbose,
    )
    if prepared is None:
        return None
    train, val, test, label_col = prepared

    model = fit_lightgbm(train[features], train[label_col], val[features], val[label_col])
    calibrator = calibrate_on_validation(model, val[features], val[label_col])
    model_scores = calibrator.predict_proba(test[features])[:, 1]

    y = test[label_col].to_numpy()

    results = {
        "horizon_days": int(horizon_days),
        "features_used": features,
        "test_rows": int(len(test)),
        "test_positives": int(y.sum()),
        "test_prevalence": float(y.mean()),
        "model_best_iteration": int(model.best_iteration_),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "confidence": CONFIDENCE_LEVEL,
        "point_metrics": {"model": _point_metrics(y, model_scores)},
        "paired_differences": {},
    }

    for baseline_name, column in BASELINE_COLUMNS.items():
        baseline_scores = test[column].to_numpy()
        results["point_metrics"][baseline_name] = _point_metrics(y, baseline_scores)
        results["paired_differences"][baseline_name] = paired_comparison_block(
            y, model_scores, baseline_scores,
            test["cell_id"].to_numpy(), test["date"], n_bootstrap, seed,
        )

    if verbose:
        print(f"\n--- {hazard_name}: model minus baseline, paired cluster bootstrap "
              f"({n_bootstrap} replicates, seed {seed}) ---")
        print("Brier: negative difference = model better; other metrics: positive = model better.")
        for baseline_name, by_clustering in results["paired_differences"].items():
            for clustering_name, per_metric in by_clustering.items():
                for metric_key, r in per_metric.items():
                    ci = r["difference_ci"]
                    ci_text = "undefined" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    print(f"{baseline_name:<20}{clustering_name:<16}{metric_key:<18}"
                          f"diff {r['point_difference']:+.4f}  95% CI {ci_text}  "
                          f"valid {r['n_valid']}/{r['n_bootstrap']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("Loading feature table...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    all_results = {}

    flood_results = run_uncertainty_for_hazard(
        df, "flood_event", "FLOOD", n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    if flood_results:
        all_results["flood"] = flood_results

    for short_horizon in (1, 3):
        short_results = run_uncertainty_for_hazard(
            df, "flood_event", f"FLOOD (horizon={short_horizon}d)",
            n_bootstrap=args.n_bootstrap, seed=args.seed,
            horizon_days=short_horizon,
        )
        if short_results:
            all_results[f"flood_horizon_{short_horizon}d"] = short_results

    landslide_results = run_uncertainty_for_hazard(
        df, "landslide_event", "LANDSLIDE",
        max_usable_date=pd.Timestamp("2017-07-12"), min_test_days=365,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    if landslide_results:
        all_results["landslide"] = landslide_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "uncertainty_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")
