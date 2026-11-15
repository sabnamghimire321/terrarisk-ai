"""
Rolling-origin (multi-year) check of the flood comparison between the
model and the train-only SEASONAL baseline.

Every other experiment evaluates one test year (the last 365 days), which
is one sample of monsoon behaviour. Here the same procedure is repeated
on N consecutive annual test windows, each trained only on data before
its own validation year:

    fold 1: test = last 365 days                 (the project's main test year)
    fold 2: test = the 365 days before fold 1's test window
    fold 3: ... and so on, going back one year per fold

Within every fold the protocol is exactly the project's: 365-day
validation window right before the test window, 7-day purge gaps,
train-only baselines, the shared LightGBM recipe (early stopping and
Platt scaling on that fold's validation year only). Folds do not share
anything except the raw table.

Two feature sets are compared with the seasonal baseline, fixed before
running: the original five features, and the extended set (14/30-day
rainfall windows plus the rainfall anomaly, the anomaly climatology being
fit on each fold's own training rows).

Evidence rule, fixed before running. A feature set is reported as showing
skill beyond season only if BOTH hold:
  1. in the pooled paired bootstrap (clusters = 14-day blocks of dates,
     resampled across all folds together) the 95% interval of
     (model - seasonal) excludes 0 in the model's favour for ROC-AUC AND
     for Brier (Brier: lower is better), and
  2. the model is better than the seasonal baseline on those metrics in
     at least 4 of 5 folds (or, for a different fold count, at least
     80% of folds).
PR-AUC is reported but is not part of the rule, because it moves with the
event rate, which differs between years.

Flood only: the landslide catalog overlaps the weather record for only
about six years, too few for a multi-year design.

Secondary analysis, declared before any rolling-origin result was seen
(--weighting selected). The calibration study showed the unweighted
LightGBM scoring higher on the main test year than the class-weighted
recipe. Picking a variant because of its test score would be test-set
selection, so the question is asked this way instead: in each fold the
class weighting (on or off) is chosen by VALIDATION PR-AUC only
(src/terrarisk/models/recipe.fit_lightgbm_selected; weighting is kept
unless the unweighted model is strictly better), then the same evidence
rule is applied. Folds 2 to 5 are test years that no earlier experiment
looked at. The primary analysis (--weighting weighted, the default)
is unchanged and is reported first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from experiments.richer_precip_features.run import (
    EXTENDED_FEATURE_COLUMNS,
    ORIGINAL_FEATURE_COLUMNS,
    apply_precip_anomaly,
    build_extended_feature_table,
    compute_train_climatology,
)
from src.terrarisk.models.recipe import (
    calibrate_on_validation,
    fit_lightgbm,
    fit_lightgbm_selected,
)
from experiments.uncertainty.run import paired_comparison_block
from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.ingestion.open_meteo_flood import subsample_grid_for_flood
from src.terrarisk.models.paired_bootstrap import (
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    paired_bootstrap_difference,
    time_block_labels,
)
from src.terrarisk.models.train_model import (
    HORIZON_DAYS,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
    prepare_hazard_data,
)

RESULTS_DIR = Path(__file__).parent / "results"

N_FOLDS = 5
TEST_DAYS = 365
TIME_BLOCK_DAYS = 14
MIN_FOLD_SHARE_FAVOURING_MODEL = 0.8
WEIGHTING_MODES = ("weighted", "selected")

FEATURE_SETS = {
    "original": ORIGINAL_FEATURE_COLUMNS,
    "extended": EXTENDED_FEATURE_COLUMNS,
}
POOLED_METRICS = ("pr_auc", "roc_auc", "brier")


def fold_last_label_dates(
    last_label_date: pd.Timestamp, n_folds: int, test_days: int
) -> list:
    """Last labelled date of each fold's test window, newest fold first."""
    return [
        last_label_date - pd.Timedelta(days=test_days * i) for i in range(n_folds)
    ]


def _point_metrics(y, scores) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
    }


def _model_is_better(metric: str, difference: float) -> bool:
    """difference = model - baseline. Brier: lower is better."""
    return difference < 0 if metric == "brier" else difference > 0


def score_with_weighting(train, val, test, label_col, feature_columns, weighting: str):
    """
    Calibrated test probabilities from the shared recipe.
    weighting="weighted": the project recipe (class weighting on).
    weighting="selected": class weighting chosen on validation only.
    Returns (scores, details) where details is None for "weighted".
    """
    if weighting not in WEIGHTING_MODES:
        raise ValueError(f"weighting must be one of {WEIGHTING_MODES}")
    X_train, y_train = train[feature_columns], train[label_col]
    X_val, y_val = val[feature_columns], val[label_col]

    if weighting == "weighted":
        model = fit_lightgbm(X_train, y_train, X_val, y_val)
        details = None
    else:
        model, used_weight, details = fit_lightgbm_selected(X_train, y_train, X_val, y_val)
        details = dict(details, used_class_weight=bool(used_weight))

    calibrator = calibrate_on_validation(model, X_val, y_val)
    return calibrator.predict_proba(test[feature_columns])[:, 1], details


def run_fold(
    df: pd.DataFrame,
    fold_index: int,
    last_label_date: pd.Timestamp,
    horizon_days: int,
    n_bootstrap: int,
    seed: int,
    weighting: str = "weighted",
) -> dict | None:
    prepared = prepare_hazard_data(
        df, "flood_event", f"FLOOD rolling-origin fold {fold_index}",
        max_usable_date=last_label_date + pd.Timedelta(days=horizon_days),
        min_test_days=TEST_DAYS, horizon_days=horizon_days, verbose=False,
    )
    if prepared is None:
        return None
    train, val, test, label_col = prepared

    climatology = compute_train_climatology(train)
    train = apply_precip_anomaly(train, climatology)
    val = apply_precip_anomaly(val, climatology)
    test = apply_precip_anomaly(test, climatology)

    y = test[label_col].to_numpy()
    seasonal = test[TRAIN_ONLY_SEASONAL_BASELINE_COL].to_numpy()

    fold = {
        "fold": fold_index,
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "val_start": str(val["date"].min().date()),
        "val_end": str(val["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "test_rows": int(len(test)),
        "test_positives": int(y.sum()),
        "test_prevalence": float(y.mean()),
        "seasonal_baseline": _point_metrics(y, seasonal),
        "feature_sets": {},
        "_arrays": {"y": y, "seasonal": seasonal, "cells": test["cell_id"].to_numpy(),
                    "dates": test["date"], "scores": {}},
    }

    for name, columns in FEATURE_SETS.items():
        scores, weighting_details = score_with_weighting(
            train, val, test, label_col, columns, weighting
        )
        fold["_arrays"]["scores"][name] = scores
        point = _point_metrics(y, scores)
        baseline = fold["seasonal_baseline"]
        fold["feature_sets"][name] = {
            "class_weighting": weighting_details,
            "model": point,
            "difference_vs_seasonal": {m: point[m] - baseline[m] for m in point},
            "model_better_than_seasonal": {
                m: _model_is_better(m, point[m] - baseline[m]) for m in point
            },
            "paired_vs_seasonal": paired_comparison_block(
                y, scores, seasonal, test["cell_id"].to_numpy(), test["date"],
                n_bootstrap=n_bootstrap, seed=seed, metric_keys=POOLED_METRICS,
            ),
        }
    return fold


def pooled_analysis(folds: list, n_bootstrap: int, seed: int) -> dict:
    """Paired bootstrap on all folds' test rows together, clusters = date blocks."""
    y = np.concatenate([f["_arrays"]["y"] for f in folds])
    seasonal = np.concatenate([f["_arrays"]["seasonal"] for f in folds])
    groups = np.concatenate([
        f["fold"] * 1000 + time_block_labels(f["_arrays"]["dates"], TIME_BLOCK_DAYS)
        for f in folds
    ])

    pooled = {}
    for name in FEATURE_SETS:
        scores = np.concatenate([f["_arrays"]["scores"][name] for f in folds])
        per_metric = {}
        for metric in POOLED_METRICS:
            result = paired_bootstrap_difference(
                y, scores, seasonal, groups, metric=metric,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            result["metric"] = metric
            fold_wins = sum(
                1 for f in folds
                if f["feature_sets"][name]["model_better_than_seasonal"][metric]
            )
            result["folds_favouring_model"] = int(fold_wins)
            result["n_folds"] = len(folds)
            per_metric[metric] = result

        def _favours_model(metric: str) -> bool:
            r = per_metric[metric]
            ci = r["difference_ci"]
            if ci is None:
                return False
            excludes_in_favour = ci[1] < 0 if metric == "brier" else ci[0] > 0
            enough_folds = (
                r["folds_favouring_model"] >= MIN_FOLD_SHARE_FAVOURING_MODEL * len(folds)
            )
            return bool(excludes_in_favour and enough_folds)

        per_metric["shows_skill_beyond_season"] = bool(
            _favours_model("roc_auc") and _favours_model("brier")
        )
        pooled[name] = per_metric
    return pooled


def run_rolling_origin(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    horizon_days: int = HORIZON_DAYS,
    verbose: bool = True,
    weighting: str = "weighted",
) -> dict:
    last_label_date = df["date"].max() - pd.Timedelta(days=horizon_days)
    anchors = fold_last_label_dates(last_label_date, n_folds, TEST_DAYS)

    folds = []
    for index, anchor in enumerate(anchors, start=1):
        fold = run_fold(df, index, anchor, horizon_days, n_bootstrap, seed, weighting)
        if fold is None:
            if verbose:
                print(f"Fold {index} (ending {anchor.date()}) cannot be evaluated; stopping.")
            break
        folds.append(fold)
        if verbose:
            b = fold["seasonal_baseline"]
            print(f"\nFold {index}: test {fold['test_start']} .. {fold['test_end']} "
                  f"(train ends {fold['train_end']}), prevalence {fold['test_prevalence']:.3f}")
            print(f"  seasonal baseline  PR {b['pr_auc']:.4f}  ROC {b['roc_auc']:.4f}  Brier {b['brier']:.4f}")
            for name, r in fold["feature_sets"].items():
                m = r["model"]
                chosen = ""
                if r["class_weighting"] is not None:
                    chosen = f"  [class weighting {'on' if r['class_weighting']['used_class_weight'] else 'off'}]"
                print(f"  {name:<18} PR {m['pr_auc']:.4f}  ROC {m['roc_auc']:.4f}  Brier {m['brier']:.4f}{chosen}")

    pooled = pooled_analysis(folds, n_bootstrap, seed) if folds else {}

    if verbose and pooled:
        print(f"\n--- Pooled over {len(folds)} folds: model minus seasonal baseline "
              f"(clusters = {TIME_BLOCK_DAYS}-day blocks, {n_bootstrap} replicates) ---")
        print("Brier: negative = model better; other metrics: positive = model better.")
        for name, per_metric in pooled.items():
            for metric in POOLED_METRICS:
                r = per_metric[metric]
                ci = r["difference_ci"]
                ci_text = "undefined" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                print(f"{name:<10}{metric:<8} diff {r['point_difference']:+.4f}  "
                      f"95% CI {ci_text}  folds favouring model "
                      f"{r['folds_favouring_model']}/{r['n_folds']}")
            print(f"{name:<10}shows skill beyond season (pre-set rule): "
                  f"{per_metric['shows_skill_beyond_season']}")

    for fold in folds:
        fold.pop("_arrays")

    return {
        "n_folds_requested": int(n_folds),
        "n_folds_run": len(folds),
        "test_days": TEST_DAYS,
        "time_block_days": TIME_BLOCK_DAYS,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "horizon_days": int(horizon_days),
        "weighting": weighting,
        "folds": folds,
        "pooled": pooled,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--weighting", choices=WEIGHTING_MODES, default="weighted")
    args = parser.parse_args()

    print("Building extended feature table...")
    grid_72 = subsample_grid_for_flood(build_grid(), every_n=3)
    df = build_extended_feature_table(grid_72)

    results = run_rolling_origin(
        df, n_bootstrap=args.n_bootstrap, seed=args.seed, weighting=args.weighting
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = (
        "rolling_origin_results.json" if args.weighting == "weighted"
        else "rolling_origin_selected_weighting_results.json"
    )
    results_path = RESULTS_DIR / file_name
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")
