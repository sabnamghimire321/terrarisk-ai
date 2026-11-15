"""
Paired cluster bootstrap for the DIFFERENCE between two scoring methods
(for example the LightGBM model and a climatological baseline).

Why a paired difference: two confidence intervals that overlap do not
prove the methods are equal, and two that do not overlap are not the
only way a difference can be clear. The right question is "how much
does the metric change when we swap method B for method A on the SAME
data?", so every bootstrap replicate scores both methods on the same
resampled rows and we keep the difference.

Why clusters: rows are not independent. Resampling whole clusters
keeps rows that belong together, together. Choose the clustering to
match the dependence you want to respect:
  - cluster = grid cell: rows of one cell are correlated in time.
  - cluster = block of calendar days (time_block_labels): all cells on
    nearby days are correlated, because one storm hits many cells.
Run both and report the wider interval.

Predictions are FIXED: the models are trained once and their test
scores are resampled. Models are not retrained per replicate. The
intervals therefore describe uncertainty from the finite, dependent
test sample only, not uncertainty from training.

Replicates in which the resampled rows contain only one class are
undefined for PR-AUC, ROC-AUC and Brier here. They are dropped for both
methods together, and counted in n_invalid so they are never hidden.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 42
DEFAULT_CONFIDENCE = 0.95

METRICS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "brier": brier_score_loss,
}


def time_block_labels(dates, block_days: int) -> np.ndarray:
    """Integer block id per row: days since the earliest date // block_days."""
    if block_days < 1:
        raise ValueError("block_days must be at least 1")
    series = pd.to_datetime(pd.Series(np.asarray(dates)))
    return ((series - series.min()).dt.days // block_days).to_numpy()


def _cluster_positions(groups) -> list:
    """List with one array of row positions per distinct group label."""
    groups = np.asarray(groups)
    _, codes = np.unique(groups, return_inverse=True)
    codes = np.asarray(codes).ravel()
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes))[:-1]
    return np.split(order, boundaries)


def sample_cluster_positions(cluster_positions: list, rng) -> np.ndarray:
    """Draws as many clusters as exist, with replacement; returns all their rows."""
    chosen = rng.integers(0, len(cluster_positions), size=len(cluster_positions))
    return np.concatenate([cluster_positions[i] for i in chosen])


def _safe_metric(metric_fn, y_true, scores) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(metric_fn(y_true, scores))
    except ValueError:
        return np.nan


def _percentile_interval(values: np.ndarray, confidence: float) -> tuple:
    alpha = 1.0 - confidence
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def paired_bootstrap_difference(
    y_true,
    scores_a,
    scores_b,
    groups,
    metric="pr_auc",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """
    Paired cluster bootstrap of metric(A) - metric(B).

    `groups` gives each row's cluster label (cell id, or the output of
    time_block_labels). `metric` is "pr_auc", "roc_auc", "brier" or any
    callable f(y_true, scores). Fully deterministic for a given seed.

    Returns a JSON-friendly dict with the point estimates on the full
    test set, a percentile interval for each method and for the
    difference, how many replicates were valid or invalid, and
    `difference_ci_excludes_zero`. That last flag only says the interval
    does not contain 0 for this sample and clustering; it is not a
    significance claim by itself.
    """
    y = np.asarray(y_true)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    groups = np.asarray(groups)

    if not (len(y) == len(a) == len(b) == len(groups)):
        raise ValueError("y_true, scores_a, scores_b and groups must have the same length")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1)")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "custom")
    metric_fn = METRICS[metric] if isinstance(metric, str) else metric

    clusters = _cluster_positions(groups)
    rng = np.random.default_rng(seed)

    values_a = np.full(n_bootstrap, np.nan)
    values_b = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        positions = sample_cluster_positions(clusters, rng)
        y_boot = y[positions]
        values_a[i] = _safe_metric(metric_fn, y_boot, a[positions])
        values_b[i] = _safe_metric(metric_fn, y_boot, b[positions])

    valid = ~np.isnan(values_a) & ~np.isnan(values_b)
    n_valid = int(valid.sum())

    point_a = float(metric_fn(y, a))
    point_b = float(metric_fn(y, b))

    result = {
        "metric": metric_name,
        "n_clusters": len(clusters),
        "n_bootstrap": int(n_bootstrap),
        "n_valid": n_valid,
        "n_invalid": int(n_bootstrap - n_valid),
        "seed": int(seed),
        "confidence": float(confidence),
        "point_a": point_a,
        "point_b": point_b,
        "point_difference": point_a - point_b,
        "ci_a": None,
        "ci_b": None,
        "difference_ci": None,
        "fraction_difference_positive": None,
        "difference_ci_excludes_zero": None,
    }
    if n_valid == 0:
        return result

    differences = values_a[valid] - values_b[valid]
    lower, upper = _percentile_interval(differences, confidence)
    result["ci_a"] = _percentile_interval(values_a[valid], confidence)
    result["ci_b"] = _percentile_interval(values_b[valid], confidence)
    result["difference_ci"] = (lower, upper)
    result["fraction_difference_positive"] = float(np.mean(differences > 0))
    result["difference_ci_excludes_zero"] = bool(lower > 0 or upper < 0)
    return result
