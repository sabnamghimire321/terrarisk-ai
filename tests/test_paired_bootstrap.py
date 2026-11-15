"""
Deterministic tests for the paired cluster bootstrap in
src/terrarisk/models/paired_bootstrap.py. Synthetic data only; no network.
"""

import numpy as np
import pandas as pd
import pytest

from src.terrarisk.models.paired_bootstrap import (
    _cluster_positions,
    paired_bootstrap_difference,
    sample_cluster_positions,
    time_block_labels,
)


def _synthetic(n_clusters=40, rows_per_cluster=100, seed=0):
    rng = np.random.default_rng(seed)
    n = n_clusters * rows_per_cluster
    groups = np.repeat(np.arange(n_clusters), rows_per_cluster)
    y = (rng.random(n) < 0.2).astype(int)
    good = y + rng.normal(0, 0.3, n)
    random_scores = rng.random(n)
    return y, good, random_scores, groups


def test_time_block_labels_are_exact():
    dates = pd.date_range("2020-01-01", periods=10)
    assert time_block_labels(dates, 4).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]


def test_time_block_labels_reject_bad_block_size():
    with pytest.raises(ValueError):
        time_block_labels(pd.date_range("2020-01-01", periods=3), 0)


def test_cluster_positions_partition_all_rows_without_overlap():
    groups = np.array(["b", "a", "b", "c", "a", "b"])
    clusters = _cluster_positions(groups)
    assert [c.tolist() for c in clusters] == [[1, 4], [0, 2, 5], [3]]


def test_resampling_keeps_whole_clusters_together():
    groups = np.array([0, 1, 1, 2, 2, 2])
    clusters = _cluster_positions(groups)
    rng = np.random.default_rng(7)
    for _ in range(50):
        positions = sample_cluster_positions(clusters, rng)
        counts = np.bincount(positions, minlength=len(groups))
        assert counts[1] == counts[2]
        assert counts[3] == counts[4] == counts[5]


def test_same_seed_gives_identical_results_and_different_seed_differs():
    y, good, rand, groups = _synthetic()
    first = paired_bootstrap_difference(y, good, rand, groups, n_bootstrap=100, seed=3)
    second = paired_bootstrap_difference(y, good, rand, groups, n_bootstrap=100, seed=3)
    other = paired_bootstrap_difference(y, good, rand, groups, n_bootstrap=100, seed=4)
    assert first == second
    assert first["difference_ci"] != other["difference_ci"]


def test_identical_methods_give_exactly_zero_difference():
    y, good, _, groups = _synthetic()
    result = paired_bootstrap_difference(y, good, good, groups, n_bootstrap=50, seed=1)
    assert result["point_difference"] == 0.0
    assert result["difference_ci"] == (0.0, 0.0)
    assert result["difference_ci_excludes_zero"] is False


def test_clearly_better_method_has_interval_above_zero():
    y, good, rand, groups = _synthetic()
    for metric in ("pr_auc", "roc_auc"):
        result = paired_bootstrap_difference(
            y, good, rand, groups, metric=metric, n_bootstrap=200, seed=5
        )
        assert result["difference_ci"][0] > 0
        assert result["difference_ci_excludes_zero"] is True
        assert result["fraction_difference_positive"] == 1.0
        assert result["point_difference"] > 0


def test_brier_difference_is_negative_when_method_a_is_better():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(30), 50)
    y = (rng.random(len(groups)) < 0.3).astype(int)
    a = np.clip(y * 0.8 + 0.1, 0, 1)
    b = np.full(len(y), 0.3)
    result = paired_bootstrap_difference(y, a, b, groups, metric="brier", n_bootstrap=100, seed=1)
    assert result["point_difference"] < 0
    assert result["difference_ci"][1] < 0


def test_replicates_without_both_classes_are_counted_not_hidden():
    groups = np.repeat(np.arange(10), 5)
    y = np.zeros(50, dtype=int)
    y[[0, 1]] = 1
    a = np.linspace(0, 1, 50)
    b = np.linspace(1, 0, 50)
    result = paired_bootstrap_difference(y, a, b, groups, n_bootstrap=200, seed=11)
    assert result["n_invalid"] > 0
    assert result["n_valid"] + result["n_invalid"] == 200
    again = paired_bootstrap_difference(y, a, b, groups, n_bootstrap=200, seed=11)
    assert (again["n_valid"], again["n_invalid"]) == (result["n_valid"], result["n_invalid"])


def test_single_class_test_set_is_rejected_with_a_clear_error():
    groups = np.repeat(np.arange(5), 4)
    y = np.zeros(20, dtype=int)
    scores = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="both classes"):
        paired_bootstrap_difference(y, scores, scores, groups, n_bootstrap=10, seed=1)


def test_time_block_clustering_runs_and_is_wider_than_row_level_resampling():
    rng = np.random.default_rng(4)
    n_blocks, block_size = 40, 10
    block_event = rng.random(n_blocks) < 0.3
    y = np.repeat(block_event.astype(int), block_size)
    a = y + rng.normal(0, 0.8, len(y))
    b = rng.random(len(y))
    block_groups = np.repeat(np.arange(n_blocks), block_size)
    row_groups = np.arange(len(y))

    by_block = paired_bootstrap_difference(y, a, b, block_groups, n_bootstrap=300, seed=1)
    by_row = paired_bootstrap_difference(y, a, b, row_groups, n_bootstrap=300, seed=1)

    width_block = by_block["difference_ci"][1] - by_block["difference_ci"][0]
    width_row = by_row["difference_ci"][1] - by_row["difference_ci"][0]
    assert width_block > width_row


def test_input_validation():
    y, good, rand, groups = _synthetic(n_clusters=5, rows_per_cluster=10)
    with pytest.raises(ValueError):
        paired_bootstrap_difference(y[:-1], good, rand, groups)
    with pytest.raises(ValueError):
        paired_bootstrap_difference(y, good, rand, groups, n_bootstrap=0)
    with pytest.raises(ValueError):
        paired_bootstrap_difference(y, good, rand, groups, confidence=1.5)
