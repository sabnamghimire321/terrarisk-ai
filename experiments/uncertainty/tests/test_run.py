"""
Tests for the uncertainty experiment: whole-cell resampling, NaN-safe
percentile intervals, the top-k capture metric, and a small end-to-end
run on synthetic data (no network, no real datasets).
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.uncertainty.run import (
    block_bootstrap_indices,
    percentile_ci,
    run_uncertainty_for_hazard,
    top_k_capture_rate,
)


def test_bootstrap_resamples_whole_cells_not_individual_rows():
    cell_ids = pd.Series(["cell_a", "cell_a", "cell_a", "cell_b", "cell_b"])
    positions = block_bootstrap_indices(cell_ids, seed=0)

    from collections import Counter
    counts = Counter(positions)
    cell_a_count_0 = counts.get(0, 0)
    cell_a_count_1 = counts.get(1, 0)
    cell_a_count_2 = counts.get(2, 0)
    assert cell_a_count_0 == cell_a_count_1 == cell_a_count_2


def test_bootstrap_output_size_matches_expected_when_no_duplicates():
    cell_ids = pd.Series(["cell_a"] * 3 + ["cell_b"] * 2 + ["cell_c"] * 4)
    positions = block_bootstrap_indices(cell_ids, seed=1)
    assert len(positions) > 0


def test_different_seeds_produce_different_samples():
    cell_ids = pd.Series([f"cell_{i}" for i in range(20)])
    positions_1 = block_bootstrap_indices(cell_ids, seed=1)
    positions_2 = block_bootstrap_indices(cell_ids, seed=2)
    assert not np.array_equal(positions_1, positions_2)


def test_percentile_ci_excludes_nan_values():
    values = np.array([0.1, 0.2, 0.3, np.nan, 0.4, np.nan, 0.5])
    lower, upper, n_valid = percentile_ci(values, confidence=0.95)
    assert n_valid == 5
    assert 0.1 <= lower <= upper <= 0.5


def test_percentile_ci_narrow_for_tight_distribution():
    values = np.full(100, 0.5)
    lower, upper, n_valid = percentile_ci(values)
    assert abs(lower - 0.5) < 1e-9
    assert abs(upper - 0.5) < 1e-9


def test_top_k_capture_rate_exact_value():
    y = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    assert top_k_capture_rate(y, scores, 50) == 0.5


def test_top_k_capture_rate_breaks_ties_deterministically_by_row_order():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert top_k_capture_rate(y, scores, 25) == 1.0
    y_last = np.array([0, 0, 0, 1])
    assert top_k_capture_rate(y_last, scores, 25) == 0.0


def test_top_k_capture_rate_is_nan_without_positives():
    assert np.isnan(top_k_capture_rate(np.zeros(5), np.arange(5), 40))


def _seasonal_frame(n_days=1500, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for k, cell in enumerate(["a", "b", "c", "d"]):
        dates = pd.date_range("2020-01-01", periods=n_days)
        month = dates.month.to_numpy()
        probability = np.where((month >= 6) & (month <= 9), 0.25, 0.02)
        frames.append(pd.DataFrame({
            "cell_id": cell,
            "date": dates,
            "flood_event": (rng.random(n_days) < probability).astype(int),
            "precip_1day_sum_mm": rng.gamma(1.0, 5.0, n_days),
            "precip_3day_sum_mm": rng.gamma(1.0, 12.0, n_days),
            "precip_7day_sum_mm": rng.gamma(1.0, 25.0, n_days),
            "elevation_m": 100.0 * (k + 1),
            "day_of_year": dates.dayofyear.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def test_end_to_end_run_on_synthetic_data_is_complete_json_and_reproducible():
    df = _seasonal_frame()
    first = run_uncertainty_for_hazard(
        df, "flood_event", "SYNTH", n_bootstrap=20, seed=7, verbose=False
    )
    second = run_uncertainty_for_hazard(
        df, "flood_event", "SYNTH", n_bootstrap=20, seed=7, verbose=False
    )

    assert first["test_rows"] == 365 * 4
    assert first["n_bootstrap"] == 20 and first["seed"] == 7
    assert set(first["point_metrics"]) == {"model", "cell_rate_baseline", "seasonal_baseline"}
    assert set(first["paired_differences"]) == {"cell_rate_baseline", "seasonal_baseline"}
    for by_clustering in first["paired_differences"].values():
        assert set(by_clustering) == {"by_cell", "by_14day_block"}
        for per_metric in by_clustering.values():
            assert set(per_metric) == {"pr_auc", "roc_auc", "brier", "top10pct_capture"}
            for r in per_metric.values():
                assert r["n_valid"] + r["n_invalid"] == 20

    json.dumps(first)
    assert first == second


def test_run_returns_none_when_data_cannot_support_one_year_windows():
    df = _seasonal_frame(n_days=400)
    assert run_uncertainty_for_hazard(df, "flood_event", "SYNTH", verbose=False) is None


def test_horizon_and_feature_list_are_honoured():
    df = _seasonal_frame()
    result = run_uncertainty_for_hazard(
        df, "flood_event", "SYNTH", n_bootstrap=10, seed=1, verbose=False,
        horizon_days=3, feature_columns=["day_of_year", "elevation_m"],
    )
    assert result["horizon_days"] == 3
    assert result["features_used"] == ["day_of_year", "elevation_m"]
    assert result["test_rows"] == 365 * 4


def test_paired_comparison_block_shape_subset_and_determinism():
    from experiments.uncertainty.run import paired_comparison_block

    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.2).astype(int)
    model_scores = y * 0.5 + rng.random(n) * 0.5
    baseline_scores = rng.random(n)
    cells = np.repeat(np.arange(20), 100)
    dates = pd.Series(pd.date_range("2020-01-01", periods=100)).iloc[np.tile(np.arange(100), 20)]

    kwargs = dict(n_bootstrap=15, seed=2, metric_keys=("pr_auc", "brier"))
    first = paired_comparison_block(y, model_scores, baseline_scores, cells, dates, **kwargs)
    second = paired_comparison_block(y, model_scores, baseline_scores, cells, dates, **kwargs)

    assert first == second
    assert set(first) == {"by_cell", "by_14day_block"}
    for per_metric in first.values():
        assert set(per_metric) == {"pr_auc", "brier"}
