"""
Tests for the rolling-origin experiment on synthetic multi-year data
(no network, no real datasets): exact fold dates, no fold's training
data reaching its own test window, determinism and serialisability.
"""

import json

import numpy as np
import pandas as pd

from experiments.rolling_origin.run import (
    fold_last_label_dates,
    run_rolling_origin,
)

HORIZON = 7


def _frame(n_days=3200, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for k, cell in enumerate(["a", "b", "c", "d"]):
        dates = pd.date_range("2010-01-01", periods=n_days)
        month = dates.month.to_numpy()
        probability = np.where((month >= 6) & (month <= 9), 0.25, 0.02)
        frames.append(pd.DataFrame({
            "cell_id": cell,
            "date": dates,
            "flood_event": (rng.random(n_days) < probability).astype(int),
            "precip_1day_sum_mm": rng.gamma(1.0, 5.0, n_days),
            "precip_3day_sum_mm": rng.gamma(1.0, 12.0, n_days),
            "precip_7day_sum_mm": rng.gamma(1.0, 25.0, n_days),
            "precip_14day_sum_mm": rng.gamma(1.0, 50.0, n_days),
            "precip_30day_sum_mm": rng.gamma(1.0, 100.0, n_days),
            "elevation_m": 100.0 * (k + 1),
            "day_of_year": dates.dayofyear.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def test_fold_anchors_step_back_exactly_one_test_window_each():
    last = pd.Timestamp("2026-09-02")
    anchors = fold_last_label_dates(last, 3, 365)
    assert anchors == [
        pd.Timestamp("2026-09-02"),
        pd.Timestamp("2026-09-02") - pd.Timedelta(days=365),
        pd.Timestamp("2026-09-02") - pd.Timedelta(days=730),
    ]


def test_folds_have_exact_disjoint_test_windows_and_clean_boundaries():
    df = _frame()
    result = run_rolling_origin(df, n_folds=3, n_bootstrap=5, seed=1, verbose=False)

    assert result["n_folds_run"] == 3
    last_label = df["date"].max() - pd.Timedelta(days=HORIZON)

    previous_test_start = None
    for index, fold in enumerate(result["folds"]):
        test_end = pd.Timestamp(fold["test_end"])
        test_start = pd.Timestamp(fold["test_start"])
        assert test_end == last_label - pd.Timedelta(days=365 * index)
        assert (test_end - test_start).days == 364
        assert fold["test_rows"] == 365 * 4

        assert pd.Timestamp(fold["train_end"]) + pd.Timedelta(days=HORIZON) < pd.Timestamp(fold["val_start"])
        assert pd.Timestamp(fold["val_end"]) + pd.Timedelta(days=HORIZON) < test_start

        if previous_test_start is not None:
            assert test_end < previous_test_start
            assert (previous_test_start - test_end).days == 1
        previous_test_start = test_start


def test_results_are_deterministic_serialisable_and_complete():
    df = _frame()
    first = run_rolling_origin(df, n_folds=2, n_bootstrap=5, seed=3, verbose=False)
    second = run_rolling_origin(df, n_folds=2, n_bootstrap=5, seed=3, verbose=False)

    json.dumps(first)
    assert first == second
    assert "_arrays" not in first["folds"][0]
    for fold in first["folds"]:
        assert set(fold["feature_sets"]) == {"original", "extended"}
        for r in fold["feature_sets"].values():
            assert set(r["model"]) == {"pr_auc", "roc_auc", "brier"}
            assert set(r["paired_vs_seasonal"]) == {"by_cell", "by_14day_block"}

    pooled = first["pooled"]
    assert set(pooled) == {"original", "extended"}
    for per_metric in pooled.values():
        assert isinstance(per_metric["shows_skill_beyond_season"], bool)
        for metric in ("pr_auc", "roc_auc", "brier"):
            r = per_metric[metric]
            assert r["n_folds"] == 2
            assert 0 <= r["folds_favouring_model"] <= 2
            assert r["n_valid"] + r["n_invalid"] == 5


def test_stops_cleanly_when_older_folds_do_not_fit_in_the_data():
    df = _frame(n_days=1600)
    result = run_rolling_origin(df, n_folds=5, n_bootstrap=3, seed=1, verbose=False)
    assert 1 <= result["n_folds_run"] < 5
    assert len(result["folds"]) == result["n_folds_run"]


def test_selected_weighting_mode_records_a_validation_only_choice_per_fit():
    df = _frame()
    result = run_rolling_origin(
        df, n_folds=2, n_bootstrap=5, seed=3, verbose=False, weighting="selected"
    )
    assert result["weighting"] == "selected"
    json.dumps(result)
    for fold in result["folds"]:
        for r in fold["feature_sets"].values():
            details = r["class_weighting"]
            assert isinstance(details["used_class_weight"], bool)
            unweighted_better = (
                details["unweighted_validation_pr_auc"] > details["weighted_validation_pr_auc"]
            )
            assert details["used_class_weight"] == (not unweighted_better)


def test_default_weighting_mode_is_the_unchanged_primary_analysis():
    df = _frame()
    result = run_rolling_origin(df, n_folds=1, n_bootstrap=3, seed=1, verbose=False)
    assert result["weighting"] == "weighted"
    for r in result["folds"][0]["feature_sets"].values():
        assert r["class_weighting"] is None


def test_unknown_weighting_mode_is_rejected():
    import pytest

    from experiments.rolling_origin.run import score_with_weighting

    with pytest.raises(ValueError):
        score_with_weighting(None, None, None, "x", [], "magic")
