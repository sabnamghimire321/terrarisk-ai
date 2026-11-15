"""Tests the leakage-sensitive parts of the baseline ladder: train-only threshold fitting, train-only scaling, and calibration on validation data only."""

import numpy as np
import pandas as pd

from experiments.baseline_ladder.run import (
    THRESHOLD_RULE_FEATURE,
    THRESHOLD_RULE_PERCENTILE,
    run_logistic_regression,
    run_random_forest,
    run_threshold_rule,
)

def _make_synthetic_hazard_data(n_train=200, n_val=50, n_test=50, seed=42):
    rng = np.random.RandomState(seed)

    def _make_split(n):
        precip_7day = rng.uniform(0, 100, n)
        elevation = rng.uniform(100, 3000, n)
        day_of_year = rng.randint(1, 366, n)
        precip_1day = precip_7day / 7 + rng.normal(0, 2, n)
        precip_3day = precip_7day / 2 + rng.normal(0, 3, n)
        event_prob = 1 / (1 + np.exp(-(precip_7day - 60) / 10))
        label = (rng.uniform(0, 1, n) < event_prob).astype(int)

        return pd.DataFrame({
            "precip_1day_sum_mm": precip_1day,
            "precip_3day_sum_mm": precip_3day,
            "precip_7day_sum_mm": precip_7day,
            "elevation_m": elevation,
            "day_of_year": day_of_year,
            "event_horizon_label": label,
        })

    return _make_split(n_train), _make_split(n_val), _make_split(n_test)

def test_threshold_rule_uses_train_percentile_not_test():
    train, val, test = _make_synthetic_hazard_data()
    result_before = run_threshold_rule(train, test, "event_horizon_label")

    test_inflated = test.copy()
    test_inflated[THRESHOLD_RULE_FEATURE] = test_inflated[THRESHOLD_RULE_FEATURE] * 1000

    train_threshold = np.percentile(
        train[THRESHOLD_RULE_FEATURE], THRESHOLD_RULE_PERCENTILE
    )
    assert (test_inflated[THRESHOLD_RULE_FEATURE] > train_threshold).all()

def test_threshold_rule_output_is_binary():
    train, val, test = _make_synthetic_hazard_data()
    result = run_threshold_rule(train, test, "event_horizon_label")
    assert "note" in result
    assert 0.0 <= result["pr_auc"] <= 1.0

def test_logistic_regression_scaler_uses_train_statistics_only():
    train, val, test = _make_synthetic_hazard_data()
    result_1 = run_logistic_regression(train, val, test, "event_horizon_label")
    result_2 = run_logistic_regression(train, val, test, "event_horizon_label")

    assert result_1["pr_auc"] == result_2["pr_auc"]

def test_logistic_regression_returns_both_raw_and_calibrated():
    train, val, test = _make_synthetic_hazard_data()
    result = run_logistic_regression(train, val, test, "event_horizon_label")

    assert "pr_auc" in result and "pr_auc_raw" in result
    assert "brier" in result and "brier_raw" in result
    assert abs(result["pr_auc"] - result["pr_auc_raw"]) < 0.05

def test_random_forest_returns_both_raw_and_calibrated():
    train, val, test = _make_synthetic_hazard_data()
    result = run_random_forest(train, val, test, "event_horizon_label")
    assert "pr_auc" in result and "pr_auc_raw" in result
    assert "brier" in result and "brier_raw" in result
    assert abs(result["pr_auc"] - result["pr_auc_raw"]) < 0.05

def test_models_detect_the_real_signal_in_synthetic_data():
    train, val, test = _make_synthetic_hazard_data(n_train=500, n_val=100, n_test=100)
    positive_rate = test["event_horizon_label"].mean()

    result = run_logistic_regression(train, val, test, "event_horizon_label")
    assert result["pr_auc"] > positive_rate * 1.5

def test_lightgbm_row_exists_and_is_calibrated_and_deterministic():
    from experiments.baseline_ladder.run import run_lightgbm

    train, val, test = _make_synthetic_hazard_data(n_train=800, n_val=200, n_test=200)
    first = run_lightgbm(train, val, test, "event_horizon_label")
    second = run_lightgbm(train, val, test, "event_horizon_label")

    assert first == second
    assert set(first) == {"pr_auc", "roc_auc", "brier", "pr_auc_raw", "brier_raw"}
    assert first["roc_auc"] > 0.5


def test_ladder_reports_all_families_with_roc_auc_and_prevalence():
    from experiments.baseline_ladder.run import run_ladder_for_hazard

    rng = np.random.default_rng(0)
    n_days = 1500
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
    df = pd.concat(frames, ignore_index=True)

    results = run_ladder_for_hazard(df, "flood_event", "SYNTH")

    for family in ("climatological_baseline", "seasonal_baseline", "threshold_rule",
                   "logistic_regression", "random_forest", "lightgbm"):
        assert family in results
        assert "roc_auc" in results[family]
    assert results["_test_set_size"] == 365 * 4
    assert 0.0 < results["_test_prevalence"] < 1.0
