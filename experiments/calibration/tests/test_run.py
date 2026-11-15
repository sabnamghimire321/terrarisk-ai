"""Tests the ECE implementation and evaluate_variant()'s handling of degenerate cases (zero predicted positives, a single-class test set)."""

import numpy as np

from experiments.calibration.run import (
    compute_reliability_table,
    evaluate_variant,
    expected_calibration_error,
)

def test_ece_is_zero_for_perfect_calibration():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  
    y_prob = np.full(10, 0.6)  
    ece = expected_calibration_error(y_true, y_prob)
    assert ece < 0.01

def test_ece_is_high_for_badly_miscalibrated_predictions():
    y_true = np.array([0] * 9 + [1])
    y_prob = np.full(10, 0.95)
    ece = expected_calibration_error(y_true, y_prob)
    assert ece > 0.5

def test_reliability_table_bin_counts_sum_to_total():
    y_true = np.random.RandomState(0).randint(0, 2, 100)
    y_prob = np.random.RandomState(1).uniform(0, 1, 100)
    table = compute_reliability_table(y_true, y_prob, n_bins=10)
    assert table["count"].sum() == 100

def test_evaluate_variant_handles_zero_predicted_positives():
    y_true = np.array([0, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.1, 0.2])
    result = evaluate_variant(y_true, y_prob)
    assert result["n_predicted_positive_at_0.5"] == 0
    assert result["precision_at_0.5"] == 0.0
    assert result["recall_at_0.5"] == 0.0

def test_evaluate_variant_handles_single_class_test_set():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.15, 0.05])
    result = evaluate_variant(y_true, y_prob)
    assert np.isnan(result["roc_auc"])

def test_train_lightgbm_uses_shared_recipe_and_only_weighting_varies():
    import pandas as pd

    from experiments.calibration.run import train_lightgbm

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=1500), "b": rng.normal(size=1500)})
    y = pd.Series((X["a"] + rng.normal(size=1500) > 1.5).astype(int))
    X_val = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y_val = pd.Series((X_val["a"] + rng.normal(size=500) > 1.5).astype(int))

    unweighted = train_lightgbm(X, y, X_val, y_val, use_weighting=False)
    weighted = train_lightgbm(X, y, X_val, y_val, use_weighting=True)

    assert unweighted.get_params()["scale_pos_weight"] == 1.0
    assert weighted.get_params()["scale_pos_weight"] == (len(y) - y.sum()) / y.sum()
    for model in (unweighted, weighted):
        assert model.get_params()["n_estimators"] == 200
        assert model.get_params()["num_leaves"] == 15
        assert model.get_params()["learning_rate"] == 0.05
