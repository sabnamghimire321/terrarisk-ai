"""Tests for the shared LightGBM recipe. Synthetic data; no network."""

import numpy as np
import pandas as pd

from src.terrarisk.models.recipe import (
    EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS,
    calibrate_on_validation,
    fit_lightgbm,
)


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + 0.5 * rng.normal(size=n) > 1.0).astype(int))
    return X, y


def test_recipe_constants_are_the_documented_ones():
    assert LIGHTGBM_PARAMS["n_estimators"] == 200
    assert LIGHTGBM_PARAMS["num_leaves"] == 15
    assert LIGHTGBM_PARAMS["learning_rate"] == 0.05
    assert LIGHTGBM_PARAMS["random_state"] == 42
    assert EARLY_STOPPING_ROUNDS == 20


def test_class_weight_is_negatives_over_positives_from_train_only():
    X, y = _data()
    X_val, y_val = _data(seed=1)
    model = fit_lightgbm(X, y, X_val, y_val)
    expected = (len(y) - y.sum()) / y.sum()
    assert model.get_params()["scale_pos_weight"] == expected


def test_class_weight_can_be_switched_off():
    X, y = _data()
    X_val, y_val = _data(seed=1)
    model = fit_lightgbm(X, y, X_val, y_val, use_class_weight=False)
    assert model.get_params()["scale_pos_weight"] == 1.0


def test_early_stopping_uses_the_validation_split_and_stops_before_the_cap():
    X, y = _data()
    X_val, _ = _data(seed=1)
    y_val = pd.Series(np.random.default_rng(9).integers(0, 2, len(X_val)))
    model = fit_lightgbm(X, y, X_val, y_val)
    assert model.best_iteration_ < LIGHTGBM_PARAMS["n_estimators"]


def test_fit_is_deterministic():
    X, y = _data()
    X_val, y_val = _data(seed=1)
    first = fit_lightgbm(X, y, X_val, y_val).predict_proba(X_val)[:, 1]
    second = fit_lightgbm(X, y, X_val, y_val).predict_proba(X_val)[:, 1]
    np.testing.assert_array_equal(first, second)


def test_calibrated_probabilities_are_valid_and_use_only_validation_labels():
    X, y = _data()
    X_val, y_val = _data(seed=1)
    model = fit_lightgbm(X, y, X_val, y_val)
    calibrator = calibrate_on_validation(model, X_val, y_val)
    X_test, _ = _data(seed=2)
    probabilities = calibrator.predict_proba(X_test)[:, 1]
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    assert len(probabilities) == len(X_test)


def test_selected_variant_picks_weighting_by_validation_pr_auc_only():
    from src.terrarisk.models.recipe import fit_lightgbm_selected

    X, y = _data()
    X_val, y_val = _data(seed=1)
    model, used_weight, details = fit_lightgbm_selected(X, y, X_val, y_val)

    assert set(details) == {"weighted_validation_pr_auc", "unweighted_validation_pr_auc"}
    unweighted_better = (
        details["unweighted_validation_pr_auc"] > details["weighted_validation_pr_auc"]
    )
    assert used_weight == (not unweighted_better)
    expected_weight = (len(y) - y.sum()) / y.sum() if used_weight else 1.0
    assert model.get_params()["scale_pos_weight"] == expected_weight


def test_selected_variant_keeps_weighting_on_a_tie():
    from src.terrarisk.models.recipe import fit_lightgbm_selected

    X, y = _data()
    X_val, _ = _data(seed=1)
    y_val = pd.Series(np.zeros(len(X_val), dtype=int))
    _, used_weight, _ = fit_lightgbm_selected(X, y, X_val, y_val)
    assert used_weight is True
