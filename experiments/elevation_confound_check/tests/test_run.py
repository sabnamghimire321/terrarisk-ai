"""
Tests for experiments/elevation_confound_check/run.py - focused on
check_constant_per_cell(), the correlation-comparison logic and the
premise check, using small synthetic data with a known, constructed answer.
"""

import numpy as np
import pandas as pd

from experiments.elevation_confound_check.run import (
    check_constant_per_cell,
    compare_correlation_with_realized_rate,
    premise_check,
    train_terrain_only_model,
)

def test_detects_truly_constant_per_cell_predictions():
    test = pd.DataFrame({
        "cell_id": ["a", "a", "a", "b", "b"],
    })
    predictions = np.array([0.5, 0.5, 0.5, 0.2, 0.2])

    result = check_constant_per_cell(test, predictions)
    assert result["is_constant_per_cell"] is True
    assert result["n_distinct_prediction_values"] == 2


def test_detects_non_constant_predictions():
    test = pd.DataFrame({
        "cell_id": ["a", "a", "a"],
    })
    predictions = np.array([0.5, 0.6, 0.4])  

    result = check_constant_per_cell(test, predictions)
    assert result["is_constant_per_cell"] is False


def test_correlation_check_identifies_the_better_correlated_score():
    test = pd.DataFrame({
        "cell_id": ["a", "a", "b", "b", "c", "c"],
        "label": [1, 0, 1, 1, 0, 0],  
    })
    elevation_predictions = np.array([0.5, 0.5, 1.0, 1.0, 0.0, 0.0])
    baseline_predictions = pd.Series([0.9, 0.9, 0.1, 0.1, 0.5, 0.5])

    result = compare_correlation_with_realized_rate(
        test, "label", elevation_predictions, baseline_predictions
    )
    assert result["elevation_correlates_more_closely"] is True


def test_premise_check_holds_only_when_elevation_clearly_beats_baseline_and_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    strong = y * 0.6 + rng.random(2000) * 0.4
    weak = rng.random(2000)

    holds = premise_check(y, strong, weak)
    assert holds["premise_holds"] is True
    assert holds["elevation_only_pr_auc"] > holds["baseline_pr_auc"]

    does_not_hold = premise_check(y, weak, strong)
    assert does_not_hold["premise_holds"] is False
    near_chance = premise_check(y, rng.random(2000), rng.random(2000))
    assert near_chance["premise_holds"] is False
    assert abs(near_chance["test_prevalence"] - y.mean()) < 1e-12


def test_terrain_only_model_uses_the_shared_recipe_and_is_constant_per_cell():
    rng = np.random.default_rng(1)
    rows = []
    for k, cell in enumerate(["a", "b", "c", "d"]):
        n = 600
        rows.append(pd.DataFrame({
            "cell_id": cell,
            "elevation_m": 100.0 * (k + 1),
            "label": (rng.random(n) < 0.1 * (k + 1)).astype(int),
        }))
    data = pd.concat(rows, ignore_index=True)
    train, val = data.iloc[::2], data.iloc[1::2]

    calibrator = train_terrain_only_model(train, val, "label")
    predictions = calibrator.predict_proba(val[["elevation_m"]])[:, 1]

    assert check_constant_per_cell(val, predictions)["is_constant_per_cell"] is True
    assert predictions.min() >= 0.0 and predictions.max() <= 1.0
