"""Tests the SHAP additivity invariant (base value plus contributions equals the raw model output) and the global-importance ranking."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from experiments.shap_explainability.run import (
    explain_single_row,
    get_shap_values,
    global_shap_summary,
)

def _train_tiny_model():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "feature_a": rng.uniform(0, 100, 200),
        "feature_b": rng.uniform(0, 100, 200),
    })
    y = (X["feature_a"] > 50).astype(int)
    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, random_state=0, verbosity=-1)
    model.fit(X, y)
    return model, X

def test_shap_values_satisfy_additivity():
    model, X = _train_tiny_model()
    shap_values, base_value = get_shap_values(model, X)

    raw_margin = model.predict(X, raw_score=True)

    reconstructed = base_value + shap_values.sum(axis=1)
    assert np.allclose(reconstructed, raw_margin, atol=1e-4)

def test_global_summary_ranks_the_dominant_feature_first():
    model, X = _train_tiny_model()
    shap_values, _ = get_shap_values(model, X)

    summary = global_shap_summary(shap_values, ["feature_a", "feature_b"])
    assert summary.iloc[0]["feature"] == "feature_a"

def test_explain_single_row_contributions_sum_to_raw_output_minus_base():
    model, X = _train_tiny_model()
    shap_values, base_value = get_shap_values(model, X)

    explanation = explain_single_row(0, X, shap_values, base_value, ["feature_a", "feature_b"])
    contribution_sum = sum(c["shap_contribution"] for c in explanation["contributions"])
    assert abs((explanation["base_value"] + contribution_sum) - explanation["raw_model_output"]) < 1e-4

def test_explain_single_row_sorts_by_absolute_contribution():
    model, X = _train_tiny_model()
    shap_values, base_value = get_shap_values(model, X)

    explanation = explain_single_row(0, X, shap_values, base_value, ["feature_a", "feature_b"])
    abs_contributions = [abs(c["shap_contribution"]) for c in explanation["contributions"]]
    assert abs_contributions == sorted(abs_contributions, reverse=True)