"""Tests the two perturbation functions: missingness produces the requested fraction and is reproducible; precipitation noise only touches precipitation columns and never goes negative."""

import numpy as np
import pandas as pd

from experiments.robustness.run import inject_missingness, inject_precip_noise

def test_missingness_produces_approximately_the_requested_fraction():
    X = pd.DataFrame(np.ones((1000, 5)), columns=[f"f{i}" for i in range(5)])
    X_perturbed = inject_missingness(X, missing_frac=0.20, seed=0)
    actual_frac = X_perturbed.isna().sum().sum() / X.size
    assert abs(actual_frac - 0.20) < 0.02  

def test_missingness_is_reproducible_with_same_seed():
    X = pd.DataFrame(np.ones((100, 3)), columns=["a", "b", "c"])
    result_1 = inject_missingness(X, 0.1, seed=7)
    result_2 = inject_missingness(X, 0.1, seed=7)
    pd.testing.assert_frame_equal(result_1, result_2)

def test_zero_missingness_leaves_data_unchanged():
    X = pd.DataFrame(np.random.RandomState(0).uniform(0, 100, (50, 4)),
                      columns=["a", "b", "c", "d"])
    result = inject_missingness(X, missing_frac=0.0, seed=0)
    pd.testing.assert_frame_equal(result, X)

def test_precip_noise_only_affects_precip_columns():
    X = pd.DataFrame({
        "precip_1day_sum_mm": [10.0] * 100,
        "precip_3day_sum_mm": [20.0] * 100,
        "precip_7day_sum_mm": [50.0] * 100,
        "elevation_m": [1000.0] * 100,
        "day_of_year": [150] * 100,
    })
    result = inject_precip_noise(X, relative_frac=0.5, seed=0)

    pd.testing.assert_series_equal(result["elevation_m"], X["elevation_m"])
    pd.testing.assert_series_equal(result["day_of_year"], X["day_of_year"])
    assert not result["precip_1day_sum_mm"].equals(X["precip_1day_sum_mm"])


def test_precip_noise_never_produces_negative_values():
    X = pd.DataFrame({
        "precip_1day_sum_mm": [1.0] * 200,  
        "precip_3day_sum_mm": [1.0] * 200,
        "precip_7day_sum_mm": [1.0] * 200,
    })
    result = inject_precip_noise(X, relative_frac=2.0, seed=0)  
    assert (result[["precip_1day_sum_mm", "precip_3day_sum_mm", "precip_7day_sum_mm"]] >= 0).all().all()