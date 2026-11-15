"""Tests the leakage-sensitive fit/transform pair this experiment depends on: climatology fit on train only, applied unchanged to other splits."""

import pandas as pd

from experiments.richer_precip_features.run import (
    apply_precip_anomaly,
    compute_train_climatology,
)

def test_climatology_uses_only_train_rows():
    train = pd.DataFrame({
        "cell_id": ["a", "a", "a"],
        "date": pd.to_datetime(["2020-01-05", "2020-01-10", "2020-01-15"]),
        "precip_7day_sum_mm": [10.0, 20.0, 30.0],
    })
    climatology = compute_train_climatology(train)
    assert abs(climatology[("a", 1)] - 20.0) < 1e-9

def test_anomaly_is_zero_when_value_equals_climatology():
    train = pd.DataFrame({
        "cell_id": ["a", "a"],
        "date": pd.to_datetime(["2020-01-05", "2020-01-10"]),
        "precip_7day_sum_mm": [10.0, 10.0],
    })
    climatology = compute_train_climatology(train)

    test_row = pd.DataFrame({
        "cell_id": ["a"],
        "date": pd.to_datetime(["2021-01-20"]),  
        "precip_7day_sum_mm": [10.0],  
    })
    result = apply_precip_anomaly(test_row, climatology)
    assert abs(result["precip_7day_anomaly_mm"].iloc[0]) < 1e-9

def test_anomaly_is_positive_when_value_exceeds_climatology():
    train = pd.DataFrame({
        "cell_id": ["a"],
        "date": pd.to_datetime(["2020-01-05"]),
        "precip_7day_sum_mm": [10.0],
    })
    climatology = compute_train_climatology(train)

    test_row = pd.DataFrame({
        "cell_id": ["a"],
        "date": pd.to_datetime(["2021-01-20"]),
        "precip_7day_sum_mm": [50.0],  
    })
    result = apply_precip_anomaly(test_row, climatology)
    assert result["precip_7day_anomaly_mm"].iloc[0] > 0

def test_unseen_cell_month_falls_back_to_overall_mean_not_nan():
    train = pd.DataFrame({
        "cell_id": ["a", "b"],
        "date": pd.to_datetime(["2020-01-05", "2020-06-05"]),
        "precip_7day_sum_mm": [10.0, 30.0],
    })
    climatology = compute_train_climatology(train)

    test_row = pd.DataFrame({
        "cell_id": ["a"],
        "date": pd.to_datetime(["2021-06-15"]),
        "precip_7day_sum_mm": [15.0],
    })
    result = apply_precip_anomaly(test_row, climatology)
    assert not result["precip_7day_anomaly_mm"].isna().any()

def test_climatology_never_sees_val_or_test_rows():
    train = pd.DataFrame({
        "cell_id": ["a"],
        "date": pd.to_datetime(["2020-01-05"]),
        "precip_7day_sum_mm": [10.0],
    })
    val = pd.DataFrame({
        "cell_id": ["a"],
        "date": pd.to_datetime(["2020-01-06"]),
        "precip_7day_sum_mm": [1000.0],  
    })
    climatology = compute_train_climatology(train)
    assert abs(climatology[("a", 1)] - 10.0) < 1e-9

def test_end_to_end_paired_comparison_against_seasonal_baseline_on_synthetic_data():
    import json

    import numpy as np

    from experiments.richer_precip_features.run import (
        PAIRED_METRIC_KEYS,
        run_for_hazard,
    )

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
            "precip_14day_sum_mm": rng.gamma(1.0, 50.0, n_days),
            "precip_30day_sum_mm": rng.gamma(1.0, 100.0, n_days),
            "elevation_m": 100.0 * (k + 1),
            "day_of_year": dates.dayofyear.to_numpy(),
        }))
    df = pd.concat(frames, ignore_index=True)

    result = run_for_hazard(df, "flood_event", "SYNTH", n_bootstrap=10, seed=3)

    paired = result["paired_vs_seasonal_baseline"]
    assert set(paired) == {"original", "windows_only", "extended"}
    for by_clustering in paired.values():
        assert set(by_clustering) == {"by_cell", "by_14day_block"}
        for per_metric in by_clustering.values():
            assert set(per_metric) == set(PAIRED_METRIC_KEYS)
            for r in per_metric.values():
                assert r["n_valid"] + r["n_invalid"] == 10
    assert result["n_bootstrap"] == 10 and result["seed"] == 3
    assert 0.0 < result["test_prevalence"] < 1.0
    json.dumps(result)
