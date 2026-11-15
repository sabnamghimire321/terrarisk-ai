"""Tests recompute_flood_labels(): stricter thresholds should never produce more positives, and unrelated columns must be unaffected. Also an end-to-end run on synthetic data (no network)."""

import json

import numpy as np
import pandas as pd

from experiments.label_sensitivity.run import recompute_flood_labels, run_for_percentile

def _make_discharge_and_features():
    dates = pd.date_range("2020-01-01", periods=100)
    discharge = pd.DataFrame({
        "cell_id": "cell_a",
        "date": dates.astype(str),
        "river_discharge_m3s": [100.0 if i % 10 == 0 else 1.0 for i in range(100)],
    })
    features = pd.DataFrame({
        "cell_id": "cell_a",
        "date": dates,
        "river_discharge_m3s": discharge["river_discharge_m3s"].values,
        "flood_event": 0,  
    })
    return discharge, features

def test_higher_percentile_produces_fewer_or_equal_positive_labels():
    discharge, features = _make_discharge_and_features()

    relabeled_loose = recompute_flood_labels(features, discharge, percentile=50)
    relabeled_strict = recompute_flood_labels(features, discharge, percentile=95)

    assert relabeled_strict["flood_event"].sum() <= relabeled_loose["flood_event"].sum()

def test_recompute_does_not_mutate_the_original_dataframe():
    discharge, features = _make_discharge_and_features()
    original_events = features["flood_event"].copy()

    recompute_flood_labels(features, discharge, percentile=90)

    pd.testing.assert_series_equal(features["flood_event"], original_events)


def test_other_columns_are_unaffected_by_relabeling():
    discharge, features = _make_discharge_and_features()
    features["elevation_m"] = 500.0

    relabeled = recompute_flood_labels(features, discharge, percentile=90)

    assert (relabeled["elevation_m"] == 500.0).all()
    assert (relabeled["river_discharge_m3s"] == features["river_discharge_m3s"]).all()


def _seasonal_synthetic(n_days=1500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days)
    month = dates.month.to_numpy()
    frames = []
    for k, cell in enumerate(["a", "b", "c", "d"]):
        discharge = 10 + np.where((month >= 6) & (month <= 9), 40, 0) + rng.gamma(2.0, 3.0, n_days)
        frames.append(pd.DataFrame({
            "cell_id": cell,
            "date": dates,
            "river_discharge_m3s": discharge,
            "flood_event": 0,
            "precip_1day_sum_mm": rng.gamma(1.0, 5.0, n_days),
            "precip_3day_sum_mm": rng.gamma(1.0, 12.0, n_days),
            "precip_7day_sum_mm": rng.gamma(1.0, 25.0, n_days),
            "elevation_m": 100.0 * (k + 1),
            "day_of_year": dates.dayofyear.to_numpy(),
        }))
    features = pd.concat(frames, ignore_index=True)
    discharge_df = features[["cell_id", "date", "river_discharge_m3s"]].copy()
    discharge_df["date"] = discharge_df["date"].astype(str)
    return features, discharge_df


def test_run_for_percentile_reports_prevalence_and_both_baselines():
    features, discharge_df = _seasonal_synthetic()
    result = run_for_percentile(features, discharge_df, percentile=90)

    assert result["evaluable"] is True
    for key in ("model", "cell_rate_baseline", "seasonal_baseline"):
        assert set(result[key]) == {"pr_auc", "roc_auc", "brier"}
    assert 0.0 < result["test_prevalence"] <= 1.0
    assert result["test_prevalence"] != result["same_day_positive_rate"]
    assert result["test_positives"] == round(result["test_prevalence"] * 365 * 4)
    json.dumps(result)


def test_run_for_percentile_is_not_evaluable_when_data_is_too_short():
    features, discharge_df = _seasonal_synthetic(n_days=400)
    assert run_for_percentile(features, discharge_df, percentile=90) == {
        "percentile": 90, "evaluable": False,
    }
