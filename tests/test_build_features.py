"""Tests build_features.py: per-cell rolling windows, per-cell flood thresholds, and nearest-cell landslide re-joining."""

import pandas as pd

from src.terrarisk.features.build_features import (
    _nearest_cell_id,
    add_flood_labels,
    add_landslide_labels,
    add_rolling_precipitation_features,
)

def test_rolling_precipitation_is_per_cell_not_global():
    df = pd.DataFrame({
        "cell_id": ["cell_a"] * 3 + ["cell_b"] * 3,
        "date": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03"] * 2
        ),
        "precipitation_sum_mm": [10, 10, 10, 100, 100, 100],
    })

    result = add_rolling_precipitation_features(df, windows=[3])

    cell_a_last = result[
        (result["cell_id"] == "cell_a") & (result["date"] == "2020-01-03")
    ]["precip_3day_sum_mm"].iloc[0]
    cell_b_last = result[
        (result["cell_id"] == "cell_b") & (result["date"] == "2020-01-03")
    ]["precip_3day_sum_mm"].iloc[0]

    assert cell_a_last == 30  
    assert cell_b_last == 300  


def test_rolling_precipitation_window_math_is_correct():
    df = pd.DataFrame({
        "cell_id": ["cell_a"] * 5,
        "date": pd.date_range("2020-01-01", periods=5),
        "precipitation_sum_mm": [1, 2, 3, 4, 5],
    })

    result = add_rolling_precipitation_features(df, windows=[3])

    day_5_sum = result[result["date"] == "2020-01-05"]["precip_3day_sum_mm"].iloc[0]
    assert day_5_sum == 12


def test_flood_label_uses_each_cells_own_threshold():
    feature_df = pd.DataFrame({
        "cell_id": ["cell_a", "cell_a", "cell_b", "cell_b"],
        "date": pd.to_datetime(
            ["2020-01-01", "2020-01-02"] * 2
        ),
    })

    discharge_df = pd.DataFrame({
        "cell_id": ["cell_a"] * 20 + ["cell_b"] * 20,
        "date": pd.date_range("2020-01-01", periods=20).tolist() * 2,
        "river_discharge_m3s": (
            [100] + [1] * 19 + [100000] + [1000] * 19
        ),
    })

    result = add_flood_labels(feature_df, discharge_df, percentile=90)

    cell_a_day1 = result[
        (result["cell_id"] == "cell_a") & (result["date"] == "2020-01-01")
    ]["flood_event"].iloc[0]
    cell_b_day1 = result[
        (result["cell_id"] == "cell_b") & (result["date"] == "2020-01-01")
    ]["flood_event"].iloc[0]

    assert cell_a_day1 == 1
    assert cell_b_day1 == 1

def test_nearest_cell_id_picks_the_actually_closest_cell():
    grid = pd.DataFrame({
        "cell_id": ["near", "far"],
        "lat": [10.0, 20.0],
        "lon": [10.0, 20.0],
    })

    result = _nearest_cell_id(lat=10.1, lon=10.1, grid=grid)
    assert result == "near"

def test_landslide_labels_reassign_to_nearest_cell_in_given_grid():
    grid = pd.DataFrame({
        "cell_id": ["cell_a", "cell_b"],
        "lat": [10.0, 50.0],
        "lon": [10.0, 50.0],
    })

    feature_df = pd.DataFrame({
        "cell_id": ["cell_a"],
        "date": pd.to_datetime(["2020-06-15"]),
    })

    events = pd.DataFrame({
        "latitude": [10.05],   
        "longitude": [10.05],
        "event_date": ["06/15/2020 12:00:00 AM"],
    })

    result = add_landslide_labels(feature_df, events, grid)

    cell_a_label = result[
        (result["cell_id"] == "cell_a") & (result["date"] == "2020-06-15")
    ]["landslide_event"].iloc[0]

    assert cell_a_label == 1


def test_landslide_labels_are_zero_when_no_matching_date():
    grid = pd.DataFrame({
        "cell_id": ["cell_a"],
        "lat": [10.0],
        "lon": [10.0],
    })

    feature_df = pd.DataFrame({
        "cell_id": ["cell_a"],
        "date": pd.to_datetime(["2020-06-15"]),
    })

    events = pd.DataFrame({
        "latitude": [10.0],
        "longitude": [10.0],
        "event_date": ["01/01/2019 12:00:00 AM"],
    })

    result = add_landslide_labels(feature_df, events, grid)

    label = result[
        (result["cell_id"] == "cell_a") & (result["date"] == "2020-06-15")
    ]["landslide_event"].iloc[0]

    assert label == 0