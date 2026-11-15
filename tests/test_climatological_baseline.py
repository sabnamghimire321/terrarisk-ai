"""Tests the landslide climatological baseline: pooling, and that probability scales linearly with horizon length."""

from datetime import datetime, timedelta

import pandas as pd

from src.terrarisk.baseline.climatological_baseline import (
    CATALOG_START,
    compute_baseline,
)

TINY_GRID = pd.DataFrame(
    {
        "cell_id": [f"cell_{i}" for i in range(9)],
        "lat": [26.0, 26.0, 26.0, 26.5, 26.5, 26.5, 27.0, 27.0, 27.0],
        "lon": [80.0, 80.5, 81.0, 80.0, 80.5, 81.0, 80.0, 80.5, 81.0],
    }
)

def test_baseline_has_one_row_per_cell():
    events = pd.DataFrame({"cell_id": ["cell_4"]})
    result = compute_baseline(TINY_GRID, horizon_days=7, events=events, pool_size=9)
    assert len(result) == len(TINY_GRID)

def test_more_events_means_higher_probability():
    few_events = pd.DataFrame({"cell_id": ["cell_4"]})
    many_events = pd.DataFrame({"cell_id": ["cell_4"] * 10})

    result_few = compute_baseline(TINY_GRID, 7, few_events, pool_size=9)
    result_many = compute_baseline(TINY_GRID, 7, many_events, pool_size=9)

    assert result_many["baseline_probability"].mean() > result_few["baseline_probability"].mean()


def test_probability_scales_linearly_with_horizon():
    events = pd.DataFrame({"cell_id": ["cell_4"] * 5})
    result_1day = compute_baseline(TINY_GRID, 1, events, pool_size=9)
    result_7day = compute_baseline(TINY_GRID, 7, events, pool_size=9)
    ratio = (
        result_7day["baseline_probability"].mean()
        / result_1day["baseline_probability"].mean()
    )
    assert abs(ratio - 7) < 0.01

def test_zero_events_gives_zero_probability():
    events = pd.DataFrame({"cell_id": []})
    result = compute_baseline(TINY_GRID, 7, events, pool_size=9)
    assert (result["baseline_probability"] == 0).all()