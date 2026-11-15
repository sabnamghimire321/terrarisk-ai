"""Tests the flood threshold-exceedance baseline: per-cell percentile thresholds and windowed exceedance counting."""

import numpy as np
import pandas as pd

from src.terrarisk.baseline.flood_baseline import (
    compute_cell_thresholds,
    compute_flood_baseline,
)

def _make_constant_discharge(cell_id: str, n_days: int, value: float) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "date": dates.astype(str),
            "river_discharge_m3s": value,
        }
    )

def test_threshold_is_correct_percentile():
    df = pd.DataFrame(
        {
            "cell_id": ["cell_a"] * 100,
            "date": pd.date_range("2020-01-01", periods=100).astype(str),
            "river_discharge_m3s": np.arange(100),
        }
    )
    thresholds = compute_cell_thresholds(df, percentile=90)
    assert abs(thresholds["cell_a"] - 89.1) < 1.0


def test_constant_discharge_never_exceeds_its_own_threshold():
    df = _make_constant_discharge("cell_a", n_days=100, value=5.0)
    result = compute_flood_baseline(df, horizon_days=1)
    assert result["baseline_probability"].iloc[0] == 0.0


def test_one_day_spike_creates_exactly_one_event_at_horizon_one():
    df = _make_constant_discharge("cell_a", n_days=100, value=5.0)
    df.loc[50, "river_discharge_m3s"] = 1000.0

    result = compute_flood_baseline(df, horizon_days=1)
    window_count = result["window_count"].iloc[0]
    probability = result["baseline_probability"].iloc[0]
    assert abs(probability - (1 / window_count)) < 1e-9


def test_longer_horizon_gives_higher_or_equal_probability():
    df = _make_constant_discharge("cell_a", n_days=90, value=5.0)
    df.loc[10, "river_discharge_m3s"] = 1000.0
    df.loc[40, "river_discharge_m3s"] = 1000.0
    df.loc[70, "river_discharge_m3s"] = 1000.0

    result_1day = compute_flood_baseline(df, horizon_days=1)
    result_7day = compute_flood_baseline(df, horizon_days=7)
    assert (
        result_7day["baseline_probability"].iloc[0]
        >= result_1day["baseline_probability"].iloc[0]
    )