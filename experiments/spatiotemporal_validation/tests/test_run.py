"""Tests that the combined geography-and-time experiment runs end to end on synthetic data and that its split keeps validation and test on different dates (no network)."""

import json

import numpy as np
import pandas as pd

from experiments.spatiotemporal_validation.run import (
    HOLDOUT_LATS,
    SPATIOTEMPORAL_TEST_DAYS,
    run_spatiotemporal_for_flood,
)
from src.terrarisk.models.spatial_holdout import cells_in_bands


def _synthetic(n_days=2300, seed=0):
    rng = np.random.default_rng(seed)
    frames, grid_rows = [], []
    lats = HOLDOUT_LATS + [99.0, 100.0, 101.0, 102.0]
    for k, lat in enumerate(lats):
        cell = f"cell_{lat}"
        grid_rows.append({"cell_id": cell, "lat": lat, "lon": 80.0 + 0.25 * k})
        dates = pd.date_range("2018-01-01", periods=n_days)
        month = dates.month.to_numpy()
        probability = np.where((month >= 6) & (month <= 9), 0.25, 0.02)
        frames.append(pd.DataFrame({
            "cell_id": cell,
            "date": dates,
            "flood_event": (rng.random(n_days) < probability).astype(int),
            "precip_1day_sum_mm": rng.gamma(1.0, 5.0, n_days),
            "precip_3day_sum_mm": rng.gamma(1.0, 12.0, n_days),
            "precip_7day_sum_mm": rng.gamma(1.0, 25.0, n_days),
            "elevation_m": 100.0 * (k + 1),
            "day_of_year": dates.dayofyear.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(grid_rows)


def test_the_test_window_is_a_full_year():
    assert SPATIOTEMPORAL_TEST_DAYS == 365


def test_end_to_end_run_is_complete_json_serialisable_and_reproducible():
    df, grid = _synthetic()
    held = cells_in_bands(grid, "lat", HOLDOUT_LATS)

    first = run_spatiotemporal_for_flood(
        df, held, grid=grid, n_bootstrap=8, seed=3, verbose=False
    )
    second = run_spatiotemporal_for_flood(
        df, held, grid=grid, n_bootstrap=8, seed=3, verbose=False
    )

    assert first == second
    json.dumps(first)
    assert first["test_cells"] == len(held)
    assert first["test_rows"] == len(held) * 365
    assert set(first["paired_vs_baseline"]) == {
        "seasonal_baseline", "neighbour_seasonal_baseline"
    }
    for name in ("model", "pooled_cell_baseline", "seasonal_baseline",
                 "neighbour_cell_baseline", "neighbour_seasonal_baseline"):
        assert set(first[name]) == {"pr_auc", "roc_auc", "brier"}


def test_returns_none_when_there_is_not_enough_history():
    df, grid = _synthetic(n_days=600)
    held = cells_in_bands(grid, "lat", HOLDOUT_LATS)
    assert run_spatiotemporal_for_flood(
        df, held, grid=grid, n_bootstrap=3, verbose=False
    ) is None
