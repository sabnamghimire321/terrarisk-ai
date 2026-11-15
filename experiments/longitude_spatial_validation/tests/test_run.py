"""Mirrors spatial_validation's tests on the longitude axis: the held-out cells are selected by longitude and the shared runner works with them."""

import numpy as np
import pandas as pd

from experiments.longitude_spatial_validation.run import HOLDOUT_LONS
from experiments.spatial_validation.run import run_holdout_experiment
from src.terrarisk.models.spatial_holdout import cells_in_bands


def _synthetic(n_days=1500, seed=1):
    rng = np.random.default_rng(seed)
    frames, grid_rows = [], []
    lons = HOLDOUT_LONS + [99.0, 100.0, 101.0, 102.0]
    for k, lon in enumerate(lons):
        cell = f"cell_{lon}"
        grid_rows.append({"cell_id": cell, "lon": lon})
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
            "elevation_m": 100.0 * (k + 1),
            "day_of_year": dates.dayofyear.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(grid_rows)


def test_holdout_cells_are_exactly_the_held_out_longitudes():
    _, grid = _synthetic()
    assert cells_in_bands(grid, "lon", HOLDOUT_LONS) == {
        f"cell_{lon}" for lon in HOLDOUT_LONS
    }


def test_shared_runner_works_on_the_longitude_holdout():
    df, grid = _synthetic()
    held = cells_in_bands(grid, "lon", HOLDOUT_LONS)
    result = run_holdout_experiment(
        df, "flood_event", "SYNTH", held, n_bootstrap=5, seed=2, verbose=False
    )
    assert result["test_cells"] == len(HOLDOUT_LONS)
    assert 0.0 < result["test_prevalence"] < 1.0
