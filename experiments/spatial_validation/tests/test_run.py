"""Tests that the latitude-band holdout selects the right cells and that run_holdout_experiment() runs end to end on synthetic data (no network)."""

import json

import numpy as np
import pandas as pd

from experiments.spatial_validation.run import HOLDOUT_LATS, run_holdout_experiment
from src.terrarisk.models.spatial_holdout import cells_in_bands


def _synthetic(n_days=1500, seed=0):
    rng = np.random.default_rng(seed)
    frames, grid_rows = [], []
    lats = HOLDOUT_LATS + [99.0, 100.0, 101.0, 102.0]
    for k, lat in enumerate(lats):
        cell = f"cell_{lat}"
        grid_rows.append({"cell_id": cell, "lat": lat, "lon": 80.0 + 0.25 * k})
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


def test_holdout_cells_are_exactly_the_held_out_latitudes():
    _, grid = _synthetic()
    assert cells_in_bands(grid, "lat", HOLDOUT_LATS) == {
        f"cell_{lat}" for lat in HOLDOUT_LATS
    }


def test_end_to_end_run_is_complete_json_serialisable_and_reproducible():
    df, grid = _synthetic()
    held = cells_in_bands(grid, "lat", HOLDOUT_LATS)

    first = run_holdout_experiment(
        df, "flood_event", "SYNTH", held, n_bootstrap=10, seed=5, verbose=False
    )
    second = run_holdout_experiment(
        df, "flood_event", "SYNTH", held, n_bootstrap=10, seed=5, verbose=False
    )

    assert first == second
    json.dumps(first)
    assert first["test_cells"] == len(held)
    assert set(first) >= {"model", "pooled_cell_baseline", "seasonal_baseline"}
    for name in ("model", "pooled_cell_baseline", "seasonal_baseline"):
        assert set(first[name]) == {"pr_auc", "roc_auc", "brier"}
    for by_baseline in first["paired_vs_baseline"].values():
        for by_clustering in by_baseline.values():
            for r in by_clustering.values():
                assert r["n_valid"] + r["n_invalid"] == 10


def test_returns_none_when_a_held_out_block_has_a_single_class():
    df, grid = _synthetic()
    held = cells_in_bands(grid, "lat", HOLDOUT_LATS)
    df.loc[df["cell_id"].isin(held), "flood_event"] = 0
    assert run_holdout_experiment(
        df, "flood_event", "SYNTH", held, n_bootstrap=5, verbose=False
    ) is None


def test_neighbour_baselines_appear_only_when_a_grid_is_supplied():
    df, grid = _synthetic()
    held = cells_in_bands(grid, "lat", HOLDOUT_LATS)

    without = run_holdout_experiment(
        df, "flood_event", "SYNTH", held, n_bootstrap=5, seed=1, verbose=False
    )
    with_grid = run_holdout_experiment(
        df, "flood_event", "SYNTH", held, n_bootstrap=5, seed=1, verbose=False, grid=grid
    )

    assert "neighbour_seasonal_baseline" not in without
    assert set(without["paired_vs_baseline"]) == {"seasonal_baseline"}
    assert "neighbour_seasonal_baseline" in with_grid
    assert set(with_grid["paired_vs_baseline"]) == {
        "seasonal_baseline", "neighbour_seasonal_baseline"
    }
    assert with_grid["model"] == without["model"]
    assert with_grid["pooled_cell_baseline"] == without["pooled_cell_baseline"]
