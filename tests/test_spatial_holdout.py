"""
Leakage tests for the spatial-holdout split. Synthetic data only; no
network access. Exact dates, exact cell sets and exact baseline values.
"""

import numpy as np
import pandas as pd

from src.terrarisk.models.spatial_holdout import cells_in_bands, spatial_holdout_split
from src.terrarisk.models.train_model import (
    TRAIN_ONLY_BASELINE_COL,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
)

START = pd.Timestamp("2020-01-01")
N_DAYS = 1200
HORIZON = 7
HELD = ["h1", "h2"]
OTHERS = ["a", "b", "c", "d"]


def _day(i):
    return START + pd.Timedelta(days=i)


def _frame(flip_held_from=None, post_catalog_events_from=None):
    frames = []
    for k, cell in enumerate(HELD + OTHERS):
        days = np.arange(N_DAYS)
        events = ((days + 3 * k) % 11 == 0).astype(int)
        if flip_held_from is not None and cell in HELD:
            events[flip_held_from:] = 1
        if post_catalog_events_from is not None:
            events[post_catalog_events_from:] = 1
        frames.append(pd.DataFrame({
            "cell_id": cell,
            "date": START + pd.to_timedelta(days, unit="D"),
            "event": events,
        }))
    return pd.concat(frames, ignore_index=True)


def test_cells_in_bands_selects_cells_by_coordinate():
    grid = pd.DataFrame({"cell_id": ["x", "y", "z"], "lat": [26.3, 27.05, 28.0]})
    assert cells_in_bands(grid, "lat", [26.3, 27.05]) == {"x", "y"}


def test_test_contains_only_held_out_cells_and_the_others_never_leak_in():
    train, val, test, label_col = spatial_holdout_split(_frame(), "event", HELD)
    assert set(test["cell_id"]) == set(HELD)
    assert not set(train["cell_id"]) & set(HELD)
    assert not set(val["cell_id"]) & set(HELD)
    assert set(train["cell_id"]) == set(OTHERS) == set(val["cell_id"])


def test_held_out_cells_keep_every_complete_window_and_lose_the_incomplete_tail():
    _, _, test, _ = spatial_holdout_split(_frame(), "event", HELD)
    last_complete_day = N_DAYS - 1 - HORIZON
    assert test["date"].max() == _day(last_complete_day)
    assert test["date"].min() == _day(0)
    assert len(test) == len(HELD) * (last_complete_day + 1)


def test_validation_is_the_last_year_and_training_is_purged_before_it():
    train, val, test, _ = spatial_holdout_split(_frame(), "event", HELD)
    anchor = _day(N_DAYS - 1 - HORIZON)
    val_start = anchor - pd.Timedelta(days=364)

    assert val["date"].min() == val_start and val["date"].max() == anchor
    assert val["date"].nunique() == 365
    assert train["date"].max() + pd.Timedelta(days=HORIZON) < val["date"].min()
    assert train["date"].max() == val_start - pd.Timedelta(days=HORIZON + 1)


def test_baselines_come_from_training_cells_only_and_ignore_held_out_outcomes():
    clean = spatial_holdout_split(_frame(), "event", HELD)
    polluted = spatial_holdout_split(_frame(flip_held_from=200), "event", HELD)
    train_clean, _, test_clean, label_col = clean
    _, _, test_polluted, _ = polluted

    assert test_polluted[label_col].mean() > test_clean[label_col].mean()
    for column in (TRAIN_ONLY_BASELINE_COL, TRAIN_ONLY_SEASONAL_BASELINE_COL):
        np.testing.assert_array_equal(
            test_clean[column].to_numpy(), test_polluted[column].to_numpy()
        )
    np.testing.assert_allclose(
        test_clean[TRAIN_ONLY_BASELINE_COL].to_numpy(),
        train_clean[label_col].mean(),
    )


def test_anchor_for_a_catalog_with_an_end_date_ignores_later_events():
    cutoff = _day(900)
    clean = spatial_holdout_split(_frame(), "event", HELD, max_usable_date=cutoff)
    polluted = spatial_holdout_split(
        _frame(post_catalog_events_from=901), "event", HELD, max_usable_date=cutoff
    )
    _, val_c, test_c, _ = clean
    _, val_p, test_p, _ = polluted

    assert test_c["date"].max() == _day(900 - HORIZON)
    assert val_c["date"].max() == _day(900 - HORIZON)
    pd.testing.assert_frame_equal(
        test_c.reset_index(drop=True), test_p.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        val_c.reset_index(drop=True), val_p.reset_index(drop=True)
    )


def test_returns_none_when_the_data_cannot_support_the_split():
    short = _frame().query("date < @_day(300)")
    assert spatial_holdout_split(short, "event", HELD) is None
    assert spatial_holdout_split(_frame(), "event", ["not_a_cell"]) is None



from src.terrarisk.models.spatial_holdout import (
    NEIGHBOUR_BASELINE_COL,
    NEIGHBOUR_SEASONAL_BASELINE_COL,
    nearest_training_cells,
    spatiotemporal_holdout_split,
)

GRID = pd.DataFrame({
    "cell_id": ["h1", "h2", "a", "b", "c", "d"],
    "lat": [0.0, 10.0, 0.1, 0.2, 9.9, 10.2],
    "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
})


def test_nearest_training_cells_are_exact_and_deterministic():
    pools = nearest_training_cells(GRID, ["h1", "h2"], OTHERS, k=2)
    assert pools["h1"] == ["a", "b"]
    assert pools["h2"] == ["c", "d"]
    assert pools == nearest_training_cells(GRID, ["h1", "h2"], OTHERS, k=2)


def test_nearest_training_cells_rejects_cells_missing_from_the_grid():
    import pytest

    with pytest.raises(ValueError):
        nearest_training_cells(GRID.drop(index=2), ["h1"], OTHERS, k=2)


def test_neighbour_baseline_uses_only_nearby_training_cells_and_ignores_held_out_outcomes():
    clean = spatial_holdout_split(_frame(), "event", HELD, grid=GRID)
    polluted = spatial_holdout_split(_frame(flip_held_from=200), "event", HELD, grid=GRID)
    train, _, test, label_col = clean
    _, _, test_polluted, _ = polluted

    for column in (NEIGHBOUR_BASELINE_COL, NEIGHBOUR_SEASONAL_BASELINE_COL):
        np.testing.assert_array_equal(
            test[column].to_numpy(), test_polluted[column].to_numpy()
        )

    assert TRAIN_ONLY_BASELINE_COL in test.columns
    by_cell = test.groupby("cell_id")[NEIGHBOUR_BASELINE_COL].first()
    expected = train[label_col].mean()
    np.testing.assert_allclose(by_cell.to_numpy(), expected)


def test_spatiotemporal_split_has_exact_non_overlapping_windows():
    train, val, test, label_col = spatiotemporal_holdout_split(
        _frame(), "event", HELD, val_days=200, test_days=200
    )
    anchor = _day(N_DAYS - 1 - HORIZON)
    test_start = anchor - pd.Timedelta(days=199)
    val_start = test_start - pd.Timedelta(days=200)

    assert set(test["cell_id"]) == set(HELD)
    assert (test["date"].min(), test["date"].max()) == (test_start, anchor)
    assert set(val["cell_id"]) == set(OTHERS) == set(train["cell_id"])

    assert val["date"].max() + pd.Timedelta(days=HORIZON) < test_start
    assert val["date"].min() == val_start
    assert train["date"].max() + pd.Timedelta(days=HORIZON) < val_start
    assert not (set(val["date"]) & set(test["date"]))


def test_spatiotemporal_outcomes_in_the_test_period_cannot_reach_train_val_or_baselines():
    clean = spatiotemporal_holdout_split(_frame(), "event", HELD, val_days=200, test_days=200)
    polluted = spatiotemporal_holdout_split(
        _frame(post_catalog_events_from=N_DAYS - 200 - HORIZON),
        "event", HELD, val_days=200, test_days=200,
    )
    train_c, val_c, test_c, _ = clean
    train_p, val_p, test_p, _ = polluted

    pd.testing.assert_frame_equal(train_c.reset_index(drop=True), train_p.reset_index(drop=True))
    pd.testing.assert_frame_equal(val_c.reset_index(drop=True), val_p.reset_index(drop=True))
    for column in (TRAIN_ONLY_BASELINE_COL, TRAIN_ONLY_SEASONAL_BASELINE_COL):
        np.testing.assert_array_equal(test_c[column].to_numpy(), test_p[column].to_numpy())


def test_spatiotemporal_returns_none_when_history_is_too_short():
    short = _frame().query("date < @_day(300)")
    assert spatiotemporal_holdout_split(short, "event", HELD, val_days=200, test_days=200) is None
