"""
Tests that climatological baselines use TRAIN rows only, are pooled and
seasonal as documented, and are attached to the test frame returned by
prepare_hazard_data(). Synthetic data only; no network access.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.terrarisk.grid.build_grid import build_grid
from src.terrarisk.models.train_model import (
    TRAIN_ONLY_BASELINE_COL,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
    _pools_for_hazard,
    compute_train_only_baselines,
    get_baseline_predictions,
    prepare_hazard_data,
)

START = pd.Timestamp("2020-01-01")


def _rows(cell_id, dates, labels):
    return pd.DataFrame({"cell_id": cell_id, "date": pd.to_datetime(dates), "y": labels})


def _test_frame(cell_ids, month_date="2021-01-15"):
    return pd.DataFrame(
        {"cell_id": cell_ids, "date": pd.to_datetime([month_date] * len(cell_ids)),
         "y": [1] * len(cell_ids)}
    )


def test_cell_rate_is_the_training_mean_and_ignores_test_labels():
    train = _rows("a", ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], [1, 0, 0, 0])
    test_ones = _test_frame(["a"])
    test_zeros = test_ones.assign(y=0)

    out_ones = compute_train_only_baselines(train, test_ones, "y")
    out_zeros = compute_train_only_baselines(train, test_zeros, "y")

    assert out_ones[TRAIN_ONLY_BASELINE_COL].iloc[0] == pytest.approx(0.25)
    pd.testing.assert_frame_equal(out_ones, out_zeros)


def test_unseen_cell_falls_back_to_overall_training_rate():
    train = pd.concat([
        _rows("a", ["2020-01-01", "2020-01-02"], [1, 1]),
        _rows("b", ["2020-01-01", "2020-01-02"], [0, 0]),
    ])
    out = compute_train_only_baselines(train, _test_frame(["zzz"]), "y")
    assert out[TRAIN_ONLY_BASELINE_COL].iloc[0] == pytest.approx(0.5)
    assert out[TRAIN_ONLY_SEASONAL_BASELINE_COL].iloc[0] == pytest.approx(0.5)


def test_pooled_rate_combines_the_cells_in_each_pool():
    train = pd.concat([
        _rows("a", pd.date_range("2020-01-01", periods=4), [1, 0, 0, 0]),
        _rows("b", pd.date_range("2020-01-01", periods=4), [1, 1, 1, 0]),
        _rows("c", pd.date_range("2020-01-01", periods=4), [0, 0, 0, 0]),
    ])
    pools = {"a": ["a", "b"], "b": ["a", "b"], "c": ["c"]}
    out = compute_train_only_baselines(train, _test_frame(["a", "b", "c"]), "y", pools)
    assert out[TRAIN_ONLY_BASELINE_COL].tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_seasonal_baseline_uses_calendar_month_with_exact_shrinkage():
    train = _rows(
        "a",
        ["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"],
        [1, 1, 0, 0],
    )
    test = pd.DataFrame({
        "cell_id": ["a", "a"],
        "date": pd.to_datetime(["2021-01-10", "2021-02-10"]),
        "y": [0, 0],
    })

    no_shrink = compute_train_only_baselines(train, test, "y", shrinkage_rows=0)
    assert no_shrink[TRAIN_ONLY_SEASONAL_BASELINE_COL].tolist() == pytest.approx([1.0, 0.0])

    shrunk = compute_train_only_baselines(train, test, "y", shrinkage_rows=2)
    assert shrunk[TRAIN_ONLY_SEASONAL_BASELINE_COL].tolist() == pytest.approx([0.75, 0.25])
    assert shrunk[TRAIN_ONLY_BASELINE_COL].tolist() == pytest.approx([0.5, 0.5])


def test_empty_training_set_is_rejected():
    empty = pd.DataFrame({"cell_id": [], "date": pd.to_datetime([]), "y": []})
    with pytest.raises(ValueError):
        compute_train_only_baselines(empty, _test_frame(["a"]), "y")


def test_pools_are_singletons_for_flood_and_nearest_cells_for_landslide():
    cell_ids = build_grid()["cell_id"].tolist()[:20]
    flood_pools = _pools_for_hazard("flood_event", cell_ids)
    assert all(members == [cell] for cell, members in flood_pools.items())

    landslide_pools = _pools_for_hazard("landslide_event", cell_ids)
    for cell, members in landslide_pools.items():
        assert len(members) == 15
        assert members[0] == cell
    assert landslide_pools == _pools_for_hazard("landslide_event", cell_ids)


def test_landslide_pools_reject_cells_that_are_not_on_the_grid():
    with pytest.raises(ValueError):
        _pools_for_hazard("landslide_event", ["not_a_grid_cell"])


def _multi_cell_frame(n_days, flip_from=None):
    frames = []
    for k, cell in enumerate(["a", "b", "c"]):
        days = np.arange(n_days)
        events = ((days + 3 * k) % 11 == 0).astype(int)
        if flip_from is not None:
            events[flip_from:] = 1
        frames.append(pd.DataFrame({
            "cell_id": cell,
            "date": START + pd.to_timedelta(days, unit="D"),
            "event": events,
        }))
    return pd.concat(frames, ignore_index=True)


def test_prepare_hazard_data_attaches_baselines_computed_from_train_only():
    df = _multi_cell_frame(1500)
    train, val, test, label_col = prepare_hazard_data(
        df, "event", "SYNTH", verbose=False
    )
    assert TRAIN_ONLY_BASELINE_COL in test.columns
    assert TRAIN_ONLY_SEASONAL_BASELINE_COL in test.columns

    expected = train.groupby("cell_id")[label_col].mean()
    attached = test.groupby("cell_id")[TRAIN_ONLY_BASELINE_COL].first()
    pd.testing.assert_series_equal(attached, expected, check_names=False)


def test_test_period_outcomes_cannot_change_the_baseline():
    clean = _multi_cell_frame(1500)
    polluted = _multi_cell_frame(1500, flip_from=1135)

    _, _, test_clean, label_col = prepare_hazard_data(clean, "event", "SYNTH", verbose=False)
    _, _, test_polluted, _ = prepare_hazard_data(polluted, "event", "SYNTH", verbose=False)

    assert test_polluted[label_col].mean() > test_clean[label_col].mean()
    for column in (TRAIN_ONLY_BASELINE_COL, TRAIN_ONLY_SEASONAL_BASELINE_COL):
        np.testing.assert_array_equal(
            test_clean[column].to_numpy(), test_polluted[column].to_numpy()
        )


def test_get_baseline_predictions_prefers_attached_train_only_baseline():
    df = _multi_cell_frame(1500)
    _, _, test, _ = prepare_hazard_data(df, "event", "SYNTH", verbose=False)
    csv_baseline = pd.DataFrame({"cell_id": ["a", "b", "c"], "baseline_probability": 0.99})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = get_baseline_predictions(test, csv_baseline)

    np.testing.assert_array_equal(result.to_numpy(), test[TRAIN_ONLY_BASELINE_COL].to_numpy())


def test_legacy_csv_baseline_still_works_but_warns():
    test = pd.DataFrame({"cell_id": ["a", "b", "zzz"]})
    csv_baseline = pd.DataFrame({"cell_id": ["a", "b"], "baseline_probability": [0.1, 0.3]})

    with pytest.warns(UserWarning, match="full-history"):
        result = get_baseline_predictions(test, csv_baseline)

    assert result.tolist() == pytest.approx([0.1, 0.3, 0.2])


def test_missing_baseline_everywhere_raises():
    with pytest.raises(ValueError):
        get_baseline_predictions(pd.DataFrame({"cell_id": ["a"]}))
