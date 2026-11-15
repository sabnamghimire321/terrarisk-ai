"""
Leakage regression tests for the horizon-label and split logic in
src/terrarisk/models/train_model.py.

All data is small and synthetic: no network access and no real
weather, flood or landslide files are needed. Each test checks exact
retained dates and exact labels, not just that a function runs.
"""

import numpy as np
import pandas as pd
import pytest

from src.terrarisk.models.train_model import (
    add_horizon_label,
    prepare_hazard_data,
    time_based_split,
)

START = pd.Timestamp("2020-01-01")


def _day(i: int) -> pd.Timestamp:
    return START + pd.Timedelta(days=i)


def _series(cell_id: str, n_days: int, event_days=(), skip_days=()) -> pd.DataFrame:
    days = [i for i in range(n_days) if i not in set(skip_days)]
    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "date": [_day(i) for i in days],
            "event": [1 if i in set(event_days) else 0 for i in days],
        }
    )


def _labels_by_day(result: pd.DataFrame, cell_id: str = None) -> dict:
    """Returns {day_index: label} for rows whose label is not NaN."""
    if cell_id is not None:
        result = result[result["cell_id"] == cell_id]
    kept = result.dropna(subset=["label"])
    return {(d - START).days: int(v) for d, v in zip(kept["date"], kept["label"])}



def test_complete_window_exact_labels_and_retained_dates():
    df = _series("a", 20, event_days={10})
    result = add_horizon_label(df, "event", 3, "label")
    labels = _labels_by_day(result)

    assert sorted(labels) == list(range(0, 17))
    assert {d for d, v in labels.items() if v == 1} == {7, 8, 9}
    assert labels[10] == 0


def test_incomplete_future_window_is_nan_even_if_event_is_inside_it():
    df = _series("a", 20, event_days={18})
    result = add_horizon_label(df, "event", 3, "label")
    labels = _labels_by_day(result)

    assert sorted(labels) == list(range(0, 17))
    assert {d for d, v in labels.items() if v == 1} == {15, 16}
    row_17 = result[result["date"] == _day(17)].iloc[0]
    assert np.isnan(row_17["label"])


def test_last_horizon_rows_of_every_cell_are_dropped():
    df = pd.concat(
        [_series("a", 15), _series("b", 15)], ignore_index=True
    )
    result = add_horizon_label(df, "event", 7, "label").dropna(subset=["label"])
    for cell in ("a", "b"):
        cell_rows = result[result["cell_id"] == cell]
        assert cell_rows["date"].max() == _day(7)
        assert len(cell_rows) == 8


def test_cells_are_independent_time_series():
    a = _series("a", 10)
    b = _series("b", 20, event_days={10})
    b = b[b["date"] >= _day(10)]
    df = pd.concat([b, a], ignore_index=True)

    result = add_horizon_label(df, "event", 3, "label")

    a_labels = _labels_by_day(result, "a")
    assert sorted(a_labels) == list(range(0, 7))
    assert set(a_labels.values()) == {0}

    b_labels = _labels_by_day(result, "b")
    assert b_labels[10] == 0 and b_labels[11] == 0


def test_missing_dates_make_affected_windows_incomplete():
    df = _series("a", 10, event_days={6}, skip_days={5})
    result = add_horizon_label(df, "event", 3, "label")
    labels = _labels_by_day(result)

    assert labels == {0: 0, 1: 0, 6: 0}


def test_missing_event_value_inside_window_gives_nan_label():
    df = _series("a", 12, event_days=set())
    df["event"] = df["event"].astype(float)
    df.loc[df["date"] == _day(5), "event"] = np.nan
    result = add_horizon_label(df, "event", 3, "label")
    labels = _labels_by_day(result)

    assert 2 not in labels and 3 not in labels and 4 not in labels
    assert 1 in labels and 5 in labels


def test_duplicate_dates_raise_clear_error():
    df = _series("a", 10)
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        add_horizon_label(df, "event", 3, "label")


def test_unsorted_input_gives_same_labels_as_sorted_input():
    df = _series("a", 20, event_days={10})
    shuffled = df.sample(frac=1.0, random_state=0)
    result_sorted = add_horizon_label(df, "event", 3, "label")
    result_shuffled = add_horizon_label(shuffled, "event", 3, "label")
    assert _labels_by_day(result_sorted) == _labels_by_day(result_shuffled)



def test_split_has_exact_boundaries_and_purge_gaps():
    df = _series("a", 100)
    train, val, test = time_based_split(
        df, test_days=10, val_days=10, max_usable_date=_day(99), purge_days=7
    )

    assert (test["date"].min(), test["date"].max()) == (_day(90), _day(99))
    assert (val["date"].min(), val["date"].max()) == (_day(80), _day(82))
    assert (train["date"].min(), train["date"].max()) == (_day(0), _day(72))


def test_no_training_label_window_crosses_into_validation():
    df = _series("a", 100)
    train, val, test = time_based_split(
        df, test_days=10, val_days=10, max_usable_date=_day(99), purge_days=7
    )
    last_train_label_day = train["date"].max() + pd.Timedelta(days=7)
    assert last_train_label_day < val["date"].min()


def test_no_validation_label_window_crosses_into_test():
    df = _series("a", 100)
    train, val, test = time_based_split(
        df, test_days=10, val_days=10, max_usable_date=_day(99), purge_days=7
    )
    last_val_label_day = val["date"].max() + pd.Timedelta(days=7)
    assert last_val_label_day < test["date"].min()


def test_test_outcome_cannot_reach_train_or_validation_labels():
    df = _series("a", 100, event_days={83})
    labeled = add_horizon_label(df, "event", 7, "label").dropna(subset=["label"])

    train, val, test = time_based_split(
        labeled, 10, 10, _day(92), purge_days=7
    )
    assert train["label"].sum() == 0
    assert val["label"].sum() == 0

    _, val_unpurged, _ = time_based_split(labeled, 10, 10, _day(92), purge_days=0)
    assert val_unpurged["label"].sum() > 0


def test_split_period_lengths_are_exactly_the_requested_days():
    df = _series("a", 100)
    train, val, test = time_based_split(df, 10, 10, _day(99), purge_days=0)
    assert test["date"].nunique() == 10
    assert val["date"].nunique() == 10



def _multi_cell_frame(n_days: int, post_event_days_from=None) -> pd.DataFrame:
    frames = []
    for k, cell in enumerate(["a", "b", "c"]):
        events = {d for d in range(n_days) if (d + 3 * k) % 11 == 0}
        if post_event_days_from is not None:
            events |= set(range(post_event_days_from, n_days))
        frames.append(_series(cell, n_days, event_days=events))
    return pd.concat(frames, ignore_index=True)


def test_prepare_hazard_data_boundaries_without_anchor():
    df = _multi_cell_frame(400)
    train, val, test, label_col = prepare_hazard_data(
        df, "event", "SYNTH", min_test_days=20, horizon_days=7, verbose=False,
        protocol_floor_days=0,
    )

    assert label_col == "event_horizon_label"
    for part in (train, val, test):
        assert part[label_col].notna().all()
        assert set(part["cell_id"]) == {"a", "b", "c"}

    assert test["date"].max() == _day(392)
    assert train["date"].max() + pd.Timedelta(days=7) < val["date"].min()
    assert val["date"].max() + pd.Timedelta(days=7) < test["date"].min()


def test_prepare_hazard_data_anchor_ignores_events_after_catalog_end():
    cutoff = _day(300)
    clean = _multi_cell_frame(400)
    polluted = _multi_cell_frame(400, post_event_days_from=301)

    out_clean = prepare_hazard_data(
        clean, "event", "SYNTH", max_usable_date=cutoff,
        min_test_days=20, horizon_days=7, verbose=False,
        protocol_floor_days=0,
    )
    out_polluted = prepare_hazard_data(
        polluted, "event", "SYNTH", max_usable_date=cutoff,
        min_test_days=20, horizon_days=7, verbose=False,
        protocol_floor_days=0,
    )

    train, val, test, label_col = out_clean
    assert test["date"].max() == _day(293)
    assert test["date"].max() + pd.Timedelta(days=7) <= cutoff

    pd.testing.assert_frame_equal(
        test.reset_index(drop=True), out_polluted[2].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        val.reset_index(drop=True), out_polluted[1].reset_index(drop=True)
    )



def test_protocol_floor_raises_short_windows_to_one_year():
    df = _multi_cell_frame(1500)
    train, val, test, label_col = prepare_hazard_data(
        df, "event", "SYNTH", min_test_days=20, horizon_days=7, verbose=False
    )
    assert test["date"].nunique() == 365
    assert val["date"].nunique() == 365 - 7
    assert test["date"].max() == _day(1492)


def test_protocol_floor_returns_none_when_data_cannot_support_it():
    df = _multi_cell_frame(400)
    out = prepare_hazard_data(
        df, "event", "SYNTH", min_test_days=20, horizon_days=7, verbose=False
    )
    assert out is None
