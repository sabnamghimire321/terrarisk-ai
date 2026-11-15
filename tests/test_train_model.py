"""Tests the leakage-sensitive core of train_model.py: horizon-label shifting, test-window sizing, and date-anchored splitting - regression tests for the two real leakage bugs found during development."""

import pandas as pd

from src.terrarisk.models.train_model import (
    add_horizon_label,
    choose_test_window_days,
    drop_boundary_leakage_rows,
    time_based_split,
)

def _make_daily_events(cell_id: str, event_days: set, n_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    events = [1 if i in event_days else 0 for i in range(n_days)]
    return pd.DataFrame({"cell_id": cell_id, "date": dates, "event": events})

def test_horizon_label_flags_future_event_within_window():
    df = _make_daily_events("cell_a", event_days={10})
    result = add_horizon_label(df, "event", horizon_days=3, new_col_name="label")

    row_index_7 = result[result["date"] == df.iloc[7]["date"]].iloc[0]
    assert row_index_7["label"] == 1

def test_horizon_label_does_not_leak_same_day_event():
    df = _make_daily_events("cell_a", event_days={10})
    result = add_horizon_label(df, "event", horizon_days=3, new_col_name="label")

    event_date = df.iloc[10]["date"]
    row_on_event_day = result[result["date"] == event_date].iloc[0]
    assert row_on_event_day["label"] == 0


def test_horizon_label_does_not_see_beyond_window():
    df = _make_daily_events("cell_a", event_days={10})
    result = add_horizon_label(df, "event", horizon_days=2, new_col_name="label")

    row_index_5 = result[result["date"] == df.iloc[5]["date"]].iloc[0]
    assert row_index_5["label"] == 0

def test_horizon_label_is_per_cell_not_global():
    df_a = _make_daily_events("cell_a", event_days=set())
    df_b = _make_daily_events("cell_b", event_days={10})
    df = pd.concat([df_a, df_b], ignore_index=True)

    result = add_horizon_label(df, "event", horizon_days=3, new_col_name="label")

    target_date = df_a.iloc[7]["date"]
    cell_a_row = result[
        (result["cell_id"] == "cell_a") & (result["date"] == target_date)
    ].iloc[0]
    assert cell_a_row["label"] == 0

def test_window_sizing_scales_inversely_with_event_rate():
    dense_df = _make_daily_events("cell_a", event_days=set(range(0, 30, 2)), n_days=30)
    sparse_df = _make_daily_events("cell_a", event_days={5}, n_days=30)

    dense_days = choose_test_window_days(dense_df, "event", min_expected_positives=3)
    sparse_days = choose_test_window_days(sparse_df, "event", min_expected_positives=3)

    assert sparse_days > dense_days

def test_window_sizing_returns_zero_for_no_events():
    df = _make_daily_events("cell_a", event_days=set(), n_days=30)
    days = choose_test_window_days(df, "event", min_expected_positives=3)
    assert days == 0

def test_split_respects_max_usable_date_anchor():
    df = _make_daily_events("cell_a", event_days={10, 20}, n_days=30)
    anchor = pd.Timestamp("2020-01-15")

    train, val, test = time_based_split(
        df, test_days=5, val_days=5, max_usable_date=anchor
    )

    assert (train["date"] <= anchor).all()
    assert (val["date"] <= anchor).all()
    assert (test["date"] <= anchor).all()
    assert test["date"].max() == anchor

def test_split_train_val_test_are_chronologically_ordered():
    df = _make_daily_events("cell_a", event_days={10, 20}, n_days=30)
    train, val, test = time_based_split(df, test_days=5, val_days=5)

    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()

def test_boundary_leakage_rows_are_dropped():
    df = _make_daily_events("cell_a", event_days=set(), n_days=30)
    trimmed = drop_boundary_leakage_rows(df, horizon_days=3)

    expected_last_date = df["date"].max() - pd.Timedelta(days=3)
    assert trimmed["date"].max() == expected_last_date
    assert len(trimmed) == len(df) - 3

def test_boundary_leakage_drop_handles_empty_train():
    empty = pd.DataFrame(columns=["cell_id", "date", "event"])
    result = drop_boundary_leakage_rows(empty, horizon_days=3)
    assert len(result) == 0

def test_window_sizing_uses_only_the_slice_it_is_given():
    full_df = _make_daily_events("cell_a", event_days={5, 15, 25}, n_days=30)
    early_slice = full_df[full_df["date"] <= full_df.iloc[14]["date"]]

    full_days = choose_test_window_days(full_df, "event", min_expected_positives=2)
    early_days = choose_test_window_days(early_slice, "event", min_expected_positives=2)
    assert early_days > full_days