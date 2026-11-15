"""Tests assign_error_category() and summarize_by_group()'s false-negative-rate calculation."""

import pandas as pd

from experiments.error_analysis.run import assign_error_category, summarize_by_group


def test_flagged_event_is_true_positive():
    y_true = pd.Series([1, 0, 1, 0], index=[10, 20, 30, 40])
    flagged = pd.Index([10, 20])
    categories = assign_error_category(y_true, flagged)
    assert categories[10] == "true_positive"


def test_flagged_non_event_is_false_positive():
    y_true = pd.Series([1, 0, 1, 0], index=[10, 20, 30, 40])
    flagged = pd.Index([10, 20])
    categories = assign_error_category(y_true, flagged)
    assert categories[20] == "false_positive"


def test_unflagged_event_is_false_negative():
    y_true = pd.Series([1, 0, 1, 0], index=[10, 20, 30, 40])
    flagged = pd.Index([10, 20])
    categories = assign_error_category(y_true, flagged)
    assert categories[30] == "false_negative"


def test_unflagged_non_event_is_true_negative():
    y_true = pd.Series([1, 0, 1, 0], index=[10, 20, 30, 40])
    flagged = pd.Index([10, 20])
    categories = assign_error_category(y_true, flagged)
    assert categories[40] == "true_negative"


def test_every_row_gets_exactly_one_category():
    y_true = pd.Series([1, 0, 1, 0, 1], index=range(5))
    flagged = pd.Index([0, 2])
    categories = assign_error_category(y_true, flagged)
    assert categories.isna().sum() == 0
    assert len(categories) == 5


def test_summarize_by_group_computes_fn_rate_correctly():
    df = pd.DataFrame({
        "group": ["a", "a", "a", "b", "b"],
        "error_category": [
            "true_positive", "false_negative", "false_negative",
            "true_positive", "true_positive",
        ],
    })
    summary = summarize_by_group(df, "group")
    assert abs(summary.loc["a", "fn_rate_of_events"] - (2 / 3)) < 0.01
    assert summary.loc["b", "fn_rate_of_events"] == 0.0