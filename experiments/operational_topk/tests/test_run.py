"""Tests evaluate_top_k(): correct selection of the highest-scoring rows, correct capture-rate arithmetic, and realistic rounding of the selected count."""

import pandas as pd

from experiments.operational_topk.run import evaluate_top_k

def test_top_k_selects_highest_scores():
    y_true = pd.Series([0, 0, 1, 0, 1], index=range(5))
    scores = pd.Series([0.1, 0.9, 0.8, 0.2, 0.95], index=range(5))
    result = evaluate_top_k(y_true, scores, k_percent=40)
    assert result["n_selected"] == 2
    assert result["n_positive_captured"] == 1


def test_capture_rate_is_fraction_of_all_positives_found():
    y_true = pd.Series([1, 1, 0, 0], index=range(4))
    scores = pd.Series([0.9, 0.1, 0.5, 0.2], index=range(4))
    result = evaluate_top_k(y_true, scores, k_percent=25)
    assert result["n_positive_total"] == 2
    assert result["n_positive_captured"] == 1
    assert result["capture_rate"] == 0.5

def test_n_selected_rounds_realistically_not_assumed_round_number():
    y_true = pd.Series([0] * 4392, index=range(4392))
    scores = pd.Series(range(4392), index=range(4392))
    result = evaluate_top_k(y_true, scores, k_percent=1)
    assert result["n_selected"] == round(4392 * 0.01)

def test_perfect_ranking_gives_perfect_precision():
    y_true = pd.Series([1, 1, 0, 0, 0], index=range(5))
    scores = pd.Series([0.9, 0.8, 0.3, 0.2, 0.1], index=range(5))
    result = evaluate_top_k(y_true, scores, k_percent=40)
    assert result["precision"] == 1.0
    assert result["capture_rate"] == 1.0
    assert result["false_alarms"] == 0


def test_at_least_one_row_always_selected():
    y_true = pd.Series([0, 1, 0], index=range(3))
    scores = pd.Series([0.1, 0.9, 0.2], index=range(3))
    result = evaluate_top_k(y_true, scores, k_percent=1)
    assert result["n_selected"] >= 1