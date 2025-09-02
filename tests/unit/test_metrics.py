"""Unit tests for the metrics module."""

import pytest

from vdbt.metrics import (
    compute_percentiles,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_compute_percentiles():
    """Test percentile computation."""
    data = list(range(1, 101))
    percentiles = compute_percentiles(data, [50, 90, 100])
    assert percentiles["p50"] == pytest.approx(50.5)
    assert percentiles["p90"] == pytest.approx(90.1)
    assert percentiles["p100"] == pytest.approx(100.0)


def test_compute_percentiles_empty():
    """Test percentile computation with no data."""
    percentiles = compute_percentiles([], [50, 90])
    assert percentiles["p50"] == 0.0
    assert percentiles["p90"] == 0.0


def test_recall_at_k():
    """Test recall@k calculation."""
    y_true = [1, 2, 3]
    y_pred = [[1, 4, 5], [4, 5, 6], [3, 1, 2]]
    assert recall_at_k(y_true, y_pred, k=1) == pytest.approx(2 / 3)
    assert recall_at_k(y_true, y_pred, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_empty():
    """Test recall@k with empty inputs."""
    assert recall_at_k([], [], k=3) == 0.0


def test_mrr_at_k():
    """Test MRR@k calculation."""
    y_true = [1, 2, 3]
    y_pred = [[1, 4, 5], [4, 5, 2], [2, 1, 3]]
    # Ranks: 1, 3, 3
    # RR: 1/1, 1/3, 1/3
    # MRR: (1 + 1/3 + 1/3) / 3 = (5/3) / 3 = 5/9
    assert mrr_at_k(y_true, y_pred, k=3) == pytest.approx(5 / 9)


def test_mrr_at_k_no_match():
    """Test MRR@k with no matches."""
    y_true = [1, 2, 3]
    y_pred = [[4, 5, 6], [4, 5, 6], [4, 5, 6]]
    assert mrr_at_k(y_true, y_pred, k=3) == 0.0


def test_ndcg_at_k():
    """Test nDCG@k calculation."""
    y_true = [1, 2, 3]
    y_pred = [[1, 4, 5], [4, 5, 2], [2, 1, 3]]
    # Ranks: 1, 3, 3
    # DCG: 1/log2(2), 1/log2(4), 1/log2(4)
    # nDCG: (1/1 + 1/2 + 1/2) / 3 = (2) / 3 = 2/3
    dcg1 = 1 / 1.0
    dcg2 = 1 / 2.0
    dcg3 = 1 / 2.0
    expected_ndcg = (dcg1 + dcg2 + dcg3) / 3.0
    assert ndcg_at_k(y_true, y_pred, k=3) == pytest.approx(expected_ndcg, 0.01)
