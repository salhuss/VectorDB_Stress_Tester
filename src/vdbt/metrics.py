"""Functions for computing and summarizing performance and accuracy metrics."""

from typing import Any, Dict, List, Optional

import numpy as np


def compute_percentiles(
    data: List[float], percentiles: Optional[List[int]] = None
) -> Dict[str, float]:
    """Computes percentiles for a list of numbers.

    Args:
        data: A list of numbers.
        percentiles: The percentiles to compute.

    Returns:
        A dictionary mapping percentile to value.
    """
    if percentiles is None:
        percentiles = [50, 95, 99]
    if not data:
        return {f"p{p}": 0.0 for p in percentiles}
    values = np.array(data)
    return {f"p{p}": np.percentile(values, p) for p in percentiles}


def recall_at_k(y_true: List[Any], y_pred: List[List[Any]], k: int) -> float:
    """Computes recall@k.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        k: The number of predictions to consider.

    Returns:
        The recall@k score.
    """
    if not y_true or not y_pred:
        return 0.0

    hits = sum(
        1
        for true_label, pred_labels in zip(y_true, y_pred, strict=True)
        if true_label in pred_labels[:k]
    )
    return hits / len(y_true)


def mrr_at_k(y_true: List[Any], y_pred: List[List[Any]], k: int) -> float:
    """Computes Mean Reciprocal Rank (MRR)@k.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        k: The number of predictions to consider.

    Returns:
        The MRR@k score.
    """
    if not y_true or not y_pred:
        return 0.0

    total_rr = 0.0
    for true_label, pred_labels in zip(y_true, y_pred, strict=True):
        try:
            rank = pred_labels[:k].index(true_label) + 1
            total_rr += 1.0 / rank
        except ValueError:
            total_rr += 0.0
    return total_rr / len(y_true)


def ndcg_at_k(y_true: List[Any], y_pred: List[List[Any]], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG)@k.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        k: The number of predictions to consider.

    Returns:
        The nDCG@k score.
    """
    if not y_true or not y_pred:
        return 0.0

    total_ndcg = 0.0
    for true_label, pred_labels in zip(y_true, y_pred, strict=True):
        dcg = 0.0
        try:
            rank = pred_labels[:k].index(true_label)
            dcg = 1.0 / np.log2(rank + 2)
        except ValueError:
            pass  # dcg remains 0

        # IDCG is always 1 in this case since there is only one relevant item.
        idcg = 1.0
        total_ndcg += dcg / idcg

    return total_ndcg / len(y_true)
