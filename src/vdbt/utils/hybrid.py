"""Utilities for creating hybrid query datasets."""

from typing import Any, Dict, List

import numpy as np


def create_hybrid_query_dataset(
    embeddings: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    num_queries: int,
    keyword_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Creates a dataset of hybrid queries.

    Args:
        embeddings: The embeddings to query.
        labels: The labels for the embeddings.
        num_queries: The number of queries to generate.
        keyword_ratio: The ratio of queries that should include a keyword filter.
        seed: The random seed.

    Returns:
        A list of queries, where each query is a dictionary with a vector and an
        optional filter.
    """
    rng = np.random.default_rng(seed)
    queries = []
    for _i in range(num_queries):
        query: Dict[str, Any] = {}
        query_idx = rng.integers(0, len(embeddings))
        query["vector"] = embeddings[query_idx]
        query["ground_truth_label"] = labels[query_idx]

        if rng.random() < keyword_ratio:
            # For simplicity, we filter by the ground truth label.
            # This ensures that there is at least one match.
            query["filter"] = {"label": int(labels[query_idx])}

        queries.append(query)

    return queries
