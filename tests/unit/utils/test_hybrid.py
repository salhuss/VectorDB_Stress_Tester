"""Unit tests for the hybrid query utilities."""

import numpy as np

from vdbt.utils.hybrid import create_hybrid_query_dataset


def test_create_hybrid_query_dataset():
    """Test that the hybrid query dataset is created correctly."""
    embeddings = np.random.rand(100, 4)
    labels = np.random.randint(0, 5, 100)

    queries = create_hybrid_query_dataset(
        embeddings=embeddings, labels=labels, num_queries=10, keyword_ratio=0.5, seed=42
    )

    assert len(queries) == 10
    num_keyword_queries = sum(1 for q in queries if "filter" in q)
    # The number of keyword queries should be roughly 5, but can vary
    # due to randomness.
    assert 0 < num_keyword_queries < 10
