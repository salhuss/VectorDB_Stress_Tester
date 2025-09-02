"""Unit tests for the data generation utilities."""

import numpy as np

from vdbt.utils.data import (
    create_synthetic_embeddings,
    inject_duplicates,
    inject_noise,
)


def test_create_synthetic_embeddings():
    """Test that synthetic embeddings are created with the correct shape and type."""
    embeddings, labels = create_synthetic_embeddings(
        num_embeddings=100, dim=10, num_classes=5, seed=42
    )
    assert embeddings.shape == (100, 10)
    assert labels.shape == (100,)
    assert embeddings.dtype == np.float32
    assert labels.dtype == np.int32
    assert len(np.unique(labels)) == 5


def test_inject_duplicates():
    """Test that duplicates are injected correctly."""
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1])
    new_embeddings, new_labels = inject_duplicates(
        embeddings, labels, duplicate_ratio=0.5, seed=42
    )
    assert new_embeddings.shape[0] == 3
    assert new_labels.shape[0] == 3


def test_inject_noise():
    """Test that noise is injected correctly."""
    embeddings = np.ones((10, 2))
    noisy_embeddings = inject_noise(embeddings.copy(), noise_ratio=0.5, seed=42)
    assert not np.all(embeddings == noisy_embeddings)
    # Count the number of rows that are not all ones
    num_changed_rows = np.sum(np.any(noisy_embeddings != 1.0, axis=1))
    assert num_changed_rows == 5
