"""Data generation utilities for creating synthetic datasets."""

from typing import Any

import numpy as np


def create_synthetic_embeddings(
    num_embeddings: int,
    dim: int,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Creates a synthetic dataset of embeddings with labeled classes.

    Args:
        num_embeddings: The number of embeddings to generate.
        dim: The dimension of the embeddings.
        num_classes: The number of classes to generate.
        seed: The random seed for reproducibility.

    Returns:
        A tuple containing the embeddings and their labels.
    """
    rng = np.random.default_rng(seed)
    class_centers = rng.standard_normal((num_classes, dim))
    embeddings = np.zeros((num_embeddings, dim), dtype=np.float32)
    labels = np.zeros(num_embeddings, dtype=np.int32)

    for i in range(num_embeddings):
        class_idx = i % num_classes
        # Add some noise to the class center
        embeddings[i] = class_centers[class_idx] + rng.standard_normal(dim) * 2.0
        labels[i] = class_idx

    return embeddings, labels


def inject_duplicates(
    embeddings: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    duplicate_ratio: float,
    seed: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Injects duplicate embeddings into the dataset.

    Args:
        embeddings: The original embeddings.
        labels: The original labels.
        duplicate_ratio: The fraction of embeddings to duplicate.
        seed: The random seed.

    Returns:
        A tuple containing the embeddings and labels with duplicates.
    """
    rng = np.random.default_rng(seed)
    num_duplicates = int(len(embeddings) * duplicate_ratio)
    if num_duplicates == 0:
        return embeddings, labels

    duplicate_indices = rng.choice(len(embeddings), size=num_duplicates, replace=True)
    new_embeddings = np.vstack([embeddings, embeddings[duplicate_indices]])
    new_labels = np.hstack([labels, labels[duplicate_indices]])

    return new_embeddings, new_labels


def inject_noise(
    embeddings: np.ndarray[Any, Any], noise_ratio: float, seed: int
) -> np.ndarray[Any, Any]:
    """Injects noisy embeddings into the dataset.

    Args:
        embeddings: The original embeddings.
        noise_ratio: The fraction of embeddings to replace with noise.
        seed: The random seed.

    Returns:
        The embeddings with noise.
    """
    rng = np.random.default_rng(seed)
    num_noise = int(len(embeddings) * noise_ratio)
    if num_noise == 0:
        return embeddings

    noise_indices = rng.choice(len(embeddings), size=num_noise, replace=False)
    noise = rng.standard_normal((num_noise, embeddings.shape[1]))
    embeddings[noise_indices] = noise

    return embeddings
