"""Noise injection scenario."""

from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from vdbt.adapters.base import VectorDB
from vdbt.metrics import recall_at_k
from vdbt.utils.data import create_synthetic_embeddings, inject_noise


class NoiseInjectionScenario:
    """Scenario to measure performance with noisy data."""

    name = "noise_injection"

    def run(self, db: VectorDB, **kwargs: Any) -> Dict[str, Any]:
        """Run the noise injection scenario.

        Args:
            db: The vector database adapter to use.
            **kwargs: Scenario-specific parameters.

        Returns:
            A dictionary of metrics.
        """
        dim = kwargs["dim"]
        num_embeddings = kwargs["num_embeddings"]
        noise_ratios = kwargs.get("noise_ratios", [0.0, 0.1, 0.2, 0.5])
        seed = kwargs["seed"]

        results = {}
        embeddings, labels = create_synthetic_embeddings(
            num_embeddings=num_embeddings, dim=dim, num_classes=10, seed=seed
        )

        for ratio in noise_ratios:
            collection_name = f"{self.name}_{ratio}"
            db.drop_collection(collection_name)
            db.create_collection(collection_name, dim)

            noisy_embeddings = inject_noise(
                embeddings.copy(), noise_ratio=ratio, seed=seed
            )
            ids = [str(i) for i in range(num_embeddings)]
            metadata = [{"label": int(label)} for label in labels]

            db.upsert(collection_name, ids, noisy_embeddings, metadata)

            # Query with original embeddings to check recall
            # Use a subset of original embeddings as query vectors
            query_indices = np.random.default_rng(seed + 1).choice(
                num_embeddings, size=100, replace=False
            )
            query_vectors = embeddings[query_indices]
            query_labels = labels[query_indices]

            predictions = []
            for vector in tqdm(query_vectors, desc=f"Querying with noise {ratio}"):
                query_results = db.query(
                    collection_name, np.expand_dims(vector, axis=0), k=10
                )
                predictions.append([res["metadata"]["label"] for res in query_results])

            recall = recall_at_k(query_labels.tolist(), predictions, k=10)
            results[str(ratio)] = {"recall@10": recall}

            db.drop_collection(collection_name)

        return results
