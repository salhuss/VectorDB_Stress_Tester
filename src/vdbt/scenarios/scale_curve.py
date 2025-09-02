"""Scale curve scenario."""

from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from vdbt.adapters.base import VectorDB
from vdbt.metrics import compute_percentiles
from vdbt.utils.data import create_synthetic_embeddings
from vdbt.utils.timing import Timer


class ScaleCurveScenario:
    """Scenario to measure performance as the dataset size increases."""

    name = "scale_curve"

    def run(self, db: VectorDB, **kwargs: Any) -> Dict[str, Any]:
        """Run the scale curve scenario.

        Args:
            db: The vector database adapter to use.
            **kwargs: Scenario-specific parameters.

        Returns:
            A dictionary of metrics.
        """
        dim = kwargs["dim"]
        scales = kwargs["scales"]
        seed = kwargs["seed"]

        results = {}
        for scale in scales:
            collection_name = f"{self.name}_{scale}"
            db.drop_collection(collection_name)
            db.create_collection(collection_name, dim)

            # Generate data
            embeddings, _ = create_synthetic_embeddings(
                num_embeddings=scale, dim=dim, num_classes=10, seed=seed
            )
            ids = [str(i) for i in range(scale)]
            metadata = [{"i": i} for i in range(scale)]

            # Indexing
            with Timer() as index_timer:
                db.upsert(collection_name, ids, embeddings, metadata)

            # Memory usage
            memory_bytes = db.memory_bytes(collection_name)

            # Querying
            query_vectors, _ = create_synthetic_embeddings(
                num_embeddings=100, dim=dim, num_classes=10, seed=seed + 1
            )
            latencies = []
            for vector in tqdm(query_vectors, desc=f"Querying {scale}"):
                with Timer() as query_timer:
                    db.query(collection_name, np.expand_dims(vector, axis=0), k=10)
                latencies.append(query_timer["duration_s"])

            results[str(scale)] = {
                "index_time_s": index_timer["duration_s"],
                "memory_bytes": memory_bytes,
                "query_latency_s": compute_percentiles(latencies),
            }

            db.drop_collection(collection_name)

        return results
