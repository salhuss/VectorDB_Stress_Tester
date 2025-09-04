"""Multi-vector Long Context scenario."""

from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from vdbt.adapters.base import VectorDB
from vdbt.metrics import compute_percentiles
from vdbt.utils.data import create_synthetic_embeddings
from vdbt.utils.timing import Timer


class MultiVectorLongContextScenario:
    """Scenario to simulate RAG over long documents with multiple sub-queries."""

    name = "multivector_longctx"

    def run(self, db: VectorDB, **kwargs: Any) -> Dict[str, Any]:
        """Run the multi-vector long context scenario.

        Args:
            db: The vector database adapter to use.
            **kwargs: Scenario-specific parameters.

        Returns:
            A dictionary of metrics.
        """
        dim = kwargs["dim"]
        num_embeddings = kwargs["num_embeddings"]
        num_sub_queries = kwargs.get("num_sub_queries", [4, 8, 16])
        seed = kwargs["seed"]

        results = {}

        # Create a base collection
        collection_name = f"{self.name}_base"
        db.drop_collection(collection_name)
        db.create_collection(collection_name, dim)

        embeddings, labels = create_synthetic_embeddings(
            num_embeddings=num_embeddings, dim=dim, num_classes=10, seed=seed
        )
        ids = [str(i) for i in range(num_embeddings)]
        metadata = [{"label": int(label)} for label in labels]
        db.upsert(collection_name, ids, embeddings, metadata)

        for n_sub_queries in num_sub_queries:
            query_latencies = []
            recalls = []

            # Generate long context queries
            for _ in tqdm(
                range(100),
                desc=f"Querying with {n_sub_queries} sub-queries",
            ):
                ground_truth_label = np.random.default_rng(seed).integers(0, 10)
                sub_query_vectors, _ = create_synthetic_embeddings(
                    num_embeddings=n_sub_queries,
                    dim=dim,
                    num_classes=1,
                    seed=seed + ground_truth_label,
                )

                # Execute sub-queries and combine results (simulated RAG)
                combined_results_ids = set()
                with Timer() as query_timer:
                    for sub_q_vec in sub_query_vectors:
                        results_from_db = db.query(
                            collection_name,
                            np.expand_dims(sub_q_vec, axis=0),
                            k=5,
                        )
                        for res in results_from_db:
                            combined_results_ids.add(res["id"])
                query_latencies.append(query_timer["duration_s"])

                # Evaluate recall (simplified: check if any result matches ground truth label)
                # This is a very simplified recall for multi-vector queries.
                # A more robust metric would involve checking if the original document
                # (represented by the ground_truth_label) was retrieved.
                retrieved_labels = []
                for retrieved_id in combined_results_ids:
                    # Find the original label for the retrieved ID
                    original_idx = int(retrieved_id)
                    retrieved_labels.append(labels[original_idx])

                # Check if the ground truth label is among the retrieved labels
                if ground_truth_label in retrieved_labels:
                    recalls.append(1.0)
                else:
                    recalls.append(0.0)

            results[str(n_sub_queries)] = {
                "query_latency_s": compute_percentiles(query_latencies),
                "recall": np.mean(recalls),
            }

        db.drop_collection(collection_name)

        return results