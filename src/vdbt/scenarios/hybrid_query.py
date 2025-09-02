"""Hybrid query scenario."""

from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from vdbt.adapters.base import VectorDB
from vdbt.metrics import recall_at_k
from vdbt.utils.data import create_synthetic_embeddings
from vdbt.utils.hybrid import create_hybrid_query_dataset


class HybridQueryScenario:
    """Scenario to measure performance with hybrid queries (vector + keyword)."""

    name = "hybrid_query"

    def run(self, db: VectorDB, **kwargs: Any) -> Dict[str, Any]:
        """Run the hybrid query scenario.

        Args:
            db: The vector database adapter to use.
            **kwargs: Scenario-specific parameters.

        Returns:
            A dictionary of metrics.
        """
        dim = kwargs["dim"]
        num_embeddings = kwargs["num_embeddings"]
        keyword_ratio = kwargs.get("keyword_ratio", 0.5)
        seed = kwargs["seed"]

        collection_name = f"{self.name}"
        db.drop_collection(collection_name)
        db.create_collection(collection_name, dim)

        embeddings, labels = create_synthetic_embeddings(
            num_embeddings=num_embeddings, dim=dim, num_classes=10, seed=seed
        )
        ids = [str(i) for i in range(num_embeddings)]
        metadata = [{"label": int(label)} for label in labels]

        db.upsert(collection_name, ids, embeddings, metadata)

        queries = create_hybrid_query_dataset(
            embeddings=embeddings,
            labels=labels,
            num_queries=100,
            keyword_ratio=keyword_ratio,
            seed=seed + 1,
        )

        predictions = []
        ground_truth = []
        for query in tqdm(queries, desc="Executing hybrid queries"):
            query_vector = np.expand_dims(query["vector"], axis=0)
            query_filter = query.get("filter")
            query_results = db.query(
                collection_name, query_vector, k=10, filter=query_filter
            )
            predictions.append([res["metadata"]["label"] for res in query_results])
            ground_truth.append(query["ground_truth_label"])

        recall = recall_at_k(ground_truth, predictions, k=10)

        db.drop_collection(collection_name)

        return {"recall@10": recall}
