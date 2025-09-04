"""Update/Delete Storm scenario."""

import random
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from vdbt.adapters.base import VectorDB
from vdbt.metrics import compute_percentiles
from vdbt.utils.data import create_synthetic_embeddings
from vdbt.utils.timing import Timer


class UpdateDeleteStormScenario:
    """Scenario to simulate concurrent updates/deletes and queries."""

    name = "update_delete_storm"

    def run(self, db: VectorDB, **kwargs: Any) -> Dict[str, Any]:
        """Run the update/delete storm scenario.

        Args:
            db: The vector database adapter to use.
            **kwargs: Scenario-specific parameters.

        Returns:
            A dictionary of metrics.
        """
        dim = kwargs["dim"]
        num_embeddings = kwargs["num_embeddings"]
        update_ratio = kwargs.get("update_ratio", 0.1)
        delete_ratio = kwargs.get("delete_ratio", 0.1)
        num_queries = kwargs.get("num_queries", 100)
        seed = kwargs["seed"]

        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        collection_name = f"{self.name}"
        db.drop_collection(collection_name)
        db.create_collection(collection_name, dim)

        # Initial data load
        embeddings, labels = create_synthetic_embeddings(
            num_embeddings=num_embeddings, dim=dim, num_classes=10, seed=seed
        )
        ids = [str(i) for i in range(num_embeddings)]
        metadata = [{"label": int(label)} for label in labels]
        db.upsert(collection_name, ids, embeddings, metadata)

        # Operations: updates, deletes, and queries interleaved
        query_latencies = []
        stale_hits = 0
        total_queries = 0

        all_ids = set(ids)
        deleted_ids: set[str] = set()

        for _ in tqdm(range(num_queries), desc="Running update/delete storm"):
            # Perform updates
            num_updates = int(num_embeddings * update_ratio)
            if num_updates > 0:
                update_indices = np_rng.choice(
                    num_embeddings, size=num_updates, replace=False
                )
                update_ids = [ids[i] for i in update_indices]
                update_vectors = np_rng.standard_normal((num_updates, dim)).astype(
                    np.float32
                )
                update_metadata = [
                    {"label": int(labels[i]), "updated": True} for i in update_indices
                ]
                db.upsert(collection_name, update_ids, update_vectors, update_metadata)

            # Perform deletes
            num_deletes = int(num_embeddings * delete_ratio)
            if num_deletes > 0:
                deletable_ids = list(all_ids - deleted_ids)
                if deletable_ids:
                    delete_ids = rng.sample(
                        deletable_ids, min(num_deletes, len(deletable_ids))
                    )
                    db.delete(collection_name, delete_ids)
                    deleted_ids.update(delete_ids)

            # Perform query and measure latency/staleness
            query_vector, query_label = create_synthetic_embeddings(
                num_embeddings=1,
                dim=dim,
                num_classes=1,
                seed=int(np_rng.integers(0, 100000)),
            )
            query_vector = query_vector[0]
            query_label = query_label[0]

            with Timer() as query_timer:
                results = db.query(
                    collection_name, np.expand_dims(query_vector, axis=0), k=10
                )
            query_latencies.append(query_timer["duration_s"])

            total_queries += 1
            for res in results:
                if res["id"] in deleted_ids:
                    stale_hits += 1

        db.drop_collection(collection_name)

        return {
            "query_latency_s": compute_percentiles(query_latencies),
            "stale_hit_rate": stale_hits / total_queries if total_queries > 0 else 0.0,
        }
