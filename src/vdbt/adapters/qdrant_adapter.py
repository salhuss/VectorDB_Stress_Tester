"""Qdrant adapter."""

from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from httpx import ConnectError
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from vdbt.adapters.base import VectorDB


class QdrantAdapter(VectorDB):
    """A Qdrant adapter for the VectorDB protocol."""

    name = "qdrant"

    def __init__(self, url: str = "http://localhost:6333"):
        self._client = QdrantClient(url=url)

    def connect(self) -> bool:
        """Connect to the Qdrant service.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            self._client.get_collections()
            return True
        except (UnexpectedResponse, ConnectError):
            return False

    def drop_collection(self, name: str) -> None:
        """Drop a collection in Qdrant."""
        self._client.delete_collection(collection_name=name)

    def create_collection(self, name: str, dim: int, **kwargs: Any) -> None:
        """Create a collection in Qdrant."""
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )

    def upsert(
        self,
        name: str,
        ids: List[str],
        vectors: np.ndarray[Any, Any],
        meta: List[Dict[str, Any]],
    ) -> None:
        """Upsert data into a Qdrant collection."""
        points = []
        for i, doc_id in enumerate(ids):
            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector=vectors[i].tolist(),
                    payload=meta[i],
                )
            )
        self._client.upsert(collection_name=name, points=points, wait=True)

    def query(
        self,
        name: str,
        vector: np.ndarray[Any, Any],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query a Qdrant collection."""
        query_filter = None
        if filter:
            # Assuming filter is a simple key-value pair for now
            # e.g., {"label": 1}
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=list(filter.keys())[0],
                        range=models.Range(gte=list(filter.values())[0]),
                    )
                ]
            )

        search_result = self._client.search(
            collection_name=name,
            query_vector=vector.tolist()[0],
            query_filter=query_filter,
            limit=k,
        )
        results = []
        for hit in search_result:
            results.append(
                {
                    "id": hit.id,
                    "distance": hit.score,
                    "metadata": hit.payload,
                }
            )
        return results

    def delete(self, name: str, ids: List[str]) -> None:
        """Delete data from a Qdrant collection."""
        self._client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=ids),
        )

    def memory_bytes(self, name: str) -> Optional[int]:
        """Get the memory usage of a Qdrant collection in bytes.

        Qdrant does not expose direct memory usage per collection via API.
        This is a placeholder.
        """
        return None

    def count(self, name: str) -> int:
        """Get the number of items in a Qdrant collection."""
        count_result = self._client.count(collection_name=name, exact=True)
        return int(count_result.count)
