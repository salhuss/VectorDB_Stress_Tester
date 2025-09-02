"""FAISS adapter."""

from typing import Any, Dict, List, Optional

import faiss
import numpy as np


class FaissAdapter:
    """A FAISS adapter for the VectorDB protocol."""

    name = "faiss"

    def __init__(self) -> None:
        self._indices: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def connect(self) -> None:
        """FAISS is an in-memory index, so no connection is needed."""
        pass

    def drop_collection(self, name: str) -> None:
        """Delete a FAISS index."""
        if name in self._indices:
            del self._indices[name]
            del self._metadata[name]

    def create_collection(self, name: str, dim: int, **kwargs: Any) -> None:
        """Create a FAISS index."""
        # For FAISS, we can use IndexFlatL2 for simplicity.
        # Other index types can be supported as well.
        index = faiss.IndexFlatL2(dim)
        self._indices[name] = index
        self._metadata[name] = {}

    def upsert(
        self,
        name: str,
        ids: List[str],
        vectors: np.ndarray[Any, Any],
        meta: List[Dict[str, Any]],
    ) -> None:
        """Add vectors to a FAISS index."""
        index = self._indices[name]
        start_index = index.ntotal
        index.add(vectors.astype(np.float32))
        for i, doc_id in enumerate(ids):
            self._metadata[name][start_index + i] = {"id": doc_id, **meta[i]}

    def query(
        self,
        name: str,
        vector: np.ndarray[Any, Any],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query a FAISS index."""
        index = self._indices[name]
        # FAISS does not support filtering directly. This is a limitation.
        if filter:
            # This is a placeholder for a more complex filtering implementation
            # which would require iterating and checking metadata.
            pass

        distances, indices = index.search(vector.astype(np.float32), k)
        results = []
        for i in range(indices.shape[0]):
            query_results = []
            for j in range(indices.shape[1]):
                idx = indices[i][j]
                if idx != -1:
                    metadata = self._metadata[name].get(idx, {})
                    query_results.append(
                        {
                            "id": metadata.get("id"),
                            "distance": distances[i][j],
                            "metadata": metadata,
                        }
                    )
            results.extend(query_results)
        return results

    def delete(self, name: str, ids: List[str]) -> None:
        """Delete vectors from a FAISS index.

        Note: FAISS IndexFlatL2 does not support efficient deletion.
        This is a placeholder for a more complex implementation.
        """
        # This is a known limitation of many FAISS indices.
        # Deletion can be handled by re-building the index.
        pass

    def memory_bytes(self, name: str) -> Optional[int]:
        """Estimate the memory usage of a FAISS index.

        This is a rough estimation.
        """
        index = self._indices.get(name)
        if not index:
            return 0
        # d * n * 4 bytes for the vectors (float32)
        return int(index.ntotal * index.d * 4)

    def count(self, name: str) -> int:
        """Get the number of vectors in a FAISS index."""
        index = self._indices.get(name)
        return index.ntotal if index else 0
