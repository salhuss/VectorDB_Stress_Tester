"""Base classes and protocols for vector database adapters."""

from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class VectorDB(Protocol):
    """A protocol for vector database operations."""

    name: str

    def connect(self) -> bool:
        """Connect to the database."""
        ...

    def drop_collection(self, name: str) -> None:
        """Drop a collection."""
        ...

    def create_collection(self, name: str, dim: int, **kwargs: Any) -> None:
        """Create a collection."""
        ...

    def upsert(
        self,
        name: str,
        ids: List[str],
        vectors: np.ndarray[Any, Any],
        meta: List[Dict[str, Any]],
    ) -> None:
        """Upsert data into a collection."""
        ...

    def query(
        self,
        name: str,
        vector: np.ndarray[Any, Any],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query a collection."""
        ...

    def delete(self, name: str, ids: List[str]) -> None:
        """Delete data from a collection."""
        ...

    def memory_bytes(self, name: str) -> Optional[int]:
        """Get the memory usage of a collection in bytes."""
        ...

    def count(self, name: str) -> int:
        """Get the number of items in a collection."""
        ...
