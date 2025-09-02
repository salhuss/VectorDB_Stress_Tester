"""Integration tests for the Qdrant adapter."""


import numpy as np
import pytest

from vdbt.adapters.qdrant_adapter import QdrantAdapter
from vdbt.config import settings


@pytest.fixture(scope="module")
def qdrant_adapter():
    """Returns a QdrantAdapter instance, skipping if not reachable."""
    adapter = QdrantAdapter(url=settings.QDRANT_URL)
    try:
        adapter.connect()
        return adapter
    except ConnectionError:
        pytest.skip(f"Qdrant not reachable at {settings.QDRANT_URL}")


def test_qdrant_adapter_smoke(qdrant_adapter: QdrantAdapter):
    """Smoke test for the Qdrant adapter."""
    collection_name = "test_collection_qdrant"
    dim = 4
    num_vectors = 10

    qdrant_adapter.drop_collection(collection_name)
    qdrant_adapter.create_collection(collection_name, dim)

    # Upsert
    ids = [str(i) for i in range(num_vectors)]
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    metadata = [{"i": i, "label": i % 2} for i in range(num_vectors)]
    qdrant_adapter.upsert(collection_name, ids, vectors, metadata)

    assert qdrant_adapter.count(collection_name) == num_vectors

    # Query
    query_vector = np.random.rand(1, dim).astype(np.float32)
    results = qdrant_adapter.query(collection_name, query_vector, k=5)
    assert len(results) == 5
    for result in results:
        assert "id" in result
        assert "distance" in result
        assert "metadata" in result

    # Query with filter
    results_filtered = qdrant_adapter.query(
        collection_name, query_vector, k=5, filter={"label": 0}
    )
    assert len(results_filtered) <= 5
    for result in results_filtered:
        assert result["metadata"]["label"] == 0

    # Delete
    qdrant_adapter.delete(collection_name, ["0", "1"])
    assert qdrant_adapter.count(collection_name) == num_vectors - 2

    # Drop collection
    qdrant_adapter.drop_collection(collection_name)
    assert qdrant_adapter.count(collection_name) == 0
