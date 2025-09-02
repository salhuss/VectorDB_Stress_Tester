"""Integration tests for the FAISS adapter."""

import numpy as np
import pytest

from vdbt.adapters.faiss_adapter import FaissAdapter


@pytest.fixture
def adapter():
    """Returns a FaissAdapter instance."""
    return FaissAdapter()


def test_faiss_adapter_smoke(adapter: FaissAdapter):
    """Smoke test for the FAISS adapter."""
    collection_name = "test_collection"
    dim = 4
    num_vectors = 100

    adapter.connect()
    adapter.create_collection(collection_name, dim)

    # Upsert
    ids = [str(i) for i in range(num_vectors)]
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    metadata = [{"i": i} for i in range(num_vectors)]
    adapter.upsert(collection_name, ids, vectors, metadata)

    assert adapter.count(collection_name) == num_vectors

    # Query
    query_vector = np.random.rand(1, dim).astype(np.float32)
    results = adapter.query(collection_name, query_vector, k=5)
    assert len(results) == 5
    for result in results:
        assert "id" in result
        assert "distance" in result

    # Delete (not implemented for this index, but should not fail)
    adapter.delete(collection_name, ["0", "1"])
    assert adapter.count(collection_name) == num_vectors

    # Drop collection
    adapter.drop_collection(collection_name)
    assert adapter.count(collection_name) == 0
