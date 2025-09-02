"""Integration tests for the hybrid query scenario."""

import pytest

from vdbt.adapters.faiss_adapter import FaissAdapter
from vdbt.scenarios.hybrid_query import HybridQueryScenario


@pytest.fixture
def adapter():
    """Returns a FaissAdapter instance."""
    return FaissAdapter()


def test_hybrid_query_scenario_smoke(adapter: FaissAdapter):
    """Smoke test for the hybrid query scenario."""
    scenario = HybridQueryScenario()
    results = scenario.run(
        db=adapter,
        dim=4,
        num_embeddings=100,
        keyword_ratio=0.5,
        seed=42,
    )

    assert "recall@10" in results
