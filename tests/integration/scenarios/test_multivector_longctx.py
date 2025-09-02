"""Integration tests for the multi-vector long context scenario."""

import pytest

from vdbt.adapters.faiss_adapter import FaissAdapter
from vdbt.scenarios.multivector_longctx import MultiVectorLongContextScenario


@pytest.fixture
def adapter():
    """Returns a FaissAdapter instance."""
    return FaissAdapter()


def test_multivector_longctx_scenario_smoke(adapter: FaissAdapter):
    """Smoke test for the multi-vector long context scenario."""
    scenario = MultiVectorLongContextScenario()
    results = scenario.run(
        db=adapter,
        dim=4,
        num_embeddings=100,
        num_sub_queries=[2, 4],
        seed=42,
    )

    assert len(results) == 2
    assert "2" in results
    assert "4" in results
    assert "query_latency_s" in results["2"]
    assert "recall" in results["2"]
