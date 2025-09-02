"""Integration tests for the update/delete storm scenario."""

import pytest

from vdbt.adapters.faiss_adapter import FaissAdapter
from vdbt.scenarios.update_delete_storm import UpdateDeleteStormScenario


@pytest.fixture
def adapter():
    """Returns a FaissAdapter instance."""
    return FaissAdapter()


def test_update_delete_storm_scenario_smoke(adapter: FaissAdapter):
    """Smoke test for the update/delete storm scenario."""
    scenario = UpdateDeleteStormScenario()
    results = scenario.run(
        db=adapter,
        dim=4,
        num_embeddings=100,
        update_ratio=0.1,
        delete_ratio=0.1,
        num_queries=10,
        seed=42,
    )

    assert "query_latency_s" in results
    assert "stale_hit_rate" in results
    assert results["stale_hit_rate"] >= 0.0
