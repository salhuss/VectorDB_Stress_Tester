"""Integration tests for the scale curve scenario."""

import pytest

from vdbt.adapters.faiss_adapter import FaissAdapter
from vdbt.scenarios.scale_curve import ScaleCurveScenario


@pytest.fixture
def adapter():
    """Returns a FaissAdapter instance."""
    return FaissAdapter()


def test_scale_curve_scenario_smoke(adapter: FaissAdapter):
    """Smoke test for the scale curve scenario."""
    scenario = ScaleCurveScenario()
    results = scenario.run(
        db=adapter,
        dim=4,
        scales=[100, 200],
        seed=42,
    )

    assert len(results) == 2
    for scale in [100, 200]:
        assert str(scale) in results
        assert "index_time_s" in results[str(scale)]
        assert "memory_bytes" in results[str(scale)]
        assert "query_latency_s" in results[str(scale)]
