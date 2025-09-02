"""Integration tests for the noise injection scenario."""

import pytest

from vdbt.adapters.faiss_adapter import FaissAdapter
from vdbt.scenarios.noise_injection import NoiseInjectionScenario


@pytest.fixture
def adapter():
    """Returns a FaissAdapter instance."""
    return FaissAdapter()


def test_noise_injection_scenario_smoke(adapter: FaissAdapter):
    """Smoke test for the noise injection scenario."""
    scenario = NoiseInjectionScenario()
    results = scenario.run(
        db=adapter,
        dim=128,
        num_embeddings=1000,
        noise_ratios=[0.0, 0.8],
        seed=42,
    )

    assert len(results) == 2
    assert "0.0" in results
    assert "0.8" in results
    assert "recall@10" in results["0.0"]
    assert "recall@10" in results["0.8"]
    # Recall should be lower with more noise
    assert results["0.8"]["recall@10"] < results["0.0"]["recall@10"]
