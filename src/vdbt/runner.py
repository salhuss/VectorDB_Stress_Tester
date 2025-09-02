"""The main runner for orchestrating benchmark scenarios."""

import logging
from typing import Any, Dict, Sequence

from vdbt.adapters.base import VectorDB
from vdbt.scenarios.base import Scenario


class Runner:
    """Orchestrates benchmark runs across adapters and scenarios."""

    def __init__(self, adapters: Sequence[VectorDB], scenarios: Sequence[Scenario]):
        self.adapters = adapters
        self.scenarios = scenarios

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Run all scenarios on all adapters.

        Returns:
            A dictionary of results.
        """
        results: Dict[str, Any] = {}
        for adapter in self.adapters:
            logging.info(f"Running scenarios on {adapter.name}...")
            adapter.connect()
            results[adapter.name] = {}
            for scenario in self.scenarios:
                logging.info(f"Running scenario: {scenario.name}...")
                try:
                    scenario_results = scenario.run(db=adapter, **kwargs)
                    results[adapter.name][scenario.name] = scenario_results
                except Exception as e:
                    logging.error(
                        f"Scenario {scenario.name} failed on {adapter.name}: {e}"
                    )
                    results[adapter.name][scenario.name] = {"error": str(e)}
        return results
