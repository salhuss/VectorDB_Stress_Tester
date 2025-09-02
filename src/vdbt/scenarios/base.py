"""Base classes for scenarios."""

from typing import Any, Dict, Protocol

from vdbt.adapters.base import VectorDB


class Scenario(Protocol):
    """A protocol for a benchmark scenario."""

    name: str

    def run(self, db: VectorDB, **kwargs: Any) -> Dict[str, Any]:
        """Run the scenario.

        Args:
            db: The vector database adapter to use.
            **kwargs: Scenario-specific parameters.

        Returns:
            A dictionary of metrics.
        """
        ...
