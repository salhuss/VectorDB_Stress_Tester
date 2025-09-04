"""Command-line interface for the VectorDB Stress Tester."""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import json

import typer

from vdbt.adapters.base import VectorDB
from vdbt.adapters.faiss_adapter import FaissAdapter
from vdbt.adapters.qdrant_adapter import QdrantAdapter
from vdbt.report import generate_report
from vdbt.runner import Runner
from vdbt.scenarios.base import Scenario
from vdbt.scenarios.hybrid_query import HybridQueryScenario
from vdbt.scenarios.noise_injection import NoiseInjectionScenario
from vdbt.scenarios.scale_curve import ScaleCurveScenario
from vdbt.scenarios.update_delete_storm import UpdateDeleteStormScenario
from vdbt.scenarios.multivector_longctx import MultiVectorLongContextScenario

app = typer.Typer()


@app.command()
def adapters() -> None:
    """List available adapters."""
    # This will be expanded to dynamically discover adapters
    typer.echo("Available adapters: faiss, qdrant")


@app.command()
def scenarios() -> None:
    """List available scenarios."""
    # This will be expanded to dynamically discover scenarios
    typer.echo(
        "Available scenarios: scale_curve, noise_injection, hybrid_query, update_delete_storm, multivector_longctx"
    )


@app.command()
def run(
    adapters_list: List[str] = typer.Option(..., "--adapters", "-a"),
    scenarios_list: List[str] = typer.Option(..., "--scenarios", "-s"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run benchmark scenarios."""
    # This is a simplified version for now.
    # It will be expanded to handle dynamic loading and configuration.
    available_adapters = {"faiss": FaissAdapter, "qdrant": QdrantAdapter}
    available_scenarios = {
        "scale_curve": ScaleCurveScenario,
        "noise_injection": NoiseInjectionScenario,
        "hybrid_query": HybridQueryScenario,
        "update_delete_storm": UpdateDeleteStormScenario,
        "multivector_longctx": MultiVectorLongContextScenario,
    }

    config: Dict[str, Any] = {}
    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)

    selected_adapters: List[VectorDB] = []
    for adapter_name in adapters_list:
        if adapter_name in available_adapters:
            selected_adapters.append(available_adapters[adapter_name]())
        else:
            typer.echo(f"Adapter {adapter_name} not found.")

    selected_scenarios: List[Scenario] = []
    for scenario_name in scenarios_list:
        if scenario_name in available_scenarios:
            selected_scenarios.append(
                cast(Scenario, available_scenarios[scenario_name]())
            )
        else:
            typer.echo(f"Scenario {scenario_name} not found.")

    if not selected_adapters or not selected_scenarios:
        typer.echo("No valid adapters or scenarios selected. Exiting.")
        raise typer.Exit(code=1)

    runner = Runner(selected_adapters, selected_scenarios)
    # Simplified kwargs for now
    results = runner.run(**config)

    typer.echo("Benchmark run completed.")
    # In the future, results will be written to artifacts.
    print(results)


@app.command()
def report(
    artifacts_dir: Path = typer.Option(Path("./artifacts"), "--artifacts-dir", "-o")
) -> None:
    """Compile artifacts into an HTML report."""
    generate_report(artifacts_dir)
    typer.echo(f"Report generated at {artifacts_dir / 'report.html'}")


if __name__ == "__main__":
    app()
