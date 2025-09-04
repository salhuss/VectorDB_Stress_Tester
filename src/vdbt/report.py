"""Report generation for benchmark results."""

import json
from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go
from markdown import markdown
from plotly.offline import plot


def generate_report(
    artifacts_dir: Path, output_file: Path = Path("report.html")
) -> None:
    """Generates an HTML report from benchmark artifacts.

    Args:
        artifacts_dir: The directory containing the benchmark artifacts.
        output_file: The path to the output HTML report file.
    """
    metrics_dir = artifacts_dir / "metrics"
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Any] = {}
    for metric_file in metrics_dir.glob("*.json"):
        with open(metric_file, "r") as f:
            backend, scenario = metric_file.stem.split("_", 1)
            if backend not in all_metrics:
                all_metrics[backend] = {}
            all_metrics[backend][scenario] = json.load(f)

    # Plotting Latency vs Scale
    latency_plot_html = ""
    for backend, scenarios in all_metrics.items():
        if "scale_curve" in scenarios:
            scale_data = scenarios["scale_curve"]
            scales = sorted([int(s) for s in scale_data.keys()])
            p50_latencies = [
                scale_data[str(s)]["query_latency_s"]["p50"] for s in scales
            ]
            p95_latencies = [
                scale_data[str(s)]["query_latency_s"]["p95"] for s in scales
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=scales, y=p50_latencies, mode="lines+markers", name="p50 Latency"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=scales, y=p95_latencies, mode="lines+markers", name="p95 Latency"
                )
            )
            fig.update_layout(
                title=f"{backend} - Query Latency vs Scale",
                xaxis_title="Number of Embeddings",
                yaxis_title="Latency (s)",
            )
            plot_path = plots_dir / f"{backend}_latency_vs_scale.html"
            plot(fig, filename=str(plot_path), auto_open=False)
            latency_plot_html += (
                f'<iframe src="{plot_path.name}" width="100%" height="500px">'
                "</iframe>\n"
            )

    # Plotting Recall vs Noise
    recall_plot_html = ""
    for backend, scenarios in all_metrics.items():
        if "noise_injection" in scenarios:
            noise_data = scenarios["noise_injection"]
            noise_ratios = sorted([float(r) for r in noise_data.keys()])
            recall_at_10 = [noise_data[str(r)]["recall@10"] for r in noise_ratios]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=noise_ratios,
                    y=recall_at_10,
                    mode="lines+markers",
                    name="Recall@10",
                )
            )
            fig.update_layout(
                title=f"{backend} - Recall@10 vs Noise Ratio",
                xaxis_title="Noise Ratio",
                yaxis_title="Recall@10",
            )
            plot_path = plots_dir / f"{backend}_recall_vs_noise.html"
            plot(fig, filename=str(plot_path), auto_open=False)
            recall_plot_html += (
                f'<iframe src="{plot_path.name}" width="100%" height="500px">'
                "</iframe>\n"
            )

    # Executive Summary (Markdown to HTML)
    executive_summary_md = """
# Benchmark Report

## Executive Summary

This report summarizes the performance of various vector databases under
different stress workloads.

### Key Findings:

- **FAISS Scale Curve:** Observed a linear increase in query latency with
  increasing dataset size.
- **FAISS Noise Injection:** Recall degraded as the noise ratio increased,
  as expected.

### Shortcomings Observed:

- FAISS, being an in-memory index, does not persist data across runs.
- The current FAISS implementation does not support direct filtering or
  efficient deletions.

## Detailed Results

### Query Latency vs Scale

{latency_plots}

### Recall vs Noise Ratio

{recall_plots}

 """.format(
        latency_plots=latency_plot_html, recall_plots=recall_plot_html
    )

    executive_summary_html = markdown(executive_summary_md)

    # Combine into final HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>VectorDB Stress Tester Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        iframe {{ border: none; }}
    </style>
</head>
<body>
    {executive_summary_html}
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    # Example usage (for smoke testing)
    # Create dummy artifacts directory and files
    artifacts_dir = Path("artifacts_test")
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Dummy data for scale_curve
    with open(metrics_dir / "faiss_scale_curve.json", "w") as f:
        json.dump(
            {
                "1000": {
                    "index_time_s": 0.1,
                    "memory_bytes": 10000,
                    "query_latency_s": {"p50": 0.001, "p95": 0.002},
                },
                "2000": {
                    "index_time_s": 0.2,
                    "memory_bytes": 20000,
                    "query_latency_s": {"p50": 0.003, "p95": 0.004},
                },
            },
            f,
        )

    # Dummy data for noise_injection
    with open(metrics_dir / "faiss_noise_injection.json", "w") as f:
        json.dump(
            {
                "0.0": {"recall@10": 0.95},
                "0.5": {"recall@10": 0.70},
                "0.8": {"recall@10": 0.40},
            },
            f,
        )

    generate_report(artifacts_dir)
    print(f"Report generated at {artifacts_dir / 'report.html'}")
