# VectorDB Stress Tester

A lightweight framework for benchmarking the real-world performance and limitations of modern vector databases under practical AI/RAG workloads.

This tool helps you answer questions like:
- How does query latency change as my dataset grows from 10k to 1M vectors?
- How does recall degrade when I have noisy or duplicate data?
- Can the database handle a storm of updates and deletes without falling over?
- What is the real-world performance of hybrid search (vector + keyword)?

## Features

- **Modular & Extensible:** Easily add new vector database backends by implementing a simple adapter interface.
- **Reproducible:** Deterministic runs using fixed seeds for synthetic data generation.
- **Realistic Scenarios:** Benchmarks for scaling, noise injection, hybrid queries, concurrent updates/deletes, and long-context RAG simulations.
- **Comprehensive Metrics:** Measures latency (p50/p95/p99), throughput, recall@k, nDCG@k, memory usage, and more.
- **Rich Reports:** Generates detailed JSON artifacts and a final HTML report with interactive Plotly charts and a narrative summary of findings.

## Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install -e .
    ```
    To run all backends, install the optional extras:
    ```bash
    pip install -e ".[qdrant,weaviate,milvus,pinecone,chroma,transformers]"
    ```

2.  **Start Backend Services (Example: Qdrant):**
    For local testing, you can use Docker to easily spin up services.
    ```bash
    docker run -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
    ```

3.  **Run a Benchmark:**
    Execute a benchmark run against FAISS (the local baseline) and a running Qdrant instance.
    ```bash
    vdbt run --adapters faiss,qdrant --scenarios scale_curve,noise_injection
    ```

4.  **Generate the Report:**
    After the run completes, compile the artifacts into an HTML report.
    ```bash
    vdbt report
    ```
    The report will be available at `artifacts/report/report.html`.

## Safety Notes

- This tool is designed to be run in a controlled environment.
- The stress tests can generate high load on vector database instances. Do not run this against production databases.
- API keys and other secrets should be managed via environment variables and not be checked into version control.
- The tool may download datasets and models from the internet. Ensure you trust the sources.