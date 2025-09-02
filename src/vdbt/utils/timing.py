"""Timing utilities for benchmarking."""

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


@contextmanager
def measure_time() -> Iterator[Dict[str, float]]:
    """A context manager to measure the wall-clock time of a block of code.

    Yields:
        A dictionary that will be populated with timing information.
    """
    result = {"start": 0.0, "end": 0.0, "duration": 0.0}
    result["start"] = time.perf_counter()
    try:
        yield result
    finally:
        result["end"] = time.perf_counter()
        result["duration"] = result["end"] - result["start"]


@contextmanager
def Timer() -> Iterator[Dict[str, Any]]:
    """A context manager to measure and report timing information.

    Yields:
        A dictionary that will be populated with timing information.
    """
    metrics: Dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        end = time.perf_counter()
        metrics["duration_s"] = end - start
