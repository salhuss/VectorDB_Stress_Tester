"""Unit tests for the timing utilities."""

import time

from vdbt.utils.timing import Timer, measure_time


def test_measure_time():
    """Test that the measure_time context manager works."""
    with measure_time() as timing_result:
        time.sleep(0.01)
    assert timing_result["end"] > timing_result["start"]
    assert timing_result["duration"] > 0.01


def test_timer():
    """Test that the Timer context manager works."""
    metrics = {}
    with Timer() as timer_metrics:
        time.sleep(0.01)
        metrics = timer_metrics

    assert "duration_s" in metrics
    assert metrics["duration_s"] > 0.01
