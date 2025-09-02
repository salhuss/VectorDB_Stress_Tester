"""Structured logging setup for the application."""

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging using rich.

    Args:
        level: The logging level to set.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
