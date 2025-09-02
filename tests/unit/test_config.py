"""Unit tests for the configuration module."""

from pathlib import Path

from vdbt.config import AppConfig


def test_app_config_defaults():
    """Test that the default AppConfig values are loaded correctly."""
    config = AppConfig()
    assert config.SEED == 42
    assert config.DIM == 384
    assert config.ARTIFACTS_DIR == Path("./artifacts")
    assert config.QDRANT_URL == "http://localhost:6333"


def test_app_config_env_vars(monkeypatch):
    """Test that AppConfig is correctly updated from environment variables."""
    monkeypatch.setenv("VDBT_SEED", "123")
    monkeypatch.setenv("VDBT_DIM", "768")
    monkeypatch.setenv("VDBT_ARTIFACTS_DIR", "/tmp/artifacts")
    monkeypatch.setenv("VDBT_QDRANT_URL", "http://remote:1234")

    config = AppConfig()
    assert config.SEED == 123
    assert config.DIM == 768
    assert config.ARTIFACTS_DIR == Path("/tmp/artifacts")
    assert config.QDRANT_URL == "http://remote:1234"
