"""Configuration management for the VectorDB Stress Tester.

Uses Pydantic's Settings management to load configuration from environment
variables. This allows for easy configuration of the application without
hardcoding values.

The settings can be overridden by command-line arguments where applicable.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Global application configuration settings.

    Attributes:
        SEED: A seed for reproducibility.
        DIM: The dimension of the embeddings.
        ARTIFACTS_DIR: The directory to store artifacts.
        QDRANT_URL: The URL for the Qdrant instance.
        WEAVIATE_URL: The URL for the Weaviate instance.
        MILVUS_URI: The URI for the Milvus instance.
        PINECONE_API_KEY: The API key for the Pinecone instance.
        PINECONE_ENV: The environment for the Pinecone instance.
    """

    model_config = SettingsConfigDict(env_prefix="VDBT_")

    SEED: int = 42
    DIM: int = 384
    ARTIFACTS_DIR: Path = Path("./artifacts")
    QDRANT_URL: str = "http://localhost:6333"
    WEAVIATE_URL: str | None = None
    MILVUS_URI: str | None = None
    PINECONE_API_KEY: str | None = None
    PINECONE_ENV: str | None = None


settings = AppConfig()
