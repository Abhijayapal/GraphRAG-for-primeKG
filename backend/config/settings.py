"""
backend/config/settings.py

Centralised configuration using Pydantic BaseSettings.
All values are read from environment variables / .env file.
No hardcoded connection strings or file paths anywhere else in the codebase.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 30

    # RotatE embedding files
    rotate_embeddings_path: str = "embeddings/rotate_data/rotate_embeddings.npy"
    rotate_entity_map_path: str = "embeddings/rotate_data/rotate_entity_map.json"

    # PyKEEN trained RotatE model (proper complex embeddings + triple scoring)
    pykeen_model_path: str = "trained_model.pkl"
    pykeen_train_csv: str  = "train.csv"
    pykeen_test_csv: str   = "test.csv"

    # Retrieval defaults
    default_top_k: int = 10
    cypher_top_k: int = 50
    rotate_top_k: int = 50

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # Groq (UI only)
    groq_api_key: str = ""

    # Logging
    log_level: str = "INFO"


# Module-level singleton — import this everywhere instead of os.getenv()
settings = Settings()
