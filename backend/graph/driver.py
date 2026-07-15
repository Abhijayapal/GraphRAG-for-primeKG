"""
backend/graph/driver.py

Neo4j driver factory with connection pooling and validation.
The driver is created once at startup and shared across all requests
via FastAPI's dependency injection system.
"""

from __future__ import annotations

import logging
from typing import Optional

from neo4j import GraphDatabase, Driver

from backend.config.settings import settings

logger = logging.getLogger(__name__)


def create_driver() -> Driver:
    """
    Create and validate a Neo4j bolt driver.

    Uses connection pooling (max_connection_pool_size) to handle
    concurrent requests without opening a new socket per request.

    Raises:
        neo4j.exceptions.ServiceUnavailable: if Neo4j is unreachable.
        neo4j.exceptions.AuthError: if credentials are wrong.
    """
    logger.info(
        "Connecting to Neo4j",
        extra={"uri": settings.neo4j_uri, "user": settings.neo4j_user},
    )

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        connection_timeout=settings.neo4j_connection_timeout,
    )

    driver.verify_connectivity()
    logger.info("Neo4j connection verified")
    return driver


def close_driver(driver: Optional[Driver]) -> None:
    """Gracefully close the driver and release pooled connections."""
    if driver is not None:
        driver.close()
        logger.info("Neo4j driver closed")
