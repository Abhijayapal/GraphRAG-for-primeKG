"""
tests/conftest.py

Shared pytest fixtures.

Key design: all Neo4j tests use a MagicMock driver so the test suite
runs WITHOUT a live database. Only integration tests (marked separately)
require a real Neo4j instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_driver():
    """Mock Neo4j Driver that returns empty results by default."""
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value.data.return_value = []
    driver.session.return_value = session
    return driver


@pytest.fixture
def mock_searcher():
    """Mock EmbeddingSearcher that returns empty results by default."""
    searcher = MagicMock()
    searcher.search.return_value = []
    return searcher


@pytest.fixture
def api_client(mock_driver, mock_searcher):
    """
    FastAPI TestClient with all external dependencies mocked.
    Allows testing routes without a real Neo4j or FAISS index.
    """
    from backend.api.main import create_app

    app = create_app()

    # Bypass lifespan — inject mocks directly into app.state
    from backend.retrieval.entity_resolver import EntityResolver
    from backend.retrieval.cypher_retriever import CypherRetriever
    from backend.retrieval.cypher_ranker import CypherRanker
    from backend.retrieval.hybrid_ranker import HybridRanker
    from backend.retrieval.novel_link_predictor import NovelLinkPredictor

    app.state.driver = mock_driver
    app.state.searcher = mock_searcher
    app.state.resolver = MagicMock()
    app.state.retriever = MagicMock()
    app.state.cypher_ranker = CypherRanker()
    app.state.hybrid_ranker = HybridRanker()
    app.state.novel_predictor = MagicMock()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
