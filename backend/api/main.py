"""
backend/api/main.py

FastAPI application entry point.

Startup strategy:
    Heavy resources (Neo4j driver, FAISS index, retrieval components)
    are initialised ONCE in the lifespan context manager and stored in
    app.state. Each request reads from app.state — no per-request
    re-initialisation.

    WHY lifespan over @app.on_event("startup"):
        on_event is deprecated since FastAPI 0.93. Lifespan is the
        modern pattern and integrates cleanly with async context managers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.utils.logger import configure_logging
from backend.graph.driver import create_driver, close_driver
from backend.retrieval.entity_resolver import EntityResolver
from backend.retrieval.cypher_retriever import CypherRetriever
from backend.retrieval.cypher_ranker import CypherRanker
from backend.retrieval.hybrid_ranker import HybridRanker
from backend.retrieval.novel_link_predictor import NovelLinkPredictor
from backend.embeddings.embedding_searcher import EmbeddingSearcher
from backend.api.routes import health, query, repurpose, explain

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — startup and graceful shutdown.

    All shared resources are attached to app.state so that FastAPI's
    dependency injection can read them via request.app.state.
    """
    logger.info("Starting GraphRAG backend")

    # Neo4j driver (connection-pooled)
    driver = create_driver()
    app.state.driver = driver

    # FAISS index over RotatE entity embeddings — loaded once into RAM
    logger.info("Loading RotatE embeddings and building FAISS index")
    searcher = EmbeddingSearcher(
        settings.rotate_embeddings_path,
        settings.rotate_entity_map_path,
    )
    searcher.load()
    app.state.searcher = searcher
    logger.info("FAISS index ready")

    # Retrieval pipeline components
    app.state.resolver = EntityResolver(driver)
    app.state.retriever = CypherRetriever(driver)
    app.state.cypher_ranker = CypherRanker()
    app.state.hybrid_ranker = HybridRanker()
    app.state.novel_predictor = NovelLinkPredictor(driver, searcher)

    logger.info("All components initialised — API ready")
    yield  # server runs here

    # Graceful shutdown
    logger.info("Shutting down — releasing resources")
    close_driver(driver)


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="GraphRAG Backend — Biomedical Knowledge Graph",
        description=(
            "Production GraphRAG backend combining Neo4j graph traversal, "
            "RotatE knowledge graph embeddings, FAISS vector search, "
            "and Reciprocal Rank Fusion over an 8M-edge biomedical graph."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(repurpose.router)
    app.include_router(explain.router)

    return app


app = create_app()
