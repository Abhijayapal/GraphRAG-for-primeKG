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
from backend.embeddings.pykeen_searcher import PyKEENSearcher
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

    # FAISS index over RotatE entity embeddings — loaded once into RAM (legacy)
    logger.info("Loading RotatE embeddings and building FAISS index")
    searcher = EmbeddingSearcher(
        settings.rotate_embeddings_path,
        settings.rotate_entity_map_path,
    )
    searcher.load()
    app.state.searcher = searcher
    logger.info("FAISS index ready")

    # PyKEEN RotatE model — proper triple scoring for drug repurposing
    import os
    if os.path.exists(settings.pykeen_model_path):
        logger.info("Loading PyKEEN RotatE model from %s", settings.pykeen_model_path)
        pykeen_searcher = PyKEENSearcher(
            settings.pykeen_model_path,
            settings.pykeen_train_csv,
            settings.pykeen_test_csv,
        )
        pykeen_searcher.load()
        app.state.pykeen_searcher = pykeen_searcher
        novel_predictor_searcher = pykeen_searcher  # use triple scoring
        logger.info("PyKEEN searcher ready — using proper triple scoring")
    else:
        logger.warning(
            "trained_model.pkl not found at '%s'. "
            "Falling back to legacy cosine similarity for novel predictions.",
            settings.pykeen_model_path,
        )
        app.state.pykeen_searcher = None
        novel_predictor_searcher = searcher  # fallback

    # Retrieval pipeline components
    app.state.resolver = EntityResolver(driver)
    app.state.retriever = CypherRetriever(driver)
    app.state.cypher_ranker = CypherRanker()
    app.state.hybrid_ranker = HybridRanker()
    app.state.novel_predictor = NovelLinkPredictor(driver, novel_predictor_searcher)

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
