"""
backend/api/routes/health.py

Health and readiness endpoints.

/health   — liveness probe (is the process running?)
/ready    — readiness probe (are all dependencies loaded?)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 if the API process is running.",
)
def health() -> dict:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Returns 200 only when Neo4j and FAISS index are both loaded.",
)
def ready(request: Request) -> dict:
    """
    Checks that startup completed successfully.
    Returns 503 if any dependency failed to load.
    """
    state = request.app.state
    neo4j_ok = getattr(state, "driver", None) is not None
    embeddings_ok = getattr(state, "searcher", None) is not None

    all_ok = neo4j_ok and embeddings_ok
    return {
        "status": "ready" if all_ok else "not_ready",
        "dependencies": {
            "neo4j": "connected" if neo4j_ok else "unavailable",
            "embeddings": "loaded" if embeddings_ok else "not_loaded",
        },
    }
