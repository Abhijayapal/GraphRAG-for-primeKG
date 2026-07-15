"""
backend/api/routes/query.py

POST /query — Hybrid graph + vector retrieval for known entity connections.

Pipeline:
  1. EntityResolver   → canonical graph node name
  2. CypherRetriever  → explicit graph paths (Cypher)
  3. CypherRanker     → score by edge type + depth
  4. EmbeddingSearcher (RotatE/FAISS) → vector similarity
  5. HybridRanker (RRF) → fused ranked list
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import (
    ResolverDep,
    RetrieverDep,
    CypherRankerDep,
    HybridRankerDep,
    SearcherDep,
)
from backend.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retrieval"])


class QueryRequest(BaseModel):
    """Request body for hybrid retrieval."""

    name: str = Field(..., description="Entity name to search for")
    entity_type: str = Field(
        default="disease",
        description="Node label filter: 'disease' | 'drug' | 'gene'",
    )
    top_k: int = Field(
        default=10, ge=1, le=50, description="Maximum results to return"
    )


@router.post(
    "/query",
    summary="Hybrid graph + vector retrieval",
    description=(
        "Retrieves known connections for an entity by fusing Cypher graph traversal "
        "and RotatE embedding similarity via Reciprocal Rank Fusion."
    ),
)
def query(
    req: QueryRequest,
    resolver: ResolverDep,
    retriever: RetrieverDep,
    c_ranker: CypherRankerDep,
    h_ranker: HybridRankerDep,
    searcher: SearcherDep,
) -> dict:
    start = time.perf_counter()

    # Step 1: Entity resolution
    matches = resolver.resolve(req.name, entity_type=req.entity_type.capitalize())
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{req.name}' not found in the knowledge graph.",
        )
    canonical = matches[0]["name"]

    # Step 2: Cypher retrieval + ranking
    cypher_ranked: list[dict] = []
    try:
        cypher_raw = retriever.retrieve(canonical, mode=req.entity_type, top_k=settings.cypher_top_k)
        cypher_ranked = c_ranker.rank(cypher_raw, top_k=settings.cypher_top_k)["candidates"]
    except Exception:
        logger.warning("Cypher retrieval failed for '%s'", canonical, exc_info=True)

    # Step 3: RotatE / FAISS similarity
    rotate_results: list[dict] = []
    try:
        rotate_results = searcher.search(canonical, top_k=settings.rotate_top_k)
    except KeyError:
        logger.debug("Entity '%s' not in RotatE embedding index", canonical)
    except Exception:
        logger.warning("Embedding search failed for '%s'", canonical, exc_info=True)

    # Step 4: RRF fusion
    result = h_ranker.rank(
        disease_name=canonical,
        cypher_results=cypher_ranked,
        rotate_results=rotate_results,
        top_k=req.top_k,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /query query='%s' canonical='%s' results=%d latency=%.1fms",
        req.name, canonical, len(result.get("candidates", [])), elapsed_ms,
    )

    return {
        "query": req.name,
        "canonical": canonical,
        "results": result,
        "meta": {"latency_ms": round(elapsed_ms, 1)},
    }
