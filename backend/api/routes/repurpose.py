"""
backend/api/routes/repurpose.py

POST /repurpose — Novel link prediction for drug discovery.

Unlike /query (which retrieves *known* edges), this endpoint scores
drug-disease pairs that do NOT yet exist in the graph and returns
ranked novel candidates.

Pipeline:
  1. EntityResolver      → canonical disease node name
  2. NovelLinkPredictor  → filter drugs with no existing edge,
                           score via RotatE + biological plausibility bonus,
                           return ranked predictions
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import ResolverDep, NovelPredictorDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Discovery"])


class RepurposeRequest(BaseModel):
    """Request body for novel drug candidate prediction."""

    disease_name: str = Field(..., description="Disease to find novel drug candidates for")
    top_k: int = Field(default=20, ge=1, le=100, description="Maximum predictions to return")


@router.post(
    "/repurpose",
    summary="Novel drug candidate discovery",
    description=(
        "Predicts drug candidates with NO existing connection to the disease. "
        "Uses knowledge graph embeddings to score unseen drug-disease pairs."
    ),
)
def repurpose(
    req: RepurposeRequest,
    resolver: ResolverDep,
    predictor: NovelPredictorDep,
) -> dict:
    start = time.perf_counter()

    matches = resolver.resolve(req.disease_name, entity_type="Disease")
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Disease '{req.disease_name}' not found in the knowledge graph.",
        )
    canonical = matches[0]["name"]

    try:
        result = predictor.predict(canonical, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Novel prediction failed for '%s': %s", canonical, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction pipeline error.") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /repurpose disease='%s' canonical='%s' predictions=%d latency=%.1fms",
        req.disease_name, canonical,
        len(result.get("predictions", [])),
        elapsed_ms,
    )

    return {
        "query": req.disease_name,
        "canonical": canonical,
        "result": result,
        "meta": {"latency_ms": round(elapsed_ms, 1)},
    }
