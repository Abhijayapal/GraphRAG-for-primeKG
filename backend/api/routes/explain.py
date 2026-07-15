"""
backend/api/routes/explain.py

POST /explain — Retrieve the biological pathway between a drug and disease.

Returns:
  - Direct drug-disease edges (if any)
  - Indirect Drug → Gene → Disease 1-hop paths
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import RetrieverDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retrieval"])


class ExplainRequest(BaseModel):
    """Request body for biological path explanation."""

    drug: str = Field(..., description="Drug node name")
    disease: str = Field(..., description="Disease node name")
    gene: Optional[str] = Field(default=None, description="Optional intermediate gene")


@router.post(
    "/explain",
    summary="Biological pathway explanation",
    description="Returns graph paths connecting a drug to a disease.",
)
def explain(req: ExplainRequest, retriever: RetrieverDep) -> dict:
    gene = req.gene or "_placeholder_"
    try:
        result = retriever.retrieve_combined(
            drug=req.drug,
            gene=gene,
            disease=req.disease,
        )
    except Exception as exc:
        logger.error("Explain failed for %s → %s: %s", req.drug, req.disease, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


# ── Entity resolution endpoint ──────────────────────────────────────────────────

from backend.api.dependencies import ResolverDep  # noqa: E402


@router.get(
    "/entity/{name}",
    summary="Entity resolution",
    description="Fuzzy-match a query string to canonical graph node names.",
)
def get_entity(name: str, resolver: ResolverDep, entity_type: str = "disease") -> dict:
    matches = resolver.resolve(name, entity_type=entity_type.capitalize(), top_k=5)
    return {"query": name, "matches": matches}
