"""
backend/retrieval/hybrid_ranker.py

Reciprocal Rank Fusion (RRF) fusion of Cypher and RotatE retrieval results.

WHY RRF instead of weighted score averaging:
    Cypher scores (0.3–1.2) and RotatE scores (0.75–0.99) live on
    incompatible scales — direct arithmetic comparison is meaningless.
    RRF uses only rank position, which is comparable across any retrieval method.

    Formula: rrf(rank) = 1 / (K + rank),  K=60 (Cormack et al., 2009)
    K=60 dampens the outsized advantage of rank-1 over rank-2.

Source weights:
    cypher: 0.6  — explicit graph paths, highest factual trust
    rotate: 0.4  — relation-aware KGE, captures latent patterns
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_RRF_K: int = 60

_WEIGHTS: dict[str, float] = {
    "cypher": 0.6,
    "rotate": 0.4,
}


class HybridRanker:
    """
    Fuses Cypher and RotatE ranked lists using Reciprocal Rank Fusion.

    No external dependencies — stateless, instantiated once at startup.
    """

    def rank(
        self,
        disease_name: str,
        cypher_results: list[dict],
        rotate_results: list[dict],
        top_k: int = 10,
    ) -> dict:
        """
        Merge and re-rank two retrieval result lists.

        Args:
            disease_name:   Query entity (used for logging/meta only).
            cypher_results: Output from CypherRanker.rank()["candidates"].
                            Each dict must contain "candidate_name" and optionally "path_str".
            rotate_results: Output from EmbeddingSearcher.search().
                            Each dict must contain "name".
            top_k:          Number of results to return.

        Returns:
            {
                "candidates": [{"name", "hybrid_score", "cypher_rank",
                                "rotate_rank", "path_str", "sources"}, ...],
                "meta": {"cypher_count", "rotate_count", "union_count"}
            }
        """
        cypher_rank_map = self._build_rank_map(
            [r["candidate_name"] for r in cypher_results]
        )
        rotate_rank_map = self._build_rank_map(
            [r["name"] for r in rotate_results]
        )

        path_lookup: dict[str, str] = {
            r["candidate_name"]: r.get("path_str", "") for r in cypher_results
        }

        all_entities = set(cypher_rank_map) | set(rotate_rank_map)
        scored: list[dict] = []

        for entity in all_entities:
            c_rank: Optional[int] = cypher_rank_map.get(entity)
            r_rank: Optional[int] = rotate_rank_map.get(entity)

            hybrid_score = (
                _WEIGHTS["cypher"] * self._rrf(c_rank)
                + _WEIGHTS["rotate"] * self._rrf(r_rank)
            )

            sources = []
            if c_rank is not None:
                sources.append("cypher")
            if r_rank is not None:
                sources.append("rotate")

            scored.append(
                {
                    "name": entity,
                    "hybrid_score": round(hybrid_score, 6),
                    "cypher_rank": c_rank,
                    "rotate_rank": r_rank,
                    "path_str": path_lookup.get(entity),
                    "sources": sources,
                }
            )

        scored.sort(key=lambda x: (-x["hybrid_score"], x["name"].lower()))

        logger.debug(
            "HybridRanker: query='%s' cypher=%d rotate=%d union=%d returning=%d",
            disease_name,
            len(cypher_rank_map),
            len(rotate_rank_map),
            len(all_entities),
            min(top_k, len(scored)),
        )

        return {
            "candidates": scored[:top_k],
            "meta": {
                "cypher_count": len(cypher_rank_map),
                "rotate_count": len(rotate_rank_map),
                "union_count": len(all_entities),
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rank_map(names: list[str]) -> dict[str, int]:
        """Convert an ordered name list to {name: rank} (1-based, deduplicated)."""
        rank_map: dict[str, int] = {}
        rank = 1
        for name in names:
            if name not in rank_map:
                rank_map[name] = rank
                rank += 1
        return rank_map

    @staticmethod
    def _rrf(rank: Optional[int]) -> float:
        """RRF score for a given rank position. Returns 0.0 if rank is None."""
        if rank is None:
            return 0.0
        return 1.0 / (_RRF_K + rank)
