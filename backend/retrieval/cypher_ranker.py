"""
backend/retrieval/cypher_ranker.py

Scores and ranks results from CypherRetriever by evidence strength.

Scoring logic:
    - Base score assigned per relationship type + path depth combination.
    - Multi-evidence bonus (+0.2) when the same drug appears in both
      a direct path AND a 1-hop path (independent lines of evidence).
    - Contraindications are separated and returned as warnings.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.retrieval.path_extractor import PathExtractor

logger = logging.getLogger(__name__)

_BASE_SCORES: dict[tuple[str, str], float] = {
    # Direct drug-disease relationships
    ("INDICATION", "direct"): 1.0,
    ("OFF_LABEL_USE", "direct"): 0.6,
    # 1-hop via gene (scored by drug-gene relationship type)
    ("TARGET", "1-hop"): 0.4,
    ("ENZYME", "1-hop"): 0.3,
    ("TRANSPORTER", "1-hop"): 0.3,
    ("CARRIER", "1-hop"): 0.3,
}

_MULTI_EVIDENCE_BONUS: float = 0.2


class CypherRanker:
    """
    Ranks CypherRetriever results by evidence type and path depth.

    Stateless — instantiated once at application startup.
    """

    def __init__(self) -> None:
        self._extractor = PathExtractor()

    def rank(self, results: list[dict], top_k: int = 10) -> dict:
        """
        Score and rank retrieval results.

        Args:
            results: Raw output from CypherRetriever.retrieve().
            top_k:   Maximum candidates to return.

        Returns:
            {
                "candidates": [ranked dicts with "score" and "path_str"],
                "warnings":   [contraindication dicts]
            }
        """
        candidates: list[dict] = []
        warnings: list[dict] = []

        for res in results:
            res["path_str"] = self._extractor.format(res)
            if res.get("is_contraindication"):
                warnings.append(res)
            else:
                res["score"] = self._compute_score(res)
                candidates.append(res)

        candidates = self._apply_multi_evidence_bonus(candidates)
        candidates.sort(key=lambda x: (-x["score"], x["candidate_name"].lower()))

        logger.debug(
            "CypherRanker: %d candidates, %d warnings, returning top %d",
            len(candidates),
            len(warnings),
            top_k,
        )

        return {
            "candidates": candidates[:top_k],
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(result: dict) -> float:
        """Look up the base score for a result by rel_type + path_type."""
        key = (result.get("rel_type", ""), result.get("path_type", ""))
        return _BASE_SCORES.get(key, 0.1)

    @staticmethod
    def _apply_multi_evidence_bonus(candidates: list[dict]) -> list[dict]:
        """
        Grant a bonus to drugs supported by both direct and 1-hop paths.

        Two independent evidence lines give higher confidence than one path alone.
        """
        name_path_types: dict[str, set] = {}
        for res in candidates:
            name = res["candidate_name"]
            name_path_types.setdefault(name, set()).add(res.get("path_type", ""))

        multi_evidence = {
            name
            for name, types in name_path_types.items()
            if "direct" in types and "1-hop" in types
        }

        for res in candidates:
            if res["candidate_name"] in multi_evidence:
                res["score"] += _MULTI_EVIDENCE_BONUS

        return candidates
