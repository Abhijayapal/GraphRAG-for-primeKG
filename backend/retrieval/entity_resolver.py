"""
backend/retrieval/entity_resolver.py

Maps raw user input (e.g. "covid", "alzheimer's") to canonical node
names stored in the Neo4j knowledge graph.

Matching strategy (priority order):
  1. Manual alias table  — handles well-known synonyms deterministically
  2. Exact match         — toLower() equality                 (score 1.0)
  3. Prefix match        — name starts with the query term    (score 0.7)
  4. Contains match      — name contains the query term       (score 0.4)

Results are deduplicated and sorted by score descending.
"""

from __future__ import annotations

import logging
from typing import Optional

from neo4j import Driver

logger = logging.getLogger(__name__)

SUPPORTED_LABELS: list[str] = ["Drug", "Gene", "Disease"]

MATCH_SCORES: dict[str, float] = {
    "exact": 1.0,
    "starts_with": 0.7,
    "contains": 0.4,
}

# Alias table: maps lowercase user input → canonical PrimeKG node name.
# Required because PrimeKG uses 2019-era terminology that does not match
# modern common names (e.g. COVID-19 was not named until January 2020).
_ALIASES: dict[str, str] = {
    "covid": "Coronavinae infectious disease",
    "covid-19": "Coronavinae infectious disease",
    "covid19": "Coronavinae infectious disease",
    "sars-cov-2": "Coronavinae infectious disease",
    "coronavirus": "Coronavinae infectious disease",
    "alzheimer's": "Alzheimer disease",
    "alzheimer": "Alzheimer disease",
    "alzheimers": "Alzheimer disease",
}


class EntityResolver:
    """
    Resolves free-text entity names to canonical graph node names.

    Args:
        driver: An active Neo4j Driver instance (injected at startup).
    """

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        query_text: str,
        entity_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Resolve a user query string to graph entity candidates.

        Args:
            query_text:  Raw user input (e.g. "covid", "Metformin").
            entity_type: Optional label filter: "Drug" | "Gene" | "Disease".
            top_k:       Maximum candidates to return.

        Returns:
            List of dicts: [{id, name, label, match_type, score}, ...],
            sorted by score descending.

        Raises:
            ValueError: if query_text is empty or entity_type is unsupported.
        """
        query_text = query_text.strip()
        if not query_text:
            raise ValueError("query_text cannot be empty.")

        if entity_type is not None and entity_type not in SUPPORTED_LABELS:
            raise ValueError(
                f"entity_type must be one of {SUPPORTED_LABELS}, got '{entity_type}'."
            )

        # Apply alias table first (deterministic, no DB round-trip)
        resolved = _ALIASES.get(query_text.lower(), query_text)
        if resolved != query_text:
            logger.debug("Alias resolved: '%s' → '%s'", query_text, resolved)

        labels_to_search = [entity_type] if entity_type else SUPPORTED_LABELS
        results: list[dict] = []
        seen_keys: set[str] = set()

        with self._driver.session() as session:
            for label in labels_to_search:
                hits = self._search_label(session, label, resolved)
                for hit in hits:
                    key = f"{hit['label']}::{hit['id']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results.append(hit)

        results.sort(key=lambda x: (-x["score"], x["name"].lower()))
        logger.debug("Resolved '%s' → %d candidates", query_text, len(results))
        return results[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_label(self, session, label: str, query_text: str) -> list[dict]:
        """Run all three matching strategies against one node label."""
        hits: list[dict] = []
        q = query_text.lower()

        # Strategy 1: exact
        rows = session.run(
            f"MATCH (n:{label}) WHERE toLower(n.name) = $q RETURN n.id AS id, n.name AS name LIMIT 10",
            q=q,
        ).data()
        for row in rows:
            hits.append(self._make_result(row["id"], row["name"], label, "exact"))

        # Strategy 2: prefix
        rows = session.run(
            f"MATCH (n:{label}) WHERE toLower(n.name) STARTS WITH $q AND toLower(n.name) <> $q RETURN n.id AS id, n.name AS name LIMIT 10",
            q=q,
        ).data()
        for row in rows:
            hits.append(self._make_result(row["id"], row["name"], label, "starts_with"))

        # Strategy 3: contains
        rows = session.run(
            f"MATCH (n:{label}) WHERE toLower(n.name) CONTAINS $q AND NOT toLower(n.name) STARTS WITH $q RETURN n.id AS id, n.name AS name LIMIT 20",
            q=q,
        ).data()
        for row in rows:
            hits.append(self._make_result(row["id"], row["name"], label, "contains"))

        return hits

    @staticmethod
    def _make_result(
        node_id: str, name: str, label: str, match_type: str
    ) -> dict:
        return {
            "id": node_id,
            "name": name,
            "label": label,
            "match_type": match_type,
            "score": MATCH_SCORES[match_type],
        }
