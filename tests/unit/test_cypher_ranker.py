"""
tests/unit/test_cypher_ranker.py

Unit tests for CypherRanker — no database required.
"""

from __future__ import annotations

import pytest
from backend.retrieval.cypher_ranker import CypherRanker


@pytest.fixture
def ranker() -> CypherRanker:
    return CypherRanker()


def _make_result(
    name: str,
    rel_type: str,
    path_type: str,
    is_contraindication: bool = False,
) -> dict:
    return {
        "candidate_name": name,
        "rel_type": rel_type,
        "path_type": path_type,
        "is_contraindication": is_contraindication,
        "nodes": [],
    }


def test_rank_empty(ranker: CypherRanker):
    result = ranker.rank([])
    assert result["candidates"] == []
    assert result["warnings"] == []


def test_contraindication_separated(ranker: CypherRanker):
    results = [
        _make_result("SafeDrug", "INDICATION", "direct"),
        _make_result("DangerDrug", "INDICATION", "direct", is_contraindication=True),
    ]
    ranked = ranker.rank(results)
    candidate_names = [c["candidate_name"] for c in ranked["candidates"]]
    warning_names = [w["candidate_name"] for w in ranked["warnings"]]

    assert "SafeDrug" in candidate_names
    assert "DangerDrug" not in candidate_names
    assert "DangerDrug" in warning_names


def test_indication_scores_higher_than_off_label(ranker: CypherRanker):
    results = [
        _make_result("DrugA", "INDICATION", "direct"),
        _make_result("DrugB", "OFF_LABEL_USE", "direct"),
    ]
    ranked = ranker.rank(results, top_k=10)
    candidates = {c["candidate_name"]: c["score"] for c in ranked["candidates"]}
    assert candidates["DrugA"] > candidates["DrugB"]


def test_multi_evidence_bonus_applied(ranker: CypherRanker):
    """A drug with both direct and 1-hop paths gets a bonus."""
    results = [
        _make_result("DrugA", "OFF_LABEL_USE", "direct"),
        _make_result("DrugA", "TARGET", "1-hop"),
        _make_result("DrugB", "TARGET", "1-hop"),
    ]
    ranked = ranker.rank(results, top_k=10)
    candidates = {c["candidate_name"]: c["score"] for c in ranked["candidates"]}
    # DrugA has both direct and 1-hop → bonus applied
    assert candidates["DrugA"] > candidates["DrugB"]


def test_top_k_respected(ranker: CypherRanker):
    results = [_make_result(f"Drug{i}", "TARGET", "1-hop") for i in range(10)]
    ranked = ranker.rank(results, top_k=3)
    assert len(ranked["candidates"]) <= 3


def test_unknown_rel_type_gets_fallback_score(ranker: CypherRanker):
    results = [_make_result("DrugX", "UNKNOWN_REL", "direct")]
    ranked = ranker.rank(results)
    assert ranked["candidates"][0]["score"] == 0.1
