"""
tests/unit/test_hybrid_ranker.py

Unit tests for HybridRanker — no Neo4j or FAISS required.
"""

from __future__ import annotations

import pytest
from backend.retrieval.hybrid_ranker import HybridRanker


@pytest.fixture
def ranker() -> HybridRanker:
    return HybridRanker()


def test_rrf_returns_zero_for_none(ranker: HybridRanker):
    assert ranker._rrf(None) == 0.0


def test_rrf_decreases_with_rank(ranker: HybridRanker):
    """Higher rank position → lower RRF score."""
    assert ranker._rrf(1) > ranker._rrf(5) > ranker._rrf(20)


def test_build_rank_map_one_based(ranker: HybridRanker):
    result = ranker._build_rank_map(["A", "B", "C"])
    assert result == {"A": 1, "B": 2, "C": 3}


def test_build_rank_map_deduplication(ranker: HybridRanker):
    """Duplicate entries keep the first rank."""
    result = ranker._build_rank_map(["A", "B", "A"])
    assert result["A"] == 1
    assert result["B"] == 2
    assert len(result) == 2


def test_rank_empty_inputs(ranker: HybridRanker):
    result = ranker.rank("disease", [], [], top_k=10)
    assert result["candidates"] == []
    assert result["meta"]["union_count"] == 0


def test_rank_cypher_only(ranker: HybridRanker):
    cypher = [{"candidate_name": "DrugA", "path_str": "DrugA → Disease"}]
    result = ranker.rank("disease", cypher, [], top_k=5)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["name"] == "DrugA"
    assert result["candidates"][0]["sources"] == ["cypher"]


def test_rank_rotate_only(ranker: HybridRanker):
    rotate = [{"name": "DrugB"}, {"name": "DrugC"}]
    result = ranker.rank("disease", [], rotate, top_k=5)
    assert len(result["candidates"]) == 2
    assert all("rotate" in c["sources"] for c in result["candidates"])


def test_rank_union_scores_drug_in_both_higher(ranker: HybridRanker):
    """A drug ranked #1 in both lists should score higher than one in only one list."""
    cypher = [{"candidate_name": "SharedDrug", "path_str": ""}]
    rotate = [{"name": "SharedDrug"}, {"name": "UniqueOnly"}]
    result = ranker.rank("disease", cypher, rotate, top_k=10)

    names = [c["name"] for c in result["candidates"]]
    scores = {c["name"]: c["hybrid_score"] for c in result["candidates"]}

    assert "SharedDrug" in names
    assert "UniqueOnly" in names
    assert scores["SharedDrug"] > scores["UniqueOnly"]


def test_rank_respects_top_k(ranker: HybridRanker):
    rotate = [{"name": f"Drug{i}"} for i in range(20)]
    result = ranker.rank("disease", [], rotate, top_k=5)
    assert len(result["candidates"]) == 5


def test_rank_meta_counts(ranker: HybridRanker):
    cypher = [{"candidate_name": "A", "path_str": ""}, {"candidate_name": "B", "path_str": ""}]
    rotate = [{"name": "B"}, {"name": "C"}]
    result = ranker.rank("d", cypher, rotate, top_k=10)
    assert result["meta"]["cypher_count"] == 2
    assert result["meta"]["rotate_count"] == 2
    assert result["meta"]["union_count"] == 3   # A, B, C
