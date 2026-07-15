"""
backend/api/dependencies.py

FastAPI dependency functions that inject shared application state
(Neo4j driver, FAISS searcher, retrieval components) into route handlers.

WHY dependency injection instead of global variables:
    - Enables mock injection in tests (replace real Neo4j with a mock)
    - Makes each route's dependencies explicit and traceable
    - Follows FastAPI's recommended pattern for shared resources
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.retrieval.entity_resolver import EntityResolver
from backend.retrieval.cypher_retriever import CypherRetriever
from backend.retrieval.cypher_ranker import CypherRanker
from backend.retrieval.hybrid_ranker import HybridRanker
from backend.retrieval.novel_link_predictor import NovelLinkPredictor
from backend.embeddings.embedding_searcher import EmbeddingSearcher


def get_resolver(request: Request) -> EntityResolver:
    return request.app.state.resolver


def get_retriever(request: Request) -> CypherRetriever:
    return request.app.state.retriever


def get_cypher_ranker(request: Request) -> CypherRanker:
    return request.app.state.cypher_ranker


def get_hybrid_ranker(request: Request) -> HybridRanker:
    return request.app.state.hybrid_ranker


def get_searcher(request: Request) -> EmbeddingSearcher:
    return request.app.state.searcher


def get_novel_predictor(request: Request) -> NovelLinkPredictor:
    return request.app.state.novel_predictor


# Type aliases for cleaner route signatures
ResolverDep = Annotated[EntityResolver, Depends(get_resolver)]
RetrieverDep = Annotated[CypherRetriever, Depends(get_retriever)]
CypherRankerDep = Annotated[CypherRanker, Depends(get_cypher_ranker)]
HybridRankerDep = Annotated[HybridRanker, Depends(get_hybrid_ranker)]
SearcherDep = Annotated[EmbeddingSearcher, Depends(get_searcher)]
NovelPredictorDep = Annotated[NovelLinkPredictor, Depends(get_novel_predictor)]
