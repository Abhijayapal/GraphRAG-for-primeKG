"""
tests/api/test_query.py

API tests for POST /query.
Uses mocked resolver and retriever to avoid database dependency.
"""

from __future__ import annotations

import pytest


def test_query_returns_404_when_entity_not_found(api_client):
    # Resolver returns empty list → 404
    api_client.app.state.resolver.resolve.return_value = []
    response = api_client.post("/query", json={"name": "nonexistent_entity"})
    assert response.status_code == 404


def test_query_returns_200_on_valid_entity(api_client):
    api_client.app.state.resolver.resolve.return_value = [
        {"id": "1", "name": "Alzheimer disease", "label": "Disease",
         "match_type": "exact", "score": 1.0}
    ]
    api_client.app.state.retriever.retrieve.return_value = []
    api_client.app.state.searcher.search.return_value = []

    response = api_client.post(
        "/query",
        json={"name": "alzheimer", "entity_type": "disease", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "canonical" in data
    assert data["canonical"] == "Alzheimer disease"
    assert "results" in data


def test_query_response_includes_meta(api_client):
    api_client.app.state.resolver.resolve.return_value = [
        {"id": "1", "name": "Test Disease", "label": "Disease",
         "match_type": "exact", "score": 1.0}
    ]
    api_client.app.state.retriever.retrieve.return_value = []
    api_client.app.state.searcher.search.return_value = []

    response = api_client.post("/query", json={"name": "test"})
    assert response.status_code == 200
    assert "meta" in response.json()
    assert "latency_ms" in response.json()["meta"]


def test_query_invalid_top_k_rejected(api_client):
    response = api_client.post("/query", json={"name": "test", "top_k": 200})
    assert response.status_code == 422   # Pydantic validation


def test_query_missing_name_rejected(api_client):
    response = api_client.post("/query", json={})
    assert response.status_code == 422
