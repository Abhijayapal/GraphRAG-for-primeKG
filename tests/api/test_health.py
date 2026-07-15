"""
tests/api/test_health.py

API tests for the /health and /ready endpoints.
"""

from __future__ import annotations

import pytest


def test_health_returns_200(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_200_when_state_populated(api_client):
    response = api_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "dependencies" in data
    assert "neo4j" in data["dependencies"]
    assert "embeddings" in data["dependencies"]


def test_health_response_structure(api_client):
    data = api_client.get("/health").json()
    assert "status" in data
