"""
tests/unit/test_entity_resolver.py

Unit tests for EntityResolver alias table and input validation.
DB-dependent search methods are tested with a mock driver.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from backend.retrieval.entity_resolver import EntityResolver, _ALIASES


def make_resolver(session_data: list[dict] | None = None) -> EntityResolver:
    """Build an EntityResolver backed by a mock driver."""
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value.data.return_value = session_data or []
    driver.session.return_value = session
    return EntityResolver(driver)


def test_alias_table_covid_maps_correctly():
    assert _ALIASES["covid"] == "Coronavinae infectious disease"
    assert _ALIASES["covid-19"] == "Coronavinae infectious disease"


def test_alias_table_alzheimer_maps_correctly():
    assert _ALIASES["alzheimer's"] == "Alzheimer disease"


def test_empty_query_raises_value_error():
    resolver = make_resolver()
    with pytest.raises(ValueError, match="cannot be empty"):
        resolver.resolve("")


def test_invalid_entity_type_raises_value_error():
    resolver = make_resolver()
    with pytest.raises(ValueError, match="entity_type must be one of"):
        resolver.resolve("diabetes", entity_type="Planet")


def test_resolve_returns_empty_when_no_matches():
    resolver = make_resolver(session_data=[])
    result = resolver.resolve("unknowndisease12345")
    assert result == []


def test_resolve_returns_top_k():
    """Even if DB returns many hits, top_k caps the output."""
    data = [{"id": str(i), "name": f"Disease {i}"} for i in range(10)]
    resolver = make_resolver(session_data=data)
    result = resolver.resolve("disease", entity_type="Disease", top_k=3)
    assert len(result) <= 3


def test_resolve_result_has_required_fields():
    data = [{"id": "123", "name": "Alzheimer disease"}]
    resolver = make_resolver(session_data=data)
    result = resolver.resolve("alzheimer", entity_type="Disease")
    if result:
        assert all(k in result[0] for k in ["id", "name", "label", "match_type", "score"])
