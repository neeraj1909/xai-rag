"""Input validation prevents blank requests from reaching the RAG pipeline."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from xai_rag.api import app as api
from xai_rag.models import QueryRequest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("query", [" ", " \t\r\n", "\u2003"])
def test_whitespace_only_query_is_rejected(query: str) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query=query)


def test_nonempty_query_preserves_whitespace() -> None:
    query = "  What is RAG?  "
    assert QueryRequest(query=query).query == query


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
def test_http_blank_query_rejected_before_service(path: str, monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("blank requests must not reach the service")

    monkeypatch.setitem(api.app.dependency_overrides, api.require_api_key, lambda: None)
    monkeypatch.setattr(api, "_build_service", fail_if_called)
    response = TestClient(api.app).post(path, json={"query": " \t"})
    assert response.status_code == 422
