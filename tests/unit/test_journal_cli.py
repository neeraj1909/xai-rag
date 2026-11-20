"""Scoped recovery uses current tenant grants."""

import hashlib
import json

import click
import pytest

from xai_rag import cli
from xai_rag.authorization import tenant_store_name
from xai_rag.config import settings

pytestmark = pytest.mark.unit


def test_cli_policy_grant_resolves_tenant_stores(tmp_path, monkeypatch):
    key = "a" * 32
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "principals": [
                    {
                        "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        "tenant": "customer-a",
                        "sources": ["one.txt"],
                        "role": "ingester",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(settings, "auth_policy_file", str(policy))
    monkeypatch.setenv("XAI_RAG_INGEST_API_KEY", key)
    scope, collection, index = cli._ingestion_identity()
    assert scope.sources == ("one.txt",)
    assert collection == tenant_store_name(settings.chroma_collection, "customer-a")
    assert index == tenant_store_name(settings.elasticsearch_index, "customer-a")
    monkeypatch.setenv("XAI_RAG_INGEST_API_KEY", "bad")
    with pytest.raises(click.ClickException):
        cli._ingestion_identity()
