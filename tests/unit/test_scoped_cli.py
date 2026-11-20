"""Scoped mode must not silently operate on shared indexes."""

import pytest
from click.testing import CliRunner

from xai_rag.authorization import AccessScope
from xai_rag.cli import main
from xai_rag.config import settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("args", [["query", "hi"]])
def test_scoped_cli_rejects_unscoped_commands_before_connecting(args, monkeypatch):
    monkeypatch.setattr(settings, "auth_policy_file", "/configured/policy.json")
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "Scoped CLI access is not configured" in result.output


def test_scoped_ingest_cannot_rebind_grant_to_another_root(tmp_path, monkeypatch):
    root = tmp_path / "configured"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (root / "allowed.txt").write_text("permitted")
    outside = other / "allowed.txt"
    outside.write_text("private")
    monkeypatch.setattr(settings, "auth_policy_file", "configured-policy")
    monkeypatch.setattr(settings, "ingest_root", str(root))
    monkeypatch.setattr(
        "xai_rag.cli._ingestion_identity",
        lambda: (AccessScope("tenant", ("allowed.txt",), True), "collection", "index"),
    )
    for arguments in ([str(outside)], [str(outside), "--document-root", str(other)]):
        result = CliRunner().invoke(main, ["ingest", *arguments])
        assert result.exit_code != 0
        assert "outside the document root" in result.output or "configured root" in result.output
