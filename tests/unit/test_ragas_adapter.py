"""The optional external evaluator must never fake a successful empty result."""

import builtins

import pytest

from xai_rag.evaluation.ragas_eval import evaluate_response

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_missing_ragas_dependency_has_explicit_status(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"datasets", "ragas"} or name.startswith("ragas."):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = await evaluate_response("query", "answer", ["context"])

    assert result == {
        "status": "unavailable",
        "scores": {},
        "error_type": "dependency_unavailable",
    }
