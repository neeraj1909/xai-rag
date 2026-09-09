"""Unit contracts for rank fusion, reranking, and explanations."""

import pytest

from xai_rag.explainability.retrieval_explainer import RetrievalExplainer
from xai_rag.models import FusedResult, RankedResult, SearchResult
from xai_rag.retrieval.hybrid import rrf_fusion
from xai_rag.retrieval.reranker import Reranker

pytestmark = pytest.mark.unit


def _raw(result_id: str, score: float, stage: str, rank: int) -> SearchResult:
    return SearchResult(
        id=result_id,
        content=f"content for {result_id}",
        score=score,
        stage=stage,
        rank=rank,
    )


def test_rrf_preserves_native_scores_and_ranks() -> None:
    vector = [_raw("both", -0.2, "vector", 1), _raw("vector-only", -0.5, "vector", 2)]
    bm25 = [_raw("bm25-only", 8.0, "bm25", 1), _raw("both", 4.0, "bm25", 2)]

    fused = rrf_fusion([vector, bm25], k=60)
    by_id = {result.id: result for result in fused}

    assert by_id["both"].vector_score == -0.2
    assert by_id["both"].vector_rank == 1
    assert by_id["both"].bm25_score == 4.0
    assert by_id["both"].bm25_rank == 2
    assert by_id["both"].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["both"].rrf_rank == 1
    assert by_id["vector-only"].bm25_score is None


def test_rrf_ties_have_deterministic_id_order() -> None:
    first = [_raw("z", 1.0, "vector", 1)]
    second = [_raw("a", 1.0, "bm25", 1)]

    assert [result.id for result in rrf_fusion([first, second], k=60)] == ["a", "z"]


@pytest.mark.asyncio
async def test_reranker_does_not_overwrite_rrf_rank(monkeypatch) -> None:
    candidates = [
        FusedResult(id="a", content="alpha", rrf_score=0.04, rrf_rank=1),
        FusedResult(id="b", content="beta", rrf_score=0.03, rrf_rank=2),
    ]
    reranker = Reranker()
    monkeypatch.setattr(reranker, "_score_pairs", lambda query, passages: [0.1, 0.9])

    ranked = await reranker.rerank("query", candidates, top_k=2)

    assert [item.id for item in ranked] == ["b", "a"]
    assert [item.rrf_rank for item in ranked] == [2, 1]
    assert [item.reranker_rank for item in ranked] == [1, 2]
    assert [item.rrf_score for item in ranked] == [0.03, 0.04]


def test_explainer_uses_rank_membership_even_for_negative_similarity() -> None:
    ranked = [
        RankedResult(
            id="negative",
            content="semantic evidence",
            vector_score=-0.25,
            vector_rank=1,
            rrf_score=1 / 61,
            rrf_rank=1,
            reranker_score=0.8,
            reranker_rank=1,
        )
    ]

    explanation = RetrievalExplainer().explain("semantic", [], [], ranked)[0]

    assert "semantic vector search" in explanation.selection_reason
    assert "not by keyword BM25" in explanation.selection_reason
