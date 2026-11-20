import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "multilingual_compare", Path(__file__).with_name("compare.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rank_and_aggregate_exclude_unlabeled_and_require_all_sources():
    documents = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
    cases = [
        {"id": "q1", "relevant": ["a", "c"]},
        {"id": "q2", "relevant": []},
    ]
    rows = module.rank_cases(documents, cases, [[0.5, 0.5, 0.4], [0.2, 0.3, 0.4]])
    assert [item["id"] for item in rows[0]["ranking"]] == ["a", "b", "c"]
    assert rows[0]["relevant_ranks"] == {"a": 1, "c": 3}
    assert module.summarize(rows) == {
        "planned": 2,
        "labeled": 1,
        "unlabeled": 1,
        "hit_at_1": 1.0,
        "hit_at_5": 1.0,
        "all_at_3": 1.0,
        "all_at_5": 1.0,
    }


def test_reject_unknown_relevant_ids():
    with pytest.raises(ValueError, match="unknown"):
        module.rank_cases([{"id": "a"}], [{"id": "q", "relevant": ["x"]}], [[1.0]])
