"""Behavior checks for the live holdout evaluator."""

from runner import aggregate, validate_fixture


def test_metrics_keep_unlabeled_cases_and_failures_out_of_quality_means():
    rows = [
        {"relevant": ["a", "b"], "stages": {"vector": {"ranking": ["a", "x", "b"]}}},
        {"relevant": ["a"], "stages": {"vector": {"error": "TimeoutError"}}},
        {"relevant": [], "stages": {"vector": {"ranking": ["a"]}}},
    ]
    result = aggregate(rows)["vector"]
    assert result["eligible"] == 2
    assert result["scored"] == 1
    assert result["failures"] == 1
    assert result["unlabeled"] == 1
    assert result["mean"]["recall_at_3"] == 1.0
    assert result["mean"]["mrr_at_5"] == 1.0


def test_fixture_rejects_unknown_relevance():
    import pytest

    with pytest.raises(ValueError, match="relevance"):
        validate_fixture(
            {
                "documents": [{"id": "a", "text": "passage"}],
                "queries": [{"id": "q", "question": "?", "relevant": ["missing"]}],
            }
        )


def test_existing_output_is_not_overwritten(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "documents": [{"id": "a", "text": "passage"}],
                "queries": [{"id": "q", "question": "question", "relevant": ["a"]}],
            }
        )
    )
    output = tmp_path / "result.json"
    output.write_text("existing evidence")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("runner.py")),
            "--fixture",
            str(fixture),
            "--chroma-port",
            "1",
            "--es-port",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert output.read_text() == "existing evidence"
