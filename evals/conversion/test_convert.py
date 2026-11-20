"""Behavior checks for conversion of captured HTTP evaluation evidence."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from xai_rag.evaluation.models import EvaluationDataset

SCRIPT = Path(__file__).with_name("to_dataset.py")


def run_conversion(tmp_path, http, fixture, scores=None):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    http["fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    http_path = tmp_path / "http.json"
    http_path.write_text(json.dumps(http), encoding="utf-8")
    output = tmp_path / "dataset.jsonl"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(http_path),
        "--fixture",
        str(fixture_path),
        "--config-revision",
        "test",
        "--output",
        str(output),
    ]
    if scores is not None:
        score_path = tmp_path / "scores.json"
        score_path.write_text(json.dumps(scores), encoding="utf-8")
        cmd.extend(["--ragas", str(score_path)])
    return subprocess.run(cmd, capture_output=True, text=True), output


def test_planned_cases_and_missing_scores_remain_visible(tmp_path):
    fixture = {
        "queries": [
            {"id": "a", "question": "Q1?", "reference_answer": "A1", "expected_behavior": "answer"},
            {
                "id": "b",
                "question": "Q2?",
                "reference_answer": "A2",
                "expected_behavior": "abstain",
            },
            {"id": "c", "question": "Q3?", "reference_answer": "A3", "expected_behavior": "answer"},
        ]
    }
    http = {
        "planned_case_ids": ["a", "b", "c"],
        "dataset_status": "fictional labels",
        "queries": [
            {
                "case_id": "a",
                "http_status": 200,
                "prompt_visible_evidence": [],
                "body": {"answer": "A1", "grounding_status": "verifier_passed"},
            },
            {"case_id": "b", "http_status": 503, "error_type": "HTTPError", "body": {}},
        ],
    }
    scores = {
        "rows": [
            {"case_id": "a", "status": "computed", "scores": {"faithfulness": 0.8}},
            {"case_id": "b", "status": "error", "error_type": "Timeout"},
        ]
    }
    result, output = run_conversion(tmp_path, http, fixture, scores)
    assert result.returncode == 0, result.stderr
    dataset = EvaluationDataset.from_jsonl(output.read_text(encoding="utf-8"))
    a, b, c = dataset.query_cases
    assert [case.case_id for case in dataset.query_cases] == ["a", "b", "c"]
    assert a.prompt_visible_ids == [] and a.judge_scores == {"faithfulness": 0.8}
    assert b.error_type == "HTTPError" and b.expected_refusal is True
    assert b.metadata["ragas_status"] == "error" and b.metadata["ragas_error_type"] == "Timeout"
    assert c.error_type == "NotAttempted" and c.prompt_visible_ids is None
    assert c.metadata["ragas_status"] == "not_recorded"
    assert dataset.manifest.metadata["label_status"] == "fictional labels"


def test_duplicate_observations_fail_without_writing(tmp_path):
    fixture = {
        "queries": [
            {"id": "a", "question": "Q?", "reference_answer": "A", "expected_behavior": "answer"}
        ]
    }
    http = {"planned_case_ids": ["a"], "queries": [{"case_id": "a"}, {"case_id": "a"}]}
    result, output = run_conversion(tmp_path, http, fixture)
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
    assert not output.exists()


def test_verified_response_does_not_infer_refusal_from_expected_label(tmp_path):
    fixture = {
        "queries": [
            {
                "id": "answer",
                "question": "Q?",
                "reference_answer": "A",
                "expected_behavior": "answer",
            },
            {
                "id": "abstain",
                "question": "Q?",
                "reference_answer": "A",
                "expected_behavior": "abstain",
            },
        ]
    }
    response = {
        "http_status": 200,
        "prompt_visible_evidence": [],
        "body": {"answer": "The record is silent.", "grounding_status": "verifier_passed"},
    }
    http = {
        "planned_case_ids": ["answer", "abstain"],
        "queries": [{"case_id": "answer", **response}, {"case_id": "abstain", **response}],
    }
    result, output = run_conversion(tmp_path, http, fixture)
    assert result.returncode == 0, result.stderr
    cases = EvaluationDataset.from_jsonl(output.read_text(encoding="utf-8")).query_cases
    assert [case.actual_refusal for case in cases] == [None, None]
