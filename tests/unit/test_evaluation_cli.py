"""CLI behavior for machine-readable offline evaluation."""

import json

import pytest
from click.testing import CliRunner

from tests.unit.test_evaluation_runner import _dataset
from xai_rag.cli import main
from xai_rag.evaluation.models import EvaluationDataset

pytestmark = pytest.mark.unit


def test_evaluate_cli_writes_machine_readable_report(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    report_path = tmp_path / "report.json"
    dataset_path.write_text(_dataset().to_jsonl(), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["evaluate", str(dataset_path), "--k", "1", "--output", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (
        report["dataset_digest"]
        == EvaluationDataset.from_jsonl(dataset_path.read_text(encoding="utf-8")).digest()
    )
    assert report["metrics"]["retrieval.reranked.recall@1"]["status"] == "computed"


def test_evaluate_cli_exits_nonzero_for_failed_blocking_gate(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    gates_path = tmp_path / "gates.toml"
    dataset_path.write_text(_dataset().to_jsonl(), encoding="utf-8")
    gates_path.write_text(
        '[blocking]\n"latency.total.p95_ms" = '
        '{ max = 1.0, owner = "sre", rationale = "test failure" }\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["evaluate", str(dataset_path), "--k", "1", "--gates", str(gates_path)],
    )

    assert result.exit_code == 1
    assert "blocking evaluation gates did not pass" in result.output
