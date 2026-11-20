"""Offline contracts for the fictional judge sensitivity runner."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("judge_eval", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_runner_keeps_failures_and_hides_labels(tmp_path):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError("sensitive provider details")
            content = json.dumps(
                {
                    "faithfulness": None,
                    "correctness": 2,
                    "faithfulness_reason": "abstained",
                    "correctness_reason": "ok",
                }
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content, refusal=None), finish_reason="stop"
                    )
                ],
                model="test-revision",
                usage=None,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    fixture = Path(__file__).with_name("cases.json")
    report = runner.evaluate(client, fixture, "test-model", tmp_path / "result.json")
    assert report["requested"] == 16
    assert len(report["rows"]) == 16
    assert report["rows"][1]["status"] == "judge_error"
    assert report["rows"][1]["error_type"] == "RuntimeError"
    assert "sensitive provider details" not in (tmp_path / "result.json").read_text()
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert set(payload) == {"question", "evidence", "reference", "answer"}
    assert report["rows"][0]["judge"]["faithfulness"] is None
    assert report["rows"][0]["model"] == "test-revision"


def test_existing_output_is_never_overwritten_or_sent(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("protected")
    with pytest.raises(FileExistsError):
        runner.evaluate(None, Path(__file__).with_name("cases.json"), "test", output)
    assert output.read_text() == "protected"


def test_archived_rubric_and_fixture_are_preserved():
    import hashlib

    archive = Path(__file__).parents[1] / "archive/interview-evidence/evaluation"
    historical = json.loads((archive / "judge-fixture-results.json").read_text())
    fixture_bytes = Path(__file__).with_name("cases.json").read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == historical["fixture_sha256"]
    assert historical["system_prompt"] == runner.SYSTEM
    assert historical["response_schema"] == runner.SCHEMA


def test_codex_backend_uses_isolated_schema_and_records_usage(monkeypatch):
    backend_spec = importlib.util.spec_from_file_location(
        "judge_codex_backend", Path(__file__).with_name("codex_backend.py")
    )
    backend = importlib.util.module_from_spec(backend_spec)
    backend_spec.loader.exec_module(backend)

    def fake_run(command, **kwargs):
        assert "--ephemeral" in command and "read-only" in command
        assert "mymodel" in command
        assert json.loads(Path(command[command.index("--output-schema") + 1]).read_text()) == {
            "type": "object"
        }
        assert "secret label" not in kwargs["input"]
        assert "HTTP_PROXY" not in kwargs["env"]
        Path(command[command.index("-o") + 1]).write_text('{"correctness": 2}')
        return SimpleNamespace(stdout=json.dumps({"usage": {"input_tokens": 123}}))

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    result = backend.CodexClient().create(
        model="mymodel",
        response_format={"json_schema": {"schema": {"type": "object"}}},
        messages=[{"content": "system"}, {"content": "fictional case"}],
    )
    assert result.model is None
    assert result.usage.model_dump() == {"input_tokens": 123}
    assert json.loads(result.choices[0].message.content) == {"correctness": 2}
