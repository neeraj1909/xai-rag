"""Compose must preserve explicit authorization and grounding configuration."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_compose_passes_policy_and_judge_settings(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("docker compose is not installed")
    env = os.environ.copy()
    env.update(
        {
            "XAI_RAG_AUTH_POLICY_MOUNT": str(tmp_path),
            "XAI_RAG_AUTH_POLICY_FILE": "/run/xai-rag-policy/policy.json",
            "XAI_RAG_GROUNDING_JUDGE_MODEL": "judge-test-model",
            "XAI_RAG_PARSER_TIMEOUT_SECONDS": "17",
            "XAI_RAG_PARSER_MEMORY_BYTES": "536870912",
        }
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    api = json.loads(result.stdout)["services"]["api"]
    assert api["environment"]["XAI_RAG_AUTH_POLICY_FILE"] == "/run/xai-rag-policy/policy.json"
    assert api["environment"]["XAI_RAG_GROUNDING_JUDGE_MODEL"] == "judge-test-model"
    assert api["environment"]["XAI_RAG_PARSER_TIMEOUT_SECONDS"] == "17"
    assert api["environment"]["XAI_RAG_PARSER_MEMORY_BYTES"] == "536870912"
    mount = next(v for v in api["volumes"] if v["target"] == "/run/xai-rag-policy")
    assert mount["source"] == str(tmp_path)
    assert mount["read_only"] is True
