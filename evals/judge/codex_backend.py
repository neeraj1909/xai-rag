"""Adapt isolated Codex CLI responses to the judge's structured output contract."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace


class CodexClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model, response_format, messages, **_):
        # These are separate ephemeral agent turns, not OpenAI Chat Completions.
        prompt = (
            messages[0]["content"] + "\n\nDo not use tools. Judge this fictional case "
            "and return the specified JSON object.\n" + messages[1]["content"]
        )
        environment = os.environ.copy()
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            environment.pop(name, None)
        with tempfile.TemporaryDirectory(prefix="xai-judge-") as directory:
            schema = Path(directory) / "schema.json"
            output = Path(directory) / "answer.json"
            schema.write_text(json.dumps(response_format["json_schema"]["schema"]))
            result = subprocess.run(
                [
                    "codex",
                    "--no-daemon",
                    "exec",
                    "--ignore-user-config",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "-s",
                    "read-only",
                    "-m",
                    model,
                    "--output-schema",
                    str(schema),
                    "-o",
                    str(output),
                    "--json",
                    "-",
                ],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=120,
                check=True,
                env=environment,
                cwd=directory,
            )
            events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
            usage = next((event["usage"] for event in reversed(events) if "usage" in event), None)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=output.read_text(), refusal=None),
                    )
                ],
                # CLI event stream does not establish a provider-returned model revision.
                model=None,
                usage=SimpleNamespace(model_dump=lambda: usage) if usage else None,
            )
