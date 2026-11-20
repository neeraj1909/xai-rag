"""Verify failures preserve evidence and isolate the ingestion journal."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parent


class FailureHandlingTests(unittest.TestCase):
    def test_partial_scoring_fails_after_saving_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "input.json", root / "output.json"
            source.write_text(json.dumps({"queries": [{"case_id": "failed", "http_status": 500}]}))
            env = dict(
                os.environ,
                XAI_EVAL_BUDGET_FILE=str(root / "ledger"),
                XAI_EVAL_MAX_PROVIDER_CALLS="1",
                XAI_EVAL_MODEL_SEED="20261120",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER / "ragas_score.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            report = json.loads(output.read_text())
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["rows"][0]["status"], "not_scored")

    def test_inherited_journal_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fakebin = root / "bin"
            fakebin.mkdir()
            fake_docker = fakebin / "docker"
            fake_docker.write_text(
                '#!/bin/sh\nif [ "$1" = info ]; then echo /tmp; exit 0; fi\n'
                'printf "%s" "$XAI_RAG_INGESTION_JOURNAL_ROOT" > "$JOURNAL_CAPTURE"\nexit 1\n'
            )
            fake_docker.chmod(0o755)
            out = root / "results"
            capture = root / "captured"
            env = dict(
                os.environ,
                PATH=f"{fakebin}:{os.environ['PATH']}",
                XAI_EVAL_RUN_ROOT=str(out),
                XAI_EVAL_RUN_NAME="isolated-test",
                XAI_EVAL_CHROMA_PORT="19310",
                XAI_EVAL_ES_PORT="19311",
                XAI_EVAL_API_PORT="19312",
                XAI_EVAL_MAX_PROVIDER_CALLS="1",
                XAI_RAG_INGESTION_JOURNAL_ROOT=str(root / "existing-user-journal"),
                JOURNAL_CAPTURE=str(capture),
            )
            result = subprocess.run(
                ["bash", str(RUNNER / "run_repeats.sh")],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(capture.read_text(), "")
            self.assertFalse((root / "existing-user-journal").exists())


if __name__ == "__main__":
    unittest.main()
