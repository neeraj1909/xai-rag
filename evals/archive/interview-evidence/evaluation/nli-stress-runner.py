"""Run single-chunk, thresholded NLI cases with the project's pinned checker."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from xai_rag.config import settings
from xai_rag.explainability import faithfulness
from xai_rag.models import Claim, RankedResult

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
fixture = Path(args.fixture)
model = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
revision = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
assert settings.nli_model == model and settings.nli_model_revision == revision
assert settings.nli_entailment_threshold == settings.nli_contradiction_threshold == 0.5
checker = faithfulness.FaithfulnessChecker()
source = Path(faithfulness.__file__)
results = []
for case in json.loads(fixture.read_text())["cases"]:
    start = time.monotonic()
    verdict = checker._check_single_claim(
        Claim(text=case["hypothesis"], source_chunk_ids=["synthetic"]),
        [RankedResult(id="synthetic", content=case["premise"])],
    )
    results.append(
        {
            **case,
            "observed": verdict.verdict.value,
            "entailment": verdict.nli_entailment,
            "contradiction": verdict.nli_contradiction,
            "neutral_derived": 1 - verdict.nli_entailment - verdict.nli_contradiction,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
    )
    print(case["id"], case["expected"], verdict.verdict.value, flush=True)
output = {
    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "source_path": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "model": model,
    "revision": revision,
    "entailment_threshold": 0.5,
    "contradiction_threshold": 0.5,
    "max_token_length": 512,
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    },
    "cases": results,
}
Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
