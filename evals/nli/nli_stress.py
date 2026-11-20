"""Run the authored NLI stress cases against the configured local checker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


def evaluate(cases, checker):
    """Keep failures and report agreement only for cases actually scored."""
    from xai_rag.models import Claim, RankedResult

    rows = []
    for case in cases:
        row = dict(case)
        try:
            verdict = checker._check_single_claim(
                Claim(text=case["hypothesis"], source_chunk_ids=["synthetic"]),
                [RankedResult(id="synthetic", content=case["premise"])],
            )
            row.update(
                observed=verdict.verdict.value,
                entailment=verdict.nli_entailment,
                contradiction=verdict.nli_contradiction,
            )
        except Exception as exc:
            row["error"] = type(exc).__name__
        rows.append(row)
    scored = [row for row in rows if "observed" in row]
    return {
        "planned": len(cases),
        "scored": len(scored),
        "matched": sum(row["observed"] == row["expected"] for row in scored),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=Path(__file__).with_name("stress-cases.json")
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output file; never overwritten"
    )
    args = parser.parse_args()
    # Set before importing transformers/huggingface_hub: this command cannot download a model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from xai_rag.config import settings
    from xai_rag.explainability import faithfulness

    data = args.fixture.read_bytes()
    cases = json.loads(data)["cases"]
    if not cases:
        parser.error("fixture contains no cases")
    with args.output.open("x", encoding="utf-8") as output:
        report = {
            "fixture_sha256": hashlib.sha256(data).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checker_sha256": hashlib.sha256(Path(faithfulness.__file__).read_bytes()).hexdigest(),
            "model": settings.nli_model,
            "revision": settings.nli_model_revision,
            "entailment_threshold": settings.nli_entailment_threshold,
            "contradiction_threshold": settings.nli_contradiction_threshold,
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "offline": True,
            **evaluate(cases, faithfulness.FaithfulnessChecker()),
        }
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(
        f"Scored {report['scored']}/{report['planned']}; "
        f"matched {report['matched']}/{report['scored']}."
    )
    return int(report["scored"] != report["planned"])


if __name__ == "__main__":
    raise SystemExit(main())
