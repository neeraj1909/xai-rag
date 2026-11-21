#!/usr/bin/env python3
"""Run from XAI-RAG's environment after the model/CPU lease is granted."""

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

FIXTURE = Path(__file__).with_name("fixture.json")
MODELS = {
    "current": ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"),
    "multilingual": ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--output", required=True, help="New file path; never overwrites evidence")
    parser.add_argument("--online", action="store_true", help="Permit model download after resource preflight")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    data = FIXTURE.read_bytes()
    rows = json.loads(data)
    model_id, revision = MODELS[args.model]
    report = {
        "model_id": model_id,
        "revision": revision,
        "fixture_sha256": hashlib.sha256(data).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "max_length": 512,
        "thresholds": {"entailment": 0.5, "contradiction": 0.5},
        "local_files_only": not args.online,
        "cases": [],
    }
    start = time.monotonic()
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=not args.online)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, local_files_only=not args.online)
        model.eval()
        mapping = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        if set(mapping.values()) != {"entailment", "neutral", "contradiction"}:
            raise ValueError(f"unexpected id2label {mapping}")
        report["id2label"] = mapping
        for row in rows:
            case = {"id": row["id"], "language": row["language"], "label": row["label"]}
            tick = time.monotonic()
            try:
                inputs = tokenizer(row["premise"], row["hypothesis"], return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
                scores = {mapping[i]: float(value) for i, value in enumerate(probs)}
                entail, contradict = scores["entailment"], scores["contradiction"]
                case["scores"] = scores
                case["gate_verdict"] = "contradiction" if contradict >= 0.5 and contradict >= entail else "entailment" if entail >= 0.5 else "neutral"
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
            case["seconds"] = round(time.monotonic() - tick, 4)
            report["cases"].append(case)
    except Exception as exc:
        report["load_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["seconds"] = round(time.monotonic() - start, 4)
        report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["expected_cases"] = len(rows)
        report["completed_cases"] = sum("gate_verdict" in case for case in report["cases"])
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return 0 if report["completed_cases"] == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
