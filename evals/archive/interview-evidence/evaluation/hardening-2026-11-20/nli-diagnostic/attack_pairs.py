"""Measure three existing synthetic pairs using the actual current NLI checker."""

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

source = Path("/tmp/xai-judge-question-candidate/fixture.json")
output = Path("/tmp/xai-multilingual-nli/attack-pairs.json")
ids = ("benign-06-imperative", "unsafe-quoted-irrelevant", "unsafe-endorsement")
sys.path.insert(0, "/home/neeraj/Documents/xai-rag/src")
from xai_rag.config import settings
from xai_rag.explainability.faithfulness import FaithfulnessChecker

data = source.read_bytes()
rows = {row["id"]: row for row in json.loads(data)}
report = {"source_sha256": hashlib.sha256(data).hexdigest(), "model": settings.nli_model, "revision": settings.nli_model_revision, "entailment_threshold": settings.nli_entailment_threshold, "contradiction_threshold": settings.nli_contradiction_threshold, "note": "Fixture labels cover combined safety/relevance, not NLI; no human review.", "cases": []}
start = time.monotonic()
try:
    checker = FaithfulnessChecker()
    for case_id in ids:
        row = rows[case_id]
        record = {"id": case_id}
        tick = time.monotonic()
        try:
            scores = checker._score_pair(row["evidence"][0]["text"], row["claim"]["claim"])
            entail, contradict = scores["entailment"], scores["contradiction"]
            record["scores"] = scores
            record["verdict"] = "not_supported" if contradict >= settings.nli_contradiction_threshold and contradict >= entail else "supported" if entail >= settings.nli_entailment_threshold else "neutral"
        except Exception as exc:
            record["error"] = type(exc).__name__
        record["seconds"] = round(time.monotonic() - tick, 4)
        report["cases"].append(record)
except Exception as exc:
    report["load_error"] = type(exc).__name__
finally:
    report["seconds"] = round(time.monotonic() - start, 4)
    report["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

sys.exit(0 if len(report["cases"]) == len(ids) and all("verdict" in case for case in report["cases"]) else 1)
