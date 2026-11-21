"""Convert captured HTTP observations to the repository evaluator's JSONL contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from xai_rag.evaluation.models import EvaluationDataset, EvaluationManifest, QueryEvaluationCase

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--ragas", type=Path)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--config-revision", required=True)
args = parser.parse_args()
raw = json.loads(args.input.read_text())
fixture = json.loads(args.fixture.read_text())
cases = {item["id"]: item for item in fixture["queries"]}
observed = {item["case_id"]: item for item in raw["queries"]}
scores = {}
if args.ragas:
    scores = {row["case_id"]: row.get("scores", {}) for row in json.loads(args.ragas.read_text())["rows"]
              if row.get("status") == "computed"}
items = []
for case_id in raw["planned_case_ids"]:
    planned = cases[case_id]
    row = observed.get(case_id, {})
    body = row.get("body", {})
    evidence = row.get("prompt_visible_evidence")
    success = row.get("http_status") == 200
    answer = body.get("answer") if success else None
    grounding = body.get("grounding_status")
    refusal = (None if answer is None or grounding == "verification_unavailable" else
               True if grounding == "insufficient_evidence" else
               False if grounding == "verifier_passed" else
               answer == "I don't have enough information to answer this question.")
    expected = planned["expected_behavior"]
    expected_refusal = True if expected == "abstain" else False if expected == "answer" else None
    # A verified response may explain an explicit absence while also providing
    # related cited context. Classifying that semantic abstention from status
    # alone would create a false refusal-accuracy failure.
    if grounding == "verifier_passed" and expected_refusal:
        refusal = None
    error_type = ("GroundingVerificationUnavailable" if grounding == "verification_unavailable" else
                  None if success and evidence is not None else
                  row.get("error_type") or ("MissingPromptEvidence" if success else "NotAttempted"))
    items.append(QueryEvaluationCase(
        case_id=case_id, query=planned["question"], expected_answer=planned["reference_answer"],
        expected_refusal=expected_refusal,
        reranked_ids=[e["chunk_id"] for e in body.get("retrieval_explanations", [])],
        prompt_visible_ids=[e["id"] for e in evidence] if evidence is not None else None,
        answer=answer, cited_ids=body.get("sources", []),
        claim_source_ids=[claim["source_chunk_ids"] for claim in body.get("claims", [])],
        claim_verdicts=[v["verdict"] for v in body.get("faithfulness_report", [])],
        actual_refusal=refusal, judge_scores=scores.get(case_id, {}),
        stage_latency_ms=body.get("stage_latency_ms", {}), total_latency_ms=body.get("latency_ms"),
        input_tokens=body.get("input_tokens"), output_tokens=body.get("output_tokens"),
        error_type=error_type,
        metadata={"labels": "synthetic; not human reviewed", "http_status": row.get("http_status"),
                  "grounding_status": grounding,
                  "retrieval_relevance": "not captured for every corpus chunk; ranking quality unavailable",
                  "score_provenance": "RAGAS LLM-based metrics, not human scores"},
    ))
dataset = EvaluationDataset(manifest=EvaluationManifest(
    dataset_name="synthetic-http-evaluation", dataset_revision=raw["fixture_sha256"],
    corpus_revision=raw["fixture_sha256"], config_revision=args.config_revision,
    metadata={"label_source": "synthetic", "production_representative": False,
              "scope": "HTTP and prompt-visible evidence; retrieval labels unavailable"}), query_cases=items)
fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as out:
    out.write(dataset.to_jsonl() + "\n")
print(f"Wrote {len(items)} planned cases, including failed or unattempted cases")
