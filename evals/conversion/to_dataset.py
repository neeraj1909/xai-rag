"""Convert captured synthetic HTTP/RAGAS observations to evaluator JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from xai_rag.evaluation.models import EvaluationDataset, EvaluationManifest, QueryEvaluationCase


def indexed(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row[key]
        if identifier in result:
            raise ValueError(f"duplicate {key}: {identifier}")
        result[identifier] = row
    return result


def convert(raw: dict, fixture: dict, scores: dict, config_revision: str) -> EvaluationDataset:
    planned_ids = raw["planned_case_ids"]
    if len(planned_ids) != len(set(planned_ids)):
        raise ValueError("duplicate planned case ID")
    cases = indexed(fixture["queries"], "id")
    observations = indexed(raw["queries"], "case_id")
    judgments = indexed(scores.get("rows", []), "case_id")
    if set(observations).difference(planned_ids) or set(judgments).difference(planned_ids):
        raise ValueError("observation or score does not belong to a planned case")

    items = []
    for case_id in planned_ids:
        planned = cases[case_id]
        row = observations.get(case_id, {})
        body = row.get("body") or {}
        evidence = row.get("prompt_visible_evidence")
        success = row.get("http_status") == 200
        answer = body.get("answer") if success else None
        grounding = body.get("grounding_status") if success else None
        expected = planned["expected_behavior"]
        expected_refusal = {"abstain": True, "answer": False}.get(expected)
        refusal = None
        if answer is not None and grounding != "verification_unavailable":
            if grounding == "insufficient_evidence":
                refusal = True
            elif grounding == "verifier_passed":
                # Verified explanations of absence cannot be classified from status alone.
                refusal = None
            else:
                refusal = answer == "I don't have enough information to answer this question."
        if grounding == "verification_unavailable":
            error = "GroundingVerificationUnavailable"
        elif success and evidence is None:
            error = "MissingPromptEvidence"
        elif success:
            error = None
        else:
            error = row.get("error_type") or (
                f"HTTP{row['http_status']}"
                if row.get("http_status") is not None
                else "NotAttempted"
            )
        judgment = judgments.get(case_id, {})
        items.append(
            QueryEvaluationCase(
                case_id=case_id,
                query=planned["question"],
                expected_answer=planned.get("reference_answer"),
                expected_refusal=expected_refusal,
                reranked_ids=[item["chunk_id"] for item in body.get("retrieval_explanations", [])],
                prompt_visible_ids=[item["id"] for item in evidence]
                if evidence is not None
                else None,
                answer=answer,
                cited_ids=body.get("sources", []),
                claim_source_ids=[item["source_chunk_ids"] for item in body.get("claims", [])],
                claim_verdicts=[item["verdict"] for item in body.get("faithfulness_report", [])],
                actual_refusal=refusal,
                judge_scores=judgment.get("scores", {})
                if judgment.get("status") == "computed"
                else {},
                stage_latency_ms=body.get("stage_latency_ms", {}),
                total_latency_ms=body.get("latency_ms"),
                input_tokens=body.get("input_tokens"),
                output_tokens=body.get("output_tokens"),
                error_type=error,
                metadata={
                    "http_status": row.get("http_status"),
                    "grounding_status": grounding,
                    "expected_behavior": expected,
                    "category": planned.get("category"),
                    "relevant_document_ids": planned.get("relevant"),
                    "ragas_status": judgment.get("status", "not_recorded"),
                    "ragas_error_type": judgment.get("error_type"),
                },
            )
        )
    return EvaluationDataset(
        manifest=EvaluationManifest(
            dataset_name="captured-http-evaluation",
            dataset_revision=raw["fixture_sha256"],
            corpus_revision=raw["fixture_sha256"],
            config_revision=config_revision,
            metadata={
                "label_status": raw.get("dataset_status", "unknown"),
                "scope": "captured HTTP observations; chunk relevance labels unavailable",
            },
        ),
        query_cases=items,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--ragas", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-revision", required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    fixture_bytes = args.fixture.read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != raw["fixture_sha256"]:
        raise ValueError("fixture SHA-256 does not match captured HTTP metadata")
    scores = json.loads(args.ragas.read_text(encoding="utf-8")) if args.ragas else {}
    dataset = convert(raw, json.loads(fixture_bytes), scores, args.config_revision)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(dataset.to_jsonl() + "\n")
    print(f"Wrote {len(dataset.query_cases)} planned cases")


if __name__ == "__main__":
    main()
