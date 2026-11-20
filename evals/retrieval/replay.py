"""Replay saved multilingual retrieval rankings through the current RRF implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from xai_rag.evaluation.metrics import hit_rate_at_k
from xai_rag.models import SearchResult
from xai_rag.retrieval.hybrid import rrf_fusion


def replay(source: Path) -> dict:
    data = source.read_bytes()
    rows = json.loads(data)["rows"]
    if not rows:
        raise ValueError("No saved rankings to replay")
    cases = []
    seen = set()
    for row in rows:
        case_id = row["id"]
        if case_id in seen:
            raise ValueError(f"{case_id}: duplicate case ID")
        seen.add(case_id)
        if not row["relevant"]:
            raise ValueError(f"{case_id}: missing relevance labels")
        rankings = {}
        branches = []
        for source_stage, stage in (("minilm", "vector"), ("bm25", "bm25")):
            branch = row[source_stage]
            if branch.get("error") or "ranking" not in branch:
                raise ValueError(f"{case_id}: missing or failed {source_stage} ranking")
            entries = branch["ranking"][:5]
            rankings[stage] = [entry["id"] for entry in entries]
            branches.append(
                [
                    SearchResult(
                        id=entry["id"], content="", score=entry["score"], stage=stage, rank=i
                    )
                    for i, entry in enumerate(entries, 1)
                ]
            )
        rankings["fused"] = [item.id for item in rrf_fusion(branches, k=60)]
        cases.append(
            {
                "id": case_id,
                "category": row["category"],
                "relevant": row["relevant"],
                **rankings,
            }
        )

    groups = {}
    for name in sorted({"all", *(case["category"] for case in cases)}):
        selected = cases if name == "all" else [case for case in cases if case["category"] == name]
        group = {"n": len(selected)}
        for stage in ("vector", "bm25", "fused"):
            group[stage] = {
                f"hit@{k}": sum(
                    hit_rate_at_k(case[stage], dict.fromkeys(case["relevant"], 1.0), k)
                    for case in selected
                )
                / len(selected)
                for k in (1, 3, 5)
            }
        groups[name] = group
    return {
        "input": str(source),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "method": (
            "Saved source-level top-5 branch rankings; "
            "current RRF k=60; no retrieval or model calls"
        ),
        "groups": groups,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="JSON file containing saved multilingual branch rankings"
    )
    args = parser.parse_args()
    print(json.dumps(replay(args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
