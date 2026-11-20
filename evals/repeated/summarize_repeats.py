"""Summarize complete run means and matched-case variation without hiding failures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = ("faithfulness", "answer_relevancy", "context_precision")
parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("--model-seed", type=int, required=True)
args = parser.parse_args()
fixture = json.loads(args.fixture.read_text())
planned = {case["id"] for case in fixture["queries"]}
reports = []
for number in range(1, args.runs + 1):
    prefix = args.directory / f"run-{number:02d}"
    http_path = Path(str(prefix) + "-http.json")
    ragas_path = Path(str(prefix) + "-ragas.json")
    http = json.loads(http_path.read_text()) if http_path.exists() else {"queries": []}
    ragas = json.loads(ragas_path.read_text()) if ragas_path.exists() else {"rows": []}
    by_id = {row["case_id"]: row for row in ragas["rows"]}
    queries = {row["case_id"]: row for row in http["queries"]}
    run = {
        "run": number,
        "planned": len(planned),
        "http_attempted": len(queries),
        "http_200": sum(row.get("http_status") == 200 for row in queries.values()),
        "http_unattempted": len(planned - queries.keys()),
        "ragas_computed": sum(row.get("status") == "computed" for row in by_id.values()),
        "ragas_errors": sum(row.get("status") == "error" for row in by_id.values()),
        "ragas_unscored": len(
            planned - {k for k, v in by_id.items() if v.get("status") == "computed"}
        ),
        "refusals": sum(
            row.get("body", {}).get("grounding_status") == "insufficient_evidence"
            for row in queries.values()
        ),
        "expected_behavior_vs_grounding_status": {},
        "metrics": {},
    }
    for row in queries.values():
        behavior = row.get("expected_behavior", "unknown")
        observed = row.get("body", {}).get("grounding_status", "missing")
        counts = run["expected_behavior_vs_grounding_status"].setdefault(behavior, {})
        counts[observed] = counts.get(observed, 0) + 1
    for metric in METRICS:
        scores = {
            case_id: row["scores"][metric]
            for case_id, row in by_id.items()
            if row.get("status") == "computed" and metric in row.get("scores", {})
        }
        run["metrics"][metric] = {
            "n": len(scores),
            "mean": statistics.mean(scores.values()) if scores else None,
            "complete": set(scores) == planned,
        }
    run["_scores"] = by_id
    reports.append(run)
overall = {}
for metric in METRICS:
    eligible = [
        run["metrics"][metric]["mean"] for run in reports if run["metrics"][metric]["complete"]
    ]
    matched = set(planned)
    for run in reports:
        matched &= {
            case_id
            for case_id, row in run["_scores"].items()
            if row.get("status") == "computed" and metric in row.get("scores", {})
        }
    overall[metric] = {
        "complete_run_count": len(eligible),
        "mean_of_complete_run_means": statistics.mean(eligible) if eligible else None,
        "sample_sd_of_complete_run_means": statistics.stdev(eligible)
        if len(eligible) > 1
        else None,
        "min": min(eligible) if eligible else None,
        "max": max(eligible) if eligible else None,
        "matched_cases_all_runs": len(matched),
    }
for run in reports:
    del run["_scores"]
print(
    json.dumps(
        {
            "runs": reports,
            "overall": overall,
            "sd_definition": "sample standard deviation across complete per-run means",
            "model_seed": args.model_seed,
            "order_seed_scope": "case order only",
        },
        indent=2,
    )
)
