"""Ingest once, then run repeated HTTP queries against the same synthetic index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from call_budget import remaining

parser = argparse.ArgumentParser()
parser.add_argument("--port", required=True, type=int)
parser.add_argument("--root", required=True, type=Path)
parser.add_argument("--fixture", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=("ingest", "query"), required=True)
parser.add_argument("--run", type=int, default=1)
args = parser.parse_args()
if args.run < 1:
    raise SystemExit("run number must be positive")
fixture = json.loads(args.fixture.read_text())
cases = {case["id"]: case for case in fixture["queries"]}
if len(cases) != len(fixture["queries"]):
    raise ValueError("duplicate case IDs")
if len({case["question"] for case in fixture["queries"]}) != len(cases):
    raise ValueError("duplicate questions make evidence attribution ambiguous")
order = list(cases)
random.Random(args.seed).shuffle(order)
key = (args.root / "key").read_text()
args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def request(path, payload=None):
    headers = {"X-API-Key": key}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}{path}", data=body, headers=headers, method="POST"
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=360) as response:
            return {
                "http_status": response.status,
                "body": json.load(response),
                "elapsed_seconds": time.monotonic() - started,
            }
    except urllib.error.HTTPError as exc:
        return {
            "http_status": exc.code,
            "error_type": "HTTPError",
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "http_status": None,
            "error_type": type(exc).__name__,
            "elapsed_seconds": time.monotonic() - started,
        }


if args.mode == "ingest":
    if (args.output_dir / "ingestion.json").exists():
        raise SystemExit("refusing to overwrite ingestion results")
    ingestion = []
    for doc in fixture["documents"]:
        params = urllib.parse.urlencode({"source_path": doc["id"] + ".txt", "strategy": "fixed"})
        row = {"source_id": doc["id"], **request("/ingest?" + params)}
        ingestion.append(row)
        save(args.output_dir / "ingestion.json", ingestion)
        if row["http_status"] != 200 or row.get("body", {}).get("status") != "success":
            raise SystemExit("ingestion failed: inspect ingestion.json")
    raise SystemExit(0)

capture = args.root / "capture.jsonl"
for run in (args.run,):
    report = {
        "run": run,
        "seed": args.seed,
        "seed_scope": "case ordering only",
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "planned_case_ids": order,
        "queries": [],
    }
    output = args.output_dir / f"run-{run:02d}-http.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    save(output, report)
    consecutive_failures = 0
    for case_id in order:
        if remaining() < 4:
            report["halt_reason"] = "provider attempt cap reached"
            break
        case = cases[case_id]
        before = capture.stat().st_size if capture.exists() else 0
        result = request(
            "/query",
            {
                "query": case["question"],
                "top_k": 5,
                "include_explanations": True,
                "include_faithfulness": True,
                "include_ragas": False,
            },
        )
        matches = []
        if capture.exists():
            with capture.open("rb") as f:
                f.seek(before)
                matches = [json.loads(line) for line in f if line.strip()]
            matches = [record for record in matches if record["query"] == case["question"]]
        result.update(
            {
                "case_id": case_id,
                "reference_answer": case["reference_answer"],
                "expected_behavior": case["expected_behavior"],
                "relevant_sources": case["relevant"],
                "prompt_visible_evidence": matches[0]["evidence"] if len(matches) == 1 else None,
                "evidence_capture_matches": len(matches),
            }
        )
        report["queries"].append(result)
        unavailable = (
            result.get("body", {}).get("grounding_status") == "verification_unavailable"
            or len(matches) != 1
        )
        consecutive_failures = (
            consecutive_failures + 1 if result["http_status"] != 200 or unavailable else 0
        )
        save(output, report)
        print(
            f"run={run} case={case_id} http={result['http_status']} captures={len(matches)}",
            flush=True,
        )
        if consecutive_failures >= 3:
            report["halt_reason"] = "three consecutive HTTP or verifier failures"
            break
    report["completed"] = len(report["queries"]) == len(order)
    save(output, report)
    if not report["completed"]:
        raise SystemExit(f"run {run} incomplete; evidence preserved")
