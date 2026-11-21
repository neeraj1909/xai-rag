"""Exercise real HTTP ingestion and queries using only the frozen fictional corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from call_budget import remaining

parser = argparse.ArgumentParser()
parser.add_argument("--port", required=True, type=int)
parser.add_argument("--root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cases", nargs="+", required=True)
args = parser.parse_args()
fixture_path = Path("/tmp/xai-e2e-harness/holdout-cases.json")
fixture = json.loads(fixture_path.read_text())
cases = {q["id"]: q for q in fixture["queries"]}
if set(args.cases) - cases.keys() or len(set(args.cases)) != len(args.cases):
    raise SystemExit("Invalid or duplicate case IDs")
key = (args.root / "key").read_text()
output_fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.close(output_fd)
report = {"fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
          "dataset_status": "fictional, author-labeled, not human reviewed",
          "planned_case_ids": args.cases, "ingestion": [], "queries": []}


def save():
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def request(path, *, payload=None):
    headers = {"X-API-Key": key}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}{path}", data=body,
                                 headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=360) as response:
            return {"http_status": response.status, "body": json.load(response),
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, "error_type": "HTTPError",
                "elapsed_seconds": round(time.monotonic() - started, 3)}
    except Exception as exc:
        return {"http_status": None, "error_type": type(exc).__name__,
                "elapsed_seconds": round(time.monotonic() - started, 3)}


save()
for doc in fixture["documents"]:
    params = urllib.parse.urlencode({"source_path": doc["id"] + ".txt", "strategy": "fixed"})
    row = {"source_id": doc["id"], **request("/ingest?" + params)}
    report["ingestion"].append(row)
    save()
    if row["http_status"] != 200 or row.get("body", {}).get("status") != "success":
        report["halt_reason"] = "ingestion failure; no quality results claimed"
        save()
        raise SystemExit(report["halt_reason"])

capture = args.root / "capture.jsonl"
consecutive_failures = 0
for case_id in args.cases:
    if remaining() < 4:
        report["halt_reason"] = "insufficient remaining attempts for generation retries and grounding"
        break
    case = cases[case_id]
    result = request("/query", payload={"query": case["question"], "top_k": 5,
                     "include_explanations": True, "include_faithfulness": True,
                     "include_ragas": False})
    matches = []
    if capture.exists():
        matches = [r for r in map(json.loads, capture.read_text().splitlines())
                   if r["query"] == case["question"]]
    result.update({"case_id": case_id, "reference_answer": case["reference_answer"],
                   "expected_behavior": case["expected_behavior"],
                   "relevant_sources": case["relevant"],
                   "prompt_visible_evidence": matches[0]["evidence"] if len(matches) == 1 else None,
                   "evidence_capture_matches": len(matches)})
    report["queries"].append(result)
    if result["http_status"] != 200 or result.get("body", {}).get("grounding_status") == "verification_unavailable":
        consecutive_failures += 1
    else:
        consecutive_failures = 0
    save()
    print(f"{case_id}: status={result['http_status']} evidence_captures={len(matches)}", flush=True)
    if consecutive_failures >= 3:
        report["halt_reason"] = "three consecutive query or verifier failures"
        break

report["completed"] = len(report["queries"]) == len(args.cases)
save()
