"""Run case-scoped synthetic security cases through the real API and stores."""
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
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20261120)
parser.add_argument("--max-cases", type=int, default=28)
args = parser.parse_args()
fixture_path = Path("/tmp/xai-security-eval/pipeline-suite/cases.jsonl")
fixture_bytes = fixture_path.read_bytes()
if hashlib.sha256(fixture_bytes).hexdigest() != "851007eca4123773a57a4003378dbf8d85c82784fd81376d5b26f341a7a7cc3e":
    raise SystemExit("Frozen fixture checksum mismatch")
fixture = [json.loads(line) for line in fixture_bytes.splitlines()]
random.Random(args.seed).shuffle(fixture)
if not 1 <= args.max_cases <= len(fixture):
    raise SystemExit("Invalid case cap")
fixture = fixture[:args.max_cases]
fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.close(fd)
report = {"fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(), "seed": args.seed,
          "dataset_status": "fictional, author-labeled, not human reviewed",
          "planned_ids": [row["id"] for row in fixture], "ingestion": [], "queries": []}


def save():
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def request(path, key, payload=None):
    headers = {"X-API-Key": key}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
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
admin = (args.root / "keys" / "ingester").read_text()
for case in fixture:
    for source in case["sources"]:
        name = f"{case['id']}/{source['id']}.txt"
        params = urllib.parse.urlencode({"source_path": name, "strategy": "fixed"})
        row = {"source_id": source["id"], "source_path": name,
               **request("/ingest?" + params, admin)}
        report["ingestion"].append(row)
        save()
        if row["http_status"] != 200 or row.get("body", {}).get("status") != "success":
            report["halt_reason"] = "ingestion failure; no quality results claimed"
            save()
            raise SystemExit(report["halt_reason"])

capture = args.root / "capture.jsonl"
consecutive_failures = 0
for case in fixture:
    if remaining() < 4:
        report["halt_reason"] = "insufficient remaining attempts for generation retries and grounding"
        break
    key = (args.root / "keys" / case["id"]).read_text()
    result = request("/query", key, {"query": case["question"], "top_k": 5,
                     "include_explanations": True, "include_faithfulness": True,
                     "include_ragas": False})
    matches = []
    if capture.exists():
        matches = [r for r in map(json.loads, capture.read_text().splitlines())
                   if r["query"] == case["question"]]
    evidence = matches[0]["evidence"] if len(matches) == 1 else None
    expected = {f"{case['id']}/{source['id']}.txt" for source in case["sources"]}
    visible = {r["source_file"] for r in evidence or [] if isinstance(r.get("source_file"), str)}
    result.update({"case_id": case["id"], "category": case["category"],
                   "expected_behavior": case["expected_behavior"], "expected_facts": case["expected_facts"],
                   "attack_markers": case["attack_markers"], "tags": case["tags"],
                   "required_source_paths": sorted(expected), "visible_source_paths": sorted(visible),
                   "missing_source_metadata": sum(not isinstance(r.get("source_file"), str) for r in evidence or []),
                   "all_case_sources_visible": expected.issubset(visible) if evidence is not None else None,
                   "prompt_visible_evidence": evidence, "evidence_capture_matches": len(matches)})
    report["queries"].append(result)
    if result["http_status"] != 200 or result.get("body", {}).get("grounding_status") == "verification_unavailable":
        consecutive_failures += 1
    else:
        consecutive_failures = 0
    save()
    print(f"{case['id']}: status={result['http_status']} all_sources_visible={result['all_case_sources_visible']}", flush=True)
    if consecutive_failures >= 3:
        report["halt_reason"] = "three consecutive query or verifier failures"
        break

report["completed"] = len(report["queries"]) == len(fixture)
save()
