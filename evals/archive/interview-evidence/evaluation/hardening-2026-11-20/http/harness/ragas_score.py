"""Run the project's real RAGAS metrics on HTTP-captured synthetic evidence."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from call_budget import install, purpose, remaining
from xai_rag.evaluation.ragas_eval import evaluate_response

install()
purpose.set("ragas")
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--timeout-seconds", type=int, default=180)
parser.add_argument("--seed", type=int, default=20261120)
parser.add_argument("--preserve-order", action="store_true")
args = parser.parse_args()
data = json.loads(args.input.read_text())
fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.close(fd)
ordered_rows = list(data["queries"])
if not args.preserve_order:
    random.Random(args.seed).shuffle(ordered_rows)
report = {"input": str(args.input), "status": "running", "seed": args.seed,
          "ordering": "supplied" if args.preserve_order else "fixed-seed shuffled",
          "planned_order": [row["case_id"] for row in ordered_rows], "rows": []}


def save():
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


async def run():
    save()
    for row in ordered_rows:
        result = {"case_id": row["case_id"]}
        report["rows"].append(result)
        if remaining() == 0:
            result.update({"status": "not_scored", "reason": "shared provider-call cap reached"})
            report["halt_reason"] = "shared provider-call cap reached"
            save()
            continue
        if row["http_status"] != 200 or row.get("prompt_visible_evidence") is None:
            result.update({"status": "not_scored", "reason": "HTTP or evidence capture failure"})
            save()
            continue
        if row["body"].get("grounding_status") == "verification_unavailable":
            result.update({"status": "not_scored", "reason": "grounding verifier unavailable"})
            save()
            continue
        started = time.monotonic()
        try:
            result.update(await asyncio.wait_for(evaluate_response(
                query=row["body"]["query"], answer=row["body"]["answer"],
                contexts=[c["content"] for c in row["prompt_visible_evidence"]],
                ground_truth=row["reference_answer"]), timeout=args.timeout_seconds))
        except Exception as exc:
            result.update({"status": "error", "error_type": type(exc).__name__})
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save()
        print(f"{row['case_id']}: {result['status']}", flush=True)
    report["status"] = ("complete" if len(report["rows"]) == len(ordered_rows) and
                        all(row["status"] == "computed" for row in report["rows"]) else "partial")
    save()


asyncio.run(run())
