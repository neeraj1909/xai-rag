"""Combine separately preserved RAGAS rounds without hiding earlier failures."""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--original", type=Path, required=True)
parser.add_argument("--extension", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
first = json.loads(args.original.read_text())
second = json.loads(args.extension.read_text())
initial = {row["case_id"]: row for row in first["rows"]}
followup = {row["case_id"]: row for row in second["rows"]}
if set(followup) - set(initial) or any(initial[c]["status"] == "computed" for c in followup):
    raise SystemExit("Extension overlaps an already computed case or contains unknown IDs")
rows = []
for case_id in first["planned_order"]:
    old = initial.get(case_id, {"case_id": case_id, "status": "not_scored"})
    new = followup.get(case_id)
    selected = new if new and new.get("status") == "computed" else old
    result = dict(selected)
    result["round"] = "extension" if selected is new else "original"
    result["previous_status"] = old["status"] if new else None
    rows.append(result)
summary = {"case_count": len(rows), "status_counts": dict(collections.Counter(r["status"] for r in rows)),
           "scores": {}}
for name in ("faithfulness", "answer_relevancy", "context_precision"):
    vals = [r["scores"][name] for r in rows if r["status"] == "computed" and name in r["scores"]]
    summary["scores"][name] = {"n": len(vals), "mean": statistics.mean(vals) if vals else None}
payload = {"status": "complete" if all(r["status"] == "computed" for r in rows) else "partial",
           "original": str(args.original), "extension": str(args.extension),
           "label_status": "synthetic author-labeled; no human review", "summary": summary,
           "planned_order": first["planned_order"], "rows": rows}
fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(summary, ensure_ascii=False))
