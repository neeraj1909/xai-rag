"""Summarize coverage and model outputs without inventing a human verdict."""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--ragas", type=Path)
parser.add_argument("--security-judge", type=Path)
parser.add_argument("--budget", type=Path)
args = parser.parse_args()
raw = json.loads(args.input.read_text())
rows = raw.get("queries", [])
summary = {"planned": len(raw.get("planned_ids", raw.get("planned_case_ids", []))),
           "attempted": len(rows),
           "http_200": sum(row.get("http_status") == 200 for row in rows),
           "http_errors": dict(collections.Counter(str(row.get("http_status")) for row in rows
                                                   if row.get("http_status") != 200)),
           "evidence_captured": sum(row.get("prompt_visible_evidence") is not None for row in rows),
           "grounding_statuses": dict(collections.Counter(row.get("body", {}).get("grounding_status", "unavailable")
                                                            for row in rows)),
           "synthetic_labels": True, "human_reviewed": False}
if raw.get("planned_ids"):
    summary["all_case_sources_visible"] = sum(row.get("all_case_sources_visible") is True for row in rows)
    summary["one_or_more_case_sources_missing"] = sum(row.get("all_case_sources_visible") is False for row in rows)
if args.ragas:
    scored = json.loads(args.ragas.read_text())["rows"]
    summary["ragas_statuses"] = dict(collections.Counter(row["status"] for row in scored))
    metrics = collections.defaultdict(list)
    for row in scored:
        if row["status"] == "computed":
            for name, value in row["scores"].items():
                metrics[name].append(value)
    summary["ragas_means_scored_only"] = {name: {"n": len(values), "mean": statistics.mean(values)}
                                           for name, values in metrics.items()}
if args.security_judge:
    judged = json.loads(args.security_judge.read_text())["rows"]
    summary["security_judge_statuses"] = dict(collections.Counter(row["status"] for row in judged))
    for name in ("instruction_boundary", "answer_utility", "attribution"):
        summary[f"model_judge_{name}"] = dict(collections.Counter(
            row["judgment"][name]["verdict"] for row in judged if row["status"] == "scored"))
if args.budget:
    events = [json.loads(line) for line in args.budget.read_text().splitlines()]
    attempts = [event for event in events if event["phase"] == "attempt"]
    outcomes = [event for event in events if event["phase"] == "outcome"]
    usage = [event["usage"] for event in outcomes if event.get("usage")]
    summary["provider_attempts_all_purposes"] = len(attempts)
    summary["provider_successes_all_purposes"] = sum(event["status"] == "completed" for event in outcomes)
    summary["provider_failures_all_purposes"] = sum(event["status"] == "error" for event in outcomes)
    summary["reported_prompt_tokens_all_purposes"] = sum(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0 for u in usage)
    summary["reported_completion_tokens_all_purposes"] = sum(u.get("completion_tokens", u.get("output_tokens", 0)) or 0 for u in usage)
    summary["provider_by_purpose"] = {}
    requested_by_id = {event["id"]: event.get("requested_model") for event in attempts}
    for purpose in sorted({event.get("purpose", "unclassified") for event in attempts}):
        p_attempts = [event for event in attempts if event.get("purpose", "unclassified") == purpose]
        p_outcomes = [event for event in outcomes if event.get("purpose", "unclassified") == purpose]
        p_usage = [event["usage"] for event in p_outcomes if event.get("usage")]
        priced_cost = 0.0
        unpriced = 0
        for event in p_outcomes:
            usage = event.get("usage")
            if event["status"] != "completed" or not usage:
                unpriced += 1
                continue
            model = requested_by_id.get(event["id"]) or ""
            if model.startswith("gpt-6-luna"):
                input_rate, cached_rate, write_rate, output_rate = 0.10, 0.01, 0.125, 0.50
            elif model == "text-embedding-3-small":
                input_rate, cached_rate, write_rate, output_rate = 0.02, 0.02, 0.02, 0.0
            else:
                unpriced += 1
                continue
            tokens_in = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            tokens_out = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            cached = details.get("cached_tokens", 0) or 0
            written = details.get("cache_write_tokens", 0) or 0
            priced_cost += ((tokens_in - cached - written) * input_rate + cached * cached_rate +
                            written * write_rate +
                            tokens_out * output_rate) / 1_000_000
        summary["provider_by_purpose"][purpose] = {
            "attempts": len(p_attempts),
            "completed": sum(event["status"] == "completed" for event in p_outcomes),
            "failed": sum(event["status"] == "error" for event in p_outcomes),
            "requested_models": dict(collections.Counter(str(event.get("requested_model")) for event in p_attempts)),
            "reported_prompt_tokens": sum(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0 for u in p_usage),
            "reported_completion_tokens": sum(u.get("completion_tokens", u.get("output_tokens", 0)) or 0 for u in p_usage),
            "estimated_usd_known_usage": priced_cost,
            "calls_without_price_or_usage": unpriced + max(0, len(p_attempts) - len(p_outcomes)),
            "cost_note": "Standard token prices; failed/incomplete calls may be billable, so this is not an invoice",
        }
    summary["price_source"] = "https://developers.openai.com/api/docs/pricing"
    summary["embedding_price_source"] = "https://developers.openai.com/api/docs/models/text-embedding-3-small"
print(json.dumps(summary, indent=2, ensure_ascii=False))
