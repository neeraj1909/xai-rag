"""Create per-case source grants for the frozen fictional injection suite."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

fixture_path = Path("/tmp/xai-security-eval/pipeline-suite/cases.jsonl")
if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != "851007eca4123773a57a4003378dbf8d85c82784fd81376d5b26f341a7a7cc3e":
    raise SystemExit("Frozen fixture checksum mismatch")
fixture = [json.loads(line) for line in fixture_path.read_text().splitlines()]
root = Path(os.environ["XAI_EVAL_DIR"])
if root.exists():
    raise SystemExit("Evaluation directory already exists")
root.mkdir(mode=0o700)
docs, keys = root / "documents", root / "keys"
docs.mkdir(mode=0o700)
keys.mkdir(mode=0o700)
principals, all_sources = [], []


def principal(name, sources, role):
    key = secrets.token_urlsafe(48)
    fd = os.open(keys / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    principals.append({"key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                       "tenant": "synthetic-security", "sources": sources, "role": role})


for case in fixture:
    case_id = case["id"]
    if not case_id.replace("-", "").isalnum():
        raise ValueError("Unsafe case name")
    directory = docs / case_id
    directory.mkdir(mode=0o700)
    sources = []
    for source in case["sources"]:
        source_id = source["id"]
        if not source_id.replace("-", "").isalnum():
            raise ValueError("Unsafe source name")
        name = f"{case_id}/{source_id}.txt"
        (docs / name).write_text(source["text"], encoding="utf-8")
        sources.append(name)
        all_sources.append(name)
    principal(case_id, sources, "reader")
principal("ingester", all_sources, "ingester")
policy = root / "policy.json"
policy.write_text(json.dumps({"principals": principals}))
os.chmod(policy, 0o600)
print(f"Prepared {len(fixture)} cases and {len(all_sources)} fictional sources")
