"""Prepare fictional documents and temporary scoped principals; never print keys."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

root = Path(os.environ["XAI_EVAL_DIR"])
if root.exists():
    raise SystemExit("Evaluation directory already exists")
root.mkdir(mode=0o700)
documents = root / "documents"
documents.mkdir(mode=0o700)
fixture = json.loads(Path("/tmp/xai-e2e-harness/holdout-cases.json").read_text())
sources = []
for doc in fixture["documents"]:
    name = f"{doc['id']}.txt"
    if not name.replace("_", "").replace(".", "").isalnum():
        raise ValueError("Unsafe document name")
    (documents / name).write_text(doc["text"], encoding="utf-8")
    sources.append(name)
key = secrets.token_urlsafe(48)
key_path = root / "key"
fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as f:
    f.write(key)
policy = {"principals": [{"key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                           "tenant": "synthetic-eval", "sources": sources,
                           "role": "ingester"}]}
(root / "policy.json").write_text(json.dumps(policy))
os.chmod(root / "policy.json", 0o600)
print(f"Prepared {len(sources)} fictional documents and one temporary principal")
