"""Create an isolated source tree and test-only policy from a fictional fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
args.root.mkdir(mode=0o700, parents=True, exist_ok=False)
documents = args.root / "documents"
documents.mkdir(mode=0o700)
fixture = json.loads(args.fixture.read_text())
sources = []
for doc in fixture["documents"]:
    name = f"{doc['id']}.txt"
    if not name.replace("_", "").replace(".", "").isalnum():
        raise ValueError("unsafe source name")
    (documents / name).write_text(doc["text"], encoding="utf-8")
    sources.append(name)
key = secrets.token_urlsafe(48)
policy = {
    "principals": [
        {
            "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "tenant": "synthetic-eval",
            "sources": sources,
            "role": "ingester",
        }
    ]
}
for path, text in ((args.root / "key", key), (args.root / "policy.json", json.dumps(policy))):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(text)
print(f"Prepared {len(sources)} synthetic documents and a private temporary policy")
