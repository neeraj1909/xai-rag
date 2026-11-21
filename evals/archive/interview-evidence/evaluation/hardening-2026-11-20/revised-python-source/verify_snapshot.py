"""Verify the revised evaluation's archived Python tree."""

import hashlib
from pathlib import Path
from zipfile import ZipFile


EXPECTED = "0d845a57bbbc9f695c3590407469a8dc352ca49316a1675033040c664a78ddaa"
archive = Path(__file__).with_name("revised-python-source.zip")
digest = hashlib.sha256()
with ZipFile(archive) as snapshot:
    names = sorted(name for name in snapshot.namelist() if name.endswith(".py"))
    if not names:
        raise SystemExit("Snapshot has no Python files")
    for name in names:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(snapshot.read(name)).digest())
actual = digest.hexdigest()
print(f"Python tree SHA-256: {actual}")
if actual != EXPECTED:
    raise SystemExit("Snapshot does not match the evaluation manifest")
print(f"Matches evaluation manifest ({len(names)} Python files).")
