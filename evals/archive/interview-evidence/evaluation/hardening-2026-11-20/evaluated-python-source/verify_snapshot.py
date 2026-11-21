"""Verify the Python tree hash using the original evaluation's algorithm."""

import hashlib
from pathlib import Path
from zipfile import ZipFile


EXPECTED = "6e3c87bd0f650459d490c4f0d2c975f11ee18ce47686284f70014c374dbe9abd"
archive = Path(__file__).with_name("evaluated-python-source.zip")
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
