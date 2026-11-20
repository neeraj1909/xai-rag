"""Compare scoped content/config snapshots; writers must remain quiescent."""

import json
import sys
from pathlib import Path

snapshots = [json.loads(Path(name).read_text()) for name in sys.argv[1:]]
if len(snapshots) < 2 or any(not item["record_parity"] for item in snapshots):
    raise SystemExit("Missing snapshots or record parity failure")
if any(item != snapshots[0] for item in snapshots[1:]):
    raise SystemExit("Index content, vectors or configuration changed; stopping evaluation")
