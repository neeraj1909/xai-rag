"""Resource-limited child for parsing one private document snapshot."""

from __future__ import annotations

import contextlib
import json
import math
import resource
import sys
from pathlib import Path


def main() -> None:
    path, max_characters, memory_bytes, timeout_seconds = sys.argv[1:]
    memory_limit = int(memory_bytes)
    cpu_limit = max(1, math.ceil(float(timeout_seconds)))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    with contextlib.redirect_stdout(sys.stderr):
        from xai_rag.ingestion.parser import parse_file

        text = parse_file(Path(path))
    if len(text) > int(max_characters):
        raise ValueError("extracted text exceeds limit")
    json.dump(text, sys.stdout, ensure_ascii=True)


if __name__ == "__main__":
    main()
