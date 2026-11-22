"""Demonstrate a label-leak regression without changing project source."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "evals/conversion"
RESULTS = Path(__file__).resolve().parent
FIXED = "                refusal = None\n"
MUTATION = "                refusal = expected_refusal\n"
TEST = "test_verified_response_does_not_infer_refusal_from_expected_label"


def run_case(directory: Path, label: str) -> int:
    command = [sys.executable, "-m", "pytest", "-q", f"{directory / 'test_convert.py'}::{TEST}"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (RESULTS / f"{label}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xai-converter-regression-") as temp:
        directory = Path(temp)
        for name in ("to_dataset.py", "test_convert.py"):
            shutil.copy2(SOURCE / name, directory / name)
        path = directory / "to_dataset.py"
        original = path.read_text(encoding="utf-8")
        assert original.count(FIXED) == 1, "Expected one targeted line; source may have changed."
        path.write_text(original.replace(FIXED, MUTATION), encoding="utf-8")
        red = run_case(directory, "red-mutated")
        path.write_text(original, encoding="utf-8")
        green = run_case(directory, "green-current")
    print(f"converter_sha256={hashlib.sha256(original.encode()).hexdigest()}")
    print(f"mutated_exit={red}; current_exit={green}")
    if red != 1 or green != 0:
        raise SystemExit("Unexpected regression check result; inspect saved logs.")


if __name__ == "__main__":
    main()
