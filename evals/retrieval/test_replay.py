"""Behavior checks for the offline saved-ranking replay."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("replay.py")


def run_replay(tmp_path, rows):
    source = tmp_path / "rankings.json"
    source.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True, check=False
    )


def test_replays_rankings_with_current_fusion_and_category_denominators(tmp_path):
    rows = [
        {
            "id": "one",
            "category": "cross-language",
            "relevant": ["b"],
            "minilm": {"ranking": [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]},
            "bm25": {"ranking": [{"id": "b", "score": 3.0}]},
        },
        {
            "id": "two",
            "category": "exact-code",
            "relevant": ["z"],
            "minilm": {"ranking": [{"id": "z", "score": 0.9}]},
            "bm25": {"ranking": [{"id": "a", "score": 3.0}]},
        },
    ]
    result = run_replay(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["cases"][0]["fused"] == ["b", "a"]
    assert report["cases"][1]["fused"] == ["a", "z"]  # equal score -> ID order
    assert report["groups"]["all"]["n"] == 2
    assert report["groups"]["all"]["fused"]["hit@1"] == 0.5
    assert report["groups"]["cross-language"]["vector"]["hit@1"] == 0.0
    assert report["groups"]["cross-language"]["fused"]["hit@1"] == 1.0


def test_rejects_failed_or_unlabelled_rows_instead_of_inventing_scores(tmp_path):
    row = {
        "id": "one",
        "category": "x",
        "relevant": ["a"],
        "minilm": {"error": "Timeout"},
        "bm25": {"ranking": []},
    }
    result = run_replay(tmp_path, [row])
    assert result.returncode != 0
    assert "one" in result.stderr
    row["minilm"] = {"ranking": []}
    row["relevant"] = []
    result = run_replay(tmp_path, [row])
    assert result.returncode != 0
    assert "one" in result.stderr
