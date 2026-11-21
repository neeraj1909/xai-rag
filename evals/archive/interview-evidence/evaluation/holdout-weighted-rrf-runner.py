"""Predeclared odd/even exploratory split over saved multilingual rankings."""

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "holdout-lexical-results.json"
OUTPUT = HERE / "holdout-weighted-rrf-results.json"
GRID = tuple(Fraction(n, 4) for n in range(5))
EXPECTED_DATA_HASH = "f2494b9bf4a766a374dc1fc1a6e141196ff045cf95a208d541452c94056f0887"


def rank(row, lexical_weight):
    scores = {}
    for stage, weight in (("minilm", 1 - lexical_weight), ("bm25", lexical_weight)):
        if not weight:
            continue
        for position, entry in enumerate(row[stage]["ranking"][:5], 1):
            scores[entry["id"]] = scores.get(entry["id"], Fraction(0)) + weight / (
                60 + position
            )
    order = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return order, scores


def score(rows, weight):
    out = {}
    for category in (
        "all",
        "hindi_to_hindi",
        "english_to_hindi",
        "hindi_to_english",
        "exact_code",
    ):
        selected = [r for r in rows if category == "all" or r["category"] == category]
        hit = Counter()
        ties = Counter()
        sizes = []
        traces = []
        for row in selected:
            order, values = rank(row, weight)
            wanted = set(row["relevant"])
            ranks = [i for i, x in enumerate(order, 1) if x in wanted]
            for k in (1, 3, 5):
                hit[k] += int(bool(wanted & set(order[:k])))
                if len(order) > k and values[order[k - 1]] == values[order[k]]:
                    ties[k] += 1
            sizes.append(len(order))
            traces.append(
                {
                    "id": row["id"],
                    "gold": row["relevant"],
                    "gold_ranks": ranks,
                    "ranking": order,
                    "equal_score_groups": [
                        [doc for doc in order if values[doc] == value]
                        for value in sorted(set(values.values()), reverse=True)
                        if sum(values[doc] == value for doc in order) > 1
                    ],
                }
            )
        out[category] = {
            "n": len(selected),
            "hits": {str(k): hit[k] for k in (1, 3, 5)},
            "tie_across_cutoff": {str(k): ties[k] for k in (1, 3, 5)},
            "candidate_union_size": dict(sorted(Counter(sizes).items())),
            "cases": traces,
        }
    return out


def main():
    data_bytes = INPUT.read_bytes()
    data = json.loads(data_bytes)
    if data["fixture_sha256"] != EXPECTED_DATA_HASH:
        raise ValueError("Unexpected frozen fixture")
    rows = data["rows"]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != 40 or set(ids) != {f"M{i:02}" for i in range(1, 41)}:
        raise ValueError("Expected 40 unique fixed IDs")
    for r in rows:
        if not r["relevant"] or any(
            "error" in r[stage] or "ranking" not in r[stage]
            for stage in ("minilm", "bm25")
        ):
            raise ValueError("Missing labels or saved branch results")
    train = [r for r in rows if int(r["id"][1:]) % 2]
    test = [r for r in rows if not int(r["id"][1:]) % 2]
    for subset in (train, test):
        counts = Counter(r["category"] for r in subset)
        if set(counts.values()) != {5} or len(counts) != 4:
            raise ValueError("Expected five examples per category in each split")
    train_scores = {w: score(train, w) for w in GRID}
    selected = min(GRID, key=lambda w: (-train_scores[w]["all"]["hits"]["1"], w))
    # Only after the weight has been fixed, evaluate the held-out procedure.
    test_weights = sorted({Fraction(0), Fraction(1, 2), selected})
    result = {
        "input_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "fixture_sha256": data["fixture_sha256"],
        "split": {
            "train_ids": [r["id"] for r in train],
            "test_ids": [r["id"] for r in test],
            "status": "Exploratory split; prior complete dataset/results known, not independent or blinded",
        },
        "rrf_constant": 60,
        "branch_depth": 5,
        "tie_break": "source ID ascending; application uses chunk IDs",
        "training_selection": "maximize train Hit@1; tie choose lowest lexical weight",
        "selected_lexical_weight": str(selected),
        "train_grid": {str(w): train_scores[w] for w in GRID},
        "test": {str(w): score(test, w) for w in test_weights},
    }
    with OUTPUT.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("selected lexical weight", selected)
    for weight in GRID:
        print("train", weight, train_scores[weight]["all"]["hits"])
    for weight in test_weights:
        print("test", weight, result["test"][str(weight)]["all"]["hits"])
        for category, values in result["test"][str(weight)].items():
            if category != "all":
                print(category, values["hits"], "of", values["n"])


if __name__ == "__main__":
    main()
