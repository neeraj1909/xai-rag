"""Checks for the offline NLI diagnostic's accounting."""

from types import SimpleNamespace

from nli_stress import evaluate


def test_records_each_case_and_keeps_failures_out_of_agreement() -> None:
    class Checker:
        def _check_single_claim(self, claim, chunks):
            if claim.text == "error":
                raise RuntimeError("fixture failure")
            return SimpleNamespace(
                verdict=SimpleNamespace(value="supported"),
                nli_entailment=0.9,
                nli_contradiction=0.05,
            )

    cases = [
        {"id": "ok", "slice": "basic", "premise": "p", "hypothesis": "h", "expected": "supported"},
        {
            "id": "bad",
            "slice": "basic",
            "premise": "p",
            "hypothesis": "error",
            "expected": "neutral",
        },
    ]
    report = evaluate(cases, Checker())

    assert report["planned"] == 2
    assert report["scored"] == 1
    assert report["matched"] == 1
    assert report["cases"][0]["observed"] == "supported"
    assert report["cases"][1]["error"] == "RuntimeError"
