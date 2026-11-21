# Predeclared exploratory weighted-RRF diagnostic

This plan was written before this weighted-grid calculation. The inputs are the saved [40-question lexical comparison](evaluation/holdout-lexical-results.json), including the MiniLM and real Elasticsearch top-five candidate lists. Both lists have sufficient depth to compute the first five positions of their fused union. The input's cases and earlier unweighted results are already public within this project, so the test subset is **not a blinded, independent holdout**; it is untouched by the weight-selection procedure below, but previous knowledge can influence this entire experiment.

- Assign odd-numbered question IDs to training and even-numbered IDs to test, yielding five training and five test examples in each of four categories. Do not change the split after inspecting weights or results.
- Compare lexical weights 0, 0.25, 0.5, 0.75, 1, with dense weight `1 - lexical`; use ranks from the saved top-five lists and score `dense_weight/(60+dense_rank) + lexical_weight/(60+lexical_rank)`. A document absent from a branch has zero contribution; omit a branch entirely at zero weight.
- Rank descending by score, then by the source ID used in the saved diagnostic. Select the weight that maximizes training Hit@1 across all 20 training questions. Break equal training scores in favor of the smallest lexical weight. The test data never affects weight selection.
- Report train-grid hits, then evaluate the selected weight and the fixed dense-only and equal-weight comparators on the 20 test questions and each five-question test slice. Include Hit@1, Hit@3 and Hit@5 with actual denominators, candidate-union size, exact tie counts, and selected test question ranks. Do not use this split to select per-language weights.
- Cases and labels are synthetic, AI-authored and unreviewed, and these tiny slices share templates. Scores, especially after selecting on 20 training cases, are unstable and cannot justify a production change. The ranking uses source IDs to break ties; production chunk IDs can resolve ties differently. This is a simulation on saved candidates, not a rerun of Elasticsearch, ANN retrieval, reranking or generation.

## Results

The fixed training objective selected lexical weight **0**: dense-only scored 13/20 training Hit@1. The remaining lexical weights scored 10/20 at 0.25, 10/20 at 0.5, 9/20 at 0.75, and 8/20 at 1. The procedure picked a single global weight; it did not tune each slice independently.

| Test slice, five questions each | Selected dense-only @1/@3/@5 | Equal-weight RRF @1/@3/@5 |
| --- | --- | --- |
| Hindi → Hindi | 5/5, 5/5, 5/5 | 5/5, 5/5, 5/5 |
| English → Hindi | 2/5, 4/5, 5/5 | 0/5, 1/5, 4/5 |
| Hindi → English | 1/5, 5/5, 5/5 | 0/5, 3/5, 5/5 |
| Exact code | 3/5, 5/5, 5/5 | 4/5, 5/5, 5/5 |
| All test cases, 20 | 11/20, 19/20, 20/20 | 9/20, 14/20, 19/20 |

Equal-weight RRF included 6–8 unique sources per test question from the two top-five branch lists; dense-only included exactly five. Neither method could recover sources outside its saved candidate union. Equal-weight fusion had exact score ties crossing the top-one boundary in 3/20 test cases, the top-three boundary in 10/20, and the top-five boundary in 15/20. The calculation uses exact rational arithmetic to detect ties. Source ID breaks the tie in this diagnostic; the app orders using chunk IDs, so its ordering and metrics can differ. M38, for example, moves the correct T-42 passage from rank two to rank one under equal-weight fusion, but that rank-one result is tied with the wrong T-24 passage. In M14, equal fusion moves the labelled Hindi source from rank four to rank six, outside the top-five cutoff.

The selected weight and small apparent test advantage are **not a recommendation to disable BM25**. The earlier report exposed all 40 questions and unweighted results, both subsets are tiny and template-based, and only one deterministic split was tried. We did not test production data, other language pairs, real index latency, a reranker or generated answers. A production decision needs an independently labelled, previously unseen evaluation with enough cases per language and identifier slice, explicit chunk-ID tie behavior, and separate end-to-end safety and answer checks.

The exact split, full training grid, test rankings, candidate pools and tie groups are in the [raw result](evaluation/holdout-weighted-rrf-results.json); the offline [runner](evaluation/holdout-weighted-rrf-runner.py) preserves the selection procedure. The saved ranking input SHA-256 is `cbfb989f72a12556839e77d1641f1a35d944c13af663bacf8f51db8f91519f19`. Reproduce with `python3 -B projects/xai-rag/evaluation/holdout-weighted-rrf-runner.py` after preserving the existing output; the runner uses exclusive creation to avoid overwriting it. No database, provider call, or application source modification is required.
