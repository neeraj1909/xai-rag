# Second multilingual retrieval diagnostic

This is an **exploratory synthetic stress test**, not an external holdout. An AI agent created the fictional 20-passage corpus and all 40 source-level labels together, after seeing the earlier C17 result. The case wording and relevance labels have not been independently reviewed. No model outputs were used to revise this fixture after its SHA-256 was recorded. The dataset has 10 questions each for Hindi→Hindi, English→Hindi, Hindi→English, and exact-code retrieval; the exact-code slice mixes language directions. Each source has a near-neighbor distractor on the same topic, often with a different name, number, or reversed code. Each question has one labeled relevant source.

The two pinned models and method match the [first diagnostic](multilingual-retrieval-diagnostic.md): no-prefix, L2-normalized sentence embeddings and exact cosine ranking over every full document. Both models ran offline from existing caches. This tests retrieval only: there is no BM25, fusion, reranking, answer generation, or human evaluation. It does not measure abstention, factual correctness of generated answers, or performance on real users' documents.

| Slice (10 questions each) | BGE top-1 | MiniLM top-1 | BGE top-3 | MiniLM top-3 | BGE top-5 | MiniLM top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hindi→Hindi | 3/10 | 10/10 | 6/10 | 10/10 | 7/10 | 10/10 |
| English→Hindi | 0/10 | 4/10 | 2/10 | 6/10 | 3/10 | 10/10 |
| Hindi→English | 0/10 | 2/10 | 0/10 | 7/10 | 0/10 | 10/10 |
| Exact code | 7/10 | 8/10 | 7/10 | 10/10 | 7/10 | 10/10 |

MiniLM retrieves the target in the top five for every question in this constructed set, but places it first for only 4/10 English→Hindi and 2/10 Hindi→English cases. For M12, it ranks the wrong Hindi service first; for M13, a similar English plan beats the intended Hindi plan. For M38 and M40, it ranks the T-24 service above the requested T-42 service. These failures matter if a later stage relies on only the first document or does not distinguish close identifiers. Top-five retrieval alone does not prove that a reranker or generator will use the right source.

## Every question

The entries show the labeled source's rank and the top-ranked source for each model. Source text and question wording are in the [fixture](evaluation/multilingual-challenge-cases.json); all 20 ranks and cosine scores per question are in the [raw results](evaluation/multilingual-challenge-results.json).

| Case | Slice | Gold source | BGE rank / top source | MiniLM rank / top source |
| --- | --- | --- | --- | --- |
| M01 | hindi_to_hindi | `hi_falcon` | 6 / `hi_coral` | 1 / `hi_falcon` |
| M02 | hindi_to_hindi | `hi_heron` | 10 / `hi_amber` | 1 / `hi_heron` |
| M03 | hindi_to_hindi | `hi_orbit` | 3 / `hi_coral` | 1 / `hi_orbit` |
| M04 | hindi_to_hindi | `hi_lunar` | 3 / `hi_orbit` | 1 / `hi_lunar` |
| M05 | hindi_to_hindi | `hi_amber` | 1 / `hi_amber` | 1 / `hi_amber` |
| M06 | hindi_to_hindi | `hi_coral` | 1 / `hi_coral` | 1 / `hi_coral` |
| M07 | hindi_to_hindi | `hi_neem` | 2 / `hi_amber` | 1 / `hi_neem` |
| M08 | hindi_to_hindi | `hi_peepal` | 4 / `hi_coral` | 1 / `hi_peepal` |
| M09 | hindi_to_hindi | `hi_mango` | 1 / `hi_mango` | 1 / `hi_mango` |
| M10 | hindi_to_hindi | `hi_guava` | 7 / `hi_coral` | 1 / `hi_guava` |
| M11 | english_to_hindi | `hi_falcon` | 7 / `en_cedar` | 1 / `hi_falcon` |
| M12 | english_to_hindi | `hi_heron` | 3 / `en_birch` | 3 / `hi_falcon` |
| M13 | english_to_hindi | `hi_orbit` | 12 / `en_star` | 4 / `en_star` |
| M14 | english_to_hindi | `hi_lunar` | 14 / `en_star` | 4 / `en_star` |
| M15 | english_to_hindi | `hi_amber` | 3 / `en_rose` | 1 / `hi_amber` |
| M16 | english_to_hindi | `hi_coral` | 4 / `en_rose` | 1 / `hi_coral` |
| M17 | english_to_hindi | `hi_neem` | 8 / `en_lake` | 4 / `en_lake` |
| M18 | english_to_hindi | `hi_peepal` | 12 / `en_lake` | 3 / `en_river` |
| M19 | english_to_hindi | `hi_mango` | 8 / `en_pine` | 4 / `en_pine` |
| M20 | english_to_hindi | `hi_guava` | 14 / `en_pine` | 1 / `hi_guava` |
| M21 | hindi_to_english | `en_cedar` | 14 / `hi_coral` | 4 / `hi_heron` |
| M22 | hindi_to_english | `en_birch` | 12 / `hi_amber` | 2 / `hi_heron` |
| M23 | hindi_to_english | `en_sun` | 18 / `hi_lunar` | 1 / `en_sun` |
| M24 | hindi_to_english | `en_star` | 14 / `hi_orbit` | 1 / `en_star` |
| M25 | hindi_to_english | `en_rose` | 20 / `hi_coral` | 2 / `hi_coral` |
| M26 | hindi_to_english | `en_lotus` | 19 / `hi_coral` | 3 / `hi_coral` |
| M27 | hindi_to_english | `en_river` | 11 / `hi_coral` | 4 / `hi_peepal` |
| M28 | hindi_to_english | `en_lake` | 12 / `hi_coral` | 3 / `hi_peepal` |
| M29 | hindi_to_english | `en_maple` | 13 / `hi_mango` | 4 / `hi_mango` |
| M30 | hindi_to_english | `en_pine` | 16 / `hi_mango` | 3 / `hi_mango` |
| M31 | exact_code | `hi_falcon` | 1 / `hi_falcon` | 1 / `hi_falcon` |
| M32 | exact_code | `hi_heron` | 1 / `hi_heron` | 1 / `hi_heron` |
| M33 | exact_code | `hi_falcon` | 1 / `hi_falcon` | 1 / `hi_falcon` |
| M34 | exact_code | `hi_heron` | 1 / `hi_heron` | 1 / `hi_heron` |
| M35 | exact_code | `en_cedar` | 11 / `hi_heron` | 1 / `en_cedar` |
| M36 | exact_code | `en_birch` | 11 / `hi_falcon` | 1 / `en_birch` |
| M37 | exact_code | `en_cedar` | 1 / `en_cedar` | 1 / `en_cedar` |
| M38 | exact_code | `en_birch` | 1 / `en_birch` | 2 / `en_cedar` |
| M39 | exact_code | `hi_falcon` | 1 / `hi_falcon` | 1 / `hi_falcon` |
| M40 | exact_code | `en_birch` | 11 / `hi_falcon` | 2 / `en_cedar` |

## Reproduction and limits

Fixture SHA-256: `f2494b9bf4a766a374dc1fc1a6e141196ff045cf95a208d541452c94056f0887`. Use the same [runner](evaluation/multilingual-retrieval-comparison.py) and pinned model revisions as the first diagnostic, substituting this fixture and an output path. Full rankings, package versions and timings are recorded in the JSON. The previously downloaded candidate cache was reused; no additional model download or provider call was made. Both models ran sequentially on CPU with two OpenMP threads. These 10-example slices and shared templates are not representative enough to support a production selection or a reliable confidence interval.

Next, commission independent relevance labels for diverse real or realistically sampled documents and query language pairs, include more exact identifiers and conflicting versions, and evaluate end-to-end retrieval plus reranking and answer quality before a migration. A model change also requires rebuilding a versioned vector index and checking latency, storage, rollback and licensing.
