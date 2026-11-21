# Exploratory multilingual dense-retrieval comparison

This small synthetic experiment investigates why an English question about a Hindi source (C17) was missed. It is a diagnostic, not evidence that a model is production-ready or that it will generalize to other languages or corpora. The 31 fictional passages and 28 queries were authored together by an AI agent; their relevance labels have no independent human review. The unanswerable cases C08 and C27 have no relevant-source labels and are excluded from relevance totals; this experiment does not measure abstention.

## What was run

The same full-text passages and unprefixed queries were embedded with (1) the project's configured [BGE small English model](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) and (2) [multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42). Both output 384-dimensional vectors. The project's `embed_texts` normalizes document and query embeddings and does not add a query instruction; we matched this behavior. We ranked all 31 documents by exact dot product of L2-normalized vectors (cosine similarity), with source ID as tie breaker. No Chroma, BM25, fusion, reranking, LLM generation, paid service, or API call was involved in the inference run. The saved hybrid challenge used the same fixture and one full document per indexed chunk; exact search here is not a rerun of its full pipeline.

C17 asks: “How many unused yearly vacation days can be carried forward at Surya?” Its labeled source, `hindi_leave`, says up to four days may be carried over. **BGE ranked it 22nd (cosine 0.4833); multilingual MiniLM ranked it first (0.5635).** The archived Chroma run had already missed it in its dense top five, and its fusion and reranking missed it too. The candidate's improvement only establishes that the passage can be surfaced in this example, not that answers will be correct.

| Source-level measure | BGE | Multilingual MiniLM |
| --- | ---: | ---: |
| At least one labeled source in top 1 | 22/26 | 21/26 |
| At least one labeled source in top 5 | 25/26 | 26/26 |
| All labeled sources in top 3 | 24/26 | 26/26 |
| All labeled sources in top 5 | 25/26 | 26/26 |

A question with multiple labeled sources counts as “all” only if every source appears by that rank. C15 illustrates a tradeoff: MiniLM raises one required source from rank 5 to 3, but moves the other from rank 1 to 2. C07 and C18 also move from rank 1 to 2. Cosine values are model-specific and are not calibrated confidence scores; do not compare their absolute magnitudes across models.

## Every case: ranks of labeled sources

Rank 1 is best. “Unlabeled” denotes an abstention case; see raw results for every full ranking and score.

| Case | BGE relevant-source ranks | Multilingual MiniLM relevant-source ranks |
| --- | --- | --- |
| C01 | `cooling_a17`: 1 | `cooling_a17`: 1 |
| C02 | `cooling_a71`: 1 | `cooling_a71`: 1 |
| C03 | `audit_nov24`: 1 | `audit_nov24`: 1 |
| C04 | `audit_apr22`: 2 | `audit_apr22`: 2 |
| C05 | `audit_exception`: 1 | `audit_exception`: 1 |
| C06 | `harbor_outbound`: 2, `harbor_return`: 1 | `harbor_outbound`: 1, `harbor_return`: 2 |
| C07 | `harbor_return`: 1 | `harbor_return`: 2 |
| C08 | Unlabeled | Unlabeled |
| C09 | `lab_tube_b`: 1 | `lab_tube_b`: 1 |
| C10 | `membership_family`: 1 | `membership_family`: 1 |
| C11 | `membership_family`: 2, `catalog_paper`: 1 | `membership_family`: 2, `catalog_paper`: 1 |
| C12 | `catalog_disc`: 1 | `catalog_disc`: 1 |
| C13 | `hindi_leave`: 1 | `hindi_leave`: 1 |
| C14 | `hindi_sick`: 1 | `hindi_sick`: 1 |
| C15 | `hindi_expense`: 5, `hindi_expense_new`: 1 | `hindi_expense`: 3, `hindi_expense_new`: 2 |
| C16 | `hindi_expense_old`: 1 | `hindi_expense_old`: 1 |
| C17 | `hindi_leave`: 22 | `hindi_leave`: 1 |
| C18 | `access_n81`: 1 | `access_n81`: 2 |
| C19 | `access_n18`: 1 | `access_n18`: 1 |
| C20 | `parcel_west`: 2 | `parcel_west`: 2 |
| C21 | `harbor_outbound`: 2 | `harbor_outbound`: 1 |
| C22 | `equipment_battery`: 2, `equipment_charge`: 1 | `equipment_battery`: 1, `equipment_charge`: 2 |
| C23 | `invoice_x41`: 1 | `invoice_x41`: 1 |
| C24 | `invoice_x14`: 1 | `invoice_x14`: 1 |
| C25 | `meeting_astr`: 1 | `meeting_astr`: 1 |
| C26 | `retail_silver`: 1 | `retail_silver`: 1 |
| C27 | Unlabeled | Unlabeled |
| C28 | `retail_silver`: 1, `retail_gold`: 2 | `retail_silver`: 2, `retail_gold`: 1 |

## Reproduce

The machine-readable [full output](evaluation/multilingual-retrieval-comparison.json), [runner](evaluation/multilingual-retrieval-comparison.py), and [fixture](evaluation/retrieval-challenge-cases.json) are included. Fixture SHA-256: `9d630857de2beca9ca9d270598276b0c9e0898918649a96864151f2e541e59eb`. The input and labels match the archived challenge's dataset hash. The project source-tree snapshot used by that archived challenge was `968bdafa6c1c5174f4f1385cf4f425debee1601d19bcc73f48e99807d912efb5`; this diagnostic only uses the fixture and model inference, not that running source snapshot.

Download the pinned candidate snapshot using `huggingface_hub.snapshot_download`, allowing only `config.json`, `config_sentence_transformers.json`, `sentence_bert_config.json`, `modules.json`, `1_Pooling/config.json`, `model.safetensors`, `sentencepiece.bpe.model`, `special_tokens_map.json`, `tokenizer.json`, and `tokenizer_config.json`. The main safetensors file is 470,641,600 bytes; our separate diagnostic cache used 484,801,408 bytes total. The baseline revision must also be cached. No pre-existing cache was deleted or overwritten.

From the interview-materials repository, with the XAI-RAG virtual environment, point `CANDIDATE_SNAPSHOT` to the downloaded snapshot and run:

```bash
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/neeraj/Documents/xai-rag/.venv/bin/python projects/xai-rag/evaluation/multilingual-retrieval-comparison.py \
  --fixture projects/xai-rag/evaluation/retrieval-challenge-cases.json \
  --candidate "$CANDIDATE_SNAPSHOT" --output /tmp/multilingual-retrieval-comparison.json
```

The saved run records package versions, model revisions, full per-case scores and timings. CPU inference with two OpenMP threads took 11.613 seconds for the 31 passages and 28 queries with BGE and 2.931 seconds with MiniLM; sequential load times were 1.250 and 3.648 seconds. These are single process observations under uncontrolled load, not a latency benchmark or basis for choosing a model.

## Decision and next step

Do not switch production embeddings on this sample alone. A switch would require re-embedding the corpus into a new versioned index, dual-running representative language-pair and domain slices, comparing full hybrid retrieval and answer/abstention behavior, measuring latency and storage at realistic scale, and rolling back via an index alias if the candidate regresses. Have domain reviewers independently label source relevance, include adversarial near-duplicates and missing-answer cases, and report per-language and per-slice recall with confidence intervals. English-query/Hindi-document retrieval needs its own sufficiently large slice. Review data governance and model license before deployment.
