# XAI-RAG: Explainable Retrieval-Augmented Generation

XAI-RAG answers questions over a document collection with citations and retrieval
explanations. It combines semantic and keyword search, reranks the results, and
checks generated claims before releasing an answer through a REST or streaming API.

## Business problem

Teams need to find answers across policies, manuals and other documents without
manually reading every file. A fluent answer alone is difficult to trust: users
need to see which passages support it, and operators need to understand how those
passages were retrieved. XAI-RAG addresses this by returning cited answers and
retrieval score breakdowns, while withholding answers that do not pass its checks.

The Compose configuration is intended for local deployment. Keep ChromaDB and
Elasticsearch on loopback; see the [production boundary](#production-boundary)
before exposing the service or connecting sensitive data.

## Results

On 20 November 2026, an HTTP evaluation used 40 fictional, author-labelled
questions, real retrieval stores and local models, and OpenAI generation and
judging. RAGAS scored all 40 cases; the initial capped run and a separately
budgeted extension are combined below.

| Metric | Mean | Cases |
| --- | ---: | ---: |
| Faithfulness | 0.8750 | 40 |
| Answer relevancy | 0.6355 | 40 |
| Reference-based context precision | 0.9597 | 40 |

All 40 HTTP requests returned 200. Five of 27 cases labelled directly answerable
were refused despite relevant evidence being visible. These are synthetic,
model-judged results without independent human review; HTTP success and valid
citations do not prove answer correctness. The evaluated working tree predates
the current safety-and-coverage judge, so these scores do not measure the
current implementation. Model aliases may also change over time.

The [saved cases, per-request output and RAGAS scores](evals/results/2026-11-20-synthetic-http/)
include the source digest and evaluation limitations. [Reproduce the recorded
aggregates](#reproduce-the-evaluation) from the per-case scores without making
provider calls.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│  1. Hybrid Retrieval                              │
│     ├── ChromaDB (dense / semantic search)        │
│     ├── Elasticsearch (BM25 / lexical search)     │
│     └── Reciprocal Rank Fusion (RRF)              │
├──────────────────────────────────────────────────┤
│  2. Cross-Encoder Re-Ranking (BGE-Reranker-v2)   │
├──────────────────────────────────────────────────┤
│  3. Retrieval Explainer (score breakdown)         │  ← XAI differentiator
├──────────────────────────────────────────────────┤
│  4. LLM Generation (structured Pydantic output)   │
├──────────────────────────────────────────────────┤
│  5. NLI + independent LLM grounding judge        │  ← release gate
├──────────────────────────────────────────────────┤
└──────────────────────────────────────────────────┘
    │
    ▼
Checked claims and citations, or a fixed refusal / retry message
Optional retrieval explanations, NLI report, and RAGAS quality scores
```

### Components

- **Ingestion:** parses PDF, Markdown and text files, chunks their contents and
  records source metadata. An optional persistent journal supports host-local
  recovery after interrupted writes.
- **Dense retrieval:** embeds queries and searches document vectors in ChromaDB
  for semantic matches.
- **Lexical retrieval:** uses Elasticsearch BM25 to find matching terms, including
  exact names and identifiers. BM25 is the ranking algorithm; Elasticsearch is
  the search engine that runs it.
- **Fusion and reranking:** reciprocal rank fusion combines the two ranked
  lists; a cross-encoder scores query–passage pairs to reorder candidates.
- **Retrieval explanations:** report the available stage scores, ranks and
  matching terms for retrieved passages.
- **Generation:** sends bounded, prompt-visible evidence to an LLM and requests
  structured claims with source identifiers.
- **Answer checks:** local NLI checks claim support; a separate structured LLM
  judge checks support, relevance, safety and coverage. Failed checks withhold
  the whole answer; they remain fallible and can reject supported answers.
- **API and access:** FastAPI exposes REST and SSE endpoints. An optional static
  policy scopes tenant and source access before retrieval.
- **Evaluation:** the offline evaluator aggregates saved observations; optional
  RAGAS metrics assess response faithfulness, relevance and context precision.

This is a retrieval pipeline with explicit stages, not an agent, multi-agent
or MCP workflow.

### Answer verification and streaming

After generation, each claim must have a valid source and pass local NLI. A
separate configured LLM judge then checks support, relevance, safety and question
coverage against the complete prompt-visible evidence and NLI-selected citations. The answer is rendered
from checked claims and selected citations only after both gates pass. A negative
verdict, incomplete coverage or invalid claim withholds the entire answer as `insufficient_evidence`;
invalid judge output, missing judge configuration, timeout or provider failure
returns `verification_unavailable` and a retry message. The former response says
“I can't provide a complete, relevant and safe answer that I can verify.”; the
latter says “I can't verify an answer right now. Please try again later.”
`include_faithfulness`
controls the detailed NLI report, not whether the gates run. SSE can show
retrieval progress before verification but cannot release the answer until both
checks finish. `overall_confidence` and `overall_confidence_source` are nullable.
Partial answers are withheld when coverage fails; the judge can make mistakes and its
decisions are not a guarantee of safety or correctness.

Configure the judge provider and an explicit model; an empty model fails closed.
Each answer passing NLI triggers one extra provider request, with added latency
and billable tokens. It sends the complete bounded prompt-visible evidence to
the configured provider. `input_tokens` and `output_tokens` count generation only; judge token usage is
not included, so those fields do not give total request usage or cost.
A pass is not proof of truth, safety or permission: the models can make correlated
errors or follow malicious source instructions. Validate on independently reviewed
examples, including conflicting evidence, prompt injection and unsupported numbers.

## Quick Start

```bash
# 1. Clone & Setup
git clone https://github.com/neeraj1909/xai-rag.git
cd xai-rag
cp .env.example .env
# Edit .env → add the LLM key, explicitly select a grounding judge model,
# and generate XAI_RAG_API_KEY with:
# openssl rand -hex 32
uv sync --locked --extra dev

# 2. Local stack (ChromaDB :8100, Elasticsearch :9200, API :8000)
export XAI_RAG_API_KEY="$(openssl rand -hex 32)"
docker compose up -d --build
docker compose ps

# 3. Add documents to sample_docs/
# (any PDF, MD, TXT files you want to search over)

# 4. Ingest
uv run xai-rag ingest ./sample_docs/ --strategy semantic

# 5. Query
uv run xai-rag query "your question here" --explain --faithfulness

# 6. API (protected endpoints require X-API-Key)
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H "X-API-Key: $XAI_RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"your question here","top_k":5}'
# OpenAPI docs are disabled by default; set XAI_RAG_EXPOSE_DOCS=true only locally.
```

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Default | Description |
|---|---|---|
| `XAI_RAG_CHROMA_HOST` | `localhost` | ChromaDB server host |
| `XAI_RAG_CHROMA_PORT` | `8100` | ChromaDB server port |
| `XAI_RAG_CHROMA_COLLECTION` | `xai_rag_documents` | Collection name for chunks |
| `XAI_RAG_ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `XAI_RAG_ELASTICSEARCH_API_KEY` | — | Required for secured production Elasticsearch |
| `XAI_RAG_OPENAI_API_KEY` | — | Required for generation and the grounding judge |
| `XAI_RAG_GROUNDING_JUDGE_PROVIDER` | `openai` | Grounding judge provider; unsupported values fail closed |
| `XAI_RAG_GROUNDING_JUDGE_MODEL` | empty | Required explicit model; empty fails closed |
| `XAI_RAG_GROUNDING_JUDGE_BASE_URL` | empty | Optional compatible provider endpoint |
| `XAI_RAG_GROUNDING_JUDGE_TIMEOUT_SECONDS` | `30` | Judge call timeout, within the overall query deadline |
| `XAI_RAG_RAGAS_LLM_MODEL` | `gpt-4o-mini` | RAGAS judge model |
| `XAI_RAG_RAGAS_EMBEDDING_MODEL` | `text-embedding-3-small` | RAGAS answer-relevancy embedding model |
| `XAI_RAG_API_KEY` | — | Required (32+ characters) for protected endpoints in shared-key mode |
| `XAI_RAG_AUTH_POLICY_FILE` | empty | Enables static principal, tenant and source grants; see tenant guide |
| `XAI_RAG_INGESTION_JOURNAL_ROOT` | empty | Enables host-local journal; must be provisioned for every writer |
| `XAI_RAG_MAX_REQUEST_BODY_BYTES` | `65536` | HTTP request body cap, separate from document-size limits |
| `XAI_RAG_CORS_ORIGINS` | empty | Comma-separated explicit browser origins; wildcards are ignored |
| `XAI_RAG_INGEST_ROOT` | `./sample_docs` | Allowed root for server-side ingestion paths |
| `XAI_RAG_LLM_TIMEOUT_SECONDS` | `60` | Upstream LLM request timeout |
| `XAI_RAG_RERANKER_CANDIDATE_K` | `5` | Baseline cross-encoder candidate pool; raised to a larger requested `top_k` (API maximum 100) |
| `XAI_RAG_ALLOW_REQUEST_EVALUATION` | `false` | Permit expensive optional evaluation in the request path (offline runner preferred) |
| `XAI_RAG_QUERY_TIMEOUT_SECONDS` | `120` | Total per-process queue plus query execution deadline |
| `XAI_RAG_MAX_CONCURRENT_QUERIES` | `4` | Per-process in-flight query bound |
| `XAI_RAG_MAX_INGEST_FILES` | `100` | Maximum supported documents in one ingestion request |
| `XAI_RAG_MAX_DOCUMENT_BYTES` | `20000000` | Maximum bytes in one source document |
| `XAI_RAG_MAX_INGEST_TOTAL_BYTES` | `100000000` | Maximum total bytes in one ingestion request |

The `.env.example` judge model is an example tested with one account, not a
promise of model availability. Select a model available to your configured
endpoint that supports the required structured output. Access and model aliases
can change; repeat validation whenever the model or endpoint changes.

**Isolation and recovery:** By default a shared API key and indexes are used;
this is not tenant isolation. For tenant and source grants, configure an
operator-controlled policy and follow [tenant authorization](docs/tenant-authorization.md).
Ingestion recovery is opt-in: provision a private persistent journal and follow
[ingestion recovery](docs/ingestion-recovery.md), including the quiescence
requirement. The journal and its POSIX locks do not provide a distributed
transaction or cross-host coordination.

The API rejects oversized HTTP request bodies before parsing; the default cap is
65,536 bytes, distinct from file ingestion limits. See [request body limits](docs/request-body-limits.md)
for middleware and reverse-proxy boundaries.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/live` | GET | Process liveness; does not depend on downstream services |
| `/ready` | GET | Readiness check for ChromaDB and the required Elasticsearch index |
| `/health` | GET | Backward-compatible alias for `/ready` |
| `/query` | POST | Full RAG query with explanations (API key required) |
| `/query/stream` | POST | SSE streaming (API key required) |
| `/ingest` | POST | Ingest documents under `XAI_RAG_INGEST_ROOT` (API key required) |
| `/docs` | GET | Disabled by default; enable explicitly for local development |

## Reproduce the evaluation

The recorded per-case RAGAS scores and the fictional 40-question fixture are
in [`evals/results/2026-11-20-synthetic-http/`](evals/results/2026-11-20-synthetic-http/).
From the repository root, recompute the table above using Python's standard
library; this reads the saved scores and makes no API calls:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import mean

path = Path('evals/results/2026-11-20-synthetic-http/combined-ragas.json')
results = json.loads(path.read_text())
rows = [row for row in results['rows'] if row['status'] == 'computed']
assert len(rows) == 40 and len({row['case_id'] for row in rows}) == 40
for metric in ('faithfulness', 'answer_relevancy', 'context_precision'):
    print(f"{metric}: {mean(row['scores'][metric] for row in rows):.4f} (n={len(rows)})")
PY
```

These commands reproduce the **aggregation**, not the original live model run.
The original harness relied on temporary paths and was not packaged as a
portable runner; the source was a modified working tree, and model aliases can
change. A new live evaluation requires an isolated corpus and stores, configured
provider credentials and budget, pinned dependencies/models, and fresh per-case
captures. It may produce different scores. See the [result record](evals/results/2026-11-20-synthetic-http/README.md)
for source provenance and the initial failed and capped attempts.

### Verify the evaluator contract

The offline evaluator is deterministic and makes no model or hosted-provider calls:

```bash
uv run xai-rag evaluate evals/smoke/evaluator-contract.jsonl \
  --k 1 --k 2 \
  --gates evals/gates.toml \
  --output /tmp/xai-rag-contract-report.json
```

It reports sample-sized metrics for ingestion and chunk provenance, dual-index parity, vector/BM25/fused/reranked retrieval, citation validity and coverage, reference/judge/human answer scores, claim faithfulness, refusal behavior, errors, latency percentiles, token use, and cost when observed. Missing evidence is `not_computed`; a missing blocking metric fails its gate.

The included smoke dataset validates only the evaluator contract. Follow [`evals/README.md`](evals/README.md) to build a representative, human-reviewed dataset before setting product quality or SLO gates.

### RAGAS response evaluation

Install the optional evaluation dependency set, set
`XAI_RAG_ALLOW_REQUEST_EVALUATION=true`, then request evaluation explicitly:

```bash
uv sync --locked --extra evaluation
```

Run the API from that environment when using this feature. The default Docker
image intentionally omits the optional evaluation dependencies; enabling the
flag in that image returns an explicit `dependency_unavailable` status.

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: $XAI_RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does hybrid search improve RAG?","include_ragas":true}'
```

The `ragas_scores` response contains `faithfulness`, `answer_relevancy`, and
`context_precision`. Without a reference answer, context precision uses RAGAS
`ContextUtilization`, its reference-free context-precision metric. For an
verified answer, a valid RAGAS faithfulness score supplies `overall_confidence`;
otherwise the service uses mean NLI entailment. `overall_confidence_source`
identifies the source. Both fields are null when no answer passes verification;
these scores are not calibrated probabilities. RAGAS makes additional model and
embedding calls, so request evaluation remains disabled by default. RAGAS
0.4.3 also has an unresolved upstream security advisory in a multimodal helper;
this project uses only text metrics, but the package remains outside the default
production image until a patched compatible release is available.

For the evidence-backed status of every pipeline stage, remaining production
blockers, and recommended metrics, see
[`docs/production-readiness.md`](docs/production-readiness.md). Exact local-model
revisions and licenses are recorded in
[`docs/model-inventory.md`](docs/model-inventory.md).

Check live index consistency after ingestion:

```bash
uv run xai-rag validate-indexes --json-output
```

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: $XAI_RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does hybrid search improve RAG?",
    "top_k": 5,
    "include_explanations": true,
    "include_faithfulness": true
  }'
```

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Vector DB | ChromaDB (HNSW) | Simple, embedded-friendly, no extra infra |
| BM25 Search | Elasticsearch | Industry standard lexical search |
| Embeddings | BGE-small-en-v1.5 | Pinned open model, normalized 384-dimensional embeddings |
| Re-ranker | BGE-reranker-v2-m3 | Open-source cross-encoder |
| NLI Model | DeBERTa-v3-base MNLI/FEVER/ANLI | Pinned local claim-evidence classifier |
| LLM | OpenAI structured-output API | Structured output support |
| Backend | FastAPI (async) | Production async Python |
| Evaluation | Optional RAGAS 0.4 collections API + deterministic native metrics | Model-judged response quality plus stage-level regression metrics |
| Observability | OpenTelemetry | Configure a secured external collector |

## Production boundary

This repository can prove deterministic code, evaluator, container, and local service behavior. A real production sign-off still requires a deployment target, managed/authenticated data stores, TLS/network policy, backups and restore tests, representative reviewed evaluation data, approved quality/latency/cost thresholds, and a privacy/retention policy.

The built-in rate limit and concurrency controls are per process. Multi-replica deployments must enforce quotas at a trusted gateway or shared rate-limit store. The application pipeline does not attach raw queries or document content to explicit telemetry attributes, and it does not emit query hashes. Ingestion uses an unsalted source-path hash, which permits guesses and correlation. Review the full export path, access, and retention.

The CPU container pins model revisions but does not bake their weights into the
image. Pre-populate a durable, access-controlled Hugging Face cache (or build an
approved internal model image) before disabling outbound network access in a
production environment; otherwise the first query can incur a large download
and cold-start delay.

### Validation commands

```bash
# Static, unit, package, and deterministic evaluator gates
uv lock --check
uv run ruff format --check src tests
uv run ruff check .
uv run pytest -m unit -q --cov=src/xai_rag --cov-fail-under=70
uv run xai-rag evaluate evals/smoke/evaluator-contract.jsonl \
  --k 1 --k 2 --gates evals/gates.toml \
  --output /tmp/xai-rag-contract-report.json
uv build

# Real Chroma + Elasticsearch integration boundary (services must be running)
XAI_RAG_RUN_INTEGRATION=1 uv run pytest -m integration -q

# Disposable black-box HTTP workflow with deterministic local generator and judge.
# Requires Docker Compose, free ports 8100/9200/18000, and sufficient memory, disk
# and network/cache access for container images and pinned model downloads.
# The E2E overlay supplies the stub model and judge configuration; it does not
# validate a real hosted model. Teardown removes this project's test volumes.
XAI_RAG_IMAGE=xai-rag:validation XAI_RAG_HOST_PORT=18000 \
  docker compose -p xai-rag-validation \
  -f docker-compose.yml -f docker-compose.e2e.yml up -d --build
XAI_RAG_RUN_E2E=1 XAI_RAG_E2E_BASE_URL=http://127.0.0.1:18000 \
  XAI_RAG_E2E_API_KEY=e2e-only-api-key-with-at-least-32-characters \
  uv run pytest -m e2e -q -s
docker compose -p xai-rag-validation \
  -f docker-compose.yml -f docker-compose.e2e.yml \
  down --volumes --remove-orphans
```

## Project Structure

```
xai-rag/
├── src/xai_rag/
│   ├── ingestion/          # Document processing pipeline
│   │   ├── parser.py       # PDF/DOCX/TXT → text
│   │   ├── chunker.py      # 3 strategies: fixed, semantic, parent_doc
│   │   ├── embedder.py     # BGE-small 384-dimensional embedding
│   │   └── store.py        # ChromaDB + Elasticsearch storage
│   ├── retrieval/          # Search pipeline
│   │   ├── vector_search.py # ChromaDB cosine similarity
│   │   ├── bm25_search.py  # Elasticsearch BM25
│   │   ├── hybrid.py       # Reciprocal Rank Fusion
│   │   └── reranker.py     # Cross-encoder re-ranking
│   ├── explainability/     # XAI differentiators
│   │   ├── retrieval_explainer.py  # Score breakdown per chunk
│   │   └── faithfulness.py         # NLI claim verification
│   ├── generation/         # LLM generation
│   │   └── generator.py    # Structured output with citations
│   ├── evaluation/         # Versioned datasets, metrics, gates, and adapters
│   │   ├── metrics.py      # Deterministic component metrics
│   │   ├── models.py       # Strict evaluation contracts
│   │   ├── runner.py       # Reports, comparisons, and gates
│   │   └── ragas_eval.py   # Optional external evaluator adapter
│   ├── observability/      # Tracing
│   │   └── tracing.py      # OpenTelemetry setup
│   ├── api/                # FastAPI backend
│   │   └── app.py          # Endpoints + SSE streaming
│   ├── models.py           # Pydantic data models
│   ├── config.py           # Settings from environment
│   └── cli.py              # Click CLI
├── tests/                  # pytest test suite
├── scripts/
│   └── init_db.sql         # (legacy, no longer used)
├── docker-compose.yml      # Local-only ChromaDB, ES, and API
├── Dockerfile              # Multi-stage Python build
├── Makefile                # Developer shortcuts
└── pyproject.toml          # uv project config
```

## Design Decisions

| Decision | Choice | Why | Alternative |
|---|---|---|---|
| Vector DB | ChromaDB | Simple HTTP API, auto-manages HNSW index, easy local dev | pgvector (needs Postgres), Qdrant (heavier) |
| BM25 | Elasticsearch | Full-featured, production-proven | PostgreSQL FTS (simpler but weaker) |
| Chunking | 3 strategies | Different docs need different approaches | Single strategy (less flexible) |
| Re-ranking | Cross-encoder | A measurable second-stage relevance signal | Skip (cheaper; validate the quality delta) |
| Claim diagnostics | NLI (DeBERTa) | Per-claim supported/neutral/not-supported explanations | Omit claim-level diagnostics |
| RAG evaluation | RAGAS | Standard faithfulness, answer-relevancy, and context-precision metrics | Custom aggregate faithfulness score |
| Framework | Raw Python + FastAPI | 12-Factor: own your control flow | LangChain (too much abstraction) |
| Observability | OpenTelemetry | Open standard, vendor-neutral | Vendor-specific tracing |

## Author

**Neeraj Kumar Singh** — Research Scientist specializing in XAI, NLP, and Transformer architectures.
Built on the foundation of DLBacktrace (explainability for transformers) and SafeSpeech (NLI-based hate speech mitigation in Indic languages).

- GitHub: [@neeraj1909](https://github.com/neeraj1909)
- LinkedIn: [neeraj1909](https://linkedin.com/in/neeraj1909)
