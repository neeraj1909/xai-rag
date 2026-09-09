# XAI-RAG: Explainable Retrieval-Augmented Generation

RAG system with built-in explainability: responses can include retrieval attribution and optional NLI faithfulness scores. The default runtime dependency set excludes the optional RAGAS evaluation stack so online serving does not carry evaluation-only packages.

> The included Docker Compose stack is for local learning only. Keep its ChromaDB and Elasticsearch ports on loopback, do not expose it to the internet, and replace the local ChromaDB server with a patched/private managed vector store before production deployment.

## What Makes This Different

Every RAG demo returns answers. **XAI-RAG also explains WHY:**

- **Retrieval Attribution** — For each retrieved chunk: vector similarity score, BM25 lexical match score, RRF combined rank, cross-encoder re-ranker score, and matching terms
- **NLI Faithfulness** — Every claim in the answer is checked against retrieved context using Natural Language Inference (DeBERTa-v3-base MNLI/FEVER/ANLI). Verdict: supported / not_supported / neutral

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
│  5. Faithfulness Checker (NLI per claim)          │  ← XAI differentiator
├──────────────────────────────────────────────────┤
└──────────────────────────────────────────────────┘
    │
    ▼
Answer + Retrieval Explanations + Faithfulness Report + Quality Scores
```

## Quick Start

```bash
# 1. Clone & Setup
git clone https://github.com/neeraj1909/xai-rag.git
cd xai-rag
cp .env.example .env
# Edit .env → add the LLM key and generate XAI_RAG_API_KEY with:
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
| `XAI_RAG_OPENAI_API_KEY` | — | Required for generation |
| `XAI_RAG_API_KEY` | — | Required (32+ characters) for `/query`, `/query/stream`, and `/ingest` |
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

## Component evaluation

The offline evaluator is deterministic and makes no model or hosted-provider calls:

```bash
uv run xai-rag evaluate evals/smoke/evaluator-contract.jsonl \
  --k 1 --k 2 \
  --gates evals/gates.toml \
  --output evals/smoke/baseline-report.json
```

It reports sample-sized metrics for ingestion and chunk provenance, dual-index parity, vector/BM25/fused/reranked retrieval, citation validity and coverage, reference/judge/human answer scores, claim faithfulness, refusal behavior, errors, latency percentiles, token use, and cost when observed. Missing evidence is `not_computed`; a missing blocking metric fails its gate.

The included smoke dataset validates only the evaluator contract. Follow [`evals/README.md`](evals/README.md) to build a representative, human-reviewed dataset before setting product quality or SLO gates.

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
| Evaluation | Optional local tooling | Not installed in the production runtime by default |
| Observability | OpenTelemetry | Configure a secured external collector |

## Production boundary

This repository can prove deterministic code, evaluator, container, and local service behavior. A real production sign-off still requires a deployment target, managed/authenticated data stores, TLS/network policy, backups and restore tests, representative reviewed evaluation data, approved quality/latency/cost thresholds, and a privacy/retention policy.

The built-in rate limit and concurrency controls are per process. Multi-replica deployments must enforce quotas at a trusted gateway or shared rate-limit store. Raw queries/documents are not attached to pipeline telemetry; only hashes, counts, model/config revisions, outcomes, and timings are emitted by default.

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
  --output evals/smoke/baseline-report.json
uv build

# Real Chroma + Elasticsearch integration boundary (services must be running)
XAI_RAG_RUN_INTEGRATION=1 uv run pytest -m integration -q

# Disposable black-box HTTP workflow with a deterministic local LLM boundary
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
| Faithfulness | NLI (DeBERTa) | Runs locally, no API dependency | GPT-4 judge (better but expensive) |
| Framework | Raw Python + FastAPI | 12-Factor: own your control flow | LangChain (too much abstraction) |
| Observability | OpenTelemetry | Open standard, vendor-neutral | Vendor-specific tracing |

## Author

**Neeraj Kumar Singh** — Research Scientist specializing in XAI, NLP, and Transformer architectures.
Built on the foundation of DLBacktrace (explainability for transformers) and SafeSpeech (NLI-based hate speech mitigation in Indic languages).

- GitHub: [@neeraj1909](https://github.com/neeraj1909)
- LinkedIn: [neeraj1909](https://linkedin.com/in/neeraj1909)
