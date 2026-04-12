# XAI-RAG: Explainable Retrieval-Augmented Generation

Production RAG system with built-in explainability — every answer comes with retrieval attribution, NLI faithfulness scores, and RAGAS-based quality metrics.

## What Makes This Different

Every RAG demo returns answers. **XAI-RAG also explains WHY:**

- **Retrieval Attribution** — For each retrieved chunk: vector similarity score, BM25 lexical match score, RRF combined rank, cross-encoder re-ranker score, and matching terms
- **NLI Faithfulness** — Every claim in the answer is checked against retrieved context using Natural Language Inference (DeBERTa-v3-large-mnli). Verdict: supported / not_supported / neutral
- **RAGAS Evaluation** — Automatic evaluation of faithfulness, answer relevancy, and context precision on every query

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
│  6. RAGAS Auto-Evaluation                         │
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
# Edit .env → add XAI_RAG_OPENAI_API_KEY=sk-...
uv sync

# 2. Infrastructure (ChromaDB :8100, Elasticsearch :9200, Redis :6379, Jaeger :16686)
docker compose up -d
docker compose ps   # verify 4 services healthy

# 3. Add documents to sample_docs/
# (any PDF, MD, TXT files you want to search over)

# 4. Ingest
uv run xai-rag ingest ./sample_docs/ --strategy semantic

# 5. Query
uv run xai-rag query "your question here" --explain --faithfulness

# 6. API + Observability
uv run uvicorn xai_rag.api.app:app --reload --port 8000
# API docs:  http://localhost:8000/docs
# Jaeger UI: http://localhost:16686
```

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Default | Description |
|---|---|---|
| `XAI_RAG_CHROMA_HOST` | `localhost` | ChromaDB server host |
| `XAI_RAG_CHROMA_PORT` | `8100` | ChromaDB server port |
| `XAI_RAG_CHROMA_COLLECTION` | `xai_rag_documents` | Collection name for chunks |
| `XAI_RAG_ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `XAI_RAG_OPENAI_API_KEY` | — | Required for generation |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (ChromaDB + ES status) |
| `/query` | POST | Full RAG query with explanations |
| `/query/stream` | POST | SSE streaming (retrieval → generation → faithfulness) |
| `/ingest` | POST | Ingest documents from a path |
| `/docs` | GET | OpenAPI documentation |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
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
| Embeddings | BGE-large-en-v1.5 | Open-source, MTEB top-10, 1024 dim |
| Re-ranker | BGE-reranker-v2-m3 | Open-source cross-encoder |
| NLI Model | DeBERTa-v3-large-mnli | Best open NLI for faithfulness |
| LLM | GPT-4o / Claude (configurable) | Structured output support |
| Backend | FastAPI (async) | Production async Python |
| Evaluation | RAGAS | Standard RAG evaluation framework |
| Observability | OpenTelemetry + Jaeger | Open standard tracing |
| Caching | Redis | Fast response caching |

## Project Structure

```
xai-rag/
├── src/xai_rag/
│   ├── ingestion/          # Document processing pipeline
│   │   ├── parser.py       # PDF/DOCX/TXT → text
│   │   ├── chunker.py      # 3 strategies: fixed, semantic, parent_doc
│   │   ├── embedder.py     # BGE-large embedding
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
│   ├── evaluation/         # Quality metrics
│   │   └── ragas_eval.py   # RAGAS auto-evaluation
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
├── docker-compose.yml      # ChromaDB, ES, Redis, Jaeger
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
| Re-ranking | Cross-encoder | 10-20% accuracy boost for minimal cost | Skip (cheaper but less accurate) |
| Faithfulness | NLI (DeBERTa) | Runs locally, no API dependency | GPT-4 judge (better but expensive) |
| Framework | Raw Python + FastAPI | 12-Factor: own your control flow | LangChain (too much abstraction) |
| Observability | OTEL + Jaeger | Open standard, vendor-neutral | Langfuse only (LLM-specific) |

## Author

**Neeraj Kumar Singh** — Research Scientist specializing in XAI, NLP, and Transformer architectures.
Built on the foundation of DLBacktrace (explainability for transformers) and SafeSpeech (NLI-based hate speech mitigation in Indic languages).

- GitHub: [@neeraj1909](https://github.com/neeraj1909)
- LinkedIn: [neeraj1909](https://linkedin.com/in/neeraj1909)
