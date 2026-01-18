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
│     ├── pgvector (dense / semantic search)        │
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
# 1. Clone
git clone https://github.com/neeraj1909/xai-rag.git
cd xai-rag

# 2. Setup
make setup          # installs dependencies via uv
cp .env.example .env  # edit with your API keys

# 3. Start infrastructure
make infra          # starts PostgreSQL+pgvector, Elasticsearch, Redis, Jaeger

# 4. Ingest documents
make ingest path=./sample_docs/ strategy=semantic

# 5. Query
make query q="What is the SafeSpeech pipeline?"

# 6. Start API server
make serve          # FastAPI on http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (Postgres + ES status) |
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
| Vector DB | PostgreSQL + pgvector (HNSW) | Specified in JD; no extra infra |
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
│   │   └── store.py        # pgvector + Elasticsearch storage
│   ├── retrieval/          # Search pipeline
│   │   ├── vector_search.py # pgvector cosine similarity
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
│   └── init_db.sql         # pgvector schema
├── docker-compose.yml      # Postgres, ES, Redis, Jaeger
├── Dockerfile              # Multi-stage Python build
├── Makefile                # Developer shortcuts
└── pyproject.toml          # uv project config
```

## Design Decisions

| Decision | Choice | Why | Alternative |
|---|---|---|---|
| Vector DB | pgvector | Capital Numbers JD specifies it; simplest infra | Qdrant (faster but extra service) |
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
