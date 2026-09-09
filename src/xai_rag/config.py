"""Application configuration via environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """XAI-RAG configuration. All values can be overridden via environment variables."""

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8100, ge=1, le=65535)
    chroma_collection: str = "xai_rag_documents"

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "xai_rag_documents"

    # Embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_model_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    embedding_dimension: int = Field(default=384, ge=1)

    # Reranker model
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

    # NLI model for faithfulness checking
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_model_revision: str = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
    nli_entailment_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nli_contradiction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    openai_api_key: str = ""

    # Retrieval
    vector_search_k: int = Field(default=100, ge=1, le=1000)
    bm25_search_k: int = Field(default=100, ge=1, le=1000)
    reranker_candidate_k: int = Field(default=5, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    # Chunking
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)
    semantic_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    chunking_model: str = "all-MiniLM-L6-v2"  # fast model for semantic chunking only
    chunking_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    # Observability
    otel_endpoint: str = ""
    otel_service_name: str = "xai-rag"
    otel_insecure: bool = False

    # API
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: str = ""
    cors_origins: str = ""
    expose_docs: bool = False
    ingest_root: str = "./sample_docs"
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1_000_000)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    llm_context_max_chars: int = Field(default=32_000, ge=1, le=1_000_000)
    allow_request_evaluation: bool = False
    query_timeout_seconds: float = Field(default=120.0, gt=0.0, le=900.0)
    max_concurrent_queries: int = Field(default=4, ge=1, le=1_000)
    max_ingest_files: int = Field(default=100, ge=1, le=100_000)
    max_document_bytes: int = Field(default=20_000_000, ge=1)
    max_ingest_total_bytes: int = Field(default=100_000_000, ge=1)

    # Elasticsearch authentication for managed/production deployments
    elasticsearch_api_key: str = ""

    # Ignore retired/unknown environment variables so removing a setting does
    # not make an existing deployment fail during application import.
    model_config = {
        "env_prefix": "XAI_RAG_",
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()
