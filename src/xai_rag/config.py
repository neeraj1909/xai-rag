"""Application configuration via environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """XAI-RAG configuration. All values can be overridden via environment variables."""

    # Database
    database_url: str = "postgresql+asyncpg://xai_rag:xai_rag_dev@localhost:5432/xai_rag"
    database_url_sync: str = "postgresql://xai_rag:xai_rag_dev@localhost:5432/xai_rag"

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "xai_rag_documents"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Embedding model (small variant for CPU; swap to bge-large-en-v1.5 for production)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # Reranker model (base variant for CPU; swap to bge-reranker-v2-m3 for production)
    reranker_model: str = "BAAI/bge-reranker-base"

    # NLI model for faithfulness checking (base variant; swap to DeBERTa-v3-large for production)
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    # LLM (gpt-4o-mini is 17x cheaper; swap to gpt-4o for production)
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = Field(default="", env="XAI_RAG_OPENAI_API_KEY")
    anthropic_api_key: str = ""

    # Retrieval
    vector_search_k: int = 100
    bm25_search_k: int = 100
    reranker_top_k: int = 5
    rrf_k: int = 60

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    semantic_threshold: float = 0.75
    chunking_model: str = "all-MiniLM-L6-v2"  # fast model for semantic chunking only

    # Observability
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "xai-rag"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_prefix": "XAI_RAG_", "env_file": ".env"}


settings = Settings()
