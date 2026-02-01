-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table with vector embeddings
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    chunk_index INT NOT NULL DEFAULT 0,
    parent_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    source_file TEXT,
    chunk_strategy TEXT NOT NULL DEFAULT 'fixed',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Index for metadata filtering
CREATE INDEX IF NOT EXISTS idx_documents_metadata
    ON documents USING gin (metadata);

-- Index for source file lookup
CREATE INDEX IF NOT EXISTS idx_documents_source_file
    ON documents (source_file);

-- Query log for evaluation
CREATE TABLE IF NOT EXISTS query_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    response JSONB,
    retrieval_scores JSONB,
    faithfulness_report JSONB,
    ragas_scores JSONB,
    latency_ms FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);
