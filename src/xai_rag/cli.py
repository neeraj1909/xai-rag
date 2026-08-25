"""CLI for XAI-RAG — ingest documents and run queries."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@click.group()
def main():
    """XAI-RAG: Explainable Retrieval-Augmented Generation."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--strategy", type=click.Choice(["fixed", "semantic", "parent_doc"]), default="semantic"
)
@click.option("--chunk-size", default=512, help="Target chunk size in characters")
@click.option("--chunk-overlap", default=50, help="Overlap between chunks (fixed strategy)")
def ingest(path: str, strategy: str, chunk_size: int, chunk_overlap: int):
    """Ingest documents from a file or directory into ChromaDB + Elasticsearch."""
    asyncio.run(_ingest(Path(path), strategy, chunk_size, chunk_overlap))


async def _ingest(path: Path, strategy: str, chunk_size: int, chunk_overlap: int):
    from xai_rag.ingestion.chunker import chunk_text
    from xai_rag.ingestion.embedder import embed_texts
    from xai_rag.ingestion.parser import parse_directory, parse_file
    from xai_rag.ingestion.store import (
        ensure_es_index,
        get_chroma_client,
        get_chroma_collection,
        get_es_client,
        store_chunks_chromadb,
        store_chunks_elasticsearch,
    )

    # Parse documents
    docs = [(path, parse_file(path))] if path.is_file() else parse_directory(path)

    if not docs:
        console.print("[red]No documents found.[/red]")
        return

    console.print(f"[green]Parsed {len(docs)} documents[/green]")

    # Connect to stores
    chroma_client = get_chroma_client()
    collection = get_chroma_collection(chroma_client)
    es = await get_es_client()
    await ensure_es_index(es)

    total_chunks = 0
    for file_path, text in docs:
        # Chunk
        chunks = chunk_text(
            text, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        console.print(f"  {file_path.name}: {len(chunks)} chunks ({strategy})")

        # Embed
        texts = [c.content for c in chunks]
        embeddings = await embed_texts(texts)

        # Store in both ChromaDB and Elasticsearch
        chunk_ids = store_chunks_chromadb(collection, chunks, embeddings, str(file_path.name))
        await store_chunks_elasticsearch(es, chunks, chunk_ids, str(file_path.name))

        total_chunks += len(chunks)

    await es.close()
    console.print(
        f"\n[bold green]Done! Ingested {total_chunks} chunks from {len(docs)} files.[/bold green]"
    )


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return")
@click.option("--explain/--no-explain", default=True, help="Include retrieval explanations")
@click.option("--faithfulness/--no-faithfulness", default=False, help="Run NLI faithfulness check")
def query(query: str, top_k: int, explain: bool, faithfulness: bool):
    """Run a query against the RAG pipeline."""
    asyncio.run(_query(query, top_k, explain, faithfulness))


async def _query(query_text: str, top_k: int, explain: bool, faithfulness: bool):
    import time

    from xai_rag.config import settings
    from xai_rag.explainability.faithfulness import FaithfulnessChecker
    from xai_rag.explainability.retrieval_explainer import RetrievalExplainer
    from xai_rag.generation.generator import RAGGenerator
    from xai_rag.ingestion.embedder import embed_query
    from xai_rag.ingestion.store import get_chroma_client, get_chroma_collection, get_es_client
    from xai_rag.retrieval.bm25_search import bm25_search
    from xai_rag.retrieval.hybrid import rrf_fusion
    from xai_rag.retrieval.reranker import Reranker
    from xai_rag.retrieval.vector_search import vector_search

    start = time.perf_counter()

    console.print(f"\n[bold]Query:[/bold] {query_text}\n")

    # 1. Embed query
    query_embedding = await embed_query(query_text)

    # 2. Hybrid search
    chroma_client = get_chroma_client()
    collection = get_chroma_collection(chroma_client)
    es = await get_es_client()

    vec_results = vector_search(collection, query_embedding)
    bm25_results = await bm25_search(es, query_text, settings.elasticsearch_index)
    fused = rrf_fusion([vec_results, bm25_results])

    console.print(
        f"Retrieved: {len(vec_results)} vector + {len(bm25_results)} BM25 → {len(fused)} fused"
    )

    # 3. Re-rank
    reranker = Reranker()
    ranked = await reranker.rerank(query_text, fused[:100], top_k=top_k)
    console.print(f"Re-ranked to top {len(ranked)}")

    # 4. Retrieval explanations
    if explain:
        explainer = RetrievalExplainer()
        explanations = explainer.explain(query_text, vec_results, bm25_results, ranked)
        table = Table(title="Retrieval Explanations")
        table.add_column("Rank", width=4)
        table.add_column("Content", max_width=50)
        table.add_column("Vector", width=8)
        table.add_column("BM25", width=8)
        table.add_column("Reranker", width=8)
        table.add_column("Reason", max_width=40)
        for exp in explanations:
            table.add_row(
                str(exp.rrf_rank),
                exp.content_preview[:50],
                f"{exp.vector_score:.3f}",
                f"{exp.bm25_score:.3f}",
                f"{exp.reranker_score:.3f}",
                exp.selection_reason,
            )
        console.print(table)

    # 5. Generate answer
    generator = RAGGenerator()
    gen_result = await generator.generate(query_text, ranked)
    console.print(f"\n[bold green]Answer:[/bold green]\n{gen_result.answer}\n")

    # 6. Faithfulness check
    if faithfulness and gen_result.claims:
        checker = FaithfulnessChecker()
        verdicts = await checker.check_claims(gen_result.claims, ranked)
        table = Table(title="Faithfulness Report")
        table.add_column("Claim", max_width=60)
        table.add_column("Verdict", width=14)
        table.add_column("Confidence", width=10)
        for v in verdicts:
            color = (
                "green"
                if v.verdict == "supported"
                else "red"
                if v.verdict == "not_supported"
                else "yellow"
            )
            table.add_row(
                v.claim.text[:60], f"[{color}]{v.verdict}[/{color}]", f"{v.confidence:.0%}"
            )
        console.print(table)

    elapsed = (time.perf_counter() - start) * 1000
    console.print(f"\n[dim]Latency: {elapsed:.0f}ms[/dim]")

    await es.close()


if __name__ == "__main__":
    main()
