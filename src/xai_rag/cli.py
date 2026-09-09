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
    from xai_rag.ingestion.store import (
        get_chroma_client,
        get_chroma_collection,
        get_es_client,
    )
    from xai_rag.service import RAGService

    document_root = path.parent if path.is_file() else path
    collection = get_chroma_collection(get_chroma_client())
    es = await get_es_client()
    try:
        result = await RAGService(collection=collection, es_client=es).ingest(
            path,
            document_root=document_root,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    finally:
        await es.close()

    for document in result.documents:
        console.print(
            f"  {document.source_file}: {document.status} "
            f"({document.chunk_count} chunks, {document.latency_ms:.0f}ms)"
        )
    console.print(
        f"\nIngestion {result.status}: {result.ingested_files} ingested, "
        f"{result.skipped_files} skipped, {result.failed_files} failed; "
        f"{result.total_chunks} chunks; parity={result.parity_consistent}"
    )
    if result.status != "success":
        raise click.ClickException(f"ingestion completed with status {result.status}")


@main.command("validate-indexes")
@click.option("--source", default=None, help="Restrict parity validation to one source identity")
@click.option("--json-output", is_flag=True, help="Emit the machine-readable report")
def validate_indexes(source: str | None, json_output: bool) -> None:
    """Compare chunk identities and fingerprints in ChromaDB and Elasticsearch."""
    report = asyncio.run(_validate_indexes(source))
    if json_output:
        import json

        console.print_json(json.dumps(report.to_dict()))
    else:
        status = "consistent" if report.consistent else "DIVERGED"
        issue_count = (
            len(report.missing_in_chroma)
            + len(report.missing_in_elasticsearch)
            + len(report.content_mismatches)
        )
        console.print(
            f"Index parity: {status}; Chroma={report.chroma_count}, "
            f"Elasticsearch={report.elasticsearch_count}, "
            f"missing/mismatched={issue_count}"
        )
    if not report.consistent:
        raise click.ClickException("retrieval indexes are not consistent")


async def _validate_indexes(source: str | None):
    from xai_rag.ingestion.store import (
        check_index_parity,
        get_chroma_client,
        get_chroma_collection,
        get_es_client,
    )

    collection = get_chroma_collection(get_chroma_client())
    es = await get_es_client()
    try:
        return await check_index_parity(collection, es, source_file=source)
    finally:
        await es.close()


@main.command("evaluate")
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--gates",
    "gates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="TOML file containing blocking and advisory metric thresholds",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the full JSON report to this path",
)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Compare candidate metrics with another JSON/JSONL dataset",
)
@click.option(
    "--comparison-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write candidate-minus-baseline metric deltas as JSON",
)
@click.option("--k", "k_values", type=click.IntRange(min=1), multiple=True)
def evaluate_command(
    dataset_path: Path,
    gates_path: Path | None,
    output_path: Path | None,
    baseline_path: Path | None,
    comparison_output: Path | None,
    k_values: tuple[int, ...],
) -> None:
    """Evaluate a captured RAG dataset without making model or provider calls."""
    from xai_rag.evaluation.models import EvaluationDataset
    from xai_rag.evaluation.runner import EvaluationGates, compare_reports, evaluate_dataset

    def load_dataset(path: Path) -> EvaluationDataset:
        content = path.read_text(encoding="utf-8")
        return (
            EvaluationDataset.from_jsonl(content)
            if path.suffix.casefold() == ".jsonl"
            else EvaluationDataset.model_validate_json(content)
        )

    dataset = load_dataset(dataset_path)
    gates = EvaluationGates.from_toml(gates_path) if gates_path else None
    report = evaluate_dataset(
        dataset,
        k_values=k_values or (1, 3, 5, 10),
        gates=gates,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    comparison = None
    if baseline_path:
        baseline_report = evaluate_dataset(
            load_dataset(baseline_path),
            k_values=k_values or (1, 3, 5, 10),
        )
        comparison = compare_reports(baseline_report, report)
        if comparison_output:
            comparison_output.parent.mkdir(parents=True, exist_ok=True)
            comparison_output.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    elif comparison_output:
        raise click.UsageError("--comparison-output requires --baseline")

    table = Table(title=f"Evaluation: {report.dataset_name}@{report.dataset_revision}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("N", justify="right")
    for metric_name, metric in sorted(report.metrics.items()):
        value = "not computed" if metric.value is None else f"{metric.value:.4f}"
        table.add_row(metric_name, value, str(metric.sample_size))
    console.print(table)

    if comparison is not None:
        comparison_table = Table(title="Candidate minus baseline")
        comparison_table.add_column("Metric")
        comparison_table.add_column("Delta", justify="right")
        for metric_name, metric in comparison.metrics.items():
            delta = "not computed" if metric.delta is None else f"{metric.delta:+.4f}"
            comparison_table.add_row(metric_name, delta)
        console.print(comparison_table)

    if report.gates:
        gate_table = Table(title="Evaluation gates")
        gate_table.add_column("Class")
        gate_table.add_column("Metric")
        gate_table.add_column("Status")
        for gate in report.gates:
            gate_table.add_row(
                "blocking" if gate.blocking else "advisory",
                gate.metric,
                gate.status.value,
            )
        console.print(gate_table)

    if not report.passed:
        raise click.ClickException("blocking evaluation gates did not pass")


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return")
@click.option("--explain/--no-explain", default=True, help="Include retrieval explanations")
@click.option("--faithfulness/--no-faithfulness", default=False, help="Run NLI faithfulness check")
def query(query: str, top_k: int, explain: bool, faithfulness: bool):
    """Run a query against the RAG pipeline."""
    asyncio.run(_query(query, top_k, explain, faithfulness))


async def _query(query_text: str, top_k: int, explain: bool, faithfulness: bool):
    from xai_rag.ingestion.store import get_chroma_client, get_chroma_collection, get_es_client
    from xai_rag.models import QueryRequest
    from xai_rag.service import RAGService

    console.print(f"\n[bold]Query:[/bold] {query_text}\n")
    collection = get_chroma_collection(get_chroma_client())
    es = await get_es_client()
    try:
        response = await RAGService(collection=collection, es_client=es).query(
            QueryRequest(
                query=query_text,
                top_k=top_k,
                include_explanations=explain,
                include_faithfulness=faithfulness,
            )
        )

        if response.retrieval_explanations:

            def score(value: float | None) -> str:
                return "—" if value is None else f"{value:.3f}"

            table = Table(title="Retrieval Explanations")
            table.add_column("Rank", width=4)
            table.add_column("Content", max_width=50)
            table.add_column("Vector", width=8)
            table.add_column("BM25", width=8)
            table.add_column("Reranker", width=8)
            table.add_column("Reason", max_width=40)
            for explanation in response.retrieval_explanations:
                table.add_row(
                    str(explanation.reranker_rank or explanation.rrf_rank or "—"),
                    explanation.content_preview[:50],
                    score(explanation.vector_score),
                    score(explanation.bm25_score),
                    score(explanation.reranker_score),
                    explanation.selection_reason,
                )
            console.print(table)

        console.print(f"\n[bold green]Answer:[/bold green]\n{response.answer}\n")
        if response.faithfulness_report:
            table = Table(title="Faithfulness Report")
            table.add_column("Claim", max_width=60)
            table.add_column("Verdict", width=14)
            table.add_column("Confidence", width=10)
            for verdict in response.faithfulness_report:
                color = (
                    "green"
                    if verdict.verdict == "supported"
                    else "red"
                    if verdict.verdict == "not_supported"
                    else "yellow"
                )
                table.add_row(
                    verdict.claim.text[:60],
                    f"[{color}]{verdict.verdict}[/{color}]",
                    f"{verdict.confidence:.0%}",
                )
            console.print(table)

        console.print(f"\n[dim]Latency: {response.latency_ms:.0f}ms[/dim]")
    finally:
        await es.close()


if __name__ == "__main__":
    main()
