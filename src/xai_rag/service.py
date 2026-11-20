"""Application service shared by HTTP, SSE, and command-line adapters."""

from __future__ import annotations

import asyncio
import html
import os
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xai_rag.async_utils import wait_for_thread_work
from xai_rag.authorization import granted_ingestion_paths
from xai_rag.config import settings
from xai_rag.evaluation.ragas_eval import evaluate_response
from xai_rag.explainability.faithfulness import FaithfulnessChecker
from xai_rag.explainability.retrieval_explainer import RetrievalExplainer
from xai_rag.generation.generator import RAGGenerator
from xai_rag.generation.grounding_judge import GroundingDecision, GroundingJudge
from xai_rag.generation.release import (
    REFUSAL,
    UNAVAILABLE,
    claims_are_valid,
    render_supported_claims,
)
from xai_rag.ingestion.chunker import chunk_text
from xai_rag.ingestion.embedder import embed_query, embed_texts
from xai_rag.ingestion.journal import (
    CoordinatedIndexWriter,
    IngestionJournal,
    RecoveryRequiredError,
    index_namespace,
)
from xai_rag.ingestion.parser import SUPPORTED_SUFFIXES, parse_file
from xai_rag.ingestion.secure_parser import ByteBudget, parse_staged_document, stage_document
from xai_rag.ingestion.store import (
    StorageWriteError,
    build_chunk_id,
    check_index_parity,
    get_index_fingerprint,
    intended_record_fingerprint,
    source_matches_expected,
    store_chunks_chromadb,
    store_chunks_elasticsearch,
)
from xai_rag.models import (
    IngestionDocumentResult,
    IngestionResponse,
    QueryRequest,
    Verdict,
    XAIRAGResponse,
)
from xai_rag.observability.tracing import hash_text, pipeline_stage
from xai_rag.retrieval.bm25_search import bm25_search
from xai_rag.retrieval.hybrid import rrf_fusion
from xai_rag.retrieval.reranker import Reranker
from xai_rag.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    import chromadb
    from elasticsearch import AsyncElasticsearch

EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]

# Whole-index parity must not overlap another local ingestion batch. Locks are
# scoped to one event loop; multi-worker operation still needs durable fencing.
_ingestion_locks: weakref.WeakValueDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakValueDictionary()
)
_ingestion_locks_guard = threading.Lock()


def _ingestion_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    with _ingestion_locks_guard:
        lock = _ingestion_locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _ingestion_locks[loop] = lock
        return lock


async def _finish_ingestion_write(work: Awaitable[None]) -> None:
    """Keep the batch lock until a cancelled ES request has finished locally.

    A network failure can still leave the remote write outcome unknown; this is
    not a cross-store transaction or a distributed lease.
    """
    task = asyncio.ensure_future(work)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


class RequestEvaluationDisabledError(ValueError):
    """Raised when expensive request-path evaluation has not been enabled."""


class IngestionLimitError(ValueError):
    """Raised before parsing when an ingestion request exceeds a configured bound."""


class IngestionBusyError(RuntimeError):
    """Raised without queueing when the process-local ingestion lock is held."""


def discover_document_paths(
    source: Path,
    document_root: Path,
    *,
    max_files: int,
    max_document_bytes: int,
    max_total_bytes: int,
) -> list[Path]:
    """Resolve supported files under a root and enforce count/size bounds."""
    root = document_root.expanduser().resolve()
    resolved_source = Path(os.path.abspath(source.expanduser()))
    try:
        resolved_source.relative_to(root)
    except ValueError as exc:
        raise IngestionLimitError("ingestion source is outside the document root") from exc

    if resolved_source.is_file():
        if resolved_source.suffix.casefold() not in SUPPORTED_SUFFIXES:
            raise IngestionLimitError("ingestion source has an unsupported file type")
        candidates = [resolved_source]
    elif resolved_source.is_dir():
        candidates = []
        for path in resolved_source.rglob("*"):
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES:
                candidates.append(path)
                if len(candidates) > max_files:
                    raise IngestionLimitError(
                        f"ingestion request exceeds the {max_files}-file limit"
                    )
        candidates.sort()
    else:
        raise IngestionLimitError("ingestion source does not exist")

    if len(candidates) > max_files:
        raise IngestionLimitError(f"ingestion request exceeds the {max_files}-file limit")

    total_bytes = 0
    for path in candidates:
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise IngestionLimitError("resolved document is outside the document root") from exc
        size = path.stat().st_size
        if size > max_document_bytes:
            raise IngestionLimitError(
                f"document exceeds the {max_document_bytes}-byte per-file limit"
            )
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise IngestionLimitError(
                f"ingestion request exceeds the {max_total_bytes}-byte total limit"
            )
    return candidates


class RAGService:
    """Coordinate queries and ingestion with replaceable dependencies.

    A custom ``generator_factory`` must return an object exposing async
    ``generate_with_evidence(query, chunks)`` returning a GenerationOutcome.
    The evidence is needed to evaluate only text shown to the generator.
    """

    def __init__(
        self,
        *,
        collection: chromadb.Collection,
        es_client: AsyncElasticsearch,
        embed_query_fn=embed_query,
        vector_search_fn=vector_search,
        bm25_search_fn=bm25_search,
        fusion_fn=rrf_fusion,
        reranker_factory=Reranker,
        generator_factory=RAGGenerator,
        explainer_factory=RetrievalExplainer,
        faithfulness_factory=FaithfulnessChecker,
        grounding_judge_factory=GroundingJudge,
        evaluator_fn=evaluate_response,
        parse_file_fn=parse_file,
        chunk_text_fn=chunk_text,
        embed_texts_fn=embed_texts,
        store_chroma_fn=store_chunks_chromadb,
        store_elasticsearch_fn=store_chunks_elasticsearch,
        parity_fn=check_index_parity,
        allow_request_evaluation: bool | None = None,
        index_name: str | None = None,
        tenant: str | None = None,
        ingestion_journal_root: Path | None = None,
        expected_source_fn=source_matches_expected,
    ) -> None:
        self._collection = collection
        self._es_client = es_client
        self._index_name = index_name or settings.elasticsearch_index
        self._tenant = tenant
        self._expected_source = expected_source_fn
        self._index_writer = None
        if ingestion_journal_root is not None:
            namespace = index_namespace(
                tenant=tenant,
                collection=collection.name,
                chroma_host=settings.chroma_host,
                chroma_port=settings.chroma_port,
                elastic_url=settings.elasticsearch_url,
                elastic_index=self._index_name,
            )
            self._index_writer = CoordinatedIndexWriter(
                IngestionJournal(ingestion_journal_root / namespace), namespace
            )
        self._embed_query = embed_query_fn
        self._vector_search = vector_search_fn
        self._bm25_search = bm25_search_fn
        self._fusion = fusion_fn
        self._reranker_factory = reranker_factory
        self._generator_factory = generator_factory
        self._explainer_factory = explainer_factory
        self._faithfulness_factory = faithfulness_factory
        self._grounding_judge_factory = grounding_judge_factory
        self._evaluator = evaluator_fn
        self._parse_file = parse_file_fn
        self._chunk_text = chunk_text_fn
        self._embed_texts = embed_texts_fn
        self._store_chroma = store_chroma_fn
        self._store_elasticsearch = store_elasticsearch_fn
        self._check_parity = parity_fn
        self._allow_request_evaluation = (
            settings.allow_request_evaluation
            if allow_request_evaluation is None
            else allow_request_evaluation
        )

    async def _timed(
        self,
        name: str,
        operation: Callable[[], Awaitable[Any]],
        timings: dict[str, float],
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        start = time.perf_counter()
        with pipeline_stage(f"rag.{name}", attributes):
            result = await operation()
        timings[name] = (time.perf_counter() - start) * 1000
        return result

    async def _emit(
        self,
        event_sink: EventSink | None,
        event: str,
        data: dict[str, Any],
    ) -> None:
        if event_sink is not None:
            await event_sink(event, data)

    async def query(
        self,
        request: QueryRequest,
        *,
        event_sink: EventSink | None = None,
        access_scope=None,
    ) -> XAIRAGResponse:
        """Run the canonical query pipeline and optionally publish stage events."""
        if (settings.auth_policy_file or self._tenant is not None or access_scope is not None) and (
            access_scope is None or self._tenant is None or access_scope.tenant != self._tenant
        ):
            raise PermissionError("query scope is required")
        if request.include_ragas and not self._allow_request_evaluation:
            raise RequestEvaluationDisabledError(
                "request-scoped external evaluation is disabled; use the offline evaluator"
            )

        started = time.perf_counter()
        timings: dict[str, float] = {}
        with pipeline_stage(
            "rag.query",
            {
                "xai_rag.top_k": request.top_k,
                "xai_rag.faithfulness_enabled": request.include_faithfulness,
            },
        ):
            query_embedding = await self._timed(
                "embed",
                lambda: self._embed_query(request.query),
                timings,
                attributes={
                    "xai_rag.model": settings.embedding_model,
                    "xai_rag.model_revision": settings.embedding_model_revision,
                },
            )

            retrieval_started = time.perf_counter()
            with pipeline_stage("rag.retrieve"):
                vector_task = asyncio.create_task(
                    self._timed(
                        "vector",
                        lambda: wait_for_thread_work(
                            asyncio.to_thread(
                                self._vector_search,
                                self._collection,
                                query_embedding,
                                settings.vector_search_k,
                                **(
                                    {"allowed_sources": access_scope.sources}
                                    if access_scope
                                    else {}
                                ),
                            )
                        ),
                        timings,
                    )
                )
                bm25_task = asyncio.create_task(
                    self._timed(
                        "bm25",
                        lambda: self._bm25_search(
                            self._es_client,
                            request.query,
                            self._index_name,
                            settings.bm25_search_k,
                            **({"allowed_sources": access_scope.sources} if access_scope else {}),
                        ),
                        timings,
                    )
                )
                tasks = (vector_task, bm25_task)
                try:
                    vector_results, bm25_results = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    # A cancelled vector task drains its worker before completing.
                    # Repeated cancellation must not release admission early.
                    await wait_for_thread_work(asyncio.gather(*tasks, return_exceptions=True))
                    raise
                if access_scope:
                    permitted = set(access_scope.sources)
                    if any(
                        result.metadata.get("source_file") not in permitted
                        for result in (*vector_results, *bm25_results)
                    ):
                        raise PermissionError("retrieval returned an unauthorized source")
            timings["retrieve"] = (time.perf_counter() - retrieval_started) * 1000

            fusion_started = time.perf_counter()
            with pipeline_stage("rag.fusion"):
                fused = self._fusion([vector_results, bm25_results], settings.rrf_k)
            timings["fusion"] = (time.perf_counter() - fusion_started) * 1000

            candidate_k = max(request.top_k, settings.reranker_candidate_k)
            reranker = self._reranker_factory()
            ranked = await self._timed(
                "rerank",
                lambda: reranker.rerank(
                    request.query,
                    fused[:candidate_k],
                    top_k=request.top_k,
                ),
                timings,
                attributes={
                    "xai_rag.model": settings.reranker_model,
                    "xai_rag.model_revision": settings.reranker_model_revision,
                    "xai_rag.candidate_count": min(len(fused), candidate_k),
                },
            )
            await self._emit(
                event_sink,
                "retrieval",
                {
                    "vector_count": len(vector_results),
                    "bm25_count": len(bm25_results),
                    "fused_count": len(fused),
                    "ranked_ids": [result.id for result in ranked],
                },
            )

            explanations = []
            if request.include_explanations:
                explain_started = time.perf_counter()
                with pipeline_stage("rag.explain"):
                    explanations = self._explainer_factory().explain(
                        request.query,
                        vector_results,
                        bm25_results,
                        ranked,
                    )
                timings["explain"] = (time.perf_counter() - explain_started) * 1000

            generator = self._generator_factory()
            if not callable(getattr(generator, "generate_with_evidence", None)):
                raise TypeError(
                    "generator_factory must provide async generate_with_evidence(query, chunks)"
                )
            generation = await self._timed(
                "generate",
                lambda: generator.generate_with_evidence(request.query, ranked),
                timings,
                attributes={"xai_rag.model": settings.llm_model},
            )
            generated = generation.result
            # Never publish the model's free-form answer. The only displayable
            # text is assembled from claims checked against visible evidence.
            faithfulness_report = []
            checked = []
            grounding_status = "insufficient_evidence"
            if generated.claims and claims_are_valid(generated.claims, generation.evidence):
                nli_evidence = [
                    chunk.model_copy(update={"content": html.unescape(chunk.content)})
                    for chunk in generation.evidence
                ]
                try:
                    checked = await self._timed(
                        "faithfulness",
                        lambda: self._faithfulness_factory().check_claims(
                            generated.claims, nli_evidence
                        ),
                        timings,
                        attributes={
                            "xai_rag.model": settings.nli_model,
                            "xai_rag.model_revision": settings.nli_model_revision,
                            "xai_rag.claim_count": len(generated.claims),
                        },
                    )
                    answer = render_supported_claims(generated.claims, checked)
                    # Retry only an ordinary NLI rejection, never an invalid
                    # verifier result or a source mismatch. The new answer must
                    # pass the same checks against the identical evidence.
                    repair = getattr(generator, "repair_with_evidence", None)
                    valid_rejection = (
                        answer is None
                        and len(checked) == len(generated.claims)
                        and all(
                            verdict.claim == claim
                            and (
                                verdict.supporting_chunk_id is None
                                or verdict.supporting_chunk_id in claim.source_chunk_ids
                            )
                            and (
                                verdict.contradicting_chunk_id is None
                                or verdict.contradicting_chunk_id in claim.source_chunk_ids
                            )
                            and (
                                verdict.verdict != Verdict.SUPPORTED
                                or verdict.supporting_chunk_id in claim.source_chunk_ids
                            )
                            for claim, verdict in zip(generated.claims, checked, strict=True)
                        )
                        and any(v.verdict != Verdict.SUPPORTED for v in checked)
                    )
                    if valid_rejection and callable(repair):
                        previous = generated
                        # A failed request can have consumed unknown tokens.
                        generated = generated.model_copy(
                            update={"input_tokens": None, "output_tokens": None}
                        )
                        repaired = await asyncio.wait_for(
                            self._timed(
                                "generation_repair",
                                lambda: repair(request.query, ranked),
                                timings,
                            ),
                            timeout=settings.llm_timeout_seconds,
                        )
                        generated = repaired.result.model_copy(
                            update={
                                name: (
                                    getattr(previous, name) + getattr(repaired.result, name)
                                    if getattr(previous, name) is not None
                                    and getattr(repaired.result, name) is not None
                                    else None
                                )
                                for name in ("input_tokens", "output_tokens")
                            }
                        )
                        if (
                            repaired.evidence == generation.evidence
                            and generated.claims
                            and claims_are_valid(generated.claims, generation.evidence)
                        ):
                            checked = await self._timed(
                                "faithfulness_repair",
                                lambda: self._faithfulness_factory().check_claims(
                                    generated.claims, nli_evidence
                                ),
                                timings,
                            )
                            answer = render_supported_claims(generated.claims, checked)
                    if answer is not None:
                        accepted = await asyncio.wait_for(
                            self._timed(
                                "grounding_judge",
                                lambda: self._grounding_judge_factory().check_claims(
                                    generated.claims, nli_evidence, checked, question=request.query
                                ),
                                timings,
                                attributes={"xai_rag.model": settings.grounding_judge_model},
                            ),
                            timeout=settings.grounding_judge_timeout_seconds,
                        )
                        if not isinstance(accepted, GroundingDecision):
                            raise ValueError("invalid grounding judge decision")
                        if not accepted.supported:
                            answer = None
                        elif not accepted.all_parts_addressed:
                            # Withhold even supported claims if the separate
                            # judge finds a missing or uncertain question part.
                            answer = None
                except Exception:
                    # Do not log exception text, which may contain model input.
                    grounding_status = "verification_unavailable"
                else:
                    if answer is not None:
                        grounding_status = "verifier_passed"
                        # Only the chunk selected by the checker is known to have
                        # supported this claim. Do not publish untested citations.
                        supported_claims = [
                            claim.model_copy(
                                update={"source_chunk_ids": [verdict.supporting_chunk_id]}
                            )
                            for claim, verdict in zip(generated.claims, checked, strict=True)
                        ]
                        sources = list(dict.fromkeys(v.supporting_chunk_id for v in checked))
                        generated = generated.model_copy(
                            update={
                                "answer": answer,
                                "claims": supported_claims,
                                "sources": sources,
                            }
                        )
                        if request.include_faithfulness:
                            faithfulness_report = [
                                verdict.model_copy(update={"claim": claim})
                                for claim, verdict in zip(supported_claims, checked, strict=True)
                            ]

            if grounding_status != "verifier_passed":
                generated = generated.model_copy(
                    update={
                        "answer": UNAVAILABLE
                        if grounding_status == "verification_unavailable"
                        else REFUSAL,
                        "claims": [],
                        "sources": [],
                    }
                )
            await self._emit(
                event_sink,
                "generation",
                {
                    "answer": generated.answer,
                    "claims_count": len(generated.claims),
                    "sources": generated.sources,
                    "grounding_status": grounding_status,
                },
            )

            if faithfulness_report:
                await self._emit(
                    event_sink,
                    "faithfulness",
                    {
                        "verdicts": [
                            {
                                "claim_index": index,
                                "verdict": verdict.verdict.value,
                                "confidence": verdict.confidence,
                            }
                            for index, verdict in enumerate(faithfulness_report)
                        ]
                    },
                )

            external_scores: dict[str, Any] = {}
            if request.include_ragas:
                external_scores = await self._timed(
                    "external_evaluation",
                    lambda: self._evaluator(
                        request.query,
                        generated.answer,
                        [result.content for result in generation.evidence],
                    ),
                    timings,
                )

            ragas_faithfulness = external_scores.get("scores", {}).get("faithfulness")
            overall_confidence = None
            overall_confidence_source = None
            if grounding_status == "verifier_passed":
                if (
                    external_scores.get("status") == "computed"
                    and isinstance(ragas_faithfulness, (int, float))
                    and not isinstance(ragas_faithfulness, bool)
                    and 0.0 <= ragas_faithfulness <= 1.0
                ):
                    overall_confidence = float(ragas_faithfulness)
                    overall_confidence_source = "ragas_faithfulness"
                else:
                    overall_confidence = sum(v.nli_entailment for v in checked) / len(checked)
                    overall_confidence_source = "nli_entailment"
            elapsed_ms = (time.perf_counter() - started) * 1000
            response = XAIRAGResponse(
                query=request.query,
                answer=generated.answer,
                grounding_status=grounding_status,
                claims=generated.claims,
                sources=generated.sources,
                retrieval_explanations=explanations,
                faithfulness_report=faithfulness_report,
                overall_confidence=overall_confidence,
                overall_confidence_source=overall_confidence_source,
                ragas_scores=external_scores,
                latency_ms=elapsed_ms,
                stage_latency_ms=timings,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
            )

        await self._emit(
            event_sink,
            "done",
            {"status": "complete", "latency_ms": response.latency_ms},
        )
        return response

    def _ingestion_callbacks(self):
        """Build replayable operations using the journal's saved request."""
        prepared_source: list[str] = []

        async def prepare(snapshot, request):
            if request["fingerprint"] != get_index_fingerprint():
                raise RecoveryRequiredError(
                    "index fingerprint changed; original configuration required"
                )
            if self._parse_file is parse_file:
                text = await wait_for_thread_work(
                    asyncio.to_thread(
                        parse_staged_document,
                        snapshot,
                        settings.max_extracted_characters,
                        timeout_seconds=settings.parser_timeout_seconds,
                        memory_bytes=settings.parser_memory_bytes,
                        max_document_bytes=settings.max_document_bytes,
                    )
                )
            else:
                text = await wait_for_thread_work(asyncio.to_thread(self._parse_file, snapshot))
            if len(text) > settings.max_extracted_characters:
                raise IngestionLimitError("document exceeds the extracted-character limit")
            chunks = await wait_for_thread_work(
                asyncio.to_thread(
                    self._chunk_text,
                    text,
                    request["strategy"],
                    request["chunk_size"],
                    request["chunk_overlap"],
                    request["semantic_threshold"],
                )
            )
            source_file = request["source"]
            prepared_source.append(source_file)
            expected = {
                build_chunk_id(source_file, chunk): intended_record_fingerprint(chunk, source_file)
                for chunk in chunks
            }
            if len(expected) != len(chunks):
                raise StorageWriteError("duplicate chunk identities")
            embeddings = (
                await self._embed_texts([chunk.content for chunk in chunks]) if chunks else []
            )
            return (source_file, chunks, embeddings, list(expected)), expected

        async def write(payload):
            source_file, chunks, embeddings, expected_ids = payload
            chunk_ids = await wait_for_thread_work(
                asyncio.to_thread(
                    self._store_chroma,
                    self._collection,
                    chunks,
                    embeddings,
                    source_file,
                )
            )
            if chunk_ids != expected_ids:
                raise StorageWriteError("stored chunk identities differ from intent")
            await _finish_ingestion_write(
                self._store_elasticsearch(
                    self._es_client,
                    chunks,
                    chunk_ids,
                    source_file,
                    self._index_name,
                )
            )

        async def verify(expected):
            return await self._expected_source(
                self._collection,
                self._es_client,
                source_file=prepared_source[0],
                expected=expected,
                index=self._index_name,
            )

        return prepare, write, verify

    async def recover_ingestion(self, *, access_scope=None, acknowledge_quiescence=False):
        """Replay a pending write only after an operator has stopped old requests."""
        if self._index_writer is None:
            raise RecoveryRequiredError("persistent ingestion journal is not configured")
        if (settings.auth_policy_file or self._tenant is not None or access_scope is not None) and (
            access_scope is None or self._tenant is None or access_scope.tenant != self._tenant
        ):
            raise PermissionError("ingestion scope is required")
        prepare, write, verify = self._ingestion_callbacks()

        def authorize(request):
            source = request.get("source")
            if not isinstance(source, str) or not source:
                return False
            return not access_scope or (access_scope.can_ingest and source in access_scope.sources)

        async with _ingestion_lock():
            await self._index_writer.recover(
                authorize=authorize,
                acknowledge_quiescence=acknowledge_quiescence,
                prepare=prepare,
                write=write,
                verify=verify,
            )

    async def ingest(
        self,
        source: Path,
        *,
        document_root: Path,
        strategy: str = "semantic",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        access_scope=None,
    ) -> IngestionResponse:
        """Run bounded parsing, chunking, embedding, storage, and parity validation."""
        if (settings.auth_policy_file or self._tenant is not None or access_scope is not None) and (
            access_scope is None or self._tenant is None or access_scope.tenant != self._tenant
        ):
            raise PermissionError("ingestion scope is required")
        root = document_root.expanduser().resolve()
        scoped_paths = granted_ingestion_paths(root, source, access_scope) if access_scope else None
        lock = _ingestion_lock()
        # No await between checking and acquiring the lock on this event loop.
        if lock.locked():
            raise IngestionBusyError("another ingestion batch is already running")
        async with lock:
            started = time.perf_counter()
            if access_scope:
                paths = scoped_paths
                if len(paths) > settings.max_ingest_files:
                    raise IngestionLimitError("ingestion request exceeds the file limit")
            else:
                paths = await wait_for_thread_work(
                    asyncio.to_thread(
                        discover_document_paths,
                        source,
                        root,
                        max_files=settings.max_ingest_files,
                        max_document_bytes=settings.max_document_bytes,
                        max_total_bytes=settings.max_ingest_total_bytes,
                    )
                )
            outcomes: list[IngestionDocumentResult] = []
            total_chunks = 0
            budget = ByteBudget(settings.max_ingest_total_bytes)

            for path in paths:
                document_started = time.perf_counter()
                source_file = path.relative_to(root).as_posix()
                try:
                    with pipeline_stage(
                        "rag.ingest.document",
                        {"xai_rag.source_hash": hash_text(source_file)},
                    ):

                        def parse_snapshot(path: Path = path) -> str:
                            with stage_document(
                                root, path, settings.max_document_bytes, budget=budget
                            ) as (snapshot, _):
                                if self._parse_file is parse_file:
                                    return parse_staged_document(
                                        snapshot,
                                        settings.max_extracted_characters,
                                        timeout_seconds=settings.parser_timeout_seconds,
                                        memory_bytes=settings.parser_memory_bytes,
                                        max_document_bytes=settings.max_document_bytes,
                                    )
                                # Constructor-injected parsers are trusted test adapters.
                                return self._parse_file(snapshot)

                        if self._index_writer is not None:

                            def stage_bytes(path: Path = path):
                                with stage_document(
                                    root, path, settings.max_document_bytes, budget=budget
                                ) as (snapshot, _):
                                    return snapshot.read_bytes()

                            source_bytes = await wait_for_thread_work(
                                asyncio.to_thread(stage_bytes)
                            )
                            request = {
                                "source": source_file,
                                "strategy": strategy,
                                "chunk_size": settings.chunk_size
                                if chunk_size is None
                                else chunk_size,
                                "chunk_overlap": settings.chunk_overlap
                                if chunk_overlap is None
                                else chunk_overlap,
                                "semantic_threshold": settings.semantic_threshold,
                                "fingerprint": get_index_fingerprint(),
                            }
                            prepare, write, verify = self._ingestion_callbacks()
                            payload = await self._index_writer.write(
                                source_bytes,
                                path.suffix.lower(),
                                request,
                                prepare=prepare,
                                write=write,
                                verify=verify,
                            )
                            chunks = payload[1]
                        else:
                            text = await wait_for_thread_work(asyncio.to_thread(parse_snapshot))
                            if len(text) > settings.max_extracted_characters:
                                raise IngestionLimitError(
                                    "document exceeds the extracted-character limit"
                                )
                            chunks = await wait_for_thread_work(
                                asyncio.to_thread(
                                    self._chunk_text,
                                    text,
                                    strategy,
                                    settings.chunk_size if chunk_size is None else chunk_size,
                                    chunk_overlap
                                    if chunk_overlap is not None
                                    else settings.chunk_overlap,
                                    settings.semantic_threshold,
                                )
                            )
                            embeddings = (
                                await self._embed_texts([chunk.content for chunk in chunks])
                                if chunks
                                else []
                            )
                            chunk_ids = await wait_for_thread_work(
                                asyncio.to_thread(
                                    self._store_chroma,
                                    self._collection,
                                    chunks,
                                    embeddings,
                                    source_file,
                                )
                            )
                            await _finish_ingestion_write(
                                self._store_elasticsearch(
                                    self._es_client,
                                    chunks,
                                    chunk_ids,
                                    source_file,
                                    self._index_name,
                                )
                            )
                    status = "success" if chunks else "skipped"
                    total_chunks += len(chunks)
                    outcomes.append(
                        IngestionDocumentResult(
                            source_file=source_file,
                            status=status,
                            chunk_count=len(chunks),
                            latency_ms=(time.perf_counter() - document_started) * 1000,
                        )
                    )
                except Exception as exc:
                    outcomes.append(
                        IngestionDocumentResult(
                            source_file=source_file,
                            status="failed",
                            error_type=type(exc).__name__,
                            latency_ms=(time.perf_counter() - document_started) * 1000,
                        )
                    )

            parity_consistent: bool | None = None
            try:
                if access_scope:
                    reports = [
                        await self._check_parity(
                            self._collection,
                            self._es_client,
                            index=self._index_name,
                            source_file=outcome.source_file,
                        )
                        for outcome in outcomes
                    ]
                    parity_consistent = all(report.consistent for report in reports)
                else:
                    parity = await self._check_parity(self._collection, self._es_client)
                    parity_consistent = parity.consistent
                if not parity_consistent:
                    raise StorageWriteError("retrieval indexes diverged after ingestion")
            except Exception:
                parity_consistent = False

            succeeded = sum(outcome.status == "success" for outcome in outcomes)
            skipped = sum(outcome.status == "skipped" for outcome in outcomes)
            failed = sum(outcome.status == "failed" for outcome in outcomes)
            if failed or parity_consistent is False:
                status = "failed" if succeeded == 0 else "partial"
            else:
                status = "success"
            return IngestionResponse(
                status=status,
                discovered_files=len(paths),
                ingested_files=succeeded,
                skipped_files=skipped,
                failed_files=failed,
                total_chunks=total_chunks,
                parity_consistent=parity_consistent,
                documents=outcomes,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
