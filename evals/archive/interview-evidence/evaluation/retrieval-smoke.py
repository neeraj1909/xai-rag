"""Isolated XAI-RAG synthetic retrieval probe; requires the project's .venv and two disposable stores."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import sys
import time

from xai_rag.config import settings
from xai_rag.ingestion.chunker import TextChunk
from xai_rag.ingestion.embedder import embed_texts
from xai_rag.ingestion.store import get_chroma_client, get_chroma_collection, get_es_client, store_chunks_chromadb, store_chunks_elasticsearch
from xai_rag.retrieval.hybrid import hybrid_search
from xai_rag.retrieval.reranker import Reranker

DATA = Path(__file__).with_name('retrieval-smoke-cases.json')

async def main() -> None:
    fixture = json.loads(DATA.read_text())
    docs, questions = fixture['documents'], fixture['queries']
    collection = get_chroma_collection(get_chroma_client())
    es = await get_es_client()
    embeddings = await embed_texts([d['text'] for d in docs])
    mapping = {}
    try:
        for d, embedding in zip(docs, embeddings, strict=True):
            text = d['text']
            chunk = TextChunk(content=text, index=0, start_char=0, end_char=len(text), strategy='fixed')
            ids = store_chunks_chromadb(collection, [chunk], [embedding], f"synthetic/{d['id']}.txt")
            await store_chunks_elasticsearch(es, [chunk], ids, f"synthetic/{d['id']}.txt", settings.elasticsearch_index)
            mapping[ids[0]] = d['id']
        if os.environ.get('XAI_EVAL_INGEST_ONLY') == '1':
            print(f'INGESTED {len(docs)} documents')
            return
        query_embeddings = await embed_texts([q['question'] for q in questions])
        rows=[]
        for q, embedding in zip(questions, query_embeddings, strict=True):
            start=time.monotonic()
            vector, lexical, fused = await hybrid_search(collection, es, q['question'], embedding, k=5, es_index=settings.elasticsearch_index)
            ranked = await Reranker().rerank(q['question'], fused[:5], top_k=5)
            results={name:[mapping[x.id] for x in hits] for name,hits in [('vector',vector),('bm25',lexical),('rrf',fused),('rerank',ranked)]}
            rows.append({'id':q['id'],'question':q['question'],'relevant':q['relevant'],'rankings':results,'elapsed_seconds':round(time.monotonic()-start,3)})
        metrics={}
        for stage in ['vector','bm25','rrf','rerank']:
            rr=[]; hit1=[]; hit3=[]
            for row in rows:
                ranks=[i for i,x in enumerate(row['rankings'][stage],1) if x in row['relevant']]
                best=min(ranks) if ranks else None
                rr.append(1/best if best else 0); hit1.append(bool(best and best<=1)); hit3.append(bool(best and best<=3))
            metrics[stage]={'n':len(rows),'hit_at_1':sum(hit1)/len(rows),'hit_at_3':sum(hit3)/len(rows),'mrr':sum(rr)/len(rows)}
        out={'source_commit':'f4c374d51d090beba61d34aba81fcc23b8ef700a','data':DATA.name,'embedding_model':settings.embedding_model,'embedding_revision':settings.embedding_model_revision,'reranker_model':settings.reranker_model,'reranker_revision':settings.reranker_model_revision,'k_per_branch':5,'reranker_candidates':5,'results':rows,'metrics':metrics}
        Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(metrics,indent=2))
    finally:
        await es.close()

if __name__ == '__main__': asyncio.run(main())
