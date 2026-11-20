"""Capture scoped synthetic stores after a refresh; caller must quiesce all writers."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from index_snapshot import snapshot

from xai_rag.authorization import tenant_store_name
from xai_rag.config import settings
from xai_rag.ingestion.store import get_chroma_client, get_es_client

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
fixture = json.loads(args.fixture.read_text())
sources = [d["id"] + ".txt" for d in fixture["documents"]]
tenant = "synthetic-eval"
chroma_name = tenant_store_name(settings.chroma_collection, tenant)
es_index = tenant_store_name(settings.elasticsearch_index, tenant)


async def main():
    chroma = get_chroma_client().get_collection(chroma_name)
    es = await get_es_client()
    try:
        await es.indices.refresh(index=es_index)
        data = await snapshot(chroma, es, tenant=tenant, sources=sources, es_index=es_index)
        if not data["record_parity"]:
            raise ValueError("Chroma and Elasticsearch record manifests differ")
        mappings = (await es.indices.get_mapping(index=es_index)).body
        settings_data = (await es.indices.get_settings(index=es_index)).body
        configs = {
            "mappings": mappings,
            "settings": settings_data,
            "chroma_configuration": chroma.configuration_json,
            "chroma_metadata": chroma.metadata,
        }
        encoded = json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()
        data["config_sha256"] = hashlib.sha256(encoded).hexdigest()
        data["config"] = configs
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    finally:
        await es.close()


asyncio.run(main())
