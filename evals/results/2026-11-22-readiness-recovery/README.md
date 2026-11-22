# Dependency readiness failure and recovery

Execution date: **22 November 2026 (Asia/Kolkata)**. The tested API image was `xai-rag:grounding-repair-final`, image ID `sha256:b482d389cbdcd908124d7404b7a1bc6a78d87977141b1bade01ee798242f8013`, against Git checkout `bebe3bd`. Chroma and Elasticsearch were the pinned images in `docker-compose.yml`; the Compose E2E overlay disabled the disk watermark only for these disposable test volumes.

The [raw assertions and timings](results.json) and [run output](run.log) show:

- Initially `/ready` returned HTTP 200 and both dependencies were `ok`; `/live` returned 200.
- Stopping disposable Chroma produced `/ready` 503 with `chromadb: error`, while Elasticsearch remained `ok` and `/live` remained 200. Starting it restored `/ready` 200 after 3.27 seconds.
- Stopping disposable Elasticsearch produced `/ready` 503 with `elasticsearch: error`, while Chroma remained `ok` and `/live` remained 200. Starting it restored `/ready` 200 after 165.72 seconds under concurrent machine load and CPU limits.

This checks existing clients recovering after a service restart, not first-time API startup with a missing dependency, write durability during an outage, fault tolerance under all network failures, or provider availability. It uses a deterministic E2E provider stub; it does not invoke generation, NLI or the grounding judge. Both services were stopped sequentially, not together. The long restart time is one observation on this machine, not a guarantee.

## Reproduce on this machine

The Compose project has separate loopback ports and named volumes. It does not touch the `xai-rag-local` stack. The [override](compose-override.yml) contains a machine-specific read-only model cache path; adapt it on another system. The `--env-file /dev/null` prevents loading the user's `.env`; Compose uses the explicitly synthetic E2E credentials. Commands run from the repository root:

```sh
XAI_RAG_IMAGE=xai-rag:grounding-repair-final XAI_RAG_HOST_PORT=18143 \
  docker compose --env-file /dev/null -p xai-ready-20261122 \
  -f docker-compose.yml -f docker-compose.e2e.yml \
  -f evals/results/2026-11-22-readiness-recovery/compose-override.yml \
  up -d --no-build --pull never

docker update --cpus 2 --memory 2g --memory-swap 3g xai-ready-20261122-elasticsearch-1
docker update --cpus 1 --memory 1g --memory-swap 2g xai-ready-20261122-chromadb-1
docker update --cpus 2 --memory 3g --memory-swap 4g xai-ready-20261122-api-1
docker update --cpus 0.5 --memory 256m --memory-swap 512m xai-ready-20261122-llm-stub-1
python3 evals/results/2026-11-22-readiness-recovery/check.py

XAI_RAG_IMAGE=xai-rag:grounding-repair-final XAI_RAG_HOST_PORT=18143 \
  docker compose --env-file /dev/null -p xai-ready-20261122 \
  -f docker-compose.yml -f docker-compose.e2e.yml \
  -f evals/results/2026-11-22-readiness-recovery/compose-override.yml \
  down --volumes --remove-orphans
```

`check.py` controls only containers named for `xai-ready-20261122`; it sequentially stops and starts each store and checks `/live` at each matched readiness state. The recorded run reported `PASS`. Cleanup returned success; checking the Compose project label showed zero remaining containers, volumes or networks. Afterwards the persistent `xai-rag-local` API and Elasticsearch were healthy, Chroma was running, and its `/ready` returned 200 with both stores `ok`. No persistent volume was removed.
