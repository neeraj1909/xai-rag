# Browser check of the API documentation

XAI-RAG is an API, not a separate web application. Its Swagger interface is disabled by default; this test enabled it explicitly for an isolated loopback instance. The browser was `cdp` 0.1.0 in headed mode. The managed headless daemon reported degraded health even after its permitted repair command, so the test used a newly created, task-owned headed tab and closed it afterward.

## What was tested

The instance used a copied then-uncommitted source snapshot, a synthetic API key, an in-process fixture for Chroma and Elasticsearch readiness, and a fixture query service. No user documents or real provider credentials were used. This establishes browser/API behavior only; it does **not** test actual retrieval, generation, external search services or OpenAI. `app.py` in the corrected copy had SHA-256 `1b7476ec4bd8c62e586a8836a11de28828858a50df85645ddf4570245d5413b9`. The [stub](evaluation/cdp-api-stub.py) shows the fixture boundary.

1. Swagger rendered the `GET /live`, `GET /ready`, `POST /query`, `POST /query/stream` and `POST /ingest` operations. Using **Authorize**, **Try it out**, a synthetic key and **Execute**, a valid question produced HTTP 200 and the fixture answer with `[fixture-1]`.
2. An initial copied snapshot returned HTTP 422 with `{"detail":"Invalid request"}` for a whitespace-only query, while Swagger still documented the old FastAPI array-shaped validation error. This was a documentation/contract mismatch, not a failing validation check.
3. After the local schema fix, a fresh source copy was loaded in the browser. Swagger showed a 422 object with required string `detail` constrained to `Invalid request`; a browser request returned exactly `{"detail":"Invalid request"}` with HTTP 422. A valid request returned 200 and an unauthenticated request returned 401.

The test was performed on 2026-11-19 UTC. The tab and the two disposable Uvicorn processes were closed. Real-store and live-model behavior are covered separately in [HTTP end-to-end evidence](http-e2e-evidence.md).

## Search dependency failure

A later copied source snapshot (`app.py` SHA-256 `7cfdafd2195439238f8a5fc389ac32dcb407c41a6f7db024eb3230a35e47d271`) used a fixture service that raised a synthetic Elasticsearch `ConnectionError`. Through the browser, `POST /query` returned HTTP 503 with `{"detail":"Search dependency unavailable"}`, without the underlying error text. `POST /query/stream` returned HTTP 200 and an SSE `error` event with `error_type` set to `dependency_unavailable`. The 503 response was omitted from that snapshot's OpenAPI responses; the status itself worked. The owned browser tab and separate test process were closed afterward.

After the OpenAPI update, a fresh copied source snapshot (`app.py` SHA-256 `becce1ca1ded507b24e0d9261d733b6d627dafe8a4723506e32061574b004a85`) documented 503 for `/query` with three allowed sanitized `detail` values. Swagger rendered the 503 response, and the same fixture-induced browser request returned `{"detail":"Search dependency unavailable"}` with HTTP 503, matching the documented schema. This remains an isolated fixture-backed check, not a real Elasticsearch outage test.
