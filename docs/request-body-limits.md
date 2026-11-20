# HTTP request body limit

The API rejects HTTP request bodies over 65,536 bytes with status 413 before
FastAPI parses JSON or invokes authentication. Set
`XAI_RAG_MAX_REQUEST_BODY_BYTES` to a value from 1 to 1,048,576 to adjust it.
The API accepts small JSON bodies; file ingestion references a server-side path,
so this limit is not the document size limit. No streaming file upload is supported.

The ASGI middleware counts received body bytes even without Content-Length and
stops reading when the cap is exceeded. It buffers valid bodies up to the cap,
then dispatches them; streaming response bodies such as SSE remain streamed.
This bounds application request buffering per request, not total memory under
concurrency, HTTP server/proxy buffers, headers, slow clients, or connection
count. Configure matching limits, deadlines, TLS and admission controls at a
trusted reverse proxy and HTTP server before exposing the API to the internet.
