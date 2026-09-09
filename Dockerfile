FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6 AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.9.22@sha256:2320e6c239737dc73cccce393a8bb89eba2383d17018ee91a59773df802c20e6 /uv /usr/local/bin/uv

# Keep dependency installation cacheable while still requiring an exact lock.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6 AS runtime

ARG VCS_REF="unknown"
LABEL org.opencontainers.image.title="XAI-RAG" \
      org.opencontainers.image.description="Explainable hybrid retrieval-augmented generation API" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}"

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

COPY --from=builder /app/.venv /app/.venv

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/sample_docs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app/sample_docs /home/appuser/.cache
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"]

CMD ["uvicorn", "xai_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
