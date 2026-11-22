FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS builder

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


FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS runtime

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

# The pinned base predates the patched Debian zlib package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade \
        'zlib1g=1:1.3.dfsg+really1.3.1-1+deb13u1' \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/sample_docs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app/sample_docs /home/appuser/.cache
USER appuser

EXPOSE 8000
# Readiness checks two dependencies sequentially (up to 2s each).
HEALTHCHECK --interval=30s --timeout=8s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=7)"]

# A longer idle window avoids the observed five-second client/server boundary.
# It retains idle sockets longer; proxy timeouts and other launchers are separate.
CMD ["uvicorn", "xai_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "15", "--no-access-log"]
