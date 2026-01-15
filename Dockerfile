FROM python:3.12-slim AS base

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --no-dev --no-install-project

# Copy source
COPY src/ src/
COPY scripts/ scripts/

# Install project
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "xai_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
