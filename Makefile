.PHONY: setup infra down ingest query test lint format clean

# Setup
setup:
	uv sync --all-extras
	cp -n .env.example .env || true

# Infrastructure
infra:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 5
	@docker compose ps

down:
	docker compose down

# Ingestion
ingest:
	uv run xai-rag ingest $(path) --strategy $(or $(strategy),semantic)

# Query
query:
	uv run xai-rag query "$(q)"

# Development
test:
	uv run pytest -v --tb=short

test-cov:
	uv run pytest --cov=xai_rag --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# API
serve:
	uv run uvicorn xai_rag.api.app:app --reload --host 0.0.0.0 --port 8000

# Clean
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
