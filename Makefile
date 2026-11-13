# Shortcuts for humans and agents. CI calls the underlying commands directly (see .github/workflows/ci.yml);
# `make check` runs the same set. Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help setup check check-py check-web test handbook fmt dev

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install Python and JavaScript dependencies
	uv sync
	pnpm install

check: check-py check-web ## Everything CI runs for Python and web

eval-check: ## Fast deterministic benchmark checks: case ground truth, scoring, reports, replay
	uv run python scripts/eval_models.py --check-cases
	uv run pytest services/api/tests/test_evaluation.py scripts/tests/test_replay_trace.py

eval: ## Real-model benchmark (needs MINERVA_EVAL_BASE_URL and MINERVA_EVAL_MODELS; LABEL=name)
	uv run python scripts/eval_models.py --repeat 3 --seed 7 --label "$(or $(LABEL),run)" \
		--json-out "docs/benchmarks/$(or $(LABEL),run).json" --out "docs/benchmarks/$(or $(LABEL),run).md"

check-py: ## Ruff, mypy, handbook verification, agent files, pytest
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy packages/engine/src packages/knowledge/src services/api/src
	uv run python -m minerva_knowledge verify
	uv run python scripts/check_agent_files.py
	uv run pytest

check-web: ## Prettier, eslint, tsc, vitest, production build
	pnpm format:check
	pnpm lint
	pnpm typecheck
	pnpm test
	MINERVA_API_URL=http://localhost:8100 pnpm build

test: ## Python and JavaScript tests only
	uv run pytest
	pnpm test

handbook: ## Verify every handbook entry at its declared level
	uv run python -m minerva_knowledge verify

fmt: ## Format Python and JavaScript
	uv run ruff format .
	uv run ruff check --fix .
	pnpm format

dev: ## Web app on :3000 (start the API on :8100 separately, see AGENTS.md)
	pnpm --filter @minerva/web dev
