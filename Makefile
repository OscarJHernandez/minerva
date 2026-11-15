# Shortcuts for humans and agents. CI calls the underlying commands directly (see .github/workflows/ci.yml);
# `make check` runs the same set. Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help setup check check-py check-web test handbook matrix-benchmark stat-benchmark statmech-benchmark bench fmt dev bench-geometry bench-variational eval-check eval-reliability reference reference-extended test-extended verification-report verification-compare eval bench-telescoping

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install Python and JavaScript dependencies
	uv sync
	pnpm install

check: check-py check-web ## Everything CI runs for Python and web

eval-check: ## Fast deterministic benchmark checks: case ground truth (both suites), scoring, reports, replay
	uv run python scripts/eval_models.py --suite all --check-cases
	uv run pytest services/api/tests/test_evaluation.py scripts/tests/test_replay_trace.py

eval-reliability: ## Reliability suite without a model: ideal and adversarial trajectories, restrictions carried across steps
	uv run python scripts/eval_models.py --suite reliability --check-cases
	uv run pytest services/api/tests/test_evaluation_reliability.py services/api/tests/test_agent_restrictions.py

statmech-benchmark: ## Statistical mechanics corpus (answers from closed forms); fails on a wrong answer or a lost refusal
	uv run python -m minerva_engine.statmech.benchmark --check

reference: ## Independent reference problems of the pull-request tier; fails on a wrong answer or a missed refusal
	uv run python -m minerva_engine.reference_benchmark --check

reference-extended: ## The same, every tier (what the nightly workflow runs)
	uv run python -m minerva_engine.reference_benchmark --check --tier extended

test-extended: ## Extended tier: tests marked `extended` and property tests at 100 random examples (slow, prints the seed)
	MINERVA_TEST_TIER=extended MINERVA_HYPOTHESIS_PROFILE=extended \
		uv run pytest packages/engine packages/knowledge -n 3 --dist load --hypothesis-seed=$(or $(SEED),$(shell date +%s))

verification-report: ## Rewrite docs/verification-report.md and its JSON, or fail with CHECK=1 if they are stale
	uv run python scripts/verification_report.py $(if $(CHECK),--check,)

verification-compare: ## Before/after: this verifier against the saved baseline (verifier without probes, original reference report)
	uv run python scripts/verification_report.py --compare docs/benchmarks/verification-summary-baseline.json

eval: ## Real-model benchmark (needs MINERVA_EVAL_BASE_URL and MINERVA_EVAL_MODELS; LABEL=name)
	uv run python scripts/eval_models.py --repeat 3 --seed 7 --label "$(or $(LABEL),run)" \
		--json-out "docs/benchmarks/$(or $(LABEL),run).json" --out "docs/benchmarks/$(or $(LABEL),run).md"

check-py: ## Ruff, mypy, handbook verification, agent files, pytest
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy packages/engine/src packages/knowledge/src services/api/src
	uv run python -m minerva_knowledge verify
	uv run python scripts/verification_report.py --check
	uv run python scripts/check_agent_files.py
	uv run pytest

bench-geometry: ## Re-run the geometry benchmark and rewrite docs/benchmarks/geometry.json and the report
	uv run python -m minerva_engine.geometry.benchmark

bench-telescoping: ## creative telescoping against independent references (writes docs/creative-telescoping-benchmark.md)
	uv run python -m minerva_engine.creative_telescoping.benchmark

bench-variational: ## Re-run the variational calculus corpus and rewrite docs/benchmarks/variational.json and the report
	uv run python -m minerva_engine.variational.benchmark

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

bench: ## Transformation benchmark with latency and effort (local only; CI asserts correctness, never time)
	uv run python -m minerva_knowledge step-benchmark --adapter steps --repeat 5
matrix-benchmark: ## Matrix calculus benchmark (docs/matrix-benchmark.md)
	uv run python -m minerva_knowledge matrix-benchmark --repeat 5

stat-benchmark: ## Statistical derivation benchmark against the committed baseline (docs/statistics-benchmark.md)
	uv run python -m minerva_knowledge stat-benchmark --compare docs/benchmarks/statistics.json

fmt: ## Format Python and JavaScript
	uv run ruff format .
	uv run ruff check --fix .
	pnpm format

dev: ## Web app on :3000 (start the API on :8100 separately, see AGENTS.md)
	pnpm --filter @minerva/web dev
