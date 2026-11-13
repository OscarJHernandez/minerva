#!/usr/bin/env python3
"""Evaluate language models on Minerva's agent benchmark. See `minerva_api.agent.evaluation.cli`.

    uv run python scripts/eval_models.py --base-url http://HOST:11434 --models qwen3.5:35b,gemma4:26b \
        --repeat 3 --seed 7 --label baseline --json-out docs/benchmarks/baseline.json \
        --out docs/benchmarks/baseline.md
    uv run python scripts/eval_models.py --compare-only OLD.json NEW.json
    uv run python scripts/eval_models.py --check-cases

Each case runs through the real agent loop (real tools, verification and claim checking). The host name is
never written to a report. Nothing here changes any data: tools are pure.
"""

from minerva_api.agent.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
