"""Command line of the agent benchmark (`scripts/eval_models.py` is a thin wrapper around `main`).

    uv run python scripts/eval_models.py --base-url http://HOST:11434 --models a,b \
        --repeat 3 --seed 7 --label baseline --json-out baseline.json --out baseline.md
    uv run python scripts/eval_models.py --compare-only baseline.json new.json
    uv run python scripts/eval_models.py --check-cases        # the benchmark's own ground truth

The host name is read from the command line or MINERVA_EVAL_BASE_URL and is never written to a report.
Expensive: every case is a real model run. The fast deterministic checks live in the test suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import sympy
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from ..llm import LLMError, OllamaClient
from ..loop import AgentLimits, prompt_sha
from ..tools import ToolContext
from .cases import EvalCase, cases_digest, load_cases
from .reference import check_reference
from .report import build_document, compare, render_markdown
from .runner import describe_limits, filter_cases, run_suite
from .score import CaseResult


def _git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[5]
        sha = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _ollama_version(base_url: str) -> str | None:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=10.0)
        return str(response.json().get("version")) if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def _meta(args: argparse.Namespace, cases: list[EvalCase], limits: AgentLimits) -> dict[str, Any]:
    return {
        "label": args.label,
        "date": date.today().isoformat(),
        "git_sha": args.code_ref or _git_sha(),
        "prompt_sha": prompt_sha(),
        "cases_digest": cases_digest(cases),
        "case_count": len(cases),
        "categories": sorted({c.category for c in cases}),
        "repeats": args.repeat,
        "seed": args.seed,
        "temperature": args.temperature,
        "think": args.think,
        "strip_thinking": args.strip_thinking,
        "request_timeout_seconds": args.timeout,
        "limits": describe_limits(limits),
        "python": platform.python_version(),
        "sympy": sympy.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "ollama": _ollama_version(args.base_url),
    }


async def _evaluate_model(
    args: argparse.Namespace,
    model: str,
    ctx: ToolContext,
    cases: list[EvalCase],
    limits: AgentLimits,
) -> list[CaseResult] | None:
    def client(repeat: int) -> OllamaClient:
        seed = None if args.seed is None else args.seed + repeat
        return OllamaClient(
            args.base_url,
            model,
            timeout_seconds=args.timeout,
            think=args.think,
            strip_thinking=args.strip_thinking,
            temperature=args.temperature,
            seed=seed,
        )

    # Warm-up: loads the model so load time is not counted against the first case.
    try:
        async for _ in client(0).chat([{"role": "user", "content": "Say ok."}], []):
            pass
    except LLMError as exc:
        print(f"  {model}: cannot use ({exc})", file=sys.stderr)
        return None

    def progress(result: CaseResult) -> None:
        mark = "pass" if result.passed else "FAIL"
        note = f"[{result.failure_class}] " if result.failure_class else ""
        print(
            f"  {model:<22} {result.case_id:<32} r{result.repeat} {mark} {result.seconds:5.1f}s "
            f"{note}{'; '.join(result.notes)}",
            flush=True,
        )

    results: list[CaseResult] = []
    for repeat in range(args.repeat):
        results += await run_suite(
            client(repeat),
            ctx,
            cases,
            limits,
            repeats=1,
            first_repeat=repeat,
            parallel=args.parallel,
            on_result=progress,
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default=os.environ.get("MINERVA_EVAL_BASE_URL", ""))
    parser.add_argument("--models", default=os.environ.get("MINERVA_EVAL_MODELS", ""))
    parser.add_argument("--label", default="run", help="name of this run, recorded in the report")
    parser.add_argument(
        "--code-ref", default="", help="what code was measured, when it is not a git checkout"
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per model read")
    parser.add_argument(
        "--case-timeout", type=float, default=0.0, help="seconds per turn (default: 2 x --timeout)"
    )
    parser.add_argument("--think", action="store_true", help="let reasoning models think first")
    parser.add_argument("--strip-thinking", action="store_true", help="drop leaked reasoning")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--categories", help="comma-separated categories (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="runs of every case (default 1)")
    parser.add_argument("--seed", type=int, default=None, help="base seed; repeat i uses seed + i")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--parallel", type=int, default=1, help="cases run at once per model")
    parser.add_argument("--json-out", help="write the machine-readable report here")
    parser.add_argument("--out", help="write the markdown report here")
    parser.add_argument("--compare", metavar="BASE.json", help="compare this run with a saved one")
    parser.add_argument(
        "--compare-only", nargs=2, metavar=("BASE.json", "NEW.json"), help="compare two saved runs"
    )
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument(
        "--check-cases",
        action="store_true",
        help="run every case's reference call against the engine and exit",
    )
    return parser


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or doc.get("schema") != 1:
        raise SystemExit(f"{path}: not a benchmark report this version can read")
    return doc


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _context() -> ToolContext:
    ctx = ToolContext(engine=Engine(Limits(timeout_seconds=30)), kb=KnowledgeBase.load())
    ctx.engine.warm_up()
    return ctx


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.compare_only:
        text = compare(_load(args.compare_only[0]), _load(args.compare_only[1]))
        if args.out:
            _write(args.out, text)
        print(text)
        return 0

    try:
        cases = filter_cases(
            load_cases(),
            ids=[c.strip() for c in (args.cases or "").split(",") if c.strip()],
            categories=[c.strip() for c in (args.categories or "").split(",") if c.strip()],
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.list:
        for case in cases:
            print(f"{case.category:<14} {case.id:<34} expects {case.expected_tool}")
        return 0

    if args.check_cases:
        ctx = _context()
        bad = 0
        for case in cases:
            for problem in check_reference(case, ctx):
                bad += 1
                print(f"{case.id}: {problem}")
        refs = sum(1 for c in cases if c.reference)
        print(f"{len(cases)} cases, {refs} reference calls, {bad} problem(s)")
        return 1 if bad else 0

    if not args.base_url or not args.models:
        parser.error(
            "--base-url and --models are required (or MINERVA_EVAL_BASE_URL / MINERVA_EVAL_MODELS)"
        )
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    limits = AgentLimits(max_steps=6, max_seconds=args.case_timeout or args.timeout * 2)
    ctx = _context()
    meta = _meta(args, cases, limits)
    results: dict[str, list[CaseResult] | None] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"== {model}", flush=True)
        results[model] = asyncio.run(_evaluate_model(args, model, ctx, cases, limits))
    doc = build_document(meta, results)
    text = render_markdown(doc)
    if args.json_out:
        _write(args.json_out, json.dumps(doc, indent=2, sort_keys=True) + "\n")
    if args.out:
        _write(args.out, text)
    print("\n" + text)
    if args.compare:
        print(compare(_load(args.compare), doc))
    return 0
