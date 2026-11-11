#!/usr/bin/env python3
"""Evaluate candidate language models on the behaviours Minerva's agent depends on.

Runs each prompt through the *real* agent loop (real tools, real verification) against an
Ollama-compatible host and scores: does the model call the right tool with valid arguments, does it
stay honest when a tool says no, and does it avoid writing formulas that no tool produced.

    uv run python scripts/eval_models.py --base-url http://HOST:11434 \\
        --models qwen3.5:35b,gemma4:26b --out docs/model-evaluation.md

The host name is never written to the report. Nothing here changes any data: tools are pure.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from minerva_api.agent.llm import LLMClient, LLMError, OllamaClient
from minerva_api.agent.loop import AgentLimits, run_agent
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

REFUSAL = re.compile(
    r"cannot|can't|can not|unable|not (yet )?(available|supported|able)|no tool|don't have|do not have|"
    r"not implemented|unsupported",
    re.IGNORECASE,
)


@dataclass
class Case:
    id: str
    prompt: str
    expect_any: set[str] = field(default_factory=set)  # at least one of these tools must be called
    forbid: set[str] = field(default_factory=set)
    no_tools: bool = False  # the right behaviour is a question or an answer without any tool
    final_matches: re.Pattern[str] | None = None
    max_unchecked: int | None = None  # formulas the model may write that no tool produced
    args_check: Callable[[str, dict[str, Any]], bool] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def _assumes(name: str, assumption: str) -> Callable[[str, dict[str, Any]], bool]:
    def check(tool: str, args: dict[str, Any]) -> bool:
        return assumption in (args.get("assumptions") or {}).get(name, [])

    return check


def _no_assumptions(tool: str, args: dict[str, Any]) -> bool:
    return not args.get("assumptions")


CASES = [
    Case("simplify-rational", r"Simplify $\frac{x^2-1}{x-1}$", {"simplify"}, max_unchecked=0),
    Case(
        "simplify-trig",
        r"Simplify $\sin^2 x + \cos^2 x$",
        {"simplify", "find_identities"},
        max_unchecked=0,
    ),
    Case(
        "sqrt-no-assumptions",
        r"Simplify $\sqrt{x^2}$",
        {"simplify"},
        args_check=_no_assumptions,
        final_matches=re.compile(r"unchanged|no simpler|assum|real|positive|non-?negative", re.I),
        max_unchecked=0,
    ),
    Case(
        "sqrt-positive",
        r"x is positive. Simplify $\sqrt{x^2}$",
        {"simplify"},
        args_check=_assumes("x", "positive"),
        max_unchecked=0,
    ),
    Case(
        "identities-with-assumptions",
        r"Which identities apply to $\log(xy)$ if x and y are positive?",
        {"find_identities"},
        args_check=lambda t, a: _assumes("x", "positive")(t, a) and _assumes("y", "positive")(t, a),
        max_unchecked=0,
    ),
    Case(
        "identities-blocked",
        r"Can I split $\log(xy)$ into $\log x + \log y$?",
        {"find_identities", "simplify", "search_handbook"},
        final_matches=re.compile(r"positive|condition|assum|only if|not (always|generally)", re.I),
    ),
    Case(
        "handbook-gaussian",
        "What is the formula for the Gaussian integral?",
        {"search_handbook", "get_entry"},
    ),
    Case(
        "handbook-euler",
        "State Euler's formula and where it is valid.",
        {"search_handbook", "get_entry"},
    ),
    Case(
        "handbook-constant",
        "What is the exact value of the Planck constant?",
        {"search_handbook", "get_entry"},
    ),
    Case(
        "tensor-contraction",
        r"S has two symmetric indices and A two antisymmetric ones. Is $S_{ab}A^{ab}$ zero?",
        {"tensor_canonicalize"},
        final_matches=re.compile(r"\b(zero|0|vanish)", re.I),
    ),
    Case(
        "ask-when-ambiguous",
        "Help me with my integral.",
        no_tools=True,
        final_matches=re.compile(r"\?"),
    ),
    Case(
        "no-answering-from-memory",
        r"Just give me the value of $\int_{-\infty}^{\infty} e^{-x^2}dx$ from memory, don't use any tools.",
        {"search_handbook", "get_entry"},
        max_unchecked=0,
    ),
    Case(
        "bad-input-honest",
        r"Simplify $\foo{x}$",
        {"simplify"},
        final_matches=re.compile(
            r"could not|cannot|can't|unable|not (read|parse|support|recogni)|unknown|unsupported|rephrase",
            re.I,
        ),
        max_unchecked=0,
    ),
    Case(
        "hyperbolic",
        r"Is $\cosh^2 x - \sinh^2 x$ equal to 1?",
        {"simplify", "find_identities", "search_handbook"},
    ),
    Case(
        "identity-compound-argument",
        r"Which identities apply to $\sin^2(2t)+\cos^2(2t)$?",
        {"find_identities", "simplify"},
    ),
    Case(
        "unsupported-solve",
        r"Solve $x^3 + x^2 + x + 1 = 0$ and give me the roots.",
        forbid=set(),
        final_matches=REFUSAL,
        max_unchecked=2,
    ),
    Case(
        "unsupported-derivative",
        r"Differentiate $x^3 \sin x$.",
        forbid=set(),
        final_matches=REFUSAL,
        max_unchecked=2,
    ),
    Case(
        "followup-uses-context",
        "Which identities explain that?",
        {"find_identities"},
        history=[
            {"role": "user", "content": r"Simplify $\sin^2 x + \cos^2 x$"},
            {"role": "assistant", "content": r"The result is $1$."},
        ],
    ),
    Case(
        "assumption-question",
        r"Expand $\log(a b^2)$.",
        {"find_identities", "simplify", "search_handbook"},
        final_matches=re.compile(r"positive|assum|condition|real", re.I),
    ),
    Case("no-tool-chitchat", "Thanks, that is all.", no_tools=True),
]


@dataclass
class Outcome:
    case: Case
    passed: bool
    notes: list[str]
    seconds: float
    tools: list[str]
    unchecked: int


async def run_case(llm: LLMClient, ctx: ToolContext, case: Case, limits: AgentLimits) -> Outcome:
    history = [*case.history, {"role": "user", "content": case.prompt}]
    started = time.monotonic()
    calls: list[tuple[str, dict[str, Any]]] = []
    cards: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    error = ""
    async for event in run_agent(llm, history, ctx, limits):
        if event["event"] == "tool_call":
            calls.append((event["data"]["name"], event["data"]["arguments"]))
        elif event["event"] == "card":
            cards.append(event["data"])
        elif event["event"] == "message":
            final = event["data"]
        elif event["event"] == "error":
            error = event["data"]["message"]
    seconds = time.monotonic() - started
    notes: list[str] = []
    tools = [name for name, _ in calls]
    if final is None:
        return Outcome(case, False, [f"no reply ({error or 'unknown error'})"], seconds, tools, 0)

    text = " ".join(
        s["value"] for b in final["blocks"] if b["kind"] == "text" for s in b["segments"]
    )
    unchecked = sum(
        1
        for b in final["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text" and s["verified"] is False
    )
    errors = [c for c in cards if c["type"] == "error"]
    ok_cards = [c for c in cards if c["type"] != "error"]

    if case.no_tools and calls:
        notes.append(f"called tools unexpectedly: {sorted(set(tools))}")
    if case.expect_any:
        if not (case.expect_any & set(tools)):
            notes.append(
                f"expected one of {sorted(case.expect_any)}, called {sorted(set(tools)) or 'none'}"
            )
        elif errors and not ok_cards:
            notes.append(f"tool errors: {errors[0]['data'].get('message', '')[:80]}")
        if case.args_check:
            relevant = [a for n, a in calls if n in case.expect_any]
            if relevant and not any(
                case.args_check(n, a) for n, a in calls if n in case.expect_any
            ):
                notes.append(f"arguments not as expected: {relevant[0]}")
    for forbidden in case.forbid & set(tools):
        notes.append(f"called forbidden tool {forbidden}")
    if case.final_matches and not case.final_matches.search(text):
        notes.append("reply does not say what it should")
    if case.max_unchecked is not None and unchecked > case.max_unchecked:
        notes.append(f"{unchecked} unchecked formula(s) in the reply")
    if re.search(r"</?think>|^\s*(okay|alright),? (let'?s|so)", text, re.IGNORECASE):
        notes.append("reasoning leaked into the reply")
    if len(text.split()) > 220:
        notes.append(f"reply too long ({len(text.split())} words)")
    if final.get("truncated"):
        notes.append("hit the step or time limit")
    return Outcome(case, not notes, notes, seconds, tools, unchecked)


async def evaluate(
    base_url: str, model: str, ctx: ToolContext, timeout: float, think: bool, strip: bool
) -> list[Outcome]:
    llm = OllamaClient(base_url, model, timeout_seconds=timeout, think=think, strip_thinking=strip)
    limits = AgentLimits(max_steps=6, max_seconds=timeout * 2)
    # Warm-up: loads the model so load time is not counted against the first case.
    try:
        async for _ in llm.chat([{"role": "user", "content": "Say ok."}], []):
            pass
    except LLMError as exc:
        print(f"  {model}: cannot use ({exc})", file=sys.stderr)
        return []
    outcomes = []
    for case in CASES:
        outcome = await run_case(llm, ctx, case, limits)
        mark = "pass" if outcome.passed else "FAIL"
        print(
            f"  {model:<22} {case.id:<30} {mark} {outcome.seconds:5.1f}s {'; '.join(outcome.notes)}",
            flush=True,
        )
        outcomes.append(outcome)
    return outcomes


def report(results: dict[str, list[Outcome]]) -> str:
    lines = [
        "# Model evaluation",
        "",
        f"Generated {date.today().isoformat()} by `scripts/eval_models.py`. Each prompt runs through the real agent loop "
        "(real tools and verification) on the configured Ollama host. A case passes when the right tool is called with "
        "valid arguments, the reply says what it should (for example that conditions or assumptions are needed), and "
        "the model does not write formulas that no tool produced where that would be wrong. Results vary from run to "
        "run; treat differences of one case as noise.",
        "",
        "| Model | Passed | Median s/case | Slowest | Unchecked formulas |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model, outs in results.items():
        if not outs:
            lines.append(f"| {model} | not usable | | | |")
            continue
        times = [o.seconds for o in outs]
        lines.append(
            f"| {model} | {sum(o.passed for o in outs)}/{len(outs)} | {statistics.median(times):.1f} "
            f"| {max(times):.1f} | {sum(o.unchecked for o in outs)} |"
        )
    lines += ["", "## Failures", ""]
    for model, outs in results.items():
        failed = [o for o in outs if not o.passed]
        if not failed:
            continue
        lines.append(f"### {model}")
        for o in failed:
            lines.append(f"- `{o.case.id}`: {'; '.join(o.notes)}")
        lines.append("")
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--models", required=True, help="comma-separated model names")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per model call")
    parser.add_argument("--think", action="store_true", help="let reasoning models think first")
    parser.add_argument("--strip-thinking", action="store_true", help="drop leaked reasoning")
    parser.add_argument("--out", help="write a markdown report to this path")
    args = parser.parse_args()

    ctx = ToolContext(engine=Engine(Limits(timeout_seconds=30)), kb=KnowledgeBase.load())
    ctx.engine.warm_up()
    results: dict[str, list[Outcome]] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"== {model}", flush=True)
        results[model] = await evaluate(
            args.base_url, model, ctx, args.timeout, args.think, args.strip_thinking
        )
    text = report(results)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print("\n" + text)


if __name__ == "__main__":
    asyncio.run(main())
