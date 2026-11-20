"""Record the landing page's research sessions from live runs of the real chat loop.

Two conversations (`landing-kepler`, `landing-schwarzschild` in cases_exploration.yaml) are run five times each
against a model host, through the same loop, tools, limits and scoring as `scripts/eval_models.py`. The rule, fixed
before a recording (2026-11-20): the page shows the FIRST run of each conversation in which every turn passed, and says
how many of the runs passed (with a link to the record). Nothing else is chosen, and a conversation with no passing run
is not published at all:

  apps/site/src/data/landing-sessions.json   what the page shows: each turn's text with its marks, each card's
                                             rows, numbers and evidence, and the call that produced it
  docs/benchmarks/landing-live.json          the record of all runs: pass/fail per turn, the model as the host
                                             reports it, the limits, the digest of the prompts

Equation numbers are assigned here exactly as the chat server assigns them (`equations.number_stored`, the same
reading-order rule), because the evaluation loop, like the agent loop, never numbers. The model host address is
never written anywhere; `services/api/tests/test_landing_sessions.py` re-runs every recorded call against the
engine and checks the file holds no address.

  uv run python scripts/record_landing_sessions.py --base-url http://HOST:11434 [--model minerva] [--runs 5]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from minerva_api.agent.card_equations import listed_equations
from minerva_api.agent.equations import EquationTable, number_stored
from minerva_api.agent.evaluation import cli
from minerva_api.agent.evaluation.cases import EvalCase, load_suite
from minerva_api.agent.evaluation.runner import describe_limits, run_case
from minerva_api.agent.loop import AgentLimits

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "apps" / "site" / "src" / "data" / "landing-sessions.json"
RECORD = ROOT / "docs" / "benchmarks" / "landing-live.json"
CASES = ("landing-kepler", "landing-schwarzschild")
TITLES = {
    "landing-kepler": "The Kepler problem: from the Lagrangian to a hidden symmetry",
    "landing-schwarzschild": "Schwarzschild spacetime: from the metric to its curvature",
}
# Model fields a reader may see: what the weights are, never where they are served from.
MODEL_FIELDS = (
    "name",
    "base_model",
    "family",
    "parameter_size",
    "quantization",
    "digest",
    "hosted",
    "provider",
)


def _evidence(verification: dict[str, Any] | None) -> str:
    if not verification:
        return "Not verified"
    if verification.get("symbolic_zero") is True:
        return "Proved by a symbolic zero test, and numerically checked"
    if verification.get("numeric_passed") or verification.get("passed"):
        return "Numerically checked; no symbolic proof"
    return "Not verified"


def _constant_evidence(c: dict[str, Any]) -> str:
    v = c.get("verification") or {}
    if not c.get("verified"):
        return "Not verified"
    agreed, checked = v.get("numeric_points_agreed"), v.get("numeric_points_checked")
    points = f", {agreed} of {checked} sample points" if checked else ""
    proof = "symbolic zero" if v.get("symbolic_zero") else "no symbolic proof"
    return f"Time derivative is zero on the equations of motion ({proof}{points})"


def _sections(card: dict[str, Any]) -> list[dict[str, Any]]:
    """The parts of a card the page shows, with each listed equation's row key (its number is matched by it)."""
    kind, data = card.get("type"), card.get("data") or {}
    rows = dict(listed_equations(str(kind), data))
    out: list[dict[str, Any]] = []
    if kind == "geometry":
        for q in (data.get("result") or {}).get("quantities") or []:
            if q.get("status") != "ok":
                continue
            keys = [k for k in rows if k == q["kind"] or k.startswith(q["kind"] + ":")]
            out.append(
                {
                    "title": q.get("title", q["kind"]),
                    "evidence": _evidence(q.get("verification")),
                    "rows": [{"key": k, "latex": rows[k]} for k in keys],
                    "note": "Every component is zero." if q.get("all_zero") and not keys else None,
                }
            )
    elif kind in ("discovery", "invariants", "lagrangian"):
        report = data.get("report") or {}
        if kind == "lagrangian" and report.get("equations"):
            out.append(
                {
                    "title": "Equations of motion",
                    "evidence": "Checked against numerical differentiation of L",
                    "rows": [
                        {"key": f"eq:{i}", "latex": rows[f"eq:{i}"]}
                        for i in range(len(report["equations"]))
                        if f"eq:{i}" in rows
                    ],
                    "note": None,
                }
            )
        items, prefix = {
            "discovery": (report.get("constants") or [], "const"),
            "invariants": (report.get("found") or [], "found"),
            "lagrangian": (report.get("conserved") or [], "cons"),
        }[str(kind)]
        for i, c in enumerate(items):
            key = f"{prefix}:{i}"
            if key in rows:
                out.append(
                    {
                        "title": c.get("name") or "Constant of motion",
                        "evidence": _constant_evidence(c),
                        "rows": [{"key": key, "latex": rows[key]}],
                        "note": None,
                    }
                )
    return out


def _public_turn(prompt: str, user: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    steps = {s.get("id"): s for s in final.get("trace") or [] if isinstance(s, dict)}
    blocks: list[dict[str, Any]] = []
    for block in final.get("blocks") or []:
        if block.get("kind") == "card" and isinstance(block.get("card"), dict):
            card = block["card"]
            step = steps.get(card.get("id")) or {}
            data = card.get("data") or {}
            inner = data.get("result") or data.get("report") or data
            blocks.append(
                {
                    "kind": "card",
                    "tool": card.get("tool"),
                    "title": card.get("title"),
                    "status": (inner or {}).get("status") or step.get("status"),
                    "numbers": card.get("numbers") or {},
                    "number": card.get("number"),
                    "sections": _sections(card),
                    "restrictions_latex": list((inner or {}).get("restrictions_latex") or []),
                    # the call as the tool ran it (after reference resolution and cleaning), for the guard test
                    "call": {"tool": step.get("name"), "arguments": step.get("arguments") or {}},
                }
            )
        elif block.get("kind") == "text":
            segments = [
                {k: s.get(k) for k in ("type", "value", "status", "number") if s.get(k) is not None}
                for s in block.get("segments") or []
            ]
            if segments:
                blocks.append({"kind": "text", "segments": segments})
    return {
        "user": {"text": prompt, "equations": user.get("equations") or []},
        "assistant": {"blocks": blocks, "stop_reason": final.get("stop_reason")},
    }


async def _run(
    args: argparse.Namespace, cases: list[EvalCase]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ev = cli._parser().parse_args(
        [
            "--base-url", args.base_url,
            "--models", args.model,
            "--seed", str(args.seed),
            "--compute-timeout", str(args.compute_timeout),
            "--suite", "explore",
            "--num-predict", str(args.num_predict),
            "--think" if args.think else "--no-think",
        ]
    )  # fmt: skip
    limits = AgentLimits(max_steps=6, max_seconds=ev.case_timeout or ev.timeout * 2)
    ctx = cli._context(ev.compute_timeout)
    params = cli.build_params(ev)
    runs: list[dict[str, Any]] = []
    first: dict[str, tuple[int, list[tuple[str, dict[str, Any], bool]]]] = {}
    for repeat in range(args.runs):
        client = cli.make_client(args.model, ev, repeat)[2]
        for case in cases:
            turns: list[tuple[str, dict[str, Any], bool]] = []

            def on_turn(
                cid: str, rep: int, number: int, prompt: str, obs: Any, passed: bool
            ) -> None:
                turns.append((prompt, obs.final or {}, passed))  # noqa: B023 (consumed before the next case)

            result = await run_case(
                client,
                ctx,
                case,
                limits,
                repeat,
                params=params,
                on_turn=on_turn,
                routing=ev.routing,
            )
            runs.append(
                {
                    "case": case.id,
                    "run": repeat + 1,
                    "passed": bool(result.passed),
                    "turns": [bool(p) for _, _, p in turns],
                    "failure_class": result.failure_class,
                    "notes": list(result.notes),  # fixed scoring text, never the model's words
                    "seconds": round(result.seconds, 1),
                }
            )
            mark = (
                "pass"
                if result.passed
                else f"FAIL [{result.failure_class}] {'; '.join(result.notes)}"
            )
            print(f"  run {repeat + 1} {case.id:<24} {mark}", flush=True)
            if result.passed and case.id not in first:
                first[case.id] = (repeat + 1, turns)  # the first run that passed every turn
    settings = {
        # what every model request of the run carried (the agent's own effective parameters, not a restatement)
        "temperature": params.temperature,
        "context_window_tokens": params.num_ctx,
        "reply_budget_tokens": params.num_predict,
        "thinking": params.think,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "tool_routing": bool(ev.routing),
    }
    return {
        "runs": runs,
        "limits": describe_limits(limits),
        "settings": settings,
    }, first


def _model(args: argparse.Namespace) -> dict[str, Any] | None:
    found = cli.model_identity("ollama", args.model, args.base_url)
    if not found:
        return None
    return {k: found[k] for k in MODEL_FIELDS if k in found} | {"name": args.model}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True, help="Ollama host (never written to any file)")
    parser.add_argument(
        "--model", default="qwen3.5:35b", help="the deployed chat model (llm.model)"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compute-timeout", type=float, default=10.0)
    parser.add_argument("--num-predict", type=int, default=4096, help="reply budget, as deployed")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="as deployed (MINERVA_LLM_THINK)",
    )
    args = parser.parse_args(argv)
    cases = [c for c in load_suite("explore") if c.id in CASES]
    digest = hashlib.sha256(
        json.dumps([[c.prompt, *[f.prompt for f in c.followups]] for c in cases]).encode()
    ).hexdigest()[:12]
    before = _model(args)
    record, first = asyncio.run(_run(args, cases))
    after = _model(args)
    passed = all(r["passed"] for r in record["runs"])
    passed_runs = {
        case.id: sum(1 for r in record["runs"] if r["case"] == case.id and r["passed"])
        for case in cases
    }
    meta = {
        "date": dt.date.today().isoformat(),
        "model": after,
        "model_unchanged_during_run": bool(
            before and after and before.get("digest") == after.get("digest")
        ),
        "runs_per_conversation": args.runs,
        "seed": args.seed,
        "tool_time_limit_seconds": args.compute_timeout,
        "thinking": args.think,
        "reply_budget_tokens": args.num_predict,
        "prompts_digest": digest,
    }
    RECORD.write_text(
        json.dumps(
            {
                "description": "Live runs of the landing page's research sessions on the deployed model "
                "(scripts/record_landing_sessions.py). The page shows the first run of each conversation that passed "
                "every turn, with the number of runs that passed. Contains no model host address.",
                **meta,
                "all_passed": passed,
                "passed_runs": passed_runs,
                **record,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {RECORD.relative_to(ROOT)}")
    missing = [c.id for c in cases if c.id not in first]
    if missing:
        print(f"no passing run of {missing}: the page's sessions were NOT updated", file=sys.stderr)
        return 1
    sessions = []
    for case in cases:
        table = EquationTable()
        contents: list[tuple[str, dict[str, Any]]] = []
        shown_run, transcript = first[case.id]
        for prompt, final, _ in transcript:
            contents += [("user", {"text": prompt}), ("assistant", json.loads(json.dumps(final)))]
        number_stored(contents, table)  # the server's own reading-order numbering
        turns = [
            _public_turn(contents[i][1]["text"], contents[i][1], contents[i + 1][1])
            for i in range(0, len(contents), 2)
        ]
        sessions.append(
            {
                "id": case.id,
                "title": TITLES[case.id],
                "shown_run": shown_run,
                "passed_runs": passed_runs[case.id],
                "turns": turns,
            }
        )
    SESSIONS.write_text(
        json.dumps({**meta, "settings": record["settings"], "sessions": sessions}, indent=2) + "\n"
    )
    print(f"wrote {SESSIONS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
