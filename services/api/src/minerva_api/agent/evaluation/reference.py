"""Check the benchmark's own ground truth: run each case's reference call and compare the card.

This is what keeps the expectations honest. A case expects `result_text`; the reference call (a tool call
that is correct for the prompt) must produce exactly that through the real engine. If the engine and the
hand-written expectation disagree, one of them is wrong and the test suite says so.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from minerva_engine import parse_text

from ..tools import ToolContext, execute
from .cases import EvalCase, ExpectCard


def same_expression(a: str, b: str) -> bool:
    """Equal as expressions: identical text, equal after parsing, or equal at sample points.

    Used only to score a result against the benchmark's expectation. The numeric leg compares the two
    forms at fixed complex points (a score, never a claim shown to a user).
    """
    if "".join(a.split()) == "".join(b.split()):
        return True
    try:
        left, right = parse_text(a), parse_text(b)
    except Exception:  # an unreadable form simply does not match
        return False
    if left == right:
        return True
    symbols = sorted(left.free_symbols | right.free_symbols, key=str)
    points = [
        {
            s: sp.Rational(7 + 3 * i, 5 + i) + sp.I * sp.Rational(i, 3 + i) * (j % 2)
            for i, s in enumerate(symbols)
        }
        for j in range(4)
    ]
    try:
        for point in points:
            diff = sp.N((left - right).subs(point), 30)
            scale = 1 + abs(sp.N(left.subs(point), 30))
            if not diff.is_number or abs(diff) > sp.Float("1e-12") * scale:
                return False
    except Exception:
        return False
    return True


_NESTED = ("report", "result")  # tools that wrap their engine result in the card data


def _field(data: dict[str, Any], name: str) -> Any:
    if data.get(name) is not None:
        return data[name]
    for key in _NESTED:
        inner = data.get(key)
        if isinstance(inner, dict) and inner.get(name) is not None:
            return inner[name]
    return None


def card_status(data: dict[str, Any], tool_status: str | None = None) -> str | None:
    """The engine status inside a card; tools that carry none (handbook lookups) use the tool's own."""
    status = _field(data, "status")
    return str(status) if status is not None else tool_status


def card_result_text(data: dict[str, Any]) -> str | None:
    result = _field(data, "result_text")
    return str(result) if result is not None else None


def card_meets(
    expect: ExpectCard, card: dict[str, Any], tool_status: str | None = None
) -> list[str]:
    """Why a card does not meet the expectation (empty when it does)."""
    data = card.get("data", {})
    problems: list[str] = []
    status = card_status(data, tool_status)
    if status != expect.status:
        problems.append(f"status {status!r}, expected {expect.status!r}")
    if expect.result_text is not None:
        got = card_result_text(data)
        if got is None or not same_expression(got, expect.result_text):
            problems.append(f"result {got!r}, expected {expect.result_text!r}")
    return problems


def check_reference(case: EvalCase, ctx: ToolContext) -> list[str]:
    """Run the case's reference call; report every way it fails to meet `expect_card`."""
    if case.reference is None or case.expect_card is None:
        return []
    output = execute(case.reference.tool, case.reference.arguments, ctx)
    card = output.card.model_dump(mode="json")
    if card["type"] == "error":
        return [f"reference call failed: {card['data'].get('message', '')}"]
    return card_meets(case.expect_card, card, output.status)
