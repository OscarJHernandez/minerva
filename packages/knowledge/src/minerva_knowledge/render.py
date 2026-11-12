"""Display helpers: LaTeX statements and plain-text search text for entries."""

from __future__ import annotations

import sympy as sp
from minerva_engine import parse_text
from minerva_engine.features import expression_features

from .models import Entry


def statement_latex(entry: Entry) -> str:
    """LaTeX for the entry's statement (hand-written, or generated from its rule)."""
    if entry.statement_latex:
        return entry.statement_latex
    assert entry.rule is not None  # noqa: S101 - guaranteed by the Entry validator
    lhs, rhs = parse_text(entry.rule.lhs), parse_text(entry.rule.rhs)
    return f"{sp.latex(lhs)} = {sp.latex(rhs)}"


def rule_features(entry: Entry) -> set[str]:
    """Features an entry is retrieved by: its rule's left-hand side structure plus `applies_to`."""
    features = set(entry.applies_to)
    if entry.rule is not None:
        features |= expression_features(parse_text(entry.rule.lhs))
    return features


def rhs_features(entry: Entry) -> set[str]:
    """Structure of the rule's right-hand side: what a chain can reach by applying it left to right."""
    if entry.rule is None:
        return set()
    return expression_features(parse_text(entry.rule.rhs))


def feature_token(feature: str) -> str:
    """FTS5-safe token for a feature, e.g. `f:sin` -> `featfsin`."""
    return "feat" + feature.replace(":", "").replace("_", "")


def search_text(entry: Entry) -> str:
    parts = [entry.title, entry.validity, entry.notes]
    if entry.rule is not None:
        parts += [entry.rule.lhs, entry.rule.rhs]
    parts += [s.meaning for s in entry.symbols]
    return " ".join(p for p in parts if p)
