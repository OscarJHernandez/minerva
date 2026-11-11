"""Structural features of an expression, used to retrieve candidate identities.

Features are coarse tokens such as `f:sin`, `f:log`, `op:square`. Knowledge-base rules are
indexed by the features of their left-hand side; an expression retrieves rules that share
at least one feature. Retrieval only *proposes* candidates; the engine then decides
whether each one actually matches and applies.
"""

from __future__ import annotations

import sympy as sp


def expression_features(expr: sp.Basic) -> set[str]:
    features: set[str] = set()
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Function) and not isinstance(node, sp.Derivative):
            features.add(f"f:{node.func.__name__.lower()}")
        elif isinstance(node, sp.Pow):
            exponent = node.exp
            if exponent == 2:
                features.add("op:square")
            elif exponent == 3:
                features.add("op:cube")
            elif exponent == sp.Rational(1, 2):
                features.add("op:sqrt")
            elif exponent.is_number is False or not exponent.is_Number:
                features.add("op:power")
        elif isinstance(node, sp.Mul) and any(
            isinstance(a, sp.Pow) and a.exp.is_negative for a in node.args
        ):
            features.add("op:quotient")
    return features
