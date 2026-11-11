"""Tensor-polynomial canonicalization using `sympy.tensor.tensor` (Butler-Portugal).

Scope is exactly what SymPy supports: monoterm index symmetries (symmetric, antisymmetric,
Riemann), index renaming of dummies and contraction detection. Multi-term identities such
as the first Bianchi identity are **not** applied; expressions needing them are returned
canonicalised but otherwise unreduced.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import sympy as sp
from sympy.tensor.tensor import (
    TensorHead,
    TensorIndex,
    TensorIndexType,
    TensorSymmetry,
)

from .errors import LimitError, ParseError
from .models import Limits, TensorInput

_INDEX_RE = re.compile(r"^[a-z][a-z0-9]{0,7}$")
_MAX_DEPTH = 60


def _symmetry(kind: str, rank: int) -> TensorSymmetry:
    if kind == "symmetric":
        return TensorSymmetry.fully_symmetric(rank)
    if kind == "antisymmetric":
        return TensorSymmetry.fully_symmetric(-rank)
    if kind == "riemann":
        return TensorSymmetry.riemann()
    return TensorSymmetry.no_symmetry(rank)


def canonicalize(data: TensorInput, limits: Limits | None = None) -> tuple[Any, Any]:
    """Return (parsed_expression, canonical_expression)."""
    limits = limits or Limits()
    if len(data.input) > limits.max_input_chars:
        raise LimitError("input is too long")
    index_type = TensorIndexType("L", dim=data.dimension, dummy_name="L")
    heads = {
        name: TensorHead(name, [index_type] * spec.rank, _symmetry(spec.symmetry, spec.rank))
        for name, spec in data.tensors.items()
    }
    try:
        tree = ast.parse(data.input.strip().replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ParseError("could not parse the tensor expression") from exc

    def index(node: ast.AST) -> TensorIndex:
        lower = False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            lower, node = True, node.operand
        if not isinstance(node, ast.Name) or not _INDEX_RE.match(node.id):
            raise ParseError("indices must be short lowercase names, `-` marks a lowered index")
        idx = TensorIndex(node.id, index_type)
        return -idx if lower else idx

    def conv(node: ast.AST, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            raise LimitError("expression is nested too deeply")
        d = depth + 1
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            if abs(node.value) > 10**9:
                raise LimitError("coefficient too large")
            return sp.Integer(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -conv(node.operand, d)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return conv(node.operand, d)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return conv(node.left, d) + conv(node.right, d)
            if isinstance(node.op, ast.Sub):
                return conv(node.left, d) - conv(node.right, d)
            if isinstance(node.op, ast.Mult):
                return conv(node.left, d) * conv(node.right, d)
            raise ParseError("only +, - and * are supported in tensor expressions")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in heads or node.keywords:
                raise ParseError("unknown tensor; declare it in `tensors`")
            head = heads[node.func.id]
            if len(node.args) != head.rank:
                raise ParseError(f"{node.func.id} has rank {head.rank}")
            return head(*[index(a) for a in node.args])
        raise ParseError(f"unsupported syntax: {type(node).__name__}")

    try:
        parsed = conv(tree.body, 0)
        canonical = parsed.canon_bp() if hasattr(parsed, "canon_bp") else parsed
    except (ValueError, TypeError) as exc:
        raise ParseError("invalid index structure (an index may appear at most twice)") from exc
    return parsed, canonical
