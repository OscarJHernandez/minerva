"""Property-based tests: identities that hold for every metric in a family, for random members of it.

The families are small on purpose (diagonal and conformally flat metrics with random polynomial entries)
so that each example is cheap; hypothesis shrinks a failure to a minimal metric.
"""

import time

import sympy as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minerva_engine import Limits
from minerva_engine.geometry import GeometryInput
from minerva_engine.geometry.compute import Workspace
from minerva_engine.geometry.setup import build_geometry
from minerva_engine.geometry.verify import Verifier

PROFILE = settings(
    max_examples=8,
    deadline=None,
    derandomize=True,  # fixed seed: the same examples on every run, so CI is deterministic
    suppress_health_check=[HealthCheck.too_slow],
)
LIMITS = Limits(timeout_seconds=120)
COORDS = ["x", "y", "z"]

small = st.integers(min_value=1, max_value=4)
coefficient = st.integers(min_value=-2, max_value=2)


@st.composite
def positive_entry(draw: st.DrawFn) -> str:
    """A quadratic polynomial entry in x, y, z that is positive near the origin."""
    a, b, c = draw(coefficient), draw(coefficient), draw(coefficient)
    return f"(3 + {a}*x + {b}*y^2/2 + {c}*z)"


@st.composite
def diagonal_metric(draw: st.DrawFn) -> tuple[list[str], list[list[str]]]:
    n = draw(st.integers(min_value=2, max_value=3))
    entries = [draw(positive_entry()) for _ in range(n)]
    grid = [[entries[i] if i == j else "0" for j in range(n)] for i in range(n)]
    return COORDS[:n], grid


# All entries are polynomials, so every component is a rational function and `cancel` is a complete
# zero test (no simplifier heuristics, and much cheaper than `simplify`).


def workspace(coords: list[str], grid: list[list[str]], scale: str = "1") -> Workspace:
    if scale != "1":
        grid = [[f"({scale})*{cell}" if cell != "0" else "0" for cell in row] for row in grid]
    data = GeometryInput(coordinates=coords, metric=grid, format="text", quantities=["christoffel"])
    return Workspace(build_geometry(data, LIMITS, 6), LIMITS, time.monotonic() + 600)


@PROFILE
@given(diagonal_metric())
def test_riemann_symmetries_and_bianchi_hold_for_random_diagonal_metrics(case) -> None:  # type: ignore[no-untyped-def]
    coords, grid = case
    ws = workspace(coords, grid)
    v = Verifier(ws).riemann()
    assert v.numeric_passed, [c for c in v.checks if c.passed is False]


@PROFILE
@given(diagonal_metric())
def test_christoffel_is_the_levi_civita_connection_for_random_diagonal_metrics(case) -> None:  # type: ignore[no-untyped-def]
    coords, grid = case
    ws = workspace(coords, grid)
    v = Verifier(ws).christoffel()
    assert v.numeric_passed, [c for c in v.checks if c.passed is False]


@PROFILE
@given(diagonal_metric())
def test_the_einstein_tensor_is_divergence_free_for_random_diagonal_metrics(case) -> None:  # type: ignore[no-untyped-def]
    coords, grid = case
    ws = workspace(coords, grid)
    v = Verifier(ws).einstein()
    assert v.numeric_passed, [c for c in v.checks if c.passed is False]


@PROFILE
@given(diagonal_metric(), st.integers(min_value=2, max_value=5))
def test_a_constant_rescaling_keeps_the_connection_and_scales_the_ricci_scalar(case, k) -> None:  # type: ignore[no-untyped-def]
    coords, grid = case
    base = workspace(coords, grid)
    scaled = workspace(coords, grid, scale=str(k))
    g1, g2 = base.christoffel(), scaled.christoffel()
    assert g1.nonzero().__len__() == g2.nonzero().__len__()
    for idx, value in g1.nonzero():
        assert sp.cancel(g2.get(idx) - value) == 0
    # g -> k g leaves Gamma unchanged and divides the Ricci scalar by k
    assert sp.cancel(scaled.ricci_scalar() * k - base.ricci_scalar()) == 0


@PROFILE
@given(st.integers(min_value=1, max_value=3), st.lists(coefficient, min_size=9, max_size=9))
def test_constant_metrics_are_flat(n: int, values: list[int]) -> None:
    # a constant symmetric positive-definite metric: A^T A + I
    a = sp.Matrix(n, n, lambda i, j: values[i * 3 + j])
    g = a.T * a + sp.eye(n)
    grid = [[str(g[i, j]) for j in range(n)] for i in range(n)]
    ws = workspace(COORDS[:n], grid)
    assert all(v == 0 for v in ws.christoffel().data.values())
    assert all(v == 0 for v in ws.riemann().data.values())
    assert ws.ricci_scalar() == 0


@PROFILE
@given(diagonal_metric())
def test_the_riemann_tensor_has_the_pair_and_antisymmetry_properties_symbolically(case) -> None:  # type: ignore[no-untyped-def]
    coords, grid = case
    ws = workspace(coords, grid)
    low = ws.lowered_riemann()
    n = len(coords)
    for r in range(n):
        for s in range(n):
            for m in range(n):
                for v in range(n):
                    assert sp.cancel(low.get((r, s, m, v)) + low.get((s, r, m, v))) == 0
                    assert sp.cancel(low.get((r, s, m, v)) - low.get((m, v, r, s))) == 0
