"""Gradients, Jacobians and Hessians of functions of matrices: reading, differentials, verification."""

import pytest
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.matrix import MatrixCalculusInput, MatrixDecl
from minerva_engine.matrix import mexpr as mx
from minerva_engine.matrix.gradient import Setup, build_setup, check_gradient

CTX = mx.Context(
    {"A": (3, 3), "B": (3, 3), "x": (3, 1), "y": (3, 1), "X": (4, 2), "S": (3, 3)},
    frozenset({"S"}),
    frozenset({"S"}),
)


def read(text: str) -> mx.MNode | mx.SNode:
    return mx.parse(text, CTX)


def M(name, rows, cols=None, **flags):
    return {"name": name, "rows": rows, "cols": cols or rows, **flags}


# ---------------------------------------------------------------------------------- reading


def test_a_one_by_one_product_is_a_scalar_factor() -> None:
    node = read("transpose(x)*A*x*B")  # (x^T A x) B : the 1x1 block scales B
    assert isinstance(node, mx.Scale) and isinstance(node.coeff, mx.AsScalar)
    assert node.arg == mx.Var("B")


def test_quadratic_forms_and_traces_are_scalars() -> None:
    assert isinstance(read("transpose(x)*A*x"), mx.AsScalar)
    assert isinstance(read("tr(A*B)"), mx.Tr)
    assert isinstance(read("2*logdet(S) + tr(A)"), mx.SAdd)


@pytest.mark.parametrize(
    "bad",
    [
        "A*X",  # 3x3 times 4x2
        "x*y",  # 3x1 times 3x1
        "A + x",  # shapes differ
        "A + 1",  # a scalar and a matrix
        "tr(X)",  # not square
        "inv(X)",
        "1/A",  # division by a matrix
        "A**0.5",
        "A.T",  # attribute access is not syntax here
        "A'",
        "__import__('os')",
        "lambda: 1",
        "A[0]",
    ],
)
def test_unreadable_or_ill_shaped_expressions_are_refused(bad: str) -> None:
    with pytest.raises(mx.MatrixExprError):
        read(bad)


def test_matrices_are_never_reordered() -> None:
    assert read("A*B") != read("B*A")
    node = read("A*B*A")
    assert not isinstance(node, mx.Num | mx.Sym | mx.Tr | mx.Det | mx.LogDet | mx.AsScalar)
    assert mx.text_m(node) == "A*B*A"  # type: ignore[arg-type]


def test_text_and_latex_print() -> None:
    node = read("-1/2*transpose(x)*inv(S)*x")
    assert mx.text_s(node) == "-1/2*(transpose(x)*inv(S)*x)"  # type: ignore[arg-type]
    assert mx.latex_s(node) == r"-\frac{1}{2} x^{\top} S^{-1} x"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------- the operation


@pytest.fixture(scope="module")
def run(engine: Engine):
    def go(expr, mats, var, action="gradient", **kw):
        return engine.matrix_calculus(
            MatrixCalculusInput(
                expression=ExpressionInput(input=expr, format="text"),
                matrices=[MatrixDecl(**m) for m in mats],
                variable=var,
                action=action,
                **kw,
            )
        )

    return go


GAUSSIAN = "-1/2*logdet(S) - 1/2*transpose(x - mu)*inv(S)*(x - mu)"
GAUSSIAN_MATRICES = [M("x", 3, 1), M("mu", 3, 1)]


def test_the_gaussian_gradient_in_the_mean_and_the_covariance(run) -> None:
    mean = run(GAUSSIAN, [*GAUSSIAN_MATRICES, M("S", 3, positive_definite=True)], "mu")
    assert mean.status is Status.OK and mean.result_text == "inv(S)*(x - mu)"
    free = run(GAUSSIAN, [*GAUSSIAN_MATRICES, M("S", 3)], "S", invertible=["S"])
    assert free.status is Status.OK and free.derivative is not None
    assert free.derivative.convention == "unconstrained" and free.derivative.constraint == "none"
    assert free.result_text is not None and "transpose(inv(S))" in free.result_text


def test_a_symmetric_variable_gets_the_gradient_within_the_symmetric_matrices(run) -> None:
    r = run(GAUSSIAN, [*GAUSSIAN_MATRICES, M("S", 3, positive_definite=True)], "S")
    assert r.status is Status.OK and r.derivative is not None
    assert r.derivative.convention == "symmetric_frobenius"
    assert r.derivative.constraint == "positive_definite"
    (alt,) = r.derivative.alternatives
    assert alt.convention == "symmetric_entrywise" and "diagpart" in alt.text
    assert alt.points_agreed == alt.points_checked >= 3
    assert any("symmetric matrices" in note for note in r.notes)


def test_the_hessian_is_the_jacobian_of_the_gradient(run) -> None:
    quad = run("transpose(x)*A*x", [M("A", 3), M("x", 3, 1)], "x", "hessian")
    assert quad.result_text == "A + transpose(A)"
    info = run(
        "-1/2*transpose(x - mu)*inv(S)*(x - mu)",
        [*GAUSSIAN_MATRICES, M("S", 3, positive_definite=True)],
        "mu",
        "hessian",
    )
    assert info.result_text == "-inv(S)"
    assert run("A*x", [M("A", 3), M("x", 3, 1)], "x", "jacobian").result_text == "A"


def test_inverses_and_determinants_need_a_declared_invertible_matrix(run) -> None:
    for expr in ["tr(inv(X)*A)", "det(X)", "logdet(X)"]:
        r = run(expr, [M("A", 3), M("X", 3)], "X")
        assert (r.status, r.reason) == (
            Status.CONDITIONS_NOT_SATISFIED,
            "invertibility_not_declared",
        ), expr
        assert r.result_text is None


@pytest.mark.parametrize(
    ("expr", "mats", "var", "status", "reason"),
    [
        ("X*A", [M("A", 3), M("X", 3)], "X", Status.UNSUPPORTED, "non_scalar_gradient"),
        ("tr(A*X)", [M("A", 2, 3), M("X", 2)], "X", Status.PARSE_ERROR, "parse_error"),
        ("tr(A)", [M("A", 2, 3)], "A", Status.PARSE_ERROR, "parse_error"),
        ("tr(A)", [M("A", 2)], "B", Status.PARSE_ERROR, "parse_error"),  # B is not declared
    ],
)
def test_refusals(run, expr, mats, var, status, reason) -> None:
    r = run(expr, mats, var)
    assert (r.status, r.reason) == (status, reason) and r.result_text is None


def test_declarations_are_checked_against_the_entries(run) -> None:
    singular = {"name": "S", "entries": [["1", "2"], ["2", "4"]]}
    r = run(
        "-1/2*transpose(x - mu)*inv(S)*(x - mu)",
        [M("x", 2, 1), M("mu", 2, 1), singular],
        "mu",
        invertible=["S"],
    )
    assert (r.status, r.reason) == (Status.PARSE_ERROR, "declaration_contradicted")
    skew = {"name": "S", "entries": [["1", "2"], ["3", "4"]], "symmetric": True}
    r = run("transpose(x)*S*x", [M("x", 2, 1), skew], "x")
    assert (r.status, r.reason) == (Status.PARSE_ERROR, "declaration_contradicted")


def test_explicit_symmetric_entries_are_recognised_and_used(run) -> None:
    sym = {"name": "S", "entries": [["2", "1"], ["1", "3"]]}
    r = run("-1/2*transpose(x)*inv(S)*x", [M("x", 2, 1), sym], "x")
    assert r.status is Status.OK and r.result_text == "-inv(S)*x"
    assert "S is symmetric (from the entries)" in r.assumptions_used
    assert "S is invertible (from the entries)" in r.assumptions_used


def test_the_variable_must_be_abstract_and_the_syntax_plain_text(engine: Engine) -> None:
    explicit = MatrixCalculusInput(
        expression=ExpressionInput(input="tr(S)", format="text"),
        matrices=[MatrixDecl(name="S", entries=[["1", "0"], ["0", "1"]])],
        variable="S",
        action="gradient",
    )
    assert engine.matrix_calculus(explicit).status is Status.UNSUPPORTED
    latex = MatrixCalculusInput(
        expression=ExpressionInput(input=r"\operatorname{tr}(A)", format="latex"),
        matrices=[MatrixDecl(name="A", rows=2)],
        variable="A",
        action="gradient",
    )
    assert engine.matrix_calculus(latex).status is Status.UNSUPPORTED


def test_the_trace_of_the_variable_has_the_identity_as_its_gradient(run) -> None:
    r = run("tr(A)", [M("A", 3)], "A")
    assert r.status is Status.OK and r.result_text == "eye(3)"


def test_a_function_that_does_not_depend_on_the_variable_has_a_zero_gradient(run) -> None:
    r = run("tr(A)", [M("A", 3), M("X", 3)], "X")
    assert r.status is Status.OK and r.result_text == "0*X"


# ---------------------------------------------------------------------------------- wrong candidates


def _check(engine: Engine, expr: str, mats: list[dict], var: str, candidate: str, **kw) -> Status:
    return engine.matrix_calculus(
        MatrixCalculusInput(
            expression=ExpressionInput(input=expr, format="text"),
            matrices=[MatrixDecl(**m) for m in mats],
            variable=var,
            action="gradient",
            candidate={"result": candidate},
            **kw,
        )
    ).status


@pytest.mark.parametrize(
    ("expr", "mats", "var", "kw", "right", "wrong"),
    [
        (
            "tr(transpose(X)*A*X)",
            [M("A", 3), M("X", 3, 2)],
            "X",
            {},
            "A*X + transpose(A)*X",
            ["A*X", "2*A*X", "A*X + A*X", "transpose(A)*X", "A*X - transpose(A)*X"],
        ),
        (
            "transpose(x)*A*x",
            [M("A", 3), M("x", 3, 1)],
            "x",
            {},
            "(A + transpose(A))*x",
            ["A*x", "2*A*x", "(A + transpose(A))*x*(1000001/1000000)"],
        ),
        (
            "logdet(X)",
            [M("X", 3)],
            "X",
            {"invertible": ["X"]},
            "transpose(inv(X))",
            ["inv(X)", "-transpose(inv(X))", "2*transpose(inv(X))"],
        ),
        (
            "tr(A*X*B)",
            [M("A", 3), M("X", 3), M("B", 3)],
            "X",
            {},
            "transpose(A)*transpose(B)",
            ["transpose(B)*transpose(A)", "A*B", "transpose(A*B)"],
        ),
        (
            "tr(inv(X)*A)",
            [M("A", 3), M("X", 3)],
            "X",
            {"invertible": ["X"]},
            "-transpose(inv(X))*transpose(A)*transpose(inv(X))",
            ["-inv(X)*A*inv(X)", "transpose(inv(X))*transpose(A)*transpose(inv(X))"],
        ),
    ],
)
def test_wrong_gradients_are_rejected_and_the_right_one_accepted(
    engine, expr, mats, var, kw, right, wrong
) -> None:
    assert _check(engine, expr, mats, var, right, **kw) is Status.OK
    for candidate in wrong:
        assert _check(engine, expr, mats, var, candidate, **kw) is Status.NOT_VERIFIED, candidate


def test_an_unconstrained_gradient_is_not_a_gradient_in_the_symmetric_matrices(engine) -> None:
    mats = [M("A", 3), M("S", 3, positive_definite=True)]
    right = "-1/2*inv(S)*(A + transpose(A))*inv(S)"
    unconstrained = "-inv(S)*transpose(A)*inv(S)"  # passes every symmetric directional check
    assert _check(engine, "tr(inv(S)*A)", mats, "S", right) is Status.OK
    assert _check(engine, "tr(inv(S)*A)", mats, "S", unconstrained) is Status.NOT_VERIFIED


def read_in(setup: Setup, text: str) -> mx.MNode:
    node = mx.parse(text, setup.ctx)
    assert not isinstance(
        node,
        mx.Num | mx.Sym | mx.Tr | mx.Det | mx.LogDet | mx.AsScalar | mx.SAdd | mx.SMul | mx.SPow,
    )
    return node


def test_the_symmetry_requirement_is_what_rejects_it() -> None:
    """On every symmetric direction the unconstrained and the symmetrised gradient have the same
    directional derivative; only the symmetry of the candidate separates them."""
    data = MatrixCalculusInput(
        expression=ExpressionInput(input="tr(inv(S)*A)", format="text"),
        matrices=[MatrixDecl(**M("A", 3)), MatrixDecl(**M("S", 3, positive_definite=True))],
        variable="S",
        action="gradient",
    )
    setup, _, _ = build_setup(data, Limits())
    f = mx.parse("tr(inv(S)*A)", setup.ctx)
    assert isinstance(f, mx.Tr)
    unconstrained = read_in(setup, "-inv(S)*transpose(A)*inv(S)")
    symmetrised = read_in(setup, "-1/2*inv(S)*(A + transpose(A))*inv(S)")
    kw = {"deadline": 1e12}
    assert check_gradient(f, "S", symmetrised, setup, constrained=True, **kw)[0].passed
    assert not check_gradient(f, "S", unconstrained, setup, constrained=True, **kw)[0].passed
    # Treated as an unconstrained matrix, the same expression IS the right derivative: what separates
    # the two readings is the symmetry requirement, not the directional derivatives.
    assert check_gradient(f, "S", unconstrained, setup, constrained=False, **kw)[0].passed


def test_an_oversized_request_is_bounded(run) -> None:
    r = run("tr(A*X*A*X)", [M("A", 8), M("X", 8)], "X")
    assert r.status in {Status.OK, Status.RESOURCE_LIMIT}
    if r.status is Status.OK:
        assert r.duration_ms < 20_000
