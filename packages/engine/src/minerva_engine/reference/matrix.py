"""Matrix problems: explicit matrices against `mpmath.expm`, exact products, finite differences."""

from __future__ import annotations

import mpmath as mp

from ..matrix import MatrixCalculusInput, MatrixDecl
from ..service import Engine
from .core import Judge, Problem, close, expr, mp_matrix, numeric_entries, verdict

_A = [[0, 1], [0, 0]]
_B = [[0, 0], [1, 0]]
_JORDAN = [[2, 1], [0, 2]]


def _exp_product(engine: Engine) -> tuple[str, str]:
    """exp(A) exp(B) for noncommuting nilpotent matrices: entries against mpmath.expm."""
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=expr(r"\exp(A)\exp(B)"),
            matrices=[
                MatrixDecl(name="A", entries=[["0", "1"], ["0", "0"]]),
                MatrixDecl(name="B", entries=[["0", "0"], ["1", "0"]]),
            ],
            action="evaluate",
        )
    )
    if result.status.value != "ok" or result.result_entries is None:
        return verdict("correct", False, None, f"status {result.status.value}")
    truth = mp.expm(mp_matrix(_A)) * mp.expm(mp_matrix(_B))
    return verdict(
        "correct",
        True,
        close(numeric_entries(result.result_entries), truth),
        "entries vs mpmath.expm",
    )


def _exp_merge_refused(engine: Engine) -> tuple[str, str]:
    """exp(A) exp(B) = exp(A+B) is false for these matrices, so an abstract merge must be refused."""
    a, b = mp_matrix(_A), mp_matrix(_B)
    if close(mp.expm(a) * mp.expm(b), mp.expm(a + b), 1e-6):  # the oracle: the merge is false
        raise RuntimeError("oracle: these matrices commute, the problem is ill-posed")
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=expr(r"\exp(A)\exp(B)"),
            matrices=[MatrixDecl(name="A", rows=2), MatrixDecl(name="B", rows=2)],
            action="simplify",
        )
    )
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _frechet(engine: Engine) -> tuple[str, str]:
    """d/dt exp(A + tB) for noncommuting A, B at t = 0.7: entries against a central difference of expm."""
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=expr(r"\exp(A+tB)"),
            matrices=[
                MatrixDecl(name="A", entries=[["0", "1"], ["0", "0"]]),
                MatrixDecl(name="B", entries=[["0", "0"], ["1", "0"]]),
            ],
            action="differentiate",
            variable="t",
        )
    )
    if result.status.value != "ok" or result.result_entries is None:
        return verdict("correct", False, None, f"status {result.status.value}")
    t0, h = mp.mpf("0.7"), mp.mpf("1e-8")
    a, b = mp_matrix(_A), mp_matrix(_B)
    truth = (mp.expm(a + (t0 + h) * b) - mp.expm(a + (t0 - h) * b)) / (2 * h)
    return verdict(
        "correct",
        True,
        close(numeric_entries(result.result_entries, t=0.7), truth, 1e-7),
        "entries vs central difference of expm",
    )


def _evaluate_against(latex: str, matrices: list[MatrixDecl], truth: mp.matrix, what: str) -> Judge:
    def run(engine: Engine) -> tuple[str, str]:
        result = engine.matrix_calculus(
            MatrixCalculusInput(expression=expr(latex), matrices=matrices, action="evaluate")
        )
        if result.status.value != "ok" or result.result_entries is None:
            return verdict("correct", False, None, f"status {result.status.value}")
        return verdict("correct", True, close(numeric_entries(result.result_entries), truth), what)

    return run


def _jordan_decl() -> list[MatrixDecl]:
    return [MatrixDecl(name="A", entries=[[str(v) for v in row] for row in _JORDAN])]


def _exp_jordan(engine: Engine) -> tuple[str, str]:
    """exp of a defective matrix (repeated eigenvalue, one Jordan block): e^2 [[1, 1], [0, 1]]."""
    return _evaluate_against(
        r"\exp(A)", _jordan_decl(), mp.expm(mp_matrix(_JORDAN)), "entries vs mpmath.expm"
    )(engine)


def _power_jordan(engine: Engine) -> tuple[str, str]:
    """A^5 of the Jordan block: repeated integer multiplication in mpmath."""
    m = mp_matrix(_JORDAN)
    return _evaluate_against(r"A^5", _jordan_decl(), m**5, "entries vs mpmath matrix power")(engine)


def _square_of_sum(engine: Engine) -> tuple[str, str]:
    """(A+B)^2 with noncommuting explicit A, B: the entries contain the cross terms AB + BA."""
    decls = [
        MatrixDecl(name="A", entries=[["2", "1"], ["0", "2"]]),
        MatrixDecl(name="B", entries=[["0", "1"], ["1", "0"]]),
    ]
    a, b = mp_matrix(_JORDAN), mp_matrix([[0, 1], [1, 0]])
    return _evaluate_against(r"(A+B)^2", decls, (a + b) * (a + b), "entries vs mpmath product")(
        engine
    )


def _singular_inverse_refused(engine: Engine) -> tuple[str, str]:
    """The inverse of [[1, 2], [2, 4]] does not exist: its determinant is 0."""
    m = mp_matrix([[1, 2], [2, 4]])
    if mp.det(m) != 0:
        raise RuntimeError("oracle: the matrix is invertible, the problem is ill-posed")
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=expr(r"A^{-1}"),
            matrices=[MatrixDecl(name="A", entries=[["1", "2"], ["2", "4"]])],
            action="evaluate",
        )
    )
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _derivative_of_defective_exponential(engine: Engine) -> tuple[str, str]:
    """d/dt exp(tA) for the Jordan block: the closed form A exp(tA) against a finite difference of expm."""
    result = engine.matrix_calculus(
        MatrixCalculusInput(
            expression=expr(r"\exp(tA)", t=["real"]),
            matrices=_jordan_decl(),
            action="differentiate",
            variable="t",
        )
    )
    if result.status.value != "ok" or result.result_latex is None:
        return verdict("correct", False, None, f"status {result.status.value}")
    if result.result_latex != r"A \exp\left(t A\right)":
        return verdict("correct", True, False, f"unexpected form {result.result_latex}")
    m, t0, h = mp_matrix(_JORDAN), mp.mpf(1) / 3, mp.mpf("1e-10")
    truth = (mp.expm((t0 + h) * m) - mp.expm((t0 - h) * m)) / (2 * h)
    return verdict(
        "correct", True, close(m * mp.expm(t0 * m), truth, 1e-8), "A exp(tA) vs central difference"
    )


PROBLEMS: list[Problem] = [
    Problem(
        "matrix-exp-product-noncommuting",
        "matrix",
        "Higham, Functions of Matrices (matrix exponential)",
        "correct",
        _exp_product,
        oracle="mpmath_expm",
    ),
    Problem(
        "matrix-exp-merge-refused",
        "matrix",
        "Higham, Functions of Matrices (matrix exponential: exp(A)exp(B) = exp(A+B) needs commuting A, B in general)",
        "refused",
        _exp_merge_refused,
        oracle="mpmath_expm",
        evidence="refusal",
    ),
    Problem(
        "matrix-frechet-derivative",
        "matrix",
        "Higham, Functions of Matrices (Frechet derivative of the exponential)",
        "correct",
        _frechet,
        oracle="finite_difference",
    ),
    Problem(
        "matrix-exp-jordan-block",
        "matrix",
        "Higham, Functions of Matrices (exponential of a defective matrix)",
        "correct",
        _exp_jordan,
        oracle="mpmath_expm",
    ),
    Problem(
        "matrix-power-jordan-block",
        "matrix",
        "Horn and Johnson, Matrix Analysis (powers of a Jordan block)",
        "correct",
        _power_jordan,
        oracle="arithmetic",
    ),
    Problem(
        "matrix-square-of-sum-noncommuting",
        "matrix",
        "Horn and Johnson, Matrix Analysis (matrix multiplication is not commutative)",
        "correct",
        _square_of_sum,
        oracle="arithmetic",
    ),
    Problem(
        "matrix-singular-inverse-refused",
        "matrix",
        "Horn and Johnson, Matrix Analysis (a matrix is invertible iff its determinant is nonzero)",
        "refused",
        _singular_inverse_refused,
        oracle="arithmetic",
        evidence="refusal",
    ),
    Problem(
        "matrix-derivative-of-defective-exponential",
        "matrix",
        "Higham, Functions of Matrices (d/dt exp(tA) = A exp(tA))",
        "correct",
        _derivative_of_defective_exponential,
        oracle="finite_difference",
        evidence="symbolic_proof",
    ),
]
