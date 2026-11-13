"""Minerva symbolic computation engine."""

from .claims import ClaimCheck, check_claim, check_claims, split_equation
from .differentiate import (
    DerivativeStep,
    DerivativeVerification,
    DifferentiateResult,
    DiffVariable,
)
from .discovery import DiscoveredConstant, DiscoveryReport, Pins, SearchEvent
from .distributions.models import (
    Blocked,
    Condition,
    Diagnostics,
    DistributionEvent,
    DistributionInput,
    DistributionResult,
    DistributionStep,
    DistributionVerification,
)
from .errors import LimitError, MinervaMathError, ParseError, UnsupportedInput
from .expansion import ExpansionReport
from .geometry import GeometryInput, GeometryResult
from .independence import IndependenceReport, Relation
from .indexed.models import InstanceVerification, SumsInput, SumsResult, SumStep
from .integrate import IntegrateResult, IntegrateVerification, QuadratureVerification
from .invariants import InvariantReport
from .lagrangian import LagrangianReport
from .latexparse import parse_latex
from .matrix import MatrixCalculusInput, MatrixCalculusResult, MatrixDecl
from .models import (
    Applicability,
    BlockedMove,
    ExpressionInput,
    FormulaReading,
    IdentitiesResult,
    IsolateResult,
    Limits,
    Reason,
    Rule,
    RuleApplication,
    RuleCondition,
    SimplifyResult,
    Solution,
    Status,
    Step,
    StepDiagnostics,
    StepResult,
    Subterm,
    TensorInput,
    TensorResult,
    TensorSpec,
    TransformChain,
    TransformResult,
    TransformStep,
    Verification,
)
from .motion import MotionReport, MotionRequest, Sweep
from .observe import SandboxCall, current_trace_id, trace_scope
from .ops import apply_rule, parse_expression, verify_rule
from .perturbation import CompareRequest, PerturbationReport
from .pi_groups import (
    ConstantCandidate,
    ConstantSuggestion,
    DimensionalReport,
    QuantitySpec,
    UnitsCheckReport,
)
from .series import SeriesResult, SeriesTerm, SeriesVerification
from .service import Engine
from .solve import SolveReport
from .stats.models import ExpectationInput, ExpectationResult, PmfVerification
from .textparse import parse_text

__all__ = [
    "Applicability",
    "Blocked",
    "BlockedMove",
    "ClaimCheck",
    "CompareRequest",
    "Condition",
    "ConstantCandidate",
    "ConstantSuggestion",
    "DerivativeStep",
    "DerivativeVerification",
    "Diagnostics",
    "DiffVariable",
    "DifferentiateResult",
    "DimensionalReport",
    "DiscoveredConstant",
    "DiscoveryReport",
    "DistributionEvent",
    "DistributionInput",
    "DistributionResult",
    "DistributionStep",
    "DistributionVerification",
    "Engine",
    "ExpansionReport",
    "ExpectationInput",
    "ExpectationResult",
    "ExpressionInput",
    "FormulaReading",
    "GeometryInput",
    "GeometryResult",
    "IdentitiesResult",
    "IndependenceReport",
    "InstanceVerification",
    "IntegrateResult",
    "IntegrateVerification",
    "InvariantReport",
    "IsolateResult",
    "LagrangianReport",
    "LimitError",
    "Limits",
    "MatrixCalculusInput",
    "MatrixCalculusResult",
    "MatrixDecl",
    "MinervaMathError",
    "MotionReport",
    "MotionRequest",
    "ParseError",
    "PerturbationReport",
    "Pins",
    "PmfVerification",
    "QuadratureVerification",
    "QuantitySpec",
    "Reason",
    "Relation",
    "Rule",
    "RuleApplication",
    "RuleCondition",
    "SandboxCall",
    "SearchEvent",
    "SeriesResult",
    "SeriesTerm",
    "SeriesVerification",
    "SimplifyResult",
    "Solution",
    "SolveReport",
    "Status",
    "Step",
    "StepDiagnostics",
    "StepResult",
    "Subterm",
    "SumStep",
    "SumsInput",
    "SumsResult",
    "Sweep",
    "TensorInput",
    "TensorResult",
    "TensorSpec",
    "TransformChain",
    "TransformResult",
    "TransformStep",
    "UnitsCheckReport",
    "UnsupportedInput",
    "Verification",
    "apply_rule",
    "check_claim",
    "check_claims",
    "current_trace_id",
    "parse_expression",
    "parse_latex",
    "parse_text",
    "split_equation",
    "trace_scope",
    "verify_rule",
]
