/** Types of the `variation` operation, mirroring `minerva_engine.variational.models`. */
import type { Assumptions, Status } from "./types";

export type VariationAction = "analyze" | "first" | "evaluate" | "verify" | "solve" | "second";
export type VariationEnd = "lower" | "upper";

export interface GivenFunction {
  name: string;
  args: string[];
}

export interface EndpointCondition {
  function: string;
  order: number;
  end: VariationEnd;
  kind: "fixed" | "free";
  value?: string | null;
}

export interface SampledFunction {
  t: number[];
  values: number[];
  slopes?: number[] | null;
}

export interface TrialPath {
  label?: string | null;
  analytic: Record<string, string>;
  sampled: Record<string, SampledFunction>;
}

export interface Sweep {
  name: string;
  values: string[];
}

export interface PerturbationSpec {
  label?: string | null;
  expressions: Record<string, string>;
}

/** The functional part of a request: what the model may send, and what a card carries. */
export interface FunctionalRequest {
  integrand: string;
  format?: "latex" | "text";
  variable: string;
  functions: string[];
  given_functions: GivenFunction[];
  lower: string | null;
  upper: string | null;
  lower_term: string | null;
  upper_term: string | null;
  conditions: EndpointCondition[];
  assumptions: Assumptions;
}

/** A whole request (`VariationInput`): the functional, the action and the user's numbers. */
export interface VariationRequest extends FunctionalRequest {
  action: VariationAction;
  values: Record<string, string>;
  definitions: Record<string, string>;
  paths: TrialPath[];
  sweep: Sweep | null;
  perturbations: PerturbationSpec[];
  automatic_perturbations: boolean;
  with_second: boolean;
}

export type NextStepId =
  | "first_variation"
  | "euler_lagrange"
  | "boundary_terms"
  | "evaluate_action"
  | "verify_variation"
  | "solve_stationary"
  | "second_variation"
  | "declare_conditions";
export type Need =
  "interval" | "endpoint_conditions" | "parameter_values" | "trial_path" | "initial_guess";
export type ConditionState = "fixed" | "free" | "unspecified";
export type BoundaryEffect =
  "removed_by_fixed" | "natural_condition" | "kept_unspecified" | "vanishes_identically";

export interface NextStep {
  id: NextStepId;
  needs: Need[];
}

export interface VariationAnalysis {
  functional_latex: string;
  variable: string;
  lower_latex: string;
  upper_latex: string;
  functions: { name: string; name_latex: string; order: number; appears: boolean }[];
  parameters: string[];
  given_functions: string[];
  highest_order: number;
  explicit_in_variable: boolean;
  has_endpoint_terms: boolean;
  regularity: "regular" | "conditional" | "degenerate";
  hessian_det_latex: string;
  conditions: {
    function: string;
    order: number;
    end: VariationEnd;
    state: ConditionState;
    value_latex: string | null;
  }[];
  unspecified: number;
  conditions_needed_per_end: number;
  reading: string[];
  next_steps: NextStep[];
}

export interface IbpStage {
  index: number;
  function: string | null;
  moved_order: number | null;
  integrand_latex: string;
  boundary_latex: string;
  needs_derivatives: number;
  checked: boolean | null;
}

export interface BoundaryEntry {
  end: VariationEnd;
  function: string;
  order: number;
  perturbation_latex: string;
  coefficient_latex: string;
  state: ConditionState;
  effect: BoundaryEffect;
  natural_condition_latex: string | null;
}

export interface ComparisonRow {
  step: string;
  quotient: string;
  abs_error: number;
  rel_error: number;
}

export interface Comparison {
  kind: "bubble" | "generic";
  function: string | null;
  analytic: string;
  scale: string;
  rows: ComparisonRow[];
  observed_order: number | null;
  richardson_rel_error: number;
  verdict: "agree" | "disagree" | "inconclusive";
}

export interface VariationChecks {
  deciding: "passed" | "failed" | "inconclusive";
  comparisons: Comparison[];
  pointwise_passed: boolean | null;
  stages_checked: number;
  stages_passed: boolean | null;
  sympy_agrees: boolean | null;
  seed: number;
  statement: string;
}

export interface FirstVariation {
  varied_latex: string[];
  delta_integrand_latex: string;
  stages: IbpStage[];
  equations: {
    function: string;
    equation_latex: string;
    expression_latex: string;
    order: number;
  }[];
  null_lagrangian: boolean;
  boundary_form_latex: string;
  endpoint_terms_latex: string;
  boundary: BoundaryEntry[];
  first_variation_latex: string;
  stationarity_latex: string[];
  notes: string[];
  checks: VariationChecks;
  /** The engine's `Verification` of the claimed variation; the panel uses `checks`. */
  verification: Record<string, unknown>;
}

export interface PathCondition {
  function: string;
  order: number;
  end: VariationEnd;
  required: string;
  found: string;
  met: boolean;
}

export interface TrialResult {
  label: string;
  parameter_value: string | null;
  action: string;
  integral: string;
  endpoint_terms: string;
  quadrature_error: number;
  route: "analytic" | "sampled";
  conditions: PathCondition[];
  half_data_difference: number | null;
}

export interface VariationEvaluation {
  lower: string;
  upper: string;
  parameters: Record<string, string>;
  trials: TrialResult[];
  ordering: string[] | null;
  notes: string[];
  statement: string;
}

export interface PerturbationResult {
  label: string;
  kind: "bubble" | "free_end" | "generic" | "given";
  admissible: boolean | null;
  interior_part: string;
  boundary_part: string;
  analytic: string;
  comparison: Comparison;
}

export interface VariationVerification {
  route: "integrated" | "direct";
  perturbations: PerturbationResult[];
  verdict: "passed" | "failed" | "inconclusive";
  notes: string[];
  statement: string;
}

export type SecondFact =
  | "not_a_weak_minimum"
  | "not_a_weak_maximum"
  | "legendre_violated_for_minimum"
  | "legendre_violated_for_maximum"
  | "no_negative_direction_found"
  | "no_positive_direction_found";

export interface SecondNumeric {
  stationary: "consistent" | "violated" | "not_tested";
  stationary_residual: number;
  legendre: { points: number; smallest_eigenvalue: number; largest_eigenvalue: number } | null;
  directions: {
    label: string;
    second_variation: string;
    sign: "negative" | "zero" | "positive";
    certified: boolean;
  }[];
  ritz: {
    dimension: number;
    eigenvalues: number[];
    negative: number;
    zero: number;
    positive: number;
  } | null;
  facts: SecondFact[];
  not_established: string[];
  statement: string;
}

export interface SecondVariation {
  convention_latex: string;
  quadratic_form_latex: string;
  endpoint_form_latex: string;
  legendre_matrix_latex: string;
  checks: VariationChecks;
  numeric: SecondNumeric | null;
}

export interface MethodReport {
  method: "hermite_action" | "chebyshev_euler_lagrange";
  ran: boolean;
  converged: boolean;
  levels: number[];
  successive_differences: number[];
  newton_iterations: number[];
  note: string;
}

export interface StationaryPath {
  method: "hermite_action" | "chebyshev_euler_lagrange";
  samples: {
    t: number[];
    values: Record<string, number[]>;
    slopes: Record<string, number[]>;
  };
  action: string;
  action_quadrature_error: number;
  stationarity: {
    residual: number;
    trials: number;
    tolerance: number;
    verdict: "stationary_within_tolerance" | "not_stationary";
  };
}

export interface VariationSolution {
  paths: StationaryPath[];
  methods: MethodReport[];
  agreement: number | null;
  confirmed_by_both: boolean;
  distinct_paths: boolean;
  second: SecondNumeric | null;
  notes: string[];
  statement: string;
}

export interface VariationReport {
  status: Status;
  message: string;
  action: VariationAction;
  analysis: VariationAnalysis | null;
  first: FirstVariation | null;
  evaluation: VariationEvaluation | null;
  verification: VariationVerification | null;
  second: SecondVariation | null;
  solution: VariationSolution | null;
  notes: string[];
  diagnostics: { functions: number; highest_order: number; stages: number; duration_ms: number };
  duration_ms: number;
  calculation_id?: string | null;
}
