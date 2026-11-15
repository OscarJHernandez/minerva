import type { Assumptions, Status } from "./types";

/** Mirrors `minerva_engine.ode.models`. Equations are strings in the chosen `format`. */

export interface OdeCondition {
  unknown: string;
  order: number;
  at: string;
  value: string;
}

export interface OdeInput {
  equations: string[];
  format: "latex" | "text";
  unknowns: string[];
  independent: string | null;
  assumptions: Assumptions;
  conditions: (OdeCondition | string)[];
}

export interface OdeAnalyzeInput extends OdeInput {
  detail: "structure" | "qualitative";
  parameters: Record<string, number>;
  /** [x_min, x_max, y_min, y_max] of the phase plane. */
  window: number[] | null;
}

export type SolveMethod = "closed_form" | "series" | "perturbation";

export interface OdeSolveInput extends OdeInput {
  method: SolveMethod;
  order: number;
  small_parameter: string | null;
  equilibrium: string | null;
}

export type NumericMethod = "auto" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA";

export interface OdeNumericInput extends OdeInput {
  kind: "ivp" | "bvp";
  parameters: Record<string, number>;
  /** ivp: where the interval ends. bvp: not used (the two points of the conditions are the ends). */
  t_end: number | null;
  /** A verified closed form to compare with: one right-hand side (text) per unknown. */
  compare_with: string[] | null;
  method: NumericMethod;
  rtol: number;
  atol: number;
  samples: number;
}

export interface OdeInterpretation {
  unknowns: string[];
  independent: string;
  parameters: string[];
  orders: Record<string, number>;
  equations_latex: string[];
  conditions_latex: string[];
  notes: string[];
}

export interface OdeStructure {
  kind: "ode" | "system";
  n_unknowns: number;
  n_equations: number;
  orders: Record<string, number>;
  order: number;
  state_dimension: number;
  linear: boolean;
  homogeneous: boolean | null;
  constant_coefficients: boolean | null;
  autonomous: boolean;
  solved_for_highest: boolean;
  position_only: boolean;
}

export type OdeMethodId =
  | "symbolic_closed_form"
  | "numerical_ivp"
  | "numerical_bvp"
  | "power_series"
  | "regular_perturbation"
  | "poincare_lindstedt"
  | "multiple_scales"
  | "equilibrium_analysis"
  | "phase_plane";

export interface MethodSuggestion {
  id: OdeMethodId;
  title: string;
  applicable: boolean;
  available: boolean;
  rank: number | null;
  reason: string;
  requires: string[];
  limitations: string[];
  tool: "ode_solve_symbolic" | "ode_solve_numerical" | "ode_analyze" | null;
  action: string | null;
}

export type ProblemKind =
  "general" | "initial_value" | "boundary_value" | "incomplete" | "overdetermined";

export type Classification =
  | "stable_node"
  | "stable_spiral"
  | "unstable_node"
  | "unstable_spiral"
  | "saddle"
  | "centre"
  | "non_hyperbolic"
  | "undecided";

export interface Equilibrium {
  point_latex: string[];
  point_numeric: number[];
  jacobian_latex: string;
  eigenvalues_latex: string[];
  eigenvalues_numeric: string[];
  characteristic_latex: string;
  classification: Classification;
  classification_text: string;
  decided: boolean;
}

export interface DirectionField {
  window: number[];
  x: number[];
  y: number[];
  dx: number[];
  dy: number[];
  speed: number[];
}

export interface OdeQualitative {
  status: Status;
  message: string;
  states_latex: string[];
  field_latex: string[];
  equilibria: Equilibrium[];
  complete: boolean;
  jacobian_checked: boolean;
  direction_field: DirectionField | null;
  notes: string[];
  local_note: string;
}

export interface OdeDiagnostics {
  trace_id: string | null;
  stages: { name: string; ms: number }[];
  failure_reason: string | null;
}

export interface OdeAnalysis {
  status: Status;
  message: string;
  interpretation: OdeInterpretation | null;
  structure: OdeStructure | null;
  problem_kind: ProblemKind | null;
  conditions_needed: string[];
  classes: string[];
  solved_latex: string[];
  restrictions_latex: string[];
  methods: MethodSuggestion[];
  qualitative: OdeQualitative | null;
  notes: string[];
  diagnostics: OdeDiagnostics;
  duration_ms: number;
  calculation_id?: string | null;
}

export interface OdeVerification {
  points_checked: number;
  points_agreed: number;
  points_skipped: number;
  max_residual: number | null;
  symbolic_zero: boolean | null;
  independent_constants: number | null;
  passed: boolean;
  note: string;
}

export interface ConditionCheck {
  latex: string;
  satisfied: boolean;
  residual: number | null;
}

export interface ExpansionCheck {
  name: "initial_conditions" | "residual_order" | "boundedness" | "frequency_order";
  passed: boolean;
  value: number | null;
  note: string;
}

export interface OdeExpansion {
  kind: "taylor" | "regular_perturbation" | "poincare_lindstedt";
  order: number;
  point_latex: string | null;
  small_latex: string | null;
  secular_orders: number[];
  checks: ExpansionCheck[];
  validity: string;
}

export interface OdeSolutionItem {
  kind: "explicit" | "implicit" | "unevaluated" | "series" | "perturbation";
  expansion: OdeExpansion | null;
  equations_latex: string[];
  rhs_text: string[];
  constants: string[];
  family: "general" | "particular" | "partial";
  evaluated: boolean;
  verification: OdeVerification;
  conditions: ConditionCheck[];
  restrictions_latex: string[];
}

export interface OdeSolution {
  status: Status;
  message: string;
  method: SolveMethod | null;
  interpretation: OdeInterpretation | null;
  solutions: OdeSolutionItem[];
  verification: OdeVerification | null;
  dropped: number;
  notes: string[];
  diagnostics: OdeDiagnostics;
  duration_ms: number;
  calculation_id?: string | null;
}

export interface NumericCheck {
  name:
    | "initial_conditions"
    | "independent_method"
    | "tolerance_refinement"
    | "defect"
    | "first_integral"
    | "boundary_conditions"
    | "residual"
    | "mesh_refinement"
    | "shooting";
  status: "passed" | "failed" | "skipped";
  value: number | null;
  tolerance: number | null;
  note: string;
}

export interface NumericVerification {
  checks: NumericCheck[];
  passed: boolean;
  independent_method: string | null;
  refinement_errors: number[];
  first_integral_latex: string | null;
}

export interface OdeSeries {
  label_latex: string;
  unknown: string;
  order: number;
  values: number[];
}

export type NumericWarning =
  "stiffness_indicated" | "step_collapse" | "not_converged" | "blow_up_suspected" | "domain_left";

export interface OdeComparison {
  status: Status;
  message: string;
  labels_latex: string[];
  closed_form: number[][];
  max_difference: number | null;
  worst_at: number | null;
}

export interface OdeNumericReport {
  status: Status;
  message: string;
  interpretation: OdeInterpretation | null;
  method_requested: NumericMethod | null;
  method_used: string | null;
  method_reason: string;
  t_start: number | null;
  t_end: number | null;
  parameters: Record<string, number>;
  initial: { label_latex: string; value: number }[];
  rtol: number | null;
  atol: number | null;
  steps: number | null;
  evaluations: number | null;
  t: number[];
  series: OdeSeries[];
  verification: NumericVerification | null;
  comparison: OdeComparison | null;
  warnings: NumericWarning[];
  notes: string[];
  diagnostics: OdeDiagnostics;
  duration_ms: number;
  calculation_id?: string | null;
}
