/** Types mirroring the Minerva API (see /docs on the API for the authoritative schema). */

export type Status =
  | "ok"
  | "unchanged"
  | "not_verified"
  | "conditions_not_satisfied"
  | "unsupported"
  | "parse_error"
  | "resource_limit"
  | "internal_error";

export type Assumption =
  | "real"
  | "positive"
  | "negative"
  | "nonnegative"
  | "nonpositive"
  | "nonzero"
  | "integer"
  | "rational";

export const ASSUMPTIONS: readonly Assumption[] = [
  "real",
  "positive",
  "negative",
  "nonnegative",
  "nonpositive",
  "nonzero",
  "integer",
  "rational",
];

export type Assumptions = Record<string, Assumption[]>;

export interface Verification {
  symbolic_zero: boolean;
  numeric_points_checked: number;
  numeric_points_agreed: number;
  passed: boolean;
  note: string;
}

export type SimplifyGoal = "simplify" | "expand" | "factor" | "rewrite";

/** One rewrite in a chain: a handbook identity (`kind: "rule"`) or a SymPy move (`"builtin"`). */
export interface TransformStep {
  kind: "rule" | "builtin";
  name: string;
  direction: "forward" | "reverse" | null;
  matched_latex: string | null;
  before_latex: string;
  after_latex: string;
  verification: Verification;
}

export interface TransformChain {
  result_latex: string;
  result_text: string;
  steps: TransformStep[];
  verification: Verification;
}

/** An identity that matches but whose conditions are not provable from the assumptions. Not applied. */
export interface BlockedMove {
  rule_id: string;
  direction: "forward" | "reverse";
  matched_latex: string;
  unmet_conditions: RuleCondition[];
}

export interface SimplifyResult {
  status: Status;
  input_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  verification: Verification | null;
  /** Points the result excludes that the input did not (e.g. `x \\neq 1`). */
  restrictions_latex?: string[];
  restrictions_complete?: boolean;
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
  // Present for results computed by the chain search; saved older results have none of these.
  goal?: SimplifyGoal;
  steps?: TransformStep[];
  alternatives?: TransformChain[];
  blocked?: BlockedMove[];
  entries?: Record<string, EntrySummary>;
  search_complete?: boolean;
}

export interface DiffVariable {
  name: string;
  order: number;
}

export const MAX_DERIVATIVE_ORDER = 10;

export interface DifferentiateResult {
  status: Status;
  input_latex: string | null;
  variables: DiffVariable[];
  total_order: number;
  notation_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  steps: { variable: string; order_so_far: number; latex: string }[];
  verification: {
    method: string;
    directions: ("real" | "imaginary")[];
    steps_checked: number;
    points_checked: number;
    points_agreed: number;
    passed: boolean;
    note: string;
  } | null;
  notes: string[];
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}

export const MAX_SERIES_ORDER = 12;

export interface SeriesResult {
  status: Status;
  input_latex: string | null;
  variable: string | null;
  point_latex: string | null;
  order: number;
  result_latex: string | null;
  result_text: string | null;
  remainder_latex: string | null;
  terms: { power: number; coefficient_latex: string }[];
  verification: {
    coefficient_method: string;
    remainder_method: string;
    directions: ("real" | "imaginary")[];
    parameter_sets_checked: number;
    coefficients_passed: boolean;
    remainder_passed: boolean;
    radius: number | null;
    passed: boolean;
    note: string;
  } | null;
  measured_errors: { distance: number; absolute_error: string; relative_error: string | null }[];
  notes: string[];
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}
export interface IntegrateResult {
  status: Status;
  input_latex: string | null;
  variable: string | null;
  definite: boolean;
  lower_latex: string | null;
  upper_latex: string | null;
  numeric: string | null;
  conditions_latex: string[];
  quadrature: {
    method: string;
    points_checked: number;
    points_agreed: number;
    passed: boolean;
    note: string;
  } | null;
  notation_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  verification: {
    method: string;
    directions: ("real" | "imaginary")[];
    steps_checked: number;
    branches_checked: number;
    points_checked: number;
    points_agreed: number;
    passed: boolean;
    symbolic_zero: boolean | null;
    note: string;
  } | null;
  notes: string[];
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}

export type MatrixAction = "evaluate" | "simplify" | "differentiate" | "integrate" | "evolve";
export type MatrixResultKind =
  "explicit" | "expression" | "equation" | "representation" | "numeric";

/** A declared matrix: a size (abstract) or explicit entries (LaTeX expressions). */
export interface MatrixDecl {
  name: string;
  rows?: number | null;
  cols?: number | null;
  entries?: string[][] | null;
  /** The variable this matrix is a function of, A(t); absent for a constant matrix. */
  depends_on?: string | null;
}

export interface MatrixRequest {
  matrices: MatrixDecl[];
  commuting: [string, string][];
  invertible: string[];
  /** Time-dependent matrices declared to commute with themselves at all times. */
  self_commuting?: string[];
  action: MatrixAction;
  variable: string | null;
  lower: string | null;
  upper: string | null;
}

export interface MatrixStep {
  method: string;
  title: string;
  before_latex: string | null;
  after_latex: string | null;
  note: string;
}

export interface MatrixBlocked {
  method: string;
  title: string;
  needs: string;
  latex: string | null;
}

export interface MatrixVerification {
  method: string;
  points_checked: number;
  points_agreed: number;
  passed: boolean;
  symbolic_zero: boolean | null;
  symbolic_method: string;
  note: string;
}

export type MatrixSource = "not_used" | "declared" | "from_entries" | "mixed";

/** How a result was reached. Contains no names, entries, expressions or bounds. */
export interface MatrixDiagnostics {
  trace_id: string | null;
  method: string | null;
  methods_tried: { id: string; outcome: "selected" | "blocked" | "unsupported" | "no_match" }[];
  dimensions: [number, number][];
  matrices: number;
  explicit: number;
  time_dependent: number;
  commutation: MatrixSource;
  invertibility: MatrixSource;
  stages: { name: string; ms: number }[];
  digits: number | null;
  tolerance: number | null;
  max_residual: number | null;
  points_checked: number;
  points_agreed: number;
  verification: string;
  fallbacks: string[];
  failure_reason: string | null;
}

export interface MatrixCalculusResult {
  status: Status;
  action: MatrixAction | null;
  input_latex: string | null;
  structure: string;
  features: string[];
  method: string | null;
  method_title: string | null;
  result_kind: MatrixResultKind | null;
  result_latex: string | null;
  result_entries: string[][] | null;
  equation_latex: string | null;
  evaluated: boolean;
  assumptions_used: string[];
  steps: MatrixStep[];
  blocked: MatrixBlocked[];
  verification: MatrixVerification | null;
  notes: string[];
  message: string;
  duration_ms: number;
  diagnostics?: MatrixDiagnostics | null;
  calculation_id?: string | null;
  /** Only in saved history items: the declarations the result was computed from. */
  request?: MatrixRequest;
}

export type Applicability = "applicable" | "conditions_not_satisfied" | "not_verified" | "no_match";

export interface RuleCondition {
  variable: string;
  assumption: Assumption;
}

export interface RuleApplication {
  rule_id: string;
  applicability: Applicability;
  matched_subexpression_latex: string | null;
  rewritten_latex: string | null;
  unmet_conditions: RuleCondition[];
  verification: Verification | null;
  restrictions_latex?: string[];
  restrictions_complete?: boolean;
  message: string;
}

export interface Reference {
  source: string;
  locator?: string | null;
  url?: string | null;
}

export type Verified = "symbolic" | "numeric" | "structural" | "dimensional" | "stated";

export interface EntrySummary {
  id: string;
  title: string;
  kind: string;
  chapter: string;
  statement_latex: string;
  validity: string;
  verification: Verified;
  reference: Reference | null;
}

export interface IdentityMatch {
  entry: EntrySummary;
  application: RuleApplication;
}

export interface IdentitiesResult {
  status: Status;
  input_latex: string | null;
  message: string;
  matches: IdentityMatch[];
  candidates_considered: number;
  duration_ms: number;
  calculation_id?: string | null;
}

/** One handbook identity applied to an expression (recomputed and verified by the server). */
export interface ApplyIdentityResult {
  status: Status;
  input_latex: string | null;
  rule_id: string;
  entry: EntrySummary;
  matched_subexpression_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  unmet_conditions: RuleCondition[];
  verification: Verification | null;
  restrictions_latex?: string[];
  restrictions_complete?: boolean;
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}

export type Operation =
  | "simplify"
  | "identities"
  | "differentiate"
  | "series"
  | "integrate"
  | "apply_identity"
  | "matrix_calculus";

export interface HistoryItem {
  id: string;
  operation: Operation;
  input: string;
  format: "latex" | "text";
  assumptions: Assumptions;
  status: Status;
  result: SimplifyResult | IdentitiesResult | ApplyIdentityResult;
  created_at: string;
  derivation_id?: string | null;
  parent_id?: string | null;
}

export interface DerivationSummary {
  id: string;
  updated_at: string;
  steps: number;
  first_input: string | null;
  first_format: "latex" | "text" | null;
}

export interface Derivation {
  id: string;
  updated_at: string;
  steps: HistoryItem[];
}

export interface User {
  id: string;
  email: string;
}

export interface EntryOut extends EntrySummary {
  section: string | null;
  tags: string[];
  keywords: string[];
  symbols: { name: string; meaning: string; unit: string | null }[];
  constant: { value: string; uncertainty: string | null; unit: string } | null;
  notes: string;
  applicable_as_rule: boolean;
  conditions: RuleCondition[];
}

export interface TocChapter {
  id: string;
  book: "mathematics" | "physics";
  title: string;
  summary: string;
  entry_count: number;
  sections: { id: string; title: string; entry_count: number }[];
}
