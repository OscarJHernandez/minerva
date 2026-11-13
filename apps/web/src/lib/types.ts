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

export type DistributionAction = "simplify" | "integrate" | "differentiate";
export type DistributionEndpoint = "half" | "include" | "exclude";
export type DistributionReason =
  | "unsupported_distribution"
  | "singular_root"
  | "assumptions_insufficient"
  | "verification_failed"
  | "integration_domain_ambiguous"
  | "notation_unconfirmed"
  | "resource_limit"
  | "parse_error";

/** Names that may be a Dirac delta or a step function; ordinary symbols unless the user says so. */
export type DistributionNotationName = "delta" | "theta" | "Theta";
export type DistributionReading = "dirac_delta" | "heaviside" | "symbol";
export type DistributionNotation = Partial<Record<DistributionNotationName, DistributionReading>>;

export interface DistributionQuestion {
  name: DistributionNotationName;
  candidate: "dirac_delta" | "heaviside";
}

export interface DistributionRequest {
  action: DistributionAction;
  variable: string;
  lower: string | null;
  upper: string | null;
  endpoint: DistributionEndpoint | null;
  /** What the user answered to "is that a Dirac delta?"; never set by Minerva or the model. */
  notation?: DistributionNotation;
}

/** What a chat card posts back: the whole call, with the user's answer. */
export interface DistributionAnswer extends DistributionRequest {
  latex: string;
  assumptions: Assumptions;
}

export interface DistributionCondition {
  latex: string;
  established: boolean;
  how: string;
}

export interface DistributionStep {
  entry_id: string;
  title: string;
  before_latex: string;
  after_latex: string;
  conditions: DistributionCondition[];
}

export interface DistributionVerification {
  method: string;
  test_functions: string[];
  pairings_checked: number;
  pairings_agreed: number;
  parameter_samples: number;
  widths: number[];
  convergence_observed: boolean;
  conditions_established: boolean;
  passed: boolean;
  note: string;
}

export interface DistributionResult {
  status: Status;
  action: DistributionAction | null;
  reason: DistributionReason | null;
  input_latex: string | null;
  variable: string | null;
  lower_latex: string | null;
  upper_latex: string | null;
  endpoint: DistributionEndpoint | null;
  /** Names the user has to answer for (reason `notation_unconfirmed`), and the readings applied. */
  questions?: DistributionQuestion[];
  notation?: DistributionNotation;
  result_latex: string | null;
  result_text: string | null;
  equality: "distributional" | "value" | null;
  steps: DistributionStep[];
  blocked: { entry_id: string; reason: DistributionReason; detail: string }[];
  verification: DistributionVerification | null;
  diagnostics: {
    reason: DistributionReason | null;
    events: {
      stage: string;
      status: "ok" | "failed";
      duration_ms: number;
      reason: string | null;
    }[];
  };
  notes: string[];
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
  /** Only in saved history items: the request the result was computed from. */
  request?: DistributionRequest;
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

export type Verified =
  "symbolic" | "numeric" | "structural" | "dimensional" | "distributional" | "stated";

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

/** What a controlled step did and why it ended as it did: durations, counts and codes only. */
export interface Diagnostics {
  parse_ms: number;
  candidates_ms: number;
  transform_ms: number;
  verify_ms: number;
  candidates: number;
  rejected: Record<string, number>;
  verification_method: string;
  verification_outcome: "passed" | "failed" | "not_run";
  symbolic_zero: boolean;
  restrictions_outcome: string;
  assumptions_declared: number;
  premises: number;
  /** Only with the local debug setting: rejected candidates, which may contain expressions. */
  detail: string[];
}

/** A relation the user supplied. Minerva does not check it: results that use it are conditional. */
export interface Premise {
  latex: string;
  text: string;
}

export interface StepRecord {
  action: "substitute" | "builtin" | "rule" | "collect" | "peel" | "polynomial";
  name: string;
  direction: "forward" | "reverse" | null;
  matched_latex: string | null;
  before_latex: string;
  after_latex: string;
  /** A substitution before the arithmetic done on construction: x^2 + (1 - x^2). */
  display_latex: string | null;
  evidence: "identity" | "conditional";
  given: string[];
  conditions_latex: string[];
  verification: Verification | null;
  note: string;
}

/** A part of an expression that can be chosen for a step. */
export interface Subterm {
  index: number;
  path: number[];
  args: number[] | null;
  latex: string;
  text: string;
  highlight_latex: string;
  occurrence: number;
  occurrences: number;
  blocked: boolean;
}

/** Substitute, rewrite one part, or the list of parts that can be chosen. */
export interface StepResult {
  status: Status;
  operation: "rewrite" | "substitute" | "subterms";
  input_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  steps: StepRecord[];
  evidence: "identity" | "conditional";
  given: Premise[];
  verification: Verification | null;
  restrictions_latex: string[];
  restrictions_complete: boolean;
  conditions_latex: string[];
  unmet_conditions: RuleCondition[];
  reason: string | null;
  order_dependent: boolean | null;
  subterms: Subterm[];
  truncated: boolean;
  diagnostics: Diagnostics;
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
  entry?: EntrySummary | null;
  /** What was asked, saved with the step so a reload shows it. */
  request?: Record<string, unknown>;
}

export interface Solution {
  latex: string;
  text: string;
  equation_text: string;
  conditions_latex: string[];
  branch: string;
  verification: Verification | null;
}

/** The solutions of an equation for one variable, with the conditions under which each holds. */
export interface IsolateResult {
  status: Status;
  equation_latex: string | null;
  variable: string | null;
  method: "inversion" | "polynomial" | null;
  steps: StepRecord[];
  solutions: Solution[];
  conditions_latex: string[];
  excluded_latex: string[];
  exceptional: string[];
  verification: Verification | null;
  complete: boolean;
  completeness_note: string;
  assumption_note: string;
  reason: string | null;
  diagnostics: Diagnostics;
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
  request?: Record<string, unknown>;
}

export type Operation =
  | "simplify"
  | "identities"
  | "differentiate"
  | "series"
  | "integrate"
  | "apply_identity"
  | "matrix_calculus"
  | "geometry"
  | "distribution"
  | "rewrite"
  | "substitute"
  | "isolate";

export interface HistoryItem {
  id: string;
  operation: Operation;
  input: string;
  format: "latex" | "text";
  assumptions: Assumptions;
  status: Status;
  result: SimplifyResult | IdentitiesResult | ApplyIdentityResult | StepResult | IsolateResult;
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

// --- differential geometry ------------------------------------------------------------------

export type GeometryQuantity =
  | "inverse_metric"
  | "determinant"
  | "christoffel"
  | "riemann"
  | "ricci"
  | "ricci_scalar"
  | "einstein"
  | "kretschmann"
  | "ricci_squared"
  | "geodesics"
  | "covariant_derivative"
  | "raise_index"
  | "lower_index";

export type GeometryVariance = "up" | "down";

export interface GeometryFunction {
  name: string;
  args: string[];
}

export interface GeometryTensor {
  variance: GeometryVariance[];
  components: string[];
}

/** The whole geometric problem. It is sent with every request: nothing is remembered server-side. */
export interface GeometryRequest {
  coordinates: string[];
  metric: string[][];
  format?: "latex" | "text";
  assumptions: Assumptions;
  functions: GeometryFunction[];
  quantities: GeometryQuantity[];
  tensor: GeometryTensor | null;
  tensor_index: number;
}

export interface GeometryComponent {
  indices: number[];
  label_latex: string;
  latex: string;
  text: string;
}

export interface GeometryCheck {
  name: string;
  kind: "numeric" | "symbolic";
  passed: boolean | null;
  points_checked: number;
  points_agreed: number;
  note: string;
}

export interface GeometryVerification {
  method: string;
  numeric_passed: boolean;
  symbolic_zero: boolean | null;
  checks: GeometryCheck[];
  note: string;
}

export interface GeometryQuantityResult {
  kind: GeometryQuantity;
  status: Status;
  title: string;
  variance: GeometryVariance[];
  components: GeometryComponent[];
  all_zero: boolean;
  scalar_latex: string | null;
  scalar_text: string | null;
  symmetry_note: string;
  verification: GeometryVerification | null;
  notes: string[];
  message: string;
}

export interface GeometryStage {
  stage: string;
  duration_ms: number;
  ops: number | null;
  components: number | null;
  status: "ok" | "failed" | "skipped";
  failure: string | null;
}

export interface GeometryResult {
  status: Status;
  dimension: number;
  coordinates_latex: string[];
  metric_latex: string | null;
  signature: string | null;
  signature_note: string;
  conventions: string[];
  assumptions_used: string[];
  restrictions_latex: string[];
  restrictions_complete: boolean;
  quantities: GeometryQuantityResult[];
  diagnostics: GeometryStage[];
  peak_memory_mb: number | null;
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}
