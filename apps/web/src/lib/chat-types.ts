import type {
  DifferentiateResult,
  SeriesResult,
  EntrySummary,
  IdentitiesResult,
  GeometryRequest,
  GeometryResult,
  DistributionAnswer,
  DistributionResult,
  ExpectationResult,
  IntegrateResult,
  IsolateResult,
  MatrixCalculusResult,
  MatrixRequest,
  SimplifyResult,
  StatModelResult,
  Status,
  StepResult,
  VectorRequest,
  VectorResult,
  SumsResult,
} from "./types";
import type { OdeAnalysis, OdeInput, OdeNumericReport, OdeSolution } from "./ode-types";
import type { FunctionalRequest, VariationReport } from "./variation-types";
import type { StatAnswers, StatMechResult, StatToolArgs } from "./statmech-types";
import type { TelescopingResult } from "./telescoping-types";

export interface ExpansionReport {
  status: Status;
  lagrangian_latex: string | null;
  coordinates: string[];
  equilibrium: { name: string; latex: string; name_latex: string }[];
  order: number;
  equilibrium_verification: {
    symbolic_zero: boolean;
    numeric_points_checked: number;
    numeric_points_agreed: number;
    passed: boolean;
    note: string;
  } | null;
  equations: {
    coordinate: string;
    full_latex: string;
    terms: { degree: number; latex: string }[];
    verification: SeriesResult["verification"];
  }[];
  stiffness_latex: string | null;
  velocity_latex: string | null;
  characteristic_latex: string | null;
  modes: {
    omega_squared_latex: string;
    sign: "positive" | "negative" | "zero" | "unknown";
    omega_latex: string | null;
    multiplicity: number;
    eigenvector_latex: string | null;
    verified: boolean;
  }[];
  notes: string[];
  message: string;
  duration_ms: number;
}

export interface PerturbationReport {
  status: Status;
  lagrangian_latex: string | null;
  coordinates: string[];
  equilibrium: { name: string; latex: string; name_latex: string }[];
  mode: "parameter" | "amplitude" | null;
  parameters: string[];
  small_latex: string | null;
  amplitude_latex: string | null;
  order: number;
  linear_frequency_latex: string | null;
  frequency_latex: string | null;
  period_ratio_latex: string | null;
  solution_latex: string | null;
  orders: { order: number; frequency_latex: string | null; shape_latex: string }[];
  verification: {
    residual_method: string;
    residual_passed: boolean;
    parameter_sets_checked: number;
    initial_conditions_passed: boolean;
    bounded: boolean;
    passed: boolean;
    note: string;
  } | null;
  comparison: {
    status: Status;
    message: string;
    epsilon: number;
    predicted_order: number;
    t: number[];
    numerical: number[];
    perturbative: number[];
    max_error: number | null;
    max_error_half: number | null;
    observed_order: number | null;
    scaling: string;
    notes: string[];
  } | null;
  notes: string[];
  message: string;
  duration_ms: number;
}

/** What a perturbation solution was computed from, and what the comparison form sends back. */
export interface PerturbationRequest {
  lagrangian: string;
  coordinates: string[];
  time: string;
  assumptions: Record<string, string[]>;
  small_parameter: string | null;
  equilibrium: string;
  order: number;
  compare?: {
    parameters: Record<string, number>;
    epsilon: number;
    amplitude?: number;
    t_end: number;
  };
}

export interface LagrangianReport {
  status: Status;
  lagrangian_latex: string | null;
  coordinates: string[];
  parameters: string[];
  time_dependent: boolean;
  equations: {
    coordinate: string;
    latex: string;
    verification: {
      numeric_points_checked: number;
      numeric_points_agreed: number;
      passed: boolean;
    } | null;
  }[];
  momenta: { coordinate: string; latex: string }[];
  accelerations: { coordinate: string; latex: string }[];
  conserved: {
    name: string;
    kind: "cyclic" | "energy" | "rotation";
    expression_latex: string;
    reason: string;
    verified: boolean;
    verification: {
      symbolic_zero: boolean;
      numeric_points_checked: number;
      numeric_points_agreed: number;
      passed: boolean;
      note: string;
    } | null;
  }[];
  tested: string[];
  features: string[];
  notes: string[];
  message: string;
  duration_ms: number;
}

export interface InvariantReport {
  status: Status;
  lagrangian_latex: string | null;
  coordinates: string[];
  angles: string[];
  degree: number;
  power_bound: number;
  harmonics: number;
  unknowns: number;
  ansatz: string;
  known: string[];
  found: {
    name: string;
    expression_latex: string;
    verified: boolean;
    verification: {
      symbolic_zero: boolean;
      numeric_points_checked: number;
      numeric_points_agreed: number;
      passed: boolean;
    } | null;
  }[];
  notes: string[];
  features: string[];
  message: string;
  duration_ms: number;
}

export interface DiscoveredConstant {
  id: string;
  name: string;
  expression_latex: string;
  kind: "known" | "new";
  source: string;
  tier: number | null;
  momentum_degree: number;
  independence: "independent" | "dependent" | "unresolved";
  verified: boolean;
  verification: {
    symbolic_zero: boolean;
    numeric_points_checked: number;
    numeric_points_agreed: number;
    passed: boolean;
  } | null;
  reason: string;
}

export interface DiscoveryTier {
  index: number;
  code: string;
  description: string;
  unknowns: number;
  equations: number;
  rank: number;
  nullity: number;
  new_directions: number;
  accepted: number;
  rejected: number;
  duration_ms: number;
  outcome: "searched" | "deadline" | "unsupported";
}

/** Result of the adaptive search for conserved quantities of a Lagrangian or Hamiltonian. */
export interface DiscoveryReport {
  status: Status;
  system: "lagrangian" | "hamiltonian";
  function_latex: string | null;
  coordinates: string[];
  angles: string[];
  budget: string;
  constants: DiscoveredConstant[];
  tiers: DiscoveryTier[];
  unexplored: string[];
  stop_reason: string;
  cause: string;
  independence: {
    constants: number;
    rank_lower: number;
    rank_upper: number;
    rank_proved: boolean;
    max_possible: number | null;
    relations: { latex: string; factors: number; exact: boolean }[];
    notes: string[];
  } | null;
  parameter_conditions: string[];
  limitations: string[];
  notes: string[];
  message: string;
  peak_memory_mb: number;
  duration_ms: number;
}

/** What the Lagrangian card sends to the numerical-solution endpoint (the engine's request model). */
export interface MotionRequest {
  lagrangian: string;
  coordinates: string[];
  time: string;
  assumptions: Record<string, string[]>;
  parameters: Record<string, number>;
  initial_positions: number[];
  initial_velocities: number[];
  t_start: number;
  t_end: number;
  sweep: { name: string; values: number[] } | null;
  samples?: number;
}

export interface MotionVerification {
  method_difference: number | null;
  reversal_error: number | null;
  max_drift: number | null;
  drift_checked: number;
  steps: number;
  passed: boolean;
  notes: string[];
}

export interface MotionRun {
  status: Status;
  parameter_value: number | null;
  message: string;
  /** [coordinate][sample]; empty unless the run was verified. */
  q: number[][];
  v: number[][];
  conserved: {
    name: string;
    kind: string;
    expression_latex: string;
    values: number[];
    relative_drift: number;
  }[];
  verification: MotionVerification;
}

export interface MotionReport {
  status: Status;
  lagrangian_latex: string | null;
  coordinates: string[];
  accelerations: string[];
  parameters: Record<string, number>;
  initial_positions: number[];
  initial_velocities: number[];
  t_start: number;
  t_end: number;
  sweep_name: string | null;
  t: number[];
  runs: MotionRun[];
  notes: string[];
  message: string;
  duration_ms: number;
}

export type Closure = "determined" | "up_to_function" | "missing" | "none";

export interface QuantityReading {
  name: string;
  latex: string;
  meaning: string;
  given: string;
  dimension_text: string;
  dimension_latex: string;
}

export interface DimensionalReport {
  status: Status;
  quantities: QuantityReading[];
  rank: number;
  target: string | null;
  repeating: string[];
  groups: {
    label: string;
    latex: string;
    exponents: Record<string, string>;
    contains_target: boolean;
    dimensionless_verified: boolean;
  }[];
  closure: Closure | null;
  relation_latex: string | null;
  cannot_enter: string[];
  missing: {
    target_dimension_text: string;
    target_dimension_latex: string;
    unbalanced: string[];
    missing_dimension_text: string;
    missing_dimension_latex: string;
    verified: boolean;
    explanation: string;
    suggestions: {
      ids: string[];
      latex: string[];
      closure: Closure;
      relation_latex: string | null;
      group_latex: string | null;
      verified: boolean;
    }[];
  } | null;
  tested: string[];
  notes: string[];
  features: string[];
  message: string;
  duration_ms: number;
}

export interface SideReading {
  latex: string;
  dimension_text: string;
  dimension_latex: string;
}

export interface UnitsCheckReport {
  status: Status;
  equation_latex: string | null;
  consistent: boolean | null;
  sides: SideReading[];
  missing_factor_text: string | null;
  missing_factor_latex: string | null;
  problem: string | null;
  problem_terms: SideReading[];
  quantities: QuantityReading[];
  notes: string[];
  message: string;
  duration_ms: number;
}

export interface SolveReport {
  status: Status;
  equation_latex: string | null;
  variable: string | null;
  degree: number | null;
  symbolic_coefficients: boolean;
  factors: {
    factor_latex: string;
    multiplicity: number;
    degree: number;
    radicals: "yes" | "no" | "undetermined";
    radicals_reason: string;
    exact_roots_latex: string[];
  }[];
  roots: { numeric: string; real: boolean; multiplicity: number; exact_latex: string | null }[];
  verification: {
    roots_substituted_back: boolean;
    max_residual: string;
    count_matches_degree: boolean;
    vieta_holds: boolean;
    exact_roots_match_numeric: boolean | null;
  } | null;
  features: string[];
  notes: string[];
  message: string;
  duration_ms: number;
}

/**
 * Formulas only. from_tool: appears in a verified tool result or a machine-checked handbook entry.
 * quoted: appears in a tool result without being a verified result of its own (your input as read,
 *   a handbook entry that is only stated, a candidate that was not applied); the reason is in `note`.
 * checked: an equation the model wrote that Minerva tested numerically and found true.
 * conditional: false in general but true under extra assumptions (see `note`).
 * contradicted: an equation the model wrote that Minerva tested, as an identity or against
 *   definitions stated earlier (named in `note`), and found not to hold.
 * unchecked: written by the model; Minerva could not check it (a definition, coordinate change or
 *   equation to solve is not an identity; the reason is in `note`). Never a finding that it is wrong.
 */
export type FormulaStatus =
  "from_tool" | "quoted" | "checked" | "conditional" | "contradicted" | "unchecked";

export interface Segment {
  type: "text" | "math" | "display_math";
  value: string;
  status: FormulaStatus | null;
  /** Why the mark: what must hold (conditional), what it disagrees with (contradicted), or why it was not checked. */
  note?: string | null;
  /** The equation number of a displayed formula, counted over the whole conversation. */
  number?: number | null;
}

/** A tool result card; `number` is its equation number when it has a primary result formula. */
export type Card = CardBody & { number?: number | null };

type CardBody =
  | { id: string; tool: string; type: "simplify"; title: string; data: SimplifyResult }
  | { id: string; tool: string; type: "differentiate"; title: string; data: DifferentiateResult }
  | { id: string; tool: string; type: "series"; title: string; data: SeriesResult }
  | { id: string; tool: string; type: "integrate"; title: string; data: IntegrateResult }
  | {
      id: string;
      tool: string;
      type: "matrix";
      title: string;
      data: { result: MatrixCalculusResult; methods: EntrySummary[]; request: MatrixRequest };
    }
  | {
      id: string;
      tool: string;
      type: "geometry";
      title: string;
      data: { result: GeometryResult; request: GeometryRequest };
    }
  | {
      id: string;
      tool: string;
      type: "vector";
      title: string;
      data: { result: VectorResult; request: VectorRequest };
    }
  | {
      id: string;
      tool: string;
      type: "variation";
      title: string;
      data: {
        report: VariationReport;
        /** The functional as the model sent it: the card's form adds the user's numbers to it. */
        request: FunctionalRequest;
        functional_latex?: string;
      };
    }
  | {
      id: string;
      tool: string;
      type: "distribution";
      title: string;
      data: {
        result: DistributionResult;
        methods: EntrySummary[];
        /** The call as made (with the expression), so the question's buttons can ask again. */
        request: DistributionAnswer;
      };
    }
  | {
      id: string;
      tool: string;
      type: "statmech";
      title: string;
      data: {
        result: StatMechResult;
        /** The model's call, so the card's controls can send it again with the user's choices. */
        args: StatToolArgs;
        /** What the user already chose (cell, copies, numbers): shown, and sent along again. */
        answers: StatAnswers;
        inputs: { input_latex: string }[];
      };
    }
  | {
      id: string;
      tool: string;
      type: "sums";
      title: string;
      /** `request`: the call as made (the expression and its settings). */
      data: { result: SumsResult; request: Record<string, unknown> };
    }
  | {
      id: string;
      tool: string;
      type: "expectation";
      title: string;
      data: { result: ExpectationResult; request: Record<string, unknown> };
    }
  | {
      id: string;
      tool: string;
      type: "statistics";
      title: string;
      data: { result: StatModelResult; request: Record<string, unknown> };
    }
  | { id: string; tool: string; type: "identities"; title: string; data: IdentitiesResult }
  | {
      id: string;
      tool: string;
      type: "entries";
      title: string;
      data: { query: string; entries: EntrySummary[] };
    }
  | { id: string; tool: string; type: "entry"; title: string; data: EntrySummary }
  | {
      id: string;
      tool: string;
      type: "tensor";
      title: string;
      data: {
        status: string;
        input_text: string | null;
        result_text: string | null;
        result_latex: string | null;
        message: string;
      };
    }
  | {
      id: string;
      tool: string;
      type: "solve";
      title: string;
      data: { report: SolveReport; methods: EntrySummary[] };
    }
  | {
      id: string;
      tool: string;
      type: "lagrangian";
      title: string;
      data: {
        report: LagrangianReport;
        methods: EntrySummary[];
        request?: Pick<MotionRequest, "lagrangian" | "coordinates" | "time" | "assumptions">;
      };
    }
  | {
      id: string;
      tool: string;
      type: "expansion";
      title: string;
      data: {
        report: ExpansionReport;
        request?: Pick<MotionRequest, "lagrangian" | "coordinates" | "time" | "assumptions">;
      };
    }
  | {
      id: string;
      tool: string;
      type: "perturbation";
      title: string;
      data: { report: PerturbationReport; request?: PerturbationRequest };
    }
  | {
      id: string;
      tool: string;
      type: "invariants";
      title: string;
      data: { report: InvariantReport };
    }
  | {
      id: string;
      tool: string;
      type: "telescoping";
      title: string;
      data: TelescopingResult & { request: Record<string, unknown> };
    }
  | {
      id: string;
      tool: string;
      type: "discovery";
      title: string;
      data: { report: DiscoveryReport };
    }
  | {
      id: string;
      tool: string;
      type: "motion";
      title: string;
      data: { report: MotionReport; request: MotionRequest };
    }
  | {
      id: string;
      tool: string;
      type: "ode_analysis";
      title: string;
      data: { report: OdeAnalysis; request: OdeInput };
    }
  | {
      id: string;
      tool: string;
      type: "ode_solution";
      title: string;
      data: { report: OdeSolution; request: OdeInput };
    }
  | {
      id: string;
      tool: string;
      type: "ode_numeric";
      title: string;
      data: { report: OdeNumericReport; request: OdeInput };
    }
  | {
      id: string;
      tool: string;
      type: "dimensional";
      title: string;
      data: { report: DimensionalReport; methods: EntrySummary[] };
    }
  | {
      id: string;
      tool: string;
      type: "units";
      title: string;
      data: { report: UnitsCheckReport; methods: EntrySummary[] };
    }
  | {
      id: string;
      tool: string;
      type: "step";
      title: string;
      data:
        | { operation: "substitute" | "rewrite"; result: StepResult }
        | { operation: "isolate"; result: IsolateResult };
    }
  | {
      id: string;
      tool: string;
      type: "note";
      title: string;
      data: { message: string; groups?: string[] };
    }
  | { id: string; tool: string; type: "error"; title: string; data: { message: string } };

/** One item of an assistant turn in the order it happened: prose, or a tool result card. */
export type Block =
  | { kind: "text"; segments: Segment[]; card: null }
  | { kind: "card"; segments: Segment[]; card: Card };

/** What the engine established for one tool result (numbers and flags, never an expression). */
export interface Evidence {
  check:
    | "symbolic_zero"
    | "numeric"
    | "quadrature"
    | "series_checks"
    | "structural"
    | "dimensional"
    | "lookup"
    | "none";
  /** The result may be shown as an engine result. */
  accepted: boolean;
  reason: string;
  points_checked?: number | null;
  points_agreed?: number | null;
  symbolic_zero?: boolean | null;
  restrictions: number;
  restrictions_complete?: boolean | null;
}

/** A tool call as the model made it, and what the model read back. Kept for debugging. */
export interface TraceStep {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  for_model: string;
  /** Absent on messages stored before debug data existed. */
  duration_ms?: number | null;
  status?: string | null;
  /** The provider's tool-call id, replayed when history is rebuilt. */
  call_id?: string | null;
  /** Added with tracing; absent on older messages. */
  evidence?: Evidence | null;
  arguments_valid?: boolean | null;
  error_category?: string | null;
  engine_ms?: number | null;
  repeated?: boolean;
  result_id?: string | null;
  /** What the model wrote before `@rN` references were replaced; only when it differs. */
  model_arguments?: Record<string, unknown> | null;
  assumptions_dropped?: string[];
}

export interface LLMCall {
  ms: number;
  text_chars: number;
  tool_calls: number;
  first_token_ms?: number | null;
  finish_reason?: string | null;
  /** As reported by the provider; absent when it reports none. */
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  cost_usd?: number | null;
}

/** One content-free trace event (see docs/observability.md); fields vary by `event`. */
export interface TraceEvent {
  event: string;
  trace_id: string;
  seq: number;
  t_ms: number;
  step?: number | null;
  [field: string]: unknown;
}

export interface TurnDebug {
  request_id: string;
  api_version: string;
  mode: string;
  provider?: string;
  error_kind?: string | null;
  model: string;
  prompt_sha: string;
  max_steps: number;
  max_seconds: number;
  duration_ms: number;
  llm_calls: LLMCall[];
  claims: {
    value: string;
    status: string;
    detail: string;
    kind?: string | null;
    given?: string[];
  }[];
  error: string | null;
  /** Added with tracing; absent on older messages. */
  trace_id?: string;
  stop_reason?: string | null;
  max_tool_calls?: number;
  events?: TraceEvent[];
  events_dropped?: number;
  claim_check?: string | null;
  formulas?: number;
  formulas_with_evidence?: number;
  evidence_pct?: number | null;
  /** The model parameters this turn used, the tool groups it had loaded, and the request against the window. */
  params?: Partial<ModelParams>;
  groups?: string[];
  context_window?: number;
  prompt_estimate_tokens?: number;
  /** Messages of earlier turns that were left out so the request fit the window. */
  history_dropped?: number;
}

export type Toolset = "auto" | "core" | "all";

/** What a turn may be told about the model (the server's MINERVA_LLM_* settings, agent/params.py). */
export interface ModelParams {
  num_ctx: number;
  num_predict: number | null;
  temperature: number;
  top_p: number | null;
  top_k: number | null;
  think: boolean;
  history_messages: number;
  history_tokens: number | null;
  toolset: Toolset;
}

/** What the server tells the chat about parameters: defaults, and which this account may change. */
export interface ParamsInfo {
  defaults: ModelParams;
  editable: (keyof ModelParams)[];
  num_ctx_max: number;
}

export interface AssistantContent {
  text: string;
  blocks: Block[];
  truncated: boolean;
  trace?: TraceStep[];
  debug?: TurnDebug | null;
  stop_reason?: string | null;
  /** Earlier messages left out of this turn so the request fit the model's window. */
  context_trimmed?: number;
}

/** A displayed formula of the user's own message, with the number the conversation gave it. */
export interface UserEquation {
  number: number;
  latex: string;
}

/** What "equation (3)" in the user's message was taken to mean (resolved by the server, shown to the user). */
export interface EquationReference {
  number: number;
  /** The formula, or null when the conversation has no such equation. */
  latex: string | null;
  /** from_tool: a verified result; written_by_model: not verified; yours: the user's own; unknown. */
  source: "from_tool" | "written_by_model" | "yours" | "unknown";
}

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: {
    text: string;
    equations?: UserEquation[];
    references?: EquationReference[];
  } & Partial<AssistantContent>;
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at: string;
}

export interface ChatStatus {
  enabled: boolean;
  mode: string;
  model: string;
  /** Formulas can be read from a picture ("Add image"). */
  images?: boolean;
  providers?: ProviderStatus;
  /** Parameter defaults and what this account may change; absent when not signed in. */
  params?: ParamsInfo | null;
  /** False when even the smallest request (system prompt and core tools) does not fit the window. */
  context_ok?: boolean;
}

export interface ProviderStatus {
  default: "ollama" | "openrouter";
  ollama: {
    available: boolean;
    model: string;
    reachable?: boolean;
    /** Models installed on the server's own host, recommended ones first. */
    models?: LocalModel[];
    /** Recommended families that are not installed: guidance only, not selectable. */
    suggested?: { family: string; note: string }[];
  };
  openrouter: {
    available: boolean;
    byok: boolean;
    server_key_for_you: boolean;
    default_model: string;
    models: string[];
    zdr: boolean;
    data_collection: string;
  };
}

export interface LocalModel {
  name: string;
  recommended: boolean;
  note: string;
}

/** What the browser sends to pick the provider for one turn. */
export interface ProviderChoice {
  kind: "ollama" | "openrouter";
  model?: string;
  api_key?: string;
  cloud_consent?: boolean;
}

export interface ConnectionResult {
  ok: boolean;
  message: string;
  credit_remaining?: number | null;
  supports_tools?: boolean;
}

/** One formula transcribed from a picture, and how Minerva's parser reads it. */
export interface ExtractedFormula {
  latex: string;
  /** `ok`, `parse_error` or `resource_limit` */
  status: string;
  /** The parsed formula re-rendered as LaTeX; absent when it could not be parsed. */
  read_as: string | null;
  message: string;
}

export interface ToolRun {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export type StreamEvent =
  | { event: "text"; data: { delta: string } }
  | { event: "tool_call"; data: ToolRun }
  | { event: "card"; data: Card }
  | { event: "phase"; data: { phase: "checking" } }
  | { event: "user"; data: { equations?: UserEquation[]; references?: EquationReference[] } }
  | { event: "message"; data: AssistantContent }
  | {
      event: "error";
      data: { message: string; debug?: TurnDebug; trace?: TraceStep[]; trace_id?: string };
    }
  | { event: "done"; data: { trace_id?: string } };
