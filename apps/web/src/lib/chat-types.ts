import type {
  DifferentiateResult,
  EntrySummary,
  IdentitiesResult,
  IntegrateResult,
  SimplifyResult,
  Status,
} from "./types";

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
 * Formulas only. from_tool: appears in a tool result or handbook entry of this conversation.
 * checked: an equation the model wrote that Minerva tested numerically and found true.
 * conditional: false in general but true under extra assumptions (see `note`).
 * contradicted: an equation the model wrote that Minerva tested and found false.
 * unchecked: written by the model; Minerva could not check it.
 */
export type FormulaStatus = "from_tool" | "checked" | "conditional" | "contradicted" | "unchecked";

export interface Segment {
  type: "text" | "math" | "display_math";
  value: string;
  status: FormulaStatus | null;
  /** For conditional claims: what must hold, e.g. "false in general, but holds if x is positive". */
  note?: string | null;
}

export type Card =
  | { id: string; tool: string; type: "simplify"; title: string; data: SimplifyResult }
  | { id: string; tool: string; type: "differentiate"; title: string; data: DifferentiateResult }
  | { id: string; tool: string; type: "integrate"; title: string; data: IntegrateResult }
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
      type: "motion";
      title: string;
      data: { report: MotionReport; request: MotionRequest };
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
  | { id: string; tool: string; type: "error"; title: string; data: { message: string } };

/** One item of an assistant turn in the order it happened: prose, or a tool result card. */
export type Block =
  | { kind: "text"; segments: Segment[]; card: null }
  | { kind: "card"; segments: Segment[]; card: Card };

/** A tool call as the model made it, and what the model read back. Kept for debugging. */
export interface TraceStep {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  for_model: string;
  /** Absent on messages stored before debug data existed. */
  duration_ms?: number | null;
  status?: string | null;
}

export interface TurnDebug {
  request_id: string;
  api_version: string;
  mode: string;
  model: string;
  prompt_sha: string;
  max_steps: number;
  max_seconds: number;
  duration_ms: number;
  llm_calls: { ms: number; text_chars: number; tool_calls: number }[];
  claims: { value: string; status: string; detail: string }[];
  error: string | null;
}

export interface AssistantContent {
  text: string;
  blocks: Block[];
  truncated: boolean;
  trace?: TraceStep[];
  debug?: TurnDebug | null;
}

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: { text: string } & Partial<AssistantContent>;
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
  | { event: "message"; data: AssistantContent }
  | { event: "error"; data: { message: string; debug?: TurnDebug; trace?: TraceStep[] } }
  | { event: "done"; data: Record<string, never> };
