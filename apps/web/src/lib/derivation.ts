/** Pure helpers for the Workbench derivation: what a saved step says, and what may follow it. */
import type {
  ApplyIdentityResult,
  Assumptions,
  DifferentiateResult,
  DistributionAction,
  DistributionEndpoint,
  DistributionResult,
  HistoryItem,
  IntegrateResult,
  IsolateResult,
  SeriesResult,
  SimplifyResult,
  Status,
  StepResult,
} from "./types";
import { MAX_DERIVATIVE_ORDER, MAX_SERIES_ORDER } from "./types";

/** Operations that run on the current expression and add a step. */
export type StepOperation =
  | "simplify"
  | "differentiate"
  | "integrate"
  | "series"
  | "distribution"
  | "substitute"
  | "collect"
  | "isolate";

export interface Settings {
  variable: string;
  order: number;
  point: string;
  seriesOrder: number;
  lower: string;
  upper: string;
  /** One relation `left = right` per line, written like the expression. */
  relations: string;
  /** How several relations are applied; empty means: ask if the order matters. */
  relationOrder: "" | "given" | "simultaneous";
  /** What to collect by, separated by commas. */
  collectBy: string;
  /** The unknown to solve for. */
  solveFor: string;
  /** Distribution step: what to do with the deltas and steps. */
  distAction: DistributionAction;
  /** A delta on an integration limit counts only under a convention the user chose. */
  endpoint: DistributionEndpoint | "";
}

export const DEFAULT_SETTINGS: Settings = {
  variable: "x",
  order: 1,
  point: "0",
  seriesOrder: 6,
  lower: "",
  upper: "",
  relations: "",
  relationOrder: "",
  collectBy: "x",
  solveFor: "x",
  distAction: "simplify",
  endpoint: "",
};

export const MAX_RELATIONS = 4;
export const MAX_COLLECT_TARGETS = 4;

/** The non-empty lines of the relations field. */
export function relationLines(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
}

/** The comma-separated things to collect by. */
export function collectTargets(text: string): string[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "");
}

// A finite real number as the server reads it in text syntax: 0, -2, 1/2, 0.5, pi.
export const POINT_PATTERN = /^-?(\d+(\.\d+)?(\/\d+)?|pi)$/;
export const VARIABLE_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;

/** A message for the first thing wrong with the settings of `operation`, or null. */
export function validateSettings(operation: StepOperation, s: Settings): string | null {
  if (operation === "simplify") return null;
  if (operation === "substitute") {
    const lines = relationLines(s.relations);
    if (lines.length === 0) return "Write at least one relation, for example y^2 = 1 - x^2.";
    if (lines.length > MAX_RELATIONS) return `Use at most ${MAX_RELATIONS} relations.`;
    if (lines.some((line) => (line.match(/=/g) ?? []).length !== 1)) {
      return "Write each relation as left = right, with one equals sign.";
    }
    return null;
  }
  if (operation === "collect") {
    const targets = collectTargets(s.collectBy);
    if (targets.length === 0) return "Enter what to collect by, for example x.";
    if (targets.length > MAX_COLLECT_TARGETS) {
      return `Collect by at most ${MAX_COLLECT_TARGETS} things.`;
    }
    return null;
  }
  if (operation === "isolate") {
    return VARIABLE_PATTERN.test(s.solveFor)
      ? null
      : "Enter the unknown to solve for, for example x.";
  }
  if (!VARIABLE_PATTERN.test(s.variable)) {
    return "Enter the variable, for example x.";
  }
  if (operation === "distribution" && s.distAction === "integrate") {
    if (s.lower.trim() === "" || s.upper.trim() === "") {
      return "Give both limits (use -oo and oo for the whole line).";
    }
    return null;
  }
  if (operation === "integrate" && (s.lower.trim() === "") !== (s.upper.trim() === "")) {
    return "Give both bounds for a definite integral, or leave both empty.";
  }
  if (operation === "series" && !POINT_PATTERN.test(s.point)) {
    return "Enter a finite expansion point such as 0, 1/2 or pi.";
  }
  if (operation === "differentiate" && !(s.order >= 1 && s.order <= MAX_DERIVATIVE_ORDER)) {
    return `The order must be between 1 and ${MAX_DERIVATIVE_ORDER}.`;
  }
  if (operation === "series" && !(s.seriesOrder >= 1 && s.seriesOrder <= MAX_SERIES_ORDER)) {
    return `The series order must be between 1 and ${MAX_SERIES_ORDER}.`;
  }
  return null;
}

/** The expression a step leaves behind, in the form the server can read back exactly. */
export interface Continuation {
  text: string;
  latex: string | null;
}

type Shown = Pick<SimplifyResult, "status" | "result_latex" | "result_text" | "message">;

function shown(item: HistoryItem): Shown {
  return item.result as unknown as Shown;
}

/**
 * Only the text form is carried to the next step: LaTeX produced by the engine does not always read
 * back as the same expression (`I`, `e` and some function names), the text form does or is refused.
 * A step that was not verified (or has no text form) cannot be continued.
 */
export function continuation(item: HistoryItem, which = 0): Continuation | null {
  if (item.status !== "ok" && item.status !== "unchanged") return null;
  if (item.operation === "isolate") {
    // An equation has several solutions: a step continues from one of them (its value), and with
    // more than one the choice is the user's, so nothing is continued automatically.
    const solutions = (item.result as unknown as IsolateResult).solutions ?? [];
    if (solutions.length !== 1 && which === 0 && solutions.length > 1) return null;
    const chosen = solutions[which];
    return chosen ? { text: chosen.text, latex: chosen.latex } : null;
  }
  const r = shown(item);
  if (!r.result_text) return null;
  return { text: r.result_text, latex: r.result_latex };
}

/** Every way a step can be continued: the result, or for an equation each of its solutions. */
export function continuations(item: HistoryItem): Continuation[] {
  if (item.operation === "isolate") {
    if (item.status !== "ok" && item.status !== "unchanged") return [];
    return ((item.result as unknown as IsolateResult).solutions ?? []).map((s) => ({
      text: s.text,
      latex: s.latex,
    }));
  }
  const one = continuation(item);
  return one ? [one] : [];
}

export function resultLatex(item: HistoryItem): string | null {
  if (item.operation === "isolate") return null; // the solutions are listed by the panel
  const r = shown(item);
  return item.status === "ok" || item.status === "unchanged" ? r.result_latex : null;
}

export function isStepFailure(status: Status): boolean {
  return status !== "ok" && status !== "unchanged";
}

/** What produced the step, for its heading. */
export function stepLabel(item: HistoryItem): string {
  switch (item.operation) {
    case "simplify":
      return "Simplify";
    case "differentiate": {
      const r = item.result as unknown as DifferentiateResult;
      const v = r.variables?.[0];
      if (!v) return "Differentiate";
      return v.order > 1
        ? `Differentiate by ${v.name}, ${v.order} times`
        : `Differentiate by ${v.name}`;
    }
    case "integrate": {
      const r = item.result as unknown as IntegrateResult;
      const by = r.variable ? ` by ${r.variable}` : "";
      return r.definite ? `Definite integral${by}` : `Antiderivative${by}`;
    }
    case "series": {
      const r = item.result as unknown as SeriesResult & { point_requested?: string };
      const about = r.variable ? ` in ${r.variable}` : "";
      const point = r.point_requested ? ` about ${r.point_requested}` : "";
      return `Taylor polynomial${about}${point}${r.order ? `, order ${r.order}` : ""}`;
    }
    case "distribution": {
      const r = item.result as unknown as DistributionResult;
      return (
        {
          simplify: "Simplify deltas",
          integrate: "Integrate against deltas",
          differentiate: "Differentiate steps (distributional)",
        }[r.action ?? "simplify"] ?? "Distributions"
      );
    }
    case "apply_identity":
      return (item.result as unknown as ApplyIdentityResult).entry?.title ?? "Identity";
    case "substitute": {
      const n = ((item.result as unknown as StepResult).request?.relations as string[] | undefined)
        ?.length;
      return n && n > 1 ? `Substitute ${n} relations` : "Substitute a relation";
    }
    case "isolate": {
      const v = (item.result as unknown as IsolateResult).request?.variable;
      return typeof v === "string" ? `Solve for ${v}` : "Solve for a variable";
    }
    case "rewrite": {
      const r = item.result as unknown as StepResult;
      const request = r.request ?? {};
      const part = request.part || request.selection ? " (one part)" : "";
      const action = String(request.action ?? "");
      if (action === "rule") return `${r.entry?.title ?? "Identity"}${part}`;
      const names: Record<string, string> = {
        expand: "Expand",
        factor: "Factor",
        cancel: "Cancel",
        simplify: "Simplify",
        collect: "Collect terms",
      };
      return `${names[action] ?? "Rewrite"}${part}`;
    }
    default:
      return "Identities";
  }
}

/** One line on how the result was checked. Never claims more than the stored verification. */
export function verificationSummary(item: HistoryItem): string | null {
  const r = item.result as unknown as Record<string, unknown>;
  const v = r.verification as Record<string, unknown> | null | undefined;
  if (!v) return null;
  if (item.operation === "distribution") {
    if (typeof v.pairings_checked !== "number") return null;
    return `${v.pairings_agreed}/${v.pairings_checked} test-function pairings agree (not a proof)${
      v.conditions_established ? "; conditions established exactly" : ""
    }`;
  }
  const agreed = v.points_agreed ?? v.numeric_points_agreed;
  const checked = v.points_checked ?? v.numeric_points_checked;
  if (item.operation === "series") {
    return v.passed ? "Coefficients and remainder order checked independently" : null;
  }
  if (typeof agreed !== "number" || typeof checked !== "number") return null;
  const symbolic = v.symbolic_zero === true ? "; difference simplifies to 0" : "";
  return `${agreed}/${checked} sample points agree${symbolic}`;
}

/** The variable of a distribution is a point on the real line: declare it real, visibly, in the request. */
export function withRealVariable(a: Assumptions, variable: string): Assumptions {
  const list = a[variable] ?? [];
  return list.includes("real") ? a : { ...a, [variable]: [...list, "real"] };
}

export function assumptionsSummary(a: Assumptions): string | null {
  const parts = Object.entries(a)
    .filter(([, list]) => list.length > 0)
    .map(([name, list]) => `${name} ${list.join(", ")}`);
  return parts.length ? parts.join("; ") : null;
}
