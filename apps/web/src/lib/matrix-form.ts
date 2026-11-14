import {
  GRADIENT_ACTIONS,
  type Assumptions,
  type MatrixAction,
  type MatrixDecl,
  type MatrixRequest,
} from "./types";

export const MAX_MATRICES = 6;
export const MAX_ABSTRACT_SIZE = 8;
export const MAX_EXPLICIT_SIZE = 4;
const NAME_RE = /^[A-Za-z\\][A-Za-z0-9_\\{}]{0,31}$/;

export const ACTIONS: { id: MatrixAction; label: string; hint: string }[] = [
  { id: "evaluate", label: "Evaluate", hint: "entries of the result (explicit matrices)" },
  { id: "simplify", label: "Simplify", hint: "merge, split or reorder only where it is justified" },
  { id: "differentiate", label: "Differentiate", hint: "with respect to a variable" },
  {
    id: "integrate",
    label: "Integrate",
    hint: "with respect to a variable, optionally between bounds",
  },
  { id: "evolve", label: "Evolve", hint: "solve dU/dt = A(t) U" },
  {
    id: "gradient",
    label: "Gradient",
    hint: "of a scalar function with respect to a matrix or vector (plain text, real matrices)",
  },
  {
    id: "jacobian",
    label: "Jacobian",
    hint: "of a vector function with respect to a vector (plain text, real matrices)",
  },
  {
    id: "hessian",
    label: "Hessian",
    hint: "of a scalar function with respect to a vector (plain text, real matrices)",
  },
];

/** The actions that read the plain-text syntax and differentiate with respect to a declared matrix. */
export function isGradientAction(action: MatrixAction): boolean {
  return GRADIENT_ACTIONS.includes(action);
}

/** One declared matrix as typed: a size (abstract) or a grid of entries (explicit). */
export interface MatrixDeclForm {
  name: string;
  kind: "abstract" | "explicit";
  rows: number;
  cols: number;
  entries: string[][];
  dependsOn: string;
  /** Declared symmetric: a derivative with respect to it is taken within the symmetric matrices. */
  symmetric: boolean;
  /** Declared symmetric positive definite (implies symmetric and invertible). */
  positiveDefinite: boolean;
}

/** The editable state of the matrix form: text exactly as the user typed it. */
export interface MatrixForm {
  input: string;
  format: "latex" | "text";
  matrices: MatrixDeclForm[];
  /** Pairs such as `A,B; B,C`: declared to commute. Only the user can say so. */
  commuting: string;
  invertible: string;
  selfCommuting: string;
  action: MatrixAction;
  variable: string;
  lower: string;
  upper: string;
  assumptions: Assumptions;
}

export function blankMatrix(name: string): MatrixDeclForm {
  return {
    name,
    kind: "abstract",
    rows: 2,
    cols: 2,
    entries: grid(2, 2),
    dependsOn: "",
    symmetric: false,
    positiveDefinite: false,
  };
}

export function grid(rows: number, cols: number, from: string[][] = []): string[][] {
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => from[i]?.[j] ?? ""),
  );
}

export function blankForm(): MatrixForm {
  return {
    input: "",
    format: "latex",
    matrices: [blankMatrix("A"), blankMatrix("B")],
    commuting: "",
    invertible: "",
    selfCommuting: "",
    action: "simplify",
    variable: "",
    lower: "",
    upper: "",
    assumptions: {},
  };
}

/** Resizing an explicit grid keeps what was typed. */
export function resize(m: MatrixDeclForm, rows: number, cols: number): MatrixDeclForm {
  const limit = m.kind === "explicit" ? MAX_EXPLICIT_SIZE : MAX_ABSTRACT_SIZE;
  const r = Math.min(Math.max(rows, 1), limit);
  const c = Math.min(Math.max(cols, 1), limit);
  return { ...m, rows: r, cols: c, entries: grid(r, c, m.entries) };
}

/** Switching kind clamps the size to what that kind allows. */
export function setKind(m: MatrixDeclForm, kind: "abstract" | "explicit"): MatrixDeclForm {
  return resize({ ...m, kind }, m.rows, m.cols);
}

function names(text: string): string[] {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function pairs(text: string): [string, string][] | string {
  const out: [string, string][] = [];
  for (const part of text.split(";")) {
    const items = part
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (items.length === 0) continue;
    if (items.length !== 2)
      return `“${part.trim()}” is not a pair; write commuting pairs like A,B; B,C.`;
    out.push([items[0]!, items[1]!]);
  }
  return out;
}

export type Built =
  | {
      ok: true;
      input: string;
      format: "latex" | "text";
      assumptions: Assumptions;
      request: MatrixRequest;
    }
  | { ok: false; error: string };

/** Validates the form and builds the request. The server validates again; this only saves a round trip. */
export function buildMatrixRequest(form: MatrixForm): Built {
  if (form.input.trim() === "") return { ok: false, error: "Enter the expression to work on." };
  if (form.matrices.length === 0) return { ok: false, error: "Declare at least one matrix." };
  const declared = new Set<string>();
  const matrices: MatrixDecl[] = [];
  for (const m of form.matrices) {
    const name = m.name.trim();
    if (!NAME_RE.test(name))
      return { ok: false, error: "Every matrix needs a name such as A or X_1." };
    if (declared.has(name)) return { ok: false, error: `Matrix ${name} is declared twice.` };
    declared.add(name);
    const dependsOn = m.dependsOn.trim() || null;
    const flags = {
      ...(m.symmetric ? { symmetric: true } : {}),
      ...(m.positiveDefinite ? { positive_definite: true } : {}),
    };
    if ((m.symmetric || m.positiveDefinite) && m.rows !== m.cols) {
      return { ok: false, error: `${name} is declared symmetric, so it must be square.` };
    }
    if (m.kind === "abstract") {
      matrices.push({ name, rows: m.rows, cols: m.cols, depends_on: dependsOn, ...flags });
    } else {
      const entries = grid(m.rows, m.cols, m.entries).map((row) => row.map((c) => c.trim()));
      if (entries.some((row) => row.some((c) => c === ""))) {
        return {
          ok: false,
          error: `Fill in every entry of ${name} (write 0 for zero entries), or declare only its size.`,
        };
      }
      matrices.push({ name, entries, depends_on: dependsOn, ...flags });
    }
  }
  const commuting = pairs(form.commuting);
  if (typeof commuting === "string") return { ok: false, error: commuting };
  const invertible = names(form.invertible);
  const selfCommuting = names(form.selfCommuting);
  for (const n of [...commuting.flat(), ...invertible, ...selfCommuting]) {
    if (!declared.has(n))
      return { ok: false, error: `${n} is used in a declaration but is not one of the matrices.` };
  }
  const gradient = isGradientAction(form.action);
  if (gradient) {
    if (form.format !== "text") {
      return {
        ok: false,
        error:
          "Gradients, Jacobians and Hessians read the plain-text syntax: choose Plain text and write " +
          "transpose(x), inv(S), tr(A*X), logdet(S), x - mu.",
      };
    }
    if (!declared.has(form.variable.trim())) {
      return {
        ok: false,
        error: "Name one of the declared matrices to differentiate with respect to.",
      };
    }
  }
  const needsVariable = form.action !== "simplify" && form.action !== "evaluate";
  if (needsVariable && form.variable.trim() === "") {
    return { ok: false, error: "Name the variable to differentiate, integrate or evolve in." };
  }
  if (form.action === "evolve" && form.upper.trim() === "") {
    return {
      ok: false,
      error: "Enter the end of the numerical span: only you can choose it.",
    };
  }
  const assumptions: Assumptions = {};
  for (const [name, items] of Object.entries(form.assumptions)) {
    if (items.length > 0) assumptions[name] = items;
  }
  return {
    ok: true,
    input: form.input.trim(),
    format: form.format,
    assumptions,
    request: {
      matrices,
      commuting,
      invertible,
      self_commuting: selfCommuting,
      action: form.action,
      variable: needsVariable ? form.variable.trim() : null,
      lower: needsVariable && !gradient && form.lower.trim() ? form.lower.trim() : null,
      upper: needsVariable && !gradient && form.upper.trim() ? form.upper.trim() : null,
    },
  };
}

/** Rebuilds the form from a saved request, so an earlier problem can be reopened and changed. */
export function restoreMatrixForm(
  request: MatrixRequest,
  input: string,
  format: "latex" | "text",
  assumptions: Assumptions,
): MatrixForm {
  return {
    input,
    format,
    matrices: request.matrices.map((m) => {
      const explicit = !!m.entries;
      const rows = explicit ? m.entries!.length : (m.rows ?? 2);
      const cols = explicit ? (m.entries![0]?.length ?? 1) : (m.cols ?? rows);
      return {
        name: m.name,
        kind: explicit ? "explicit" : "abstract",
        rows,
        cols,
        entries: grid(rows, cols, m.entries ?? []),
        dependsOn: m.depends_on ?? "",
        symmetric: !!m.symmetric,
        positiveDefinite: !!m.positive_definite,
      };
    }),
    commuting: request.commuting.map(([a, b]) => `${a},${b}`).join("; "),
    invertible: request.invertible.join(", "),
    selfCommuting: (request.self_commuting ?? []).join(", "),
    action: request.action,
    variable: request.variable ?? "",
    lower: request.lower ?? "",
    upper: request.upper ?? "",
    assumptions,
  };
}

export interface MatrixExample {
  id: string;
  label: string;
  form: MatrixForm;
}

/** Starting points the user chooses. They are ordinary input: nothing is assumed beyond what they declare. */
export const EXAMPLES: MatrixExample[] = [
  {
    id: "noncommuting-product",
    label: "exp(A) exp(B) for noncommuting A, B",
    form: {
      ...blankForm(),
      input: "\\exp(A)\\exp(B)",
      matrices: [
        {
          ...blankMatrix("A"),
          kind: "explicit",
          entries: [
            ["0", "1"],
            ["0", "0"],
          ],
        },
        {
          ...blankMatrix("B"),
          kind: "explicit",
          entries: [
            ["0", "0"],
            ["1", "0"],
          ],
        },
      ],
      action: "evaluate",
    },
  },
  {
    id: "merge-needs-commutation",
    label: "exp(A) exp(B), sizes only",
    form: { ...blankForm(), input: "\\exp(A)\\exp(B)", action: "simplify" },
  },
  {
    id: "derivative-of-exponential",
    label: "d/dt exp(A + tB)",
    form: { ...blankForm(), input: "\\exp(A+tB)", action: "differentiate", variable: "t" },
  },
];

/** Gaussian log-density examples: the derivative of a scalar with respect to a vector and to a matrix. */
EXAMPLES.push(
  {
    id: "gaussian-gradient-covariance",
    label: "Gaussian log-density: gradient in the covariance (positive definite)",
    form: {
      ...blankForm(),
      format: "text",
      input: "-1/2*logdet(S) - 1/2*transpose(x - mu)*inv(S)*(x - mu)",
      matrices: [
        { ...blankMatrix("x"), rows: 3, cols: 1, entries: grid(3, 1) },
        { ...blankMatrix("mu"), rows: 3, cols: 1, entries: grid(3, 1) },
        { ...blankMatrix("S"), rows: 3, cols: 3, entries: grid(3, 3), positiveDefinite: true },
      ],
      action: "gradient",
      variable: "S",
    },
  },
  {
    id: "quadratic-form-hessian",
    label: "Hessian of x^T A x",
    form: {
      ...blankForm(),
      format: "text",
      input: "transpose(x)*A*x",
      matrices: [
        { ...blankMatrix("A"), rows: 3, cols: 3, entries: grid(3, 3) },
        { ...blankMatrix("x"), rows: 3, cols: 1, entries: grid(3, 1) },
      ],
      action: "hessian",
      variable: "x",
    },
  },
);
