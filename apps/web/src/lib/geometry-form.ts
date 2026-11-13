import type {
  Assumptions,
  GeometryFunction,
  GeometryQuantity,
  GeometryRequest,
  GeometryTensor,
  GeometryVariance,
} from "./types";

export const MAX_DIMENSION = 4;
const NAME_RE = /^[A-Za-z\\][A-Za-z0-9_\\{}]{0,31}$/;
const FUNCTION_RE = /^\s*([A-Za-z\\][A-Za-z0-9_\\{}]{0,31})\s*\(([^()]*)\)\s*$/;

export interface QuantityOption {
  id: GeometryQuantity;
  label: string;
  /** Needs a tensor field (covariant derivative, raising, lowering). */
  tensor?: boolean;
}

export const QUANTITY_OPTIONS: QuantityOption[] = [
  { id: "inverse_metric", label: "Inverse metric" },
  { id: "determinant", label: "Determinant" },
  { id: "christoffel", label: "Christoffel symbols" },
  { id: "riemann", label: "Riemann tensor" },
  { id: "ricci", label: "Ricci tensor" },
  { id: "ricci_scalar", label: "Ricci scalar" },
  { id: "einstein", label: "Einstein tensor" },
  { id: "kretschmann", label: "Kretschmann scalar" },
  { id: "ricci_squared", label: "Ricci squared" },
  { id: "geodesics", label: "Geodesic equations" },
  { id: "covariant_derivative", label: "Covariant derivative", tensor: true },
  { id: "raise_index", label: "Raise an index", tensor: true },
  { id: "lower_index", label: "Lower an index", tensor: true },
];

/** The editable state of the geometry form: text exactly as the user typed it. */
export interface GeometryForm {
  coordinates: string[];
  metric: string[][];
  assumptions: Assumptions;
  /** Free text such as `f(r), a(t)`. */
  functions: string;
  quantities: GeometryQuantity[];
  tensorVariance: GeometryVariance[];
  tensorComponents: string[];
  tensorIndex: number;
}

function defaultName(i: number): string {
  return ["x", "y", "z", "w"][i] ?? `x${i}`;
}

export function blankForm(n: number): GeometryForm {
  return {
    coordinates: Array.from({ length: n }, (_, i) => defaultName(i)),
    metric: Array.from({ length: n }, () => Array.from({ length: n }, () => "")),
    assumptions: {},
    functions: "",
    quantities: ["christoffel"],
    tensorVariance: ["up"],
    tensorComponents: Array.from({ length: n }, () => ""),
    tensorIndex: 0,
  };
}

/** Grows or shrinks the grid, keeping what the user already typed. */
export function setDimension(form: GeometryForm, n: number): GeometryForm {
  const size = Math.min(Math.max(n, 1), MAX_DIMENSION);
  const coordinates = Array.from({ length: size }, (_, i) => form.coordinates[i] ?? defaultName(i));
  const metric = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => form.metric[i]?.[j] ?? ""),
  );
  return {
    ...form,
    coordinates,
    metric,
    tensorComponents: Array.from(
      { length: size ** form.tensorVariance.length },
      (_, k) => form.tensorComponents[k] ?? "",
    ),
  };
}

/** A metric is symmetric: typing g_ij also fills g_ji. */
export function setMetricCell(
  form: GeometryForm,
  i: number,
  j: number,
  value: string,
): GeometryForm {
  const metric = form.metric.map((row) => [...row]);
  metric[i]![j] = value;
  metric[j]![i] = value;
  return { ...form, metric };
}

export function setRank(form: GeometryForm, rank: number): GeometryForm {
  const n = form.coordinates.length;
  const variance = Array.from({ length: rank }, (_, k) => form.tensorVariance[k] ?? "up");
  return {
    ...form,
    tensorVariance: variance,
    tensorComponents: Array.from({ length: n ** rank }, (_, k) => form.tensorComponents[k] ?? ""),
    tensorIndex: Math.min(form.tensorIndex, rank - 1),
  };
}

/** `f(r), a(t)` -> declarations, or a message about the first part that is not one. */
export function parseFunctions(text: string): GeometryFunction[] | string {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const parts: string[] = [];
  let depth = 0;
  let current = "";
  for (const ch of trimmed) {
    if (ch === "(") depth += 1;
    if (ch === ")") depth -= 1;
    if (ch === "," && depth === 0) {
      parts.push(current);
      current = "";
    } else current += ch;
  }
  parts.push(current);
  const out: GeometryFunction[] = [];
  for (const part of parts) {
    const m = FUNCTION_RE.exec(part);
    if (!m) return `“${part.trim()}” is not a function declaration; write it like f(r) or a(t).`;
    const args = m[2]!
      .split(",")
      .map((a) => a.trim())
      .filter(Boolean);
    if (args.length === 0) return `Function ${m[1]} needs at least one argument.`;
    out.push({ name: m[1]!, args });
  }
  return out;
}

export type Built = { ok: true; request: GeometryRequest } | { ok: false; error: string };

/** Validates the form and builds the request. The server validates again; this only saves a round trip. */
export function buildGeometryRequest(form: GeometryForm): Built {
  const n = form.coordinates.length;
  const names = form.coordinates.map((c) => c.trim());
  if (names.some((c) => !NAME_RE.test(c))) {
    return { ok: false, error: "Every coordinate needs a name such as t, r or \\theta." };
  }
  if (new Set(names).size !== n) return { ok: false, error: "Coordinate names must be distinct." };
  const metric: string[][] = [];
  for (let i = 0; i < n; i += 1) {
    const row: string[] = [];
    for (let j = 0; j < n; j += 1) {
      const cell = (form.metric[i]?.[j] ?? "").trim();
      if (cell === "" && i === j) {
        return { ok: false, error: `Enter g_{${names[i]}${names[j]}}: a diagonal entry is empty.` };
      }
      row.push(cell === "" ? "0" : cell); // an empty off-diagonal entry means 0
    }
    metric.push(row);
  }
  const functions = parseFunctions(form.functions);
  if (typeof functions === "string") return { ok: false, error: functions };
  if (form.quantities.length === 0) return { ok: false, error: "Choose what to compute." };
  const needsTensor = form.quantities.some((q) => QUANTITY_OPTIONS.find((o) => o.id === q)?.tensor);
  let tensor: GeometryTensor | null = null;
  if (needsTensor) {
    const components = form.tensorComponents.map((c) => c.trim());
    if (components.some((c) => c === "")) {
      return {
        ok: false,
        error: "Fill in every component of the tensor field (write 0 for zero components).",
      };
    }
    tensor = { variance: form.tensorVariance, components };
  }
  const assumptions: Assumptions = {};
  for (const [name, items] of Object.entries(form.assumptions)) {
    if (items.length > 0) assumptions[name] = items;
  }
  return {
    ok: true,
    request: {
      coordinates: names,
      metric,
      format: "latex",
      assumptions,
      functions,
      quantities: form.quantities,
      tensor,
      tensor_index: form.tensorIndex,
    },
  };
}

/** Rebuilds the form from a saved request, so an earlier problem can be reopened and changed. */
export function restoreForm(request: GeometryRequest): GeometryForm {
  const n = request.coordinates.length;
  const tensor = request.tensor;
  return {
    coordinates: [...request.coordinates],
    metric: request.metric.map((row) => row.map((cell) => (cell === "0" ? "" : cell))),
    assumptions: request.assumptions,
    functions: request.functions.map((f) => `${f.name}(${f.args.join(", ")})`).join(", "),
    quantities: request.quantities,
    tensorVariance: tensor?.variance ?? ["up"],
    tensorComponents: tensor?.components ?? Array.from({ length: n }, () => ""),
    tensorIndex: request.tensor_index,
  };
}

export interface Example {
  id: string;
  label: string;
  form: GeometryForm;
}

function example(
  id: string,
  label: string,
  coordinates: string[],
  rows: string[][],
  extra: Partial<GeometryForm> = {},
): Example {
  return {
    id,
    label,
    form: {
      ...blankForm(coordinates.length),
      coordinates,
      metric: rows.map((row) => row.map((cell) => (cell === "0" ? "" : cell))),
      ...extra,
    },
  };
}

/** Starting points the user chooses. They are ordinary input: nothing is assumed beyond what they declare. */
export const EXAMPLES: Example[] = [
  example(
    "polar",
    "Plane, polar coordinates",
    ["r", "\\theta"],
    [
      ["1", "0"],
      ["0", "r^2"],
    ],
    {
      assumptions: { r: ["positive"] },
    },
  ),
  example(
    "sphere",
    "Sphere of radius a",
    ["\\theta", "\\phi"],
    [
      ["a^2", "0"],
      ["0", "a^2 \\sin^2\\theta"],
    ],
    { assumptions: { a: ["positive"] } },
  ),
  example(
    "minkowski",
    "Minkowski space",
    ["t", "x", "y", "z"],
    [
      ["-1", "0", "0", "0"],
      ["0", "1", "0", "0"],
      ["0", "0", "1", "0"],
      ["0", "0", "0", "1"],
    ],
  ),
  example(
    "schwarzschild",
    "Schwarzschild",
    ["t", "r", "\\theta", "\\phi"],
    [
      ["-(1-2M/r)", "0", "0", "0"],
      ["0", "(1-2M/r)^{-1}", "0", "0"],
      ["0", "0", "r^2", "0"],
      ["0", "0", "0", "r^2 \\sin^2\\theta"],
    ],
    { assumptions: { M: ["positive"], r: ["positive"] } },
  ),
];
