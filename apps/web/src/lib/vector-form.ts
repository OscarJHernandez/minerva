import type {
  Assumptions,
  VectorAction,
  VectorCoordinates,
  VectorFieldDecl,
  VectorRequest,
} from "./types";

export const MAX_FIELDS = 8;
const NAME_RE = /^[A-Za-z][A-Za-z0-9_]{0,15}$/;
const RESERVED = new Set(["x", "y", "z", "t", "r", "theta", "phi", "rho"]);

/** The actions the vector workbench offers; each runs one step the user chose. */
export const ACTIONS: { id: VectorAction; label: string; hint: string }[] = [
  { id: "evaluate", label: "Evaluate", hint: "gradient, divergence, curl, Laplacian, products" },
  { id: "identities", label: "Find identities", hint: "which handbook identities fit, and where" },
  {
    id: "apply_identity",
    label: "Apply an identity",
    hint: "one identity at a part you choose (a path)",
  },
  {
    id: "substitute",
    label: "Substitute a premise",
    hint: "replace one side of a field equation you adopted by the other",
  },
  { id: "derive", label: "Search for a derivation", hint: "toward a target, within fixed bounds" },
  {
    id: "distributional",
    label: "Distributional divergence",
    hint: "a point source: the delta term of a Coulomb field or of the Laplacian of 1/r",
  },
  {
    id: "em_derive",
    label: "Electromagnetic derivation",
    hint: "from field equations you adopt as premises",
  },
];

/** Derivations the engine can carry out (the same ids and fields as `electromagnetism.setup`). */
export const TEMPLATES: {
  id: string;
  label: string;
  fields: [string, "scalar" | "vector"][];
  constants: string[];
  smooth: string[];
  needs: string[];
}[] = [
  {
    id: "wave_equation_e",
    label: "Wave equation for E (vacuum)",
    fields: [
      ["E", "vector"],
      ["B", "vector"],
    ],
    constants: ["mu0", "epsilon0"],
    smooth: ["E", "B"],
    needs: ["div(E) = 0", "curl(E) = -dt(B)", "curl(B) = mu0*epsilon0*dt(E)"],
  },
  {
    id: "wave_equation_b",
    label: "Wave equation for B (vacuum)",
    fields: [
      ["E", "vector"],
      ["B", "vector"],
    ],
    constants: ["mu0", "epsilon0"],
    smooth: ["E", "B"],
    needs: ["div(B) = 0", "curl(B) = mu0*epsilon0*dt(E)", "curl(E) = -dt(B)"],
  },
  {
    id: "continuity",
    label: "Charge conservation",
    fields: [
      ["E", "vector"],
      ["B", "vector"],
      ["J", "vector"],
      ["rho", "scalar"],
    ],
    constants: ["mu0", "epsilon0"],
    smooth: ["E", "B"],
    needs: ["curl(B) = mu0*J + mu0*epsilon0*dt(E)", "div(E) = rho/epsilon0"],
  },
  {
    id: "poynting",
    label: "Poynting's theorem",
    fields: [
      ["E", "vector"],
      ["B", "vector"],
      ["J", "vector"],
    ],
    constants: ["mu0", "epsilon0"],
    smooth: ["E", "B"],
    needs: ["curl(E) = -dt(B)", "curl(B) = mu0*J + mu0*epsilon0*dt(E)"],
  },
  {
    id: "gauge_invariance",
    label: "Gauge invariance of E and B (no premise)",
    fields: [
      ["phi", "scalar"],
      ["A", "vector"],
      ["chi", "scalar"],
    ],
    constants: [],
    smooth: ["phi", "A", "chi"],
    needs: [],
  },
  {
    id: "poisson",
    label: "Poisson's equation",
    fields: [
      ["phi", "scalar"],
      ["E", "vector"],
      ["rho", "scalar"],
    ],
    constants: ["epsilon0"],
    smooth: ["phi", "E"],
    needs: ["div(E) = rho/epsilon0", "grad(phi) = -E"],
  },
  {
    id: "laplace",
    label: "Laplace's equation",
    fields: [
      ["phi", "scalar"],
      ["E", "vector"],
    ],
    constants: [],
    smooth: ["phi", "E"],
    needs: ["div(E) = 0", "grad(phi) = -E"],
  },
];

/** Field equations of the handbook the user may adopt. Minerva never checks that they are true. */
export const PREMISE_OPTIONS: { id: string; label: string; equation: string }[] = [
  { id: "em.maxwell.gauss", label: "Gauss's law", equation: "div(E) = rho/epsilon0" },
  { id: "em.maxwell.gauss_vacuum", label: "Gauss's law (vacuum)", equation: "div(E) = 0" },
  { id: "em.maxwell.gauss_magnetic", label: "Gauss's law for magnetism", equation: "div(B) = 0" },
  { id: "em.maxwell.faraday", label: "Faraday's law", equation: "curl(E) = -dt(B)" },
  {
    id: "em.maxwell.ampere",
    label: "Ampère–Maxwell law",
    equation: "curl(B) = mu0*J + mu0*epsilon0*dt(E)",
  },
  {
    id: "em.maxwell.ampere_vacuum",
    label: "Ampère–Maxwell law (vacuum)",
    equation: "curl(B) = mu0*epsilon0*dt(E)",
  },
  { id: "em.static.potential", label: "Electrostatic potential", equation: "grad(phi) = -E" },
];

export interface VectorFieldForm {
  name: string;
  kind: "scalar" | "vector";
  /** Comma-separated explicit components (blank: an arbitrary field). */
  components: string;
  /** Names of the variables it depends on; blank: all the coordinates (and t if time dependent). */
  dependsOn: string;
  smoothness: "" | "c1" | "c2" | "smooth";
}

export interface VectorForm {
  expression: string;
  format: "latex" | "text";
  coordinates: VectorCoordinates;
  fields: VectorFieldForm[];
  constants: string;
  assumptions: Assumptions;
  action: VectorAction;
  /** Field equations the user adopts, one `lhs = rhs` per line. */
  premises: string;
  premiseIds: string[];
  template: string;
  /** Declared by the USER for the fields of an electromagnetic derivation. */
  emSmoothness: "" | "c2" | "smooth";
  ruleId: string;
  path: string;
  reverse: boolean;
  target: string;
}

export function blankField(name = "A", kind: "scalar" | "vector" = "vector"): VectorFieldForm {
  return { name, kind, components: "", dependsOn: "", smoothness: "" };
}

export function blankForm(): VectorForm {
  return {
    expression: "",
    format: "text",
    coordinates: "cartesian",
    fields: [blankField("E"), blankField("B")],
    constants: "",
    assumptions: {},
    action: "evaluate",
    premises: "",
    premiseIds: [],
    template: "wave_equation_e",
    emSmoothness: "",
    ruleId: "",
    path: "",
    reverse: false,
    target: "",
  };
}

function words(text: string): string[] {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export type BuiltVector = { ok: true; request: VectorRequest } | { ok: false; error: string };

/** Validates the form and builds the request. The server validates again; this only saves a round trip. */
export function buildVectorRequest(form: VectorForm): BuiltVector {
  const emDerive = form.action === "em_derive";
  if (emDerive) return buildEmRequest(form);
  if (!emDerive && form.expression.trim() === "")
    return { ok: false, error: "Enter the expression to work on." };
  const seen = new Set<string>();
  const fields: VectorFieldDecl[] = [];
  for (const f of form.fields) {
    const name = f.name.trim();
    if (!NAME_RE.test(name))
      return { ok: false, error: "Every field needs a name such as E or f." };
    if (RESERVED.has(name))
      return {
        ok: false,
        error: `${name} is a coordinate name; choose another name for the field.`,
      };
    if (seen.has(name)) return { ok: false, error: `Field ${name} is declared twice.` };
    seen.add(name);
    const components = f.components
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);
    const want = f.kind === "vector" ? 3 : 1;
    if (components.length !== 0 && components.length !== want)
      return {
        ok: false,
        error: `${name} needs ${want} component${want > 1 ? "s" : ""} (comma separated), or none for an arbitrary field.`,
      };
    const dependsOn = words(f.dependsOn);
    fields.push({
      name,
      kind: f.kind,
      components: components.length ? components : null,
      depends_on: dependsOn.length ? dependsOn : null,
      smoothness: f.smoothness === "" ? null : f.smoothness,
    });
  }
  const premises: { equation: string }[] = [];
  for (const line of form.premises.split("\n")) {
    const text = line.trim();
    if (!text) continue;
    if (text.split("=").length !== 2)
      return { ok: false, error: `“${text}” is not an equation; write lhs = rhs.` };
    premises.push({ equation: text });
  }
  if (form.action === "apply_identity" && form.ruleId.trim() === "")
    return { ok: false, error: "Name the identity to apply (find identities first)." };
  if (form.action === "derive" && form.target.trim() === "")
    return { ok: false, error: "Give the expression to reach." };
  const path = form.path.trim() === "" ? null : words(form.path).map(Number);
  if (path && path.some((n) => !Number.isInteger(n) || n < 0))
    return { ok: false, error: "The path is a list of numbers such as 0, 1." };
  const request: VectorRequest = {
    expression: emDerive ? "" : form.expression.trim(),
    format: emDerive ? "text" : form.format,
    coordinates: form.coordinates,
    fields,
    constants: words(form.constants),
    assumptions: form.assumptions,
    action: form.action,
    premises,
    premise_ids: form.premiseIds,
  };
  if (emDerive) request.template = form.template;
  if (form.action === "apply_identity") {
    request.rule_id = form.ruleId.trim();
    request.path = path;
    request.reverse = form.reverse;
  }
  if (form.action === "derive") request.target = form.target.trim();
  if (form.action === "distributional" && form.target.trim() !== "")
    request.target = form.target.trim();
  return { ok: true, request };
}

function buildEmRequest(form: VectorForm): BuiltVector {
  const template = TEMPLATES.find((t) => t.id === form.template);
  if (!template) return { ok: false, error: "Choose a derivation." };
  const premises: { equation: string }[] = [];
  for (const line of form.premises.split("\n")) {
    const text = line.trim();
    if (text) premises.push({ equation: text });
  }
  return {
    ok: true,
    request: {
      expression: "",
      format: "text",
      coordinates: "cartesian",
      fields: template.fields.map(([name, kind]) => ({
        name,
        kind,
        depends_on: ["x", "y", "z", "t"],
        smoothness: template.smooth.includes(name) && form.emSmoothness ? form.emSmoothness : null,
      })),
      constants: template.constants,
      assumptions: {},
      action: "em_derive",
      template: template.id,
      premises,
      premise_ids: form.premiseIds,
    },
  };
}

/** A saved request restored into the form (the premises were expanded from ids when it was saved). */
export function restoreVectorForm(
  request: VectorRequest,
  input: string,
  format: string,
  assumptions: Assumptions,
): VectorForm {
  // premises that are handbook equations come back as ticked boxes, the others as text
  const premises = request.premises ?? [];
  const adopted = PREMISE_OPTIONS.filter((o) => premises.some((p) => p.equation === o.equation));
  const own = premises.filter((p) => !adopted.some((o) => o.equation === p.equation));
  return {
    ...blankForm(),
    premiseIds: adopted.map((o) => o.id),
    expression: request.expression || input,
    format: format === "latex" ? "latex" : "text",
    coordinates: request.coordinates,
    fields: request.fields.map((f) => ({
      name: f.name,
      kind: f.kind,
      components: (f.components ?? []).join(", "),
      dependsOn: (f.depends_on ?? []).join(", "),
      smoothness: f.smoothness ?? "",
    })),
    constants: request.constants.join(", "),
    assumptions: request.assumptions ?? assumptions,
    action: request.action,
    premises: own.map((p) => p.equation).join("\n"),
    template: request.template ?? "wave_equation_e",
    emSmoothness:
      request.action === "em_derive"
        ? request.fields.find((f) => f.smoothness)?.smoothness === "smooth"
          ? "smooth"
          : request.fields.some((f) => f.smoothness)
            ? "c2"
            : ""
        : "",
    ruleId: request.rule_id ?? "",
    path: (request.path ?? []).join(", "),
    reverse: request.reverse ?? false,
    target: request.target ?? "",
  };
}
