import { describe, expect, it } from "vitest";
import {
  BLANK_ODE_FORM,
  ODE_EXAMPLES,
  blankBoundaryForm,
  blankNumericForm,
  blankSolveForm,
  buildBoundaryRequest,
  buildSolveRequest,
  buildNumericRequest,
  buildOdeInput,
  buildSymbolicRequest,
  conditionText,
  parseNumber,
  prefillFromConditions,
  restoreOdeForm,
  stateFields,
  stateKey,
} from "./ode-form";
import type { OdeInput, OdeInterpretation } from "./ode-types";

const meaning: OdeInterpretation = {
  unknowns: ["y"],
  independent: "t",
  parameters: ["c"],
  orders: { y: 2 },
  equations_latex: ["y'' + c y' + y = 0"],
  conditions_latex: [],
  notes: [],
};

const base: OdeInput = {
  equations: ["y'' + c y' + y = 0"],
  format: "latex",
  unknowns: [],
  independent: "t",
  assumptions: {},
  conditions: [],
};

function filled() {
  const form = blankNumericForm(meaning);
  form.start = "0";
  form.initial = { "y:0": "1", "y:1": "0" };
  form.parameters = { c: "0.5" };
  form.tEnd = "10";
  return form;
}

describe("buildOdeInput", () => {
  it("reads one equation per line and an optional independent variable", () => {
    const built = buildOdeInput({
      ...BLANK_ODE_FORM,
      equations: "x' = -y\n x'' = x \n",
      independent: " t ",
      conditions: "x(0) = 1; y(0) = 0",
    });
    expect(built).toEqual({
      ok: true,
      request: {
        equations: ["x' = -y", "x'' = x"],
        format: "latex",
        unknowns: [],
        independent: "t",
        assumptions: {},
        conditions: ["x(0) = 1", "y(0) = 0"],
      },
    });
  });

  it("leaves the independent variable out when it is empty: Minerva will ask, not guess", () => {
    const built = buildOdeInput({ ...BLANK_ODE_FORM, equations: "y' = y" });
    expect(built.ok && built.request.independent).toBeNull();
  });

  it.each([
    ["", "Write an equation"],
    ["y' + y", "equals sign"],
    [Array(9).fill("y' = y").join("\n"), "At most 8"],
  ])("refuses %j", (equations, message) => {
    const built = buildOdeInput({ ...BLANK_ODE_FORM, equations });
    expect(built).toEqual({ ok: false, error: expect.stringContaining(message) });
  });

  it("refuses a condition without an equals sign", () => {
    const built = buildOdeInput({ ...BLANK_ODE_FORM, equations: "y' = y", conditions: "y(0)" });
    expect(built.ok).toBe(false);
  });

  it("has examples that all build", () => {
    for (const example of ODE_EXAMPLES) {
      expect(buildOdeInput(example.form).ok, example.id).toBe(true);
    }
  });
});

describe("parseNumber", () => {
  it.each(["1", "-0.5", ".5", "2.", "1e-3", "+4E2", " 7 "])("accepts %j", (text) => {
    expect(parseNumber(text)).not.toBeNull();
  });
  it.each(["", "1/2", "pi", "1,5", "e", "--1", "1e", "0x10", "NaN", "Infinity"])(
    "refuses %j",
    (text) => {
      expect(parseNumber(text)).toBeNull();
    },
  );
});

describe("stateFields", () => {
  it("lists every unknown and derivative below its order", () => {
    const fields = stateFields({ ...meaning, unknowns: ["x", "y"], orders: { x: 1, y: 2 } });
    expect(fields.map((f) => f.key)).toEqual(["x", "y", "y'"]);
    expect(fields.map(stateKey)).toEqual(["x:0", "y:0", "y:1"]);
  });
});

describe("buildNumericRequest", () => {
  it("sends exactly the numbers the user typed, as structured conditions", () => {
    const built = buildNumericRequest(base, meaning, filled());
    expect(built).toEqual({
      ok: true,
      request: {
        ...base,
        conditions: [
          { unknown: "y", order: 0, at: "0", value: "1" },
          { unknown: "y", order: 1, at: "0", value: "0" },
        ],
        kind: "ivp",
        parameters: { c: 0.5 },
        t_end: 10,
        method: "auto",
        rtol: 1e-8,
        atol: 1e-10,
        samples: 200,
      },
    });
  });

  it("starts from the point the user gave, which may be negative or not zero", () => {
    const form = { ...filled(), start: "-1.5", tEnd: "3" };
    const built = buildNumericRequest(base, meaning, form);
    expect(built.ok && built.request.conditions).toEqual([
      { unknown: "y", order: 0, at: "-1.5", value: "1" },
      { unknown: "y", order: 1, at: "-1.5", value: "0" },
    ]);
  });

  it("invents nothing: every missing number is an error naming it", () => {
    const blank = blankNumericForm(meaning);
    expect(buildNumericRequest(base, meaning, blank)).toEqual({
      ok: false,
      error: expect.stringContaining("point where the initial values hold"),
    });
    const noInitial = { ...blank, start: "0" };
    expect(buildNumericRequest(base, meaning, noInitial)).toEqual({
      ok: false,
      error: "Give the initial value of y as a number.",
    });
    const noDerivative = { ...noInitial, initial: { "y:0": "1", "y:1": "" } };
    expect(buildNumericRequest(base, meaning, noDerivative)).toEqual({
      ok: false,
      error: "Give the initial value of y' as a number.",
    });
    const noParameter = { ...filled(), parameters: { c: "" } };
    expect(buildNumericRequest(base, meaning, noParameter)).toEqual({
      ok: false,
      error: "Give a number for the parameter c.",
    });
    const noEnd = { ...filled(), tEnd: "" };
    expect(buildNumericRequest(base, meaning, noEnd)).toEqual({
      ok: false,
      error: "Give the end of the interval as a number.",
    });
  });

  it("refuses an empty interval and tolerances out of range", () => {
    expect(buildNumericRequest(base, meaning, { ...filled(), tEnd: "0" })).toEqual({
      ok: false,
      error: expect.stringContaining("empty"),
    });
    expect(buildNumericRequest(base, meaning, { ...filled(), rtol: "1" }).ok).toBe(false);
    expect(buildNumericRequest(base, meaning, { ...filled(), atol: "1e-20" }).ok).toBe(false);
  });

  it("keeps the equations and assumptions of the analysis unchanged", () => {
    const withAssumptions = { ...base, assumptions: { c: ["positive" as const] } };
    const built = buildNumericRequest(withAssumptions, meaning, filled());
    expect(built.ok && built.request.assumptions).toEqual({ c: ["positive"] });
    expect(built.ok && built.request.equations).toEqual(base.equations);
  });
});

describe("buildSymbolicRequest", () => {
  it("sends no conditions when none are filled in", () => {
    const built = buildSymbolicRequest({ ...base, conditions: ["y(0) = 1"] }, meaning, "", {});
    expect(built).toEqual({ ok: true, request: { ...base, conditions: [] } });
  });

  it("needs the point as soon as one value is given", () => {
    expect(buildSymbolicRequest(base, meaning, "", { "y:0": "1" }).ok).toBe(false);
    const built = buildSymbolicRequest(base, meaning, "0", { "y:0": "1", "y:1": " " });
    expect(built.ok && built.request.conditions).toEqual([
      { unknown: "y", order: 0, at: "0", value: "1" },
    ]);
  });
});

describe("prefillFromConditions", () => {
  it("takes over plain numeric conditions typed with the equation", () => {
    expect(prefillFromConditions(["y(0) = 1", "y'(0) = -0.5"], meaning)).toEqual({
      start: "0",
      initial: { "y:0": "1", "y:1": "-0.5" },
    });
  });

  it("takes over structured conditions too", () => {
    expect(
      prefillFromConditions([{ unknown: "y", order: 1, at: "2", value: "3" }], meaning),
    ).toEqual({ start: "2", initial: { "y:0": "", "y:1": "3" } });
  });

  it("leaves expressions, unknown names and too-high derivatives blank", () => {
    expect(
      prefillFromConditions(["y(0) = 1/2", "z(0) = 1", "y''(0) = 3", "y(pi) = 0"], meaning),
    ).toEqual({ start: "", initial: { "y:0": "", "y:1": "" } });
  });
});

describe("restoreOdeForm", () => {
  it("puts a saved request back into the form, structured conditions written as the user would", () => {
    const form = restoreOdeForm({
      ...base,
      equations: ["x' = -y", "y' = x"],
      independent: null,
      conditions: [{ unknown: "y", order: 1, at: "0", value: "2" }, "x(0) = 1"],
      assumptions: { k: ["positive"] },
    });
    expect(form).toEqual({
      equations: "x' = -y\ny' = x",
      format: "latex",
      independent: "",
      conditions: "y'(0) = 2\nx(0) = 1",
      assumptions: { k: ["positive"] },
    });
    expect(conditionText({ unknown: "y", order: 2, at: "1", value: "0" })).toBe("y''(1) = 0");
  });

  it("round-trips through the builder", () => {
    for (const example of ODE_EXAMPLES) {
      const built = buildOdeInput(example.form);
      if (!built.ok) throw new Error("example does not build");
      expect(restoreOdeForm(built.request)).toEqual({
        ...example.form,
        independent: example.form.independent.trim(),
        conditions: example.form.conditions
          .split(/\n|;/)
          .map((c) => c.trim())
          .filter(Boolean)
          .join("\n"),
      });
    }
  });
});

describe("buildSolveRequest", () => {
  const series = () => ({
    ...blankSolveForm("series", meaning, base),
    start: "0",
    initial: { "y:0": "1", "y:1": "0" },
  });

  it("builds a series request from exactly what was typed", () => {
    const built = buildSolveRequest(base, meaning, series());
    expect(built).toEqual({
      ok: true,
      request: {
        ...base,
        conditions: [
          { unknown: "y", order: 0, at: "0", value: "1" },
          { unknown: "y", order: 1, at: "0", value: "0" },
        ],
        method: "series",
        order: 6,
        small_parameter: null,
        equilibrium: null,
      },
    });
  });

  it("needs every initial value and the point for a series", () => {
    expect(buildSolveRequest(base, meaning, { ...series(), start: "" })).toEqual({
      ok: false,
      error: expect.stringContaining("point where the initial values hold"),
    });
    expect(
      buildSolveRequest(base, meaning, { ...series(), initial: { "y:0": "1", "y:1": "" } }),
    ).toEqual({ ok: false, error: "Give the initial value of y' as a number." });
  });

  it.each([["0"], ["13"], ["2.5"], ["x"]])("refuses the series order %j", (order) => {
    expect(buildSolveRequest(base, meaning, { ...series(), order }).ok).toBe(false);
  });

  it("never chooses the small parameter", () => {
    const form = { ...series(), method: "perturbation" as const, order: "2" };
    expect(buildSolveRequest(base, meaning, form)).toEqual({
      ok: false,
      error: expect.stringContaining("Minerva does not choose it"),
    });
    expect(buildSolveRequest(base, meaning, { ...form, smallParameter: "zeta" }).ok).toBe(false);
    const built = buildSolveRequest(base, meaning, { ...form, smallParameter: "c" });
    expect(built.ok && built.request).toMatchObject({
      method: "perturbation",
      small_parameter: "c",
      equilibrium: null,
      order: 2,
    });
  });

  it("an equilibrium makes it the periodic method: no initial values, a higher order allowed", () => {
    const form = {
      ...blankSolveForm("perturbation", meaning, base),
      smallParameter: "c",
      equilibrium: " 0 ",
      order: "5",
    };
    const built = buildSolveRequest(base, meaning, form);
    expect(built.ok && built.request).toMatchObject({
      conditions: [],
      equilibrium: "0",
      order: 5,
    });
    expect(buildSolveRequest(base, meaning, { ...form, equilibrium: "", order: "5" }).ok).toBe(
      false,
    );
  });
});

describe("buildBoundaryRequest", () => {
  const form = () => ({
    ...blankBoundaryForm({ ...base, conditions: ["y(0) = 0", "y(1) = 1"] }, meaning),
    parameters: { c: "0.5" },
  });

  it("carries the conditions as typed and leaves the interval to their two points", () => {
    expect(
      blankBoundaryForm({ ...base, conditions: ["y(0) = 0", "y(1) = 1"] }, meaning).conditions,
    ).toBe("y(0) = 0\ny(1) = 1");
    const built = buildBoundaryRequest(base, meaning, form());
    expect(built.ok && built.request).toMatchObject({
      kind: "bvp",
      conditions: ["y(0) = 0", "y(1) = 1"],
      parameters: { c: 0.5 },
      t_end: null,
      rtol: 1e-8,
    });
  });

  it("refuses missing conditions, parameters and tolerances out of range", () => {
    expect(buildBoundaryRequest(base, meaning, { ...form(), conditions: "" }).ok).toBe(false);
    expect(buildBoundaryRequest(base, meaning, { ...form(), conditions: "y(0)" }).ok).toBe(false);
    expect(buildBoundaryRequest(base, meaning, { ...form(), parameters: { c: "" } })).toEqual({
      ok: false,
      error: "Give a number for the parameter c.",
    });
    expect(buildBoundaryRequest(base, meaning, { ...form(), rtol: "1" }).ok).toBe(false);
    expect(buildBoundaryRequest(base, meaning, { ...form(), atol: "0" }).ok).toBe(false);
  });
});
