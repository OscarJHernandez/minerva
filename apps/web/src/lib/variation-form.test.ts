import { describe, expect, it } from "vitest";
import {
  EXAMPLES,
  blankCondition,
  blankForm,
  buildVariationRequest,
  parseGiven,
  restoreVariationForm,
  type VariationForm,
} from "./variation-form";

const harmonic = (): VariationForm => ({
  ...blankForm(),
  integrand: "\\frac12\\dot{x}^2-\\frac12 x^2",
  lower: "0",
  upper: "1",
  conditions: [
    { function: "x", order: "0", end: "lower", kind: "fixed", value: "0" },
    { function: "x", order: "0", end: "upper", kind: "fixed", value: "1" },
  ],
});

function ok(form: VariationForm, action: Parameters<typeof buildVariationRequest>[1]) {
  const built = buildVariationRequest(form, action);
  if (!built.ok) throw new Error(built.error);
  return built.request;
}
function error(form: VariationForm, action: Parameters<typeof buildVariationRequest>[1]) {
  const built = buildVariationRequest(form, action);
  if (built.ok) throw new Error("expected an error");
  return built.error;
}

describe("buildVariationRequest", () => {
  it("builds a symbolic request with exactly the conditions typed and no numbers", () => {
    const r = ok(harmonic(), "first");
    expect(r.action).toBe("first");
    expect(r.functions).toEqual(["x"]);
    expect(r.conditions).toEqual([
      { function: "x", order: 0, end: "lower", kind: "fixed", value: "0" },
      { function: "x", order: 0, end: "upper", kind: "fixed", value: "1" },
    ]);
    expect(r.values).toEqual({});
    expect(r.paths).toEqual([]);
  });

  it("never invents a condition, an interval or a bound", () => {
    const f = { ...blankForm(), integrand: "\\frac12\\dot{x}^2" };
    const r = ok(f, "first");
    expect(r.conditions).toEqual([]);
    expect(r.lower).toBeNull();
    expect(r.upper).toBeNull();
    expect(r.lower_term).toBeNull();
  });

  it("does not need a value for a fixed condition in a symbolic action, but does for evaluate and solve", () => {
    const f = { ...harmonic(), conditions: [{ ...blankCondition(), function: "x" }] };
    expect(ok(f, "first").conditions[0]).toEqual({
      function: "x",
      order: 0,
      end: "lower",
      kind: "fixed",
    });
    expect(error(f, "solve")).toMatch(/Give the value/);
    expect(error({ ...f, paths: { x: "t" } }, "evaluate")).toMatch(/Give the value/);
  });

  it("validates names, orders and duplicates", () => {
    expect(error({ ...harmonic(), integrand: " " }, "first")).toMatch(/integrand/);
    expect(error({ ...harmonic(), functions: "" }, "first")).toMatch(/unknown function/);
    expect(error({ ...harmonic(), functions: "x, x" }, "first")).toMatch(/different names/);
    expect(error({ ...harmonic(), functions: "1x" }, "first")).toMatch(/not a valid/);
    const c = (o: string) => [
      { function: "x", order: o, end: "lower" as const, kind: "free" as const, value: "" },
    ];
    expect(error({ ...harmonic(), conditions: c("1.5") }, "first")).toMatch(/whole number/);
    expect(error({ ...harmonic(), conditions: c("9") }, "first")).toMatch(/at most 7/);
    expect(error({ ...harmonic(), conditions: [...c("0"), ...c("0")] }, "first")).toMatch(
      /two conditions/,
    );
    const other = [{ ...c("0")[0]!, function: "y" }];
    expect(error({ ...harmonic(), conditions: other }, "first")).toMatch(/not one of the unknown/);
  });

  it("parses given functions and refuses a function that is also an unknown", () => {
    expect(parseGiven("V(x), W(x, t)")).toEqual([
      { name: "V", args: ["x"] },
      { name: "W", args: ["x", "t"] },
    ]);
    expect(parseGiven("V")).toMatch(/not a given function/);
    expect(error({ ...harmonic(), givenFunctions: "x(t)" }, "first")).toMatch(/both an unknown/);
  });

  it("numeric actions need an interval, a path and definitions the user typed", () => {
    expect(error({ ...harmonic(), upper: "", paths: { x: "t" } }, "evaluate")).toMatch(/interval/);
    expect(error(harmonic(), "evaluate")).toMatch(/trial path/);
    expect(error(harmonic(), "verify")).toMatch(/trial path/);
    const withGiven = { ...harmonic(), givenFunctions: "V(x)", paths: { x: "t" } };
    expect(error(withGiven, "evaluate")).toMatch(/Define the given function V\(x\)/);
    const r = ok(
      { ...withGiven, definitions: { V: "x^2" }, values: { m: "3", k: " " } },
      "evaluate",
    );
    expect(r.definitions).toEqual({ V: "x^2" });
    expect(r.values).toEqual({ m: "3" }); // blank values are not sent
    expect(r.paths).toEqual([{ analytic: { x: "t" }, sampled: {} }]);
  });

  it("a stationary path needs conditions but not a starting guess; a guess must cover every function", () => {
    const r = ok({ ...harmonic(), withSecond: true }, "solve");
    expect(r.paths).toEqual([]);
    expect(r.with_second).toBe(true);
    expect(error({ ...harmonic(), conditions: [] }, "solve")).toMatch(/endpoint conditions/);
    expect(error({ ...harmonic(), functions: "x, y", paths: { x: "t" } }, "solve")).toMatch(
      /every unknown function/,
    );
  });

  it("with_second and sweeps belong to their actions only", () => {
    const f = {
      ...harmonic(),
      paths: { x: "c*t" },
      sweepName: "c",
      sweepValues: "1, 2, 3",
      withSecond: true,
    };
    const r = ok(f, "evaluate");
    expect(r.sweep).toEqual({ name: "c", values: ["1", "2", "3"] });
    expect(r.with_second).toBe(false);
    expect(error({ ...f, sweepValues: "1" }, "evaluate")).toMatch(/2 to 6/);
    expect(ok(f, "verify").sweep).toBeNull();
  });
});

describe("restoreVariationForm", () => {
  it("round-trips a request", () => {
    const f = {
      ...harmonic(),
      givenFunctions: "V(x)",
      definitions: { V: "x^2" },
      paths: { x: "t" },
      values: { m: "3" },
    };
    const r = ok(f, "evaluate");
    const back = restoreVariationForm(r);
    expect(back).toEqual({ ...f, withSecond: false });
    expect(ok(back, "evaluate")).toEqual(r);
  });

  it("restores a model-made request (no numbers) and every example builds", () => {
    const back = restoreVariationForm({ integrand: "L", functions: ["x", "y"], conditions: [] });
    expect(back.functions).toBe("x, y");
    expect(back.values).toEqual({});
    for (const e of EXAMPLES) expect(buildVariationRequest(e.form, "first").ok).toBe(true);
  });
});

describe("second variation along a path", () => {
  it("is symbolic unless the user typed a path", () => {
    const symbolic = ok(harmonic(), "second");
    expect(symbolic.paths).toEqual([]);
    const along = ok({ ...harmonic(), paths: { x: "t" } }, "second");
    expect(along.paths).toEqual([{ analytic: { x: "t" }, sampled: {} }]);
    expect(error({ ...harmonic(), upper: "", paths: { x: "t" } }, "second")).toMatch(/interval/);
  });
});
