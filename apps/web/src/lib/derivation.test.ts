import { describe, expect, it } from "vitest";
import { step } from "@/test/fixtures";
import { isolateResult, stepResult } from "@/test/fixtures";
import {
  DEFAULT_SETTINGS,
  assumptionsSummary,
  collectTargets,
  continuation,
  continuations,
  relationLines,
  stepLabel,
  validateSettings,
  verificationSummary,
} from "./derivation";

describe("continuation", () => {
  it("carries the text form, not the LaTeX", () => {
    const s = step("a", "simplify", "x+x", { result_text: "2*x", result_latex: "2 x" });
    expect(continuation(s)).toEqual({ text: "2*x", latex: "2 x" });
  });

  it("is allowed for an unchanged result and refused for anything not verified", () => {
    const base = { result_text: "x", result_latex: "x" };
    expect(continuation(step("a", "simplify", "x", base, { status: "unchanged" }))).not.toBeNull();
    for (const status of [
      "not_verified",
      "unsupported",
      "parse_error",
      "conditions_not_satisfied",
    ] as const) {
      expect(continuation(step("a", "simplify", "x", base, { status }))).toBeNull();
    }
  });

  it("is refused when the result has no text form", () => {
    expect(continuation(step("a", "series", "x", { result_text: null }))).toBeNull();
  });
});

describe("validateSettings", () => {
  it("needs nothing for simplify", () => {
    expect(validateSettings("simplify", { ...DEFAULT_SETTINGS, variable: "" })).toBeNull();
  });

  it("checks the variable, a single integral bound, and the series point", () => {
    expect(validateSettings("differentiate", { ...DEFAULT_SETTINGS, variable: "2x" })).toMatch(
      /variable/,
    );
    expect(validateSettings("integrate", { ...DEFAULT_SETTINGS, lower: "0" })).toMatch(
      /both bounds/,
    );
    expect(
      validateSettings("integrate", { ...DEFAULT_SETTINGS, lower: "0", upper: "1" }),
    ).toBeNull();
    expect(validateSettings("series", { ...DEFAULT_SETTINGS, point: "x" })).toMatch(/point/);
    expect(validateSettings("series", { ...DEFAULT_SETTINGS, point: "1/2" })).toBeNull();
  });
});

describe("labels and summaries", () => {
  it("names what produced a step", () => {
    expect(
      stepLabel(step("a", "differentiate", "x**5", { variables: [{ name: "x", order: 2 }] })),
    ).toBe("Differentiate by x, 2 times");
    expect(stepLabel(step("a", "integrate", "x", { variable: "x", definite: true }))).toBe(
      "Definite integral by x",
    );
    expect(
      stepLabel(step("a", "series", "exp(x)", { variable: "x", point_requested: "0", order: 4 })),
    ).toBe("Taylor polynomial in x about 0, order 4");
    expect(
      stepLabel(step("a", "apply_identity", "x", { entry: { title: "Product rule for logs" } })),
    ).toBe("Product rule for logs");
  });

  it("reports only what the stored verification says", () => {
    const s = step("a", "simplify", "x", {
      verification: { numeric_points_checked: 12, numeric_points_agreed: 12, symbolic_zero: true },
    });
    expect(verificationSummary(s)).toBe("12/12 sample points agree; difference simplifies to 0");
    expect(verificationSummary(step("a", "simplify", "x", {}))).toBeNull();
  });

  it("lists assumptions, leaving out symbols with none", () => {
    expect(assumptionsSummary({ x: ["positive", "real"], y: [] })).toBe("x positive, real");
    expect(assumptionsSummary({})).toBeNull();
  });
});

describe("controlled steps", () => {
  it("reads relations one per line and collect targets by comma", () => {
    expect(relationLines("  y^2 = 1 - x^2 \n\n x = 2 ")).toEqual(["y^2 = 1 - x^2", "x = 2"]);
    expect(collectTargets("x, y^2 ,, ")).toEqual(["x", "y^2"]);
  });

  it("validates the settings of substitute, collect and solve", () => {
    const set = (patch: Partial<typeof DEFAULT_SETTINGS>) => ({ ...DEFAULT_SETTINGS, ...patch });
    expect(validateSettings("substitute", set({ relations: "" }))).toMatch(/at least one relation/);
    expect(validateSettings("substitute", set({ relations: "y = 1\n".repeat(5) }))).toMatch(
      /at most 4/,
    );
    expect(validateSettings("substitute", set({ relations: "x" }))).toMatch(/one equals sign/);
    expect(validateSettings("substitute", set({ relations: "a = b = c" }))).toMatch(
      /one equals sign/,
    );
    expect(validateSettings("substitute", set({ relations: "y^2 = 1 - x^2" }))).toBeNull();
    expect(validateSettings("collect", set({ collectBy: " , " }))).toMatch(/what to collect/);
    expect(validateSettings("collect", set({ collectBy: "a,b,c,d,e" }))).toMatch(/at most 4/);
    expect(validateSettings("collect", set({ collectBy: "x" }))).toBeNull();
    expect(validateSettings("isolate", set({ solveFor: "1x" }))).toMatch(/unknown/);
    expect(validateSettings("isolate", set({ solveFor: "t" }))).toBeNull();
  });

  it("does not need the derivative variable for the new operations", () => {
    expect(
      validateSettings("substitute", { ...DEFAULT_SETTINGS, variable: "", relations: "x = 1" }),
    ).toBeNull();
  });

  it("continues a substitution from its text form like any other step", () => {
    const item = step(
      "a",
      "substitute",
      "x^2+y^2",
      stepResult() as unknown as Record<string, unknown>,
    );
    expect(continuation(item)).toEqual({ text: "1", latex: "1" });
    expect(stepLabel(item)).toBe("Substitute a relation");
  });

  it("offers each solution of an equation, and continues automatically only from a single one", () => {
    const two = step(
      "a",
      "isolate",
      "a x^2 = b",
      isolateResult() as unknown as Record<string, unknown>,
    );
    expect(continuation(two)).toBeNull();
    expect(continuation(two, 1)).toEqual({
      text: "-sqrt(b/a)",
      latex: "x = - \\sqrt{\\frac{b}{a}}",
    });
    expect(continuations(two).map((c) => c.text)).toEqual(["sqrt(b/a)", "-sqrt(b/a)"]);
    const one = step("b", "isolate", "a x = b", {
      ...isolateResult(),
      solutions: [isolateResult().solutions[0]!],
    } as unknown as Record<string, unknown>);
    expect(continuation(one)?.text).toBe("sqrt(b/a)");
    const none = step("c", "isolate", "x = 1", {
      ...isolateResult(),
      solutions: [],
    } as unknown as Record<string, unknown>);
    expect(continuation(none)).toBeNull();
    expect(continuations(none)).toEqual([]);
  });

  it("never continues from a solution list that was not verified", () => {
    const failed = step(
      "a",
      "isolate",
      "x = 1",
      isolateResult({ status: "not_verified" }) as unknown as Record<string, unknown>,
      { status: "not_verified" },
    );
    expect(continuation(failed)).toBeNull();
    expect(continuations(failed)).toEqual([]);
  });

  it("labels a rewrite by what it did and whether it acted on one part", () => {
    const label = (request: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
      stepLabel(
        step("a", "rewrite", "x", {
          ...stepResult({ operation: "rewrite" }),
          request,
          ...extra,
        } as unknown as Record<string, unknown>),
      );
    expect(label({ action: "expand" })).toBe("Expand");
    expect(label({ action: "expand", selection: { path: [0] } })).toBe("Expand (one part)");
    expect(label({ action: "collect", by: ["x"] })).toBe("Collect terms");
    expect(label({ action: "rule", part: "x" }, { entry: { title: "Double angle" } })).toBe(
      "Double angle (one part)",
    );
  });

  it("labels solving by the unknown", () => {
    expect(
      stepLabel(
        step("a", "isolate", "x = 1", {
          ...isolateResult(),
          request: { variable: "t" },
        } as unknown as Record<string, unknown>),
      ),
    ).toBe("Solve for t");
  });
});
