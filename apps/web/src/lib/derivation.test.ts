import { describe, expect, it } from "vitest";
import { step } from "@/test/fixtures";
import {
  DEFAULT_SETTINGS,
  assumptionsSummary,
  continuation,
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
