import { describe, expect, it } from "vitest";
import { blankForm } from "@/lib/approximation-form";
import { buildApproximationRequest } from "./approximation-workbench";

const input = { expression: "\\exp(x)", variable: "x", positive: "", real: "", form: blankForm() };

describe("buildApproximationRequest", () => {
  it("sends only what was typed, never saves, and declares only the assumptions given", () => {
    const request = buildApproximationRequest({
      ...input,
      positive: "c, m",
      real: "v",
      form: { ...blankForm("perturbation"), small: "v/c", order: "4" },
    });
    expect(request).toEqual({
      input: "\\exp(x)",
      format: "latex",
      variable: "x",
      assumptions: { c: ["positive"], m: ["positive"], v: ["real"] },
      method: "perturbation",
      small: "v/c",
      order: 4,
      save: false,
    });
  });

  it("explains what is missing or wrong", () => {
    expect(buildApproximationRequest({ ...input, expression: " " })).toMatch(
      /Enter the expression/,
    );
    expect(buildApproximationRequest({ ...input, variable: "1x" })).toMatch(/name/);
    expect(buildApproximationRequest({ ...input, positive: "c d" })).toMatch(/not a symbol name/);
    expect(buildApproximationRequest({ ...input, form: { ...blankForm(), lower: "0" } })).toMatch(
      /both ends/,
    );
  });
});
