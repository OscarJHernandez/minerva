import { describe, expect, it } from "vitest";
import { parseInlineMath } from "./inline-math";

describe("parseInlineMath", () => {
  it("splits text and formulas", () => {
    const segs = parseInlineMath("Simplify $\\frac{x^2-1}{x-1}$ now");
    expect(segs.map((s) => [s.type, s.value])).toEqual([
      ["text", "Simplify "],
      ["math", "\\frac{x^2-1}{x-1}"],
      ["text", " now"],
    ]);
  });

  it("leaves escaped dollars and unmatched dollars as text", () => {
    expect(parseInlineMath("costs \\$5 and $x").every((s) => s.type === "text")).toBe(true);
  });
});
