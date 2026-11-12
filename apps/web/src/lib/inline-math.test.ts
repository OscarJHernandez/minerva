import { describe, expect, it } from "vitest";
import { splitMath } from "./inline-math";

describe("splitMath", () => {
  it("splits text and formulas", () => {
    const segs = splitMath("Simplify $\\frac{x^2-1}{x-1}$ now");
    expect(segs.map((s) => [s.type, s.value])).toEqual([
      ["text", "Simplify "],
      ["math", "\\frac{x^2-1}{x-1}"],
      ["text", " now"],
    ]);
  });

  it("leaves escaped dollars and unmatched dollars as text", () => {
    expect(splitMath("costs \\$5 and $x").every((s) => s.type === "text")).toBe(true);
  });

  const kinds = (t: string) =>
    splitMath(t)
      .filter((s) => s.type !== "text")
      .map((s) => [s.type, s.value]);

  it("understands every delimiter, as the server does", () => {
    expect(kinds("a $x$ b $$y$$ c \\(z\\) d \\[w\\]")).toEqual([
      ["math", "x"],
      ["display_math", "y"],
      ["math", "z"],
      ["display_math", "w"],
    ]);
  });

  it("keeps multiline display math and bare environments whole", () => {
    expect(kinds("\\[\n a = b \\\\\n c = d\n\\]")).toEqual([
      ["display_math", "a = b \\\\\n c = d"],
    ]);
    const env = "\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}";
    expect(kinds(`see ${env} done`)).toEqual([["display_math", env]]);
  });

  it("does not treat prices as formulas", () => {
    expect(kinds("It costs $5 to $10 in total")).toEqual([]);
    expect(kinds("costs \\$5 and \\$6")).toEqual([]);
  });

  it("leaves an unfinished formula as text while streaming, and renders it once it closes", () => {
    for (const partial of ["so $x^", "so $$x^2 +", "so \\[x", "\\begin{aligned} a &= b", "$$x$"]) {
      expect(kinds(partial)).toEqual([]);
    }
    expect(kinds("so $x^2$ and $$y")).toEqual([["math", "x^2"]]);
  });

  it("marks nothing as verified", () => {
    expect(splitMath("$x$ and $$y$$").every((s) => s.status === null)).toBe(true);
  });

  it("passes the constants and unsupported commands through untouched (rendering decides)", () => {
    expect(kinds("$\\exponentialE^{\\imaginaryI\\pi}$ and $\\foo{x}$")).toEqual([
      ["math", "\\exponentialE^{\\imaginaryI\\pi}"],
      ["math", "\\foo{x}"],
    ]);
  });
});
