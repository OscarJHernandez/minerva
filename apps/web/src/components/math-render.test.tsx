import { Math } from "@minerva/ui/math";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

const html = (latex: string, display = false) =>
  render(<Math latex={latex} display={display} />).container;

describe("Math", () => {
  it.each([
    ["imaginary unit", "\\exponentialE^{\\imaginaryI\\pi}"],
    ["upright constants", "\\mathrm{e}^{\\mathrm{i}x}"],
    ["greek and subscripts", "\\omega_{0}^2 = \\frac{k}{m} + \\alpha_1"],
    ["fractions", "\\frac{x^2-1}{x-1}"],
    ["integral", "\\int_0^\\infty \\mathrm{e}^{-a x}\\,dx = \\frac{1}{a}"],
    ["tensor", "R^{\\mu}{}_{\\nu\\rho\\sigma} = R_{\\rho\\sigma}{}^{\\mu}{}_{\\nu}"],
    ["differential", "\\int f\\,\\differentialD x"],
    ["multiline", "\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}"],
  ])("renders %s without an error", (_name, latex) => {
    const c = html(latex, true);
    expect(c.querySelector("[data-math-error]")).toBeNull();
    expect(c.querySelector(".katex")).not.toBeNull();
    expect(c.querySelector(".katex-error")).toBeNull();
  });

  it("renders the aliases as upright e and i", () => {
    const c = html("\\exponentialE^{\\imaginaryI}");
    expect(c.querySelector(".mathrm")).not.toBeNull();
  });

  it.each([
    ["unknown command", "\\foo{x}"],
    ["unbalanced braces", "\\frac{1"],
    ["unsupported environment", "\\begin{nonsense} x \\end{nonsense}"],
  ])("shows %s as source marked as an error, never as a formula", (_name, latex) => {
    const c = html(latex);
    const err = c.querySelector("[data-math-error]");
    expect(err?.textContent).toBe(latex);
    expect(err?.getAttribute("title")).toMatch(/cannot be displayed/);
    expect(c.querySelector(".katex")).toBeNull();
  });
});
