import { describe, expect, it } from "vitest";
import { ALL_EXAMPLES, visibleExamples } from "./homepage-examples";

/**
 * The registry is the single list of cards; the Python tests check each one against the agent evaluation suite
 * (same prompt, reference call, live record). This file freezes what the front end promises: six cards, in this
 * order, with these exact prompts, so a prompt cannot change without the evaluation case changing with it.
 */
const PROMPTS: Record<string, string> = {
  "coulomb-divergence":
    "Consider the vector field F = r/r³ in three-dimensional space. Its classical divergence vanishes everywhere away from the origin, yet its flux through an enclosing sphere is 4π. Derive its distributional divergence and explain the apparent contradiction. Show the mathematical verification.",
  "delta-two-roots":
    "Evaluate the integral of x² DiracDelta(x²−4) over the entire real line. Explain how both roots contribute and verify the result.",
  "stokes-curved-surface":
    "Verify Stokes' theorem for F = (−yz, xz, xy) on the surface r(u,v) = (u,v,uv), where 0 ≤ u,v ≤ 1. Orient the surface by ∂r/∂u × ∂r/∂v and use the induced boundary orientation. Calculate both integrals independently.",
  "heaviside-derivative":
    "For real t and positive gamma, calculate the distributional derivative of exp(−gamma*t) Heaviside(t). Explain why a Dirac delta appears at t=0 and verify the result.",
  "kepler-hidden-symmetries":
    "Find the conserved quantities of the Kepler system defined by L = m/2*(r_dot² + r²*theta_dot²) + k/r, with coordinates r and theta. Search for hidden conserved quantities beyond energy and angular momentum. Verify the discovered quantities and determine their functional independence.",
  "integral-by-symmetry":
    "Evaluate the integral of sin(x)^n / (sin(x)^n + cos(x)^n) from 0 to pi/2, for positive real n. Find an elegant symmetry-based approach and verify the result.",
};

describe("homepage example registry", () => {
  it("has the six examples, in order, with their exact prompts", () => {
    expect(ALL_EXAMPLES.map((e) => e.id)).toEqual(Object.keys(PROMPTS));
    for (const e of ALL_EXAMPLES) expect(e.prompt).toBe(PROMPTS[e.id]);
  });

  it("has no duplicate id, prompt, title or evaluation case", () => {
    for (const key of ["id", "prompt", "title", "eval_case"] as const) {
      const values = ALL_EXAMPLES.map((e) => e[key]);
      expect(new Set(values).size, key).toBe(values.length);
    }
  });

  it("gives every example the fields a release needs", () => {
    for (const e of ALL_EXAMPLES) {
      expect(e.id).toMatch(/^[a-z][a-z0-9-]+$/);
      expect(e.title.length).toBeGreaterThan(0);
      expect(e.description.length).toBeGreaterThan(10);
      expect(e.latex.length).toBeGreaterThan(0);
      expect(e.capability).toMatch(/^[a-z][a-z0-9-]+$/);
      expect(e.eval_case).toMatch(/^homepage-/);
      expect(typeof e.approved).toBe("boolean");
    }
  });

  it("shows only approved examples, in registry order", () => {
    const some = ALL_EXAMPLES.map((e, i) => ({ ...e, approved: i % 2 === 0 }));
    expect(visibleExamples(some).map((e) => e.id)).toEqual([
      "coulomb-divergence",
      "stokes-curved-surface",
      "kepler-hidden-symmetries",
    ]);
    expect(visibleExamples(ALL_EXAMPLES.map((e) => ({ ...e, approved: false })))).toEqual([]);
  });

  it("submits the plain names of the prompts, never LaTeX or a bare delta that needs confirming", () => {
    for (const e of ALL_EXAMPLES) {
      expect(e.prompt).not.toMatch(/\\[a-zA-Z]|\$/); // prose, not a formula for the parser
    }
    // the delta is written with the unambiguous name the distribution tool reads without a question
    expect(PROMPTS["delta-two-roots"]).toMatch(/DiracDelta\(/);
    expect(PROMPTS["heaviside-derivative"]).toMatch(/Heaviside\(/);
  });
});
