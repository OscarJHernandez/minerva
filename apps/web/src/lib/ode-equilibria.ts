import type { Equilibrium } from "./ode-types";

/**
 * The order of the equilibria of a qualitative analysis, and of the eigenvalues of one of them, whatever
 * order the solver found them in. The server numbers the rows by position in this order
 * (`canonical_equilibria` and `canonical_eigenvalues` in `services/api/.../card_equations.py`): the two must
 * stay identical, and a test pins both against the same fixture. Plain code-point comparison (`<`), never
 * `localeCompare`, so the result does not depend on the browser's locale.
 */
function compare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function compareLists(a: string[], b: string[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    const c = compare(a[i] ?? "", b[i] ?? "");
    if (c !== 0) return c;
  }
  return a.length - b.length;
}

export function canonicalEquilibria(items: Equilibrium[]): Equilibrium[] {
  return [...items].sort(
    (a, b) =>
      compareLists(a.point_latex, b.point_latex) ||
      compare(a.jacobian_latex ?? "", b.jacobian_latex ?? "") ||
      compare(a.characteristic_latex ?? "", b.characteristic_latex ?? ""),
  );
}

export function canonicalEigenvalues(items: string[]): string[] {
  return [...items].sort(compare);
}
