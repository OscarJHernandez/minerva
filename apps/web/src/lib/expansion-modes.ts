import type { ExpansionReport } from "./chat-types";

type Mode = ExpansionReport["modes"][number];

function compare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * The order of the normal modes of an expansion, whatever order the solver found them in. The server numbers
 * the rows by position in this order (`canonical_modes` in `services/api/.../card_equations.py`): the two must
 * stay identical, and a test pins both against the same fixture. Plain code-point comparison (`<`), never
 * `localeCompare`; `Array.prototype.sort` is stable, like Python's.
 */
export function canonicalModes(items: Mode[]): Mode[] {
  return [...items].sort(
    (a, b) =>
      compare(a.omega_squared_latex, b.omega_squared_latex) ||
      compare(a.eigenvector_latex ?? "", b.eigenvector_latex ?? "") ||
      compare(a.omega_latex ?? "", b.omega_latex ?? ""),
  );
}
