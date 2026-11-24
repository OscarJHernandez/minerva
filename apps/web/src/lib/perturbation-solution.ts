const HEAD = "u(t) = ";
const TAIL = String.raw`\omega t = \tau`;
const SEPARATOR = String.raw`,\quad `;
const REMAINDER = String.raw` + O\left(`;

/**
 * The two relations of a perturbation `solution_latex`: `u(t) = ... + O(...)` and `\omega t = \tau`, or null when
 * the text is not exactly that shape. The server numbers only the first (`split_solution` in
 * `services/api/.../card_equations.py`): the two must stay identical, and a test pins both against
 * `packages/ui/fixtures/perturbation-solution-split.json`.
 */
export function splitSolution(latex: string): { relation: string; definition: string } | null {
  let depth = 0;
  const cuts: number[] = [];
  let i = 0;
  while (i < latex.length) {
    if (latex.startsWith(String.raw`\left`, i)) {
      depth += 1;
      i += String.raw`\left`.length;
      continue;
    }
    if (latex.startsWith(String.raw`\right`, i)) {
      depth -= 1;
      i += String.raw`\right`.length;
      continue;
    }
    const ch = latex[i];
    if (ch === "{") depth += 1;
    else if (ch === "}") depth -= 1;
    else if (depth === 0 && latex.startsWith(SEPARATOR, i)) cuts.push(i);
    if (depth < 0) return null;
    i += 1;
  }
  const cut = cuts[0];
  if (depth !== 0 || cuts.length !== 1 || cut === undefined) return null;
  const head = latex.slice(0, cut);
  const tail = latex.slice(cut + SEPARATOR.length);
  if (
    tail !== TAIL ||
    !head.startsWith(HEAD) ||
    !head.endsWith(String.raw`\right)`) ||
    head.split(REMAINDER).length !== 2 ||
    head.length === HEAD.length
  ) {
    return null;
  }
  return { relation: head, definition: tail };
}
