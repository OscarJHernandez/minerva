const SEPARATOR = String.raw`,\quad `;

/**
 * The per-state formulas of the statistical-mechanics `occupations` quantity (`\langle n\rangle_{k} = ...`, joined
 * by `,\quad ` in the engine), or null when the text is not exactly that shape. The server numbers each row
 * (`split_occupations` in `services/api/.../card_equations.py`): the two must stay identical, and a test pins both
 * against `packages/ui/fixtures/statmech-occupations-split.json`.
 */
export function splitOccupations(latex: string): string[] | null {
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
  if (depth !== 0) return null;
  const bounds = [-SEPARATOR.length, ...cuts, latex.length];
  const pieces: string[] = [];
  for (let b = 0; b + 1 < bounds.length; b += 1) {
    pieces.push(latex.slice((bounds[b] ?? 0) + SEPARATOR.length, bounds[b + 1]));
  }
  for (let k = 1; k <= pieces.length; k += 1) {
    const head = String.raw`\langle n\rangle_{${k}} = `;
    const piece = pieces[k - 1] ?? "";
    if (!piece.startsWith(head) || piece.slice(head.length).trim() === "") return null;
  }
  return pieces;
}

/**
 * The key the server numbers the k-th occupation under (`card_equations.occupation_key`): `occupation:<k>`, and
 * `occupation:<k>:<m>` for the m-th `occupations` quantity (m >= 2).
 */
export function occupationKey(quantities: { name: string }[], i: number, k: number): string {
  const m = quantities.slice(0, i + 1).filter((q) => q.name === "occupations").length;
  return m === 1 ? `occupation:${k}` : `occupation:${k}:${m}`;
}
