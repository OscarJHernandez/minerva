import type { MatrixDecl } from "./types";

/** Limits that mirror the engine (`minerva_engine.matrix.models`); the server checks them again. */
export const MAX_MATRICES = 6;
export const MAX_ABSTRACT_SIZE = 8;
export const MAX_EXPLICIT_SIZE = 4;

const NAME = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;
const SIZE_LINE = /^([A-Za-z][A-Za-z0-9_]*)\s+(\d+)(?:\s*[x×]\s*(\d+))?$/;
const ENTRIES_LINE = /^([A-Za-z][A-Za-z0-9_]*)\s*=\s*\[(.*)\]$/;

export type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

/**
 * One matrix per line: `A 2x2` or `A 3` (an abstract size), or `A = [1, 0; 0, 2]` (explicit entries:
 * rows separated by `;`, entries by `,`; entries are LaTeX expressions and may contain parameters).
 */
export function parseMatrices(text: string): Parsed<MatrixDecl[]> {
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length === 0)
    return { ok: false, error: "Declare at least one matrix, for example A 2x2." };
  if (lines.length > MAX_MATRICES)
    return { ok: false, error: `Declare at most ${MAX_MATRICES} matrices.` };
  const out: MatrixDecl[] = [];
  for (const line of lines) {
    const explicit = ENTRIES_LINE.exec(line);
    if (explicit) {
      const [, name = "", body = ""] = explicit;
      const rows = body.split(";").map((r) => r.split(",").map((e) => e.trim()));
      if (rows.length > MAX_EXPLICIT_SIZE || (rows[0]?.length ?? 0) > MAX_EXPLICIT_SIZE)
        return {
          ok: false,
          error: `${name}: explicit matrices are at most ${MAX_EXPLICIT_SIZE}x${MAX_EXPLICIT_SIZE}.`,
        };
      if (rows.some((r) => r.length !== rows[0]?.length || r.some((e) => e === "")))
        return {
          ok: false,
          error: `${name}: every row needs the same number of non-empty entries.`,
        };
      out.push({ name, entries: rows });
      continue;
    }
    const sized = SIZE_LINE.exec(line);
    if (sized) {
      const [, name = "", r = "", c] = sized;
      const rows = Number(r);
      const cols = c === undefined ? rows : Number(c);
      if (rows < 1 || cols < 1 || rows > MAX_ABSTRACT_SIZE || cols > MAX_ABSTRACT_SIZE)
        return { ok: false, error: `${name}: sizes are between 1 and ${MAX_ABSTRACT_SIZE}.` };
      out.push({ name, rows, cols });
      continue;
    }
    return {
      ok: false,
      error: `Could not read “${line}”. Write A 2x2, or A = [1, 0; 0, 2].`,
    };
  }
  const names = out.map((m) => m.name);
  if (new Set(names).size !== names.length)
    return { ok: false, error: "Each matrix is declared once." };
  return { ok: true, value: out };
}

export function formatMatrices(decls: MatrixDecl[]): string {
  return decls
    .map((m) =>
      m.entries
        ? `${m.name} = [${m.entries.map((r) => r.join(", ")).join("; ")}]`
        : `${m.name} ${m.rows}x${m.cols ?? m.rows}`,
    )
    .join("\n");
}

/** `A~B, C~D` declares that A and B commute and that C and D commute. */
export function parsePairs(text: string): Parsed<[string, string][]> {
  const items = text
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const out: [string, string][] = [];
  for (const item of items) {
    const parts = item.split("~").map((p) => p.trim());
    const [a, b] = parts;
    if (parts.length !== 2 || !a || !b || !NAME.test(a) || !NAME.test(b) || a === b)
      return { ok: false, error: `Write commuting pairs as A~B, C~D (not “${item}”).` };
    out.push([a, b]);
  }
  return { ok: true, value: out };
}

export function formatPairs(pairs: [string, string][]): string {
  return pairs.map(([a, b]) => `${a}~${b}`).join(", ");
}

export function parseNames(text: string): Parsed<string[]> {
  const items = text
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const bad = items.find((n) => !NAME.test(n));
  return bad ? { ok: false, error: `“${bad}” is not a matrix name.` } : { ok: true, value: items };
}
