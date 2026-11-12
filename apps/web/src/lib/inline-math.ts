import type { Segment } from "./chat-types";

/**
 * Splits text into prose and formulas: `$$..$$`, `\\[..\\]`, `\\(..\\)`, `$..$` and bare
 * `\\begin{aligned}` style environments, across lines. The same rules as the server's
 * `agent/segments.py`, so a formula looks the same while it streams, after it, and when the user
 * typed it. Only closed delimiters match: an unfinished `$x` or `\\[` stays text until it closes.
 * A `$` follows the pandoc rule, so "$5 to $10" is text. Status is `null`: nothing here was verified.
 */
const MATH =
  /\$\$(?<d1>[\s\S]+?)\$\$|\\\[(?<d2>[\s\S]+?)\\\]|\\\((?<i2>[\s\S]+?)\\\)|(?<d3>\\begin\{(?<env>align\*?|aligned|equation\*?|gather\*?|eqnarray\*?)\}[\s\S]+?\\end\{\k<env>\})|(?<![\\$])\$(?![\s$])(?<i1>[^$\n]*?[^$\s\\])\$(?![\d$])/g;

export function splitMath(text: string): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MATH)) {
    const g = m.groups ?? {};
    const start = m.index ?? 0;
    if (start > last) segments.push({ type: "text", value: text.slice(last, start), status: null });
    const display = g.d1 !== undefined || g.d2 !== undefined || g.d3 !== undefined;
    const body = g.d1 ?? g.d2 ?? g.i2 ?? g.d3 ?? g.i1 ?? "";
    segments.push({ type: display ? "display_math" : "math", value: body.trim(), status: null });
    last = start + m[0].length;
  }
  if (last < text.length) segments.push({ type: "text", value: text.slice(last), status: null });
  return segments;
}
