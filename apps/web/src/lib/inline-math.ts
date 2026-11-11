import type { Segment } from "./chat-types";

/** Splits text on `$...$` into text and formula segments (used for what the user typed). */
export function parseInlineMath(text: string): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/g)) {
    const start = m.index ?? 0;
    if (start > last) segments.push({ type: "text", value: text.slice(last, start), status: null });
    segments.push({ type: "math", value: (m[1] ?? "").trim(), status: "from_tool" });
    last = start + m[0].length;
  }
  if (last < text.length) segments.push({ type: "text", value: text.slice(last), status: null });
  return segments;
}
