import type { AssistantContent } from "./chat-types";

/**
 * A reply as Markdown with `$...$` and `$$...$$` formulas, to copy into notes or a document.
 * Equation numbers are written after the formula as shown. Only what is on screen is copied:
 * prose, and each card's main result.
 */
export function replyToMarkdown(content: Partial<AssistantContent>): string {
  const parts: string[] = [];
  for (const block of content.blocks ?? []) {
    if (block.kind === "card") {
      const latex = (block.card.data as { result_latex?: unknown } | undefined)?.result_latex;
      if (typeof latex === "string" && latex !== "") {
        const n = block.card.number != null ? ` (${block.card.number})` : "";
        parts.push(`$$${latex}$$${n}`);
      }
      continue;
    }
    const text = block.segments
      .map((s) => {
        if (s.type === "math") return `$${s.value}$`;
        if (s.type === "display_math") {
          const n = s.number != null ? ` (${s.number})` : "";
          return `\n\n$$${s.value}$$${n}\n\n`;
        }
        return s.value;
      })
      .join("")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    if (text !== "") parts.push(text);
  }
  return parts.join("\n\n");
}
