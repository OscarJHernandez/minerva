/**
 * What each mark on a formula means: one table for the chat (apps/web) and the landing page (apps/site), so a
 * page that shows a recorded conversation says exactly what the chat says. Styling stays in each app.
 */
export type FormulaStatus =
  "from_tool" | "quoted" | "checked" | "conditional" | "contradicted" | "unchecked";

export const MARK_TEXT: Record<FormulaStatus, { label: string | null; title: string }> = {
  from_tool: {
    label: null,
    title: "Matches a formula from a tool result or the handbook",
  },
  quoted: {
    label: "quoted",
    title:
      "Shown in a tool result, but not a verified result of its own: your input as read by Minerva, or a handbook entry that is only stated. It is not marked as checked.",
  },
  checked: {
    label: "checked",
    title:
      "Written by the language model. Minerva tested this equation numerically at sample points and it held. Conditions written beside it are not tested.",
  },
  conditional: {
    label: "conditional",
    title:
      "Written by the language model. Minerva tested this equation: it is not true in general, but it holds under an extra assumption or a definition stated earlier (see the note; Minerva has not verified that assumption or definition).",
  },
  contradicted: {
    label: "contradicted",
    title:
      "Written by the language model. Minerva tested this equation, as an identity (true for every value of its symbols) or against definitions stated earlier, and it does not hold. See the note for which.",
  },
  unchecked: {
    label: "unchecked",
    title:
      "Written by the language model. Minerva did not produce this formula and could not check it. Not being checked is not a finding that it is wrong.",
  },
};

/** An equation or inequality is a claim Minerva can test; a lone symbol such as `r` or `\\theta` is not. */
const RELATION = /=|<|>|\\(?:le|ge|leq|geq|neq|ne|approx|equiv|sim|propto|to|implies)\b/;
export const isClaim = (latex: string) => RELATION.test(latex);

/** Whether a formula shows its mark: an unchecked formula that asserts nothing (a lone symbol) is not flagged. */
export const showsMark = (status: FormulaStatus, latex: string) =>
  status !== "from_tool" && !(status === "unchecked" && !isClaim(latex));
