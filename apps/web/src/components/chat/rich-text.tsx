import { Math } from "@minerva/ui/math";
import type { FormulaStatus, Segment } from "@/lib/chat-types";

const MARKS: Record<
  FormulaStatus,
  { label: string | null; title: string; underline: string; box: string; tone: string }
> = {
  from_tool: {
    label: null,
    title: "Matches a formula from a tool result or the handbook",
    underline: "",
    box: "border-rule bg-raised",
    tone: "",
  },
  checked: {
    label: "checked",
    title:
      "Written by the language model. Minerva tested this equation numerically at sample points and it held. Conditions written beside it are not tested.",
    underline: "border-b border-dotted border-ok",
    box: "border-ok/40 bg-ok-soft/40",
    tone: "text-ok",
  },
  contradicted: {
    label: "contradicted",
    title:
      "Written by the language model. Minerva tested this equation numerically and it is false (at the stated assumptions, or for complex values if none were declared).",
    underline: "border-b border-solid border-bad",
    box: "border-bad/50 bg-bad-soft/50",
    tone: "text-bad",
  },
  unchecked: {
    label: "unchecked",
    title:
      "Written by the language model. Minerva did not produce this formula and could not check it.",
    underline: "border-b border-dotted border-warn",
    box: "border-warn/50 bg-warn-soft/40",
    tone: "text-warn",
  },
};

/** `**bold**`, `` `code` `` and line breaks: the only markup the assistant may use. */
function inline(text: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let index = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    if (start > last) parts.push(text.slice(last, start));
    const token = match[0];
    parts.push(
      token.startsWith("**") ? (
        <strong key={`${keyPrefix}-b${index}`}>{token.slice(2, -2)}</strong>
      ) : (
        <code
          key={`${keyPrefix}-c${index}`}
          className="rounded bg-sunk px-1 font-mono text-[0.9em]"
        >
          {token.slice(1, -1)}
        </code>
      ),
    );
    last = start + token.length;
    index += 1;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function RichText({ segments }: { segments: Segment[] }) {
  return (
    <div className="space-y-2 leading-relaxed">
      {segments.map((s, i) => {
        if (s.type === "text") {
          return (
            <span key={i} className="whitespace-pre-wrap">
              {inline(s.value, String(i))}
            </span>
          );
        }
        const mark = MARKS[s.status ?? "unchecked"];
        if (s.type === "math") {
          return (
            <span
              key={i}
              title={mark.title}
              data-status={s.status}
              className={mark.underline || undefined}
            >
              <Math latex={s.value} />
              {mark.label && (
                <sup className={`ml-0.5 text-[10px] font-medium ${mark.tone}`}>{mark.label}</sup>
              )}
            </span>
          );
        }
        return (
          <div
            key={i}
            data-status={s.status}
            className={`my-2 rounded-md border px-3 py-1 ${mark.box}`}
          >
            <Math latex={s.value} display />
            {mark.label && (
              <div className={`pb-1 text-xs ${mark.tone}`} title={mark.title}>
                {s.status === "checked"
                  ? "Written by the model; checked numerically by Minerva"
                  : s.status === "contradicted"
                    ? "Written by the model; Minerva's check says this is false"
                    : "Not verified by Minerva (written by the model)"}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
