import { Math } from "@minerva/ui/math";
import type { Segment } from "@/lib/chat-types";

const UNCHECKED_TITLE =
  "Written by the language model. Minerva did not produce or check this formula; it does not appear in any tool result or handbook entry of this conversation.";

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
        const unchecked = s.verified === false;
        if (s.type === "math") {
          return (
            <span
              key={i}
              title={
                unchecked ? UNCHECKED_TITLE : "Matches a formula from a tool result or the handbook"
              }
              data-verified={s.verified ? "true" : "false"}
              className={unchecked ? "border-b border-dotted border-warn" : undefined}
            >
              <Math latex={s.value} />
              {unchecked && (
                <sup className="ml-0.5 text-[10px] font-medium text-warn">unchecked</sup>
              )}
            </span>
          );
        }
        return (
          <div
            key={i}
            data-verified={s.verified ? "true" : "false"}
            className={`my-2 rounded-md border px-3 py-1 ${
              unchecked ? "border-warn/50 bg-warn-soft/40" : "border-rule bg-raised"
            }`}
          >
            <Math latex={s.value} display />
            {unchecked && (
              <div className="pb-1 text-xs text-warn" title={UNCHECKED_TITLE}>
                Not verified by Minerva (written by the model)
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
