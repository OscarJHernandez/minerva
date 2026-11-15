"use client";

import { createContext, useContext, useState } from "react";

/** What a formula in a reply can do besides being read. Provided by the chat app. */
export interface FormulaActions {
  /** Put the formula in the message box, as `$...$`, ready to edit. Absent outside a conversation. */
  editInChat?: (latex: string) => void;
}

const FormulaActionsContext = createContext<FormulaActions>({});

export const FormulaActionsProvider = FormulaActionsContext.Provider;

export function useFormulaActions(): FormulaActions {
  return useContext(FormulaActionsContext);
}

/**
 * Copy the formula's LaTeX, or send it to the message box to modify. The text is exactly what
 * Minerva rendered; nothing is rewritten, so what you copy is what you see.
 */
export function FormulaButtons({ latex, subtle = false }: { latex: string; subtle?: boolean }) {
  const { editInChat } = useFormulaActions();
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);
  const visibility = subtle
    ? "opacity-0 group-hover/formula:opacity-100 group-focus-within/formula:opacity-100 focus-within:opacity-100"
    : "";
  return (
    <span className={`inline-flex items-center gap-2 text-xs ${visibility}`}>
      <button
        type="button"
        aria-label="Copy LaTeX"
        title="Copy the LaTeX of this formula"
        className="text-accent hover:underline"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(latex);
            setFailed(false);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch {
            setFailed(true);
          }
        }}
      >
        {failed ? "Copy failed" : copied ? "Copied" : "Copy LaTeX"}
      </button>
      {editInChat && (
        <button
          type="button"
          aria-label="Edit in chat"
          title="Put this formula in the message box so you can change it"
          className="text-accent hover:underline"
          onClick={() => editInChat(latex)}
        >
          Edit in chat
        </button>
      )}
    </span>
  );
}
