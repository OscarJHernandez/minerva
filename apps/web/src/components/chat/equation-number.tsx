"use client";

import { createContext, useContext } from "react";

/** Inserts a reference to an equation ("(3)") into the message being written. Absent outside the chat. */
const InsertReference = createContext<((n: number) => void) | null>(null);
export const InsertReferenceProvider = InsertReference.Provider;

/**
 * The number of an equation, like (3). In the chat, clicking it adds the reference to the message
 * being written. A number is a name for finding the equation again; it says nothing about whether
 * Minerva verified it (the mark does).
 */
export function EquationNumber({ n }: { n: number }) {
  const insert = useContext(InsertReference);
  const label = `(${n})`;
  if (!insert) {
    return (
      <span className="shrink-0 text-sm tabular-nums text-muted" data-testid="equation-number">
        {label}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={() => insert(n)}
      title={`Refer to equation ${n} in your message`}
      aria-label={`Equation ${n}: refer to it in your message`}
      data-testid="equation-number"
      className="shrink-0 rounded px-1 text-sm tabular-nums text-muted hover:bg-sunk hover:text-accent"
    >
      {label}
    </button>
  );
}
