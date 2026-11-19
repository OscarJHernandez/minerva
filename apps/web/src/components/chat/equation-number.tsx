"use client";

import { createContext, useContext, type ReactNode } from "react";

/** Inserts a reference to an equation ("(3)") into the message being written. Absent outside the chat. */
const InsertReference = createContext<((n: number) => void) | null>(null);
export const InsertReferenceProvider = InsertReference.Provider;

/** The numbers of the equations a card lists, by row key. Absent outside a chat card (the workbench). */
const RowNumbers = createContext<Record<string, number> | null>(null);
export const RowNumbersProvider = RowNumbers.Provider;

/**
 * One listed equation of a card with its number at the right edge, when the server gave it one. Without a
 * number (the workbench, a formula too long to number) it renders its children unchanged.
 */
export function Numbered({ k, children }: { k: string; children: ReactNode }) {
  const n = useContext(RowNumbers)?.[k];
  if (n == null) return <>{children}</>;
  return (
    <div className="flex items-baseline justify-between gap-3">
      <div className="min-w-0 flex-1 overflow-x-auto">{children}</div>
      <EquationNumber n={n} />
    </div>
  );
}

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
