"use client";

import type { FormulaStatus } from "@/lib/chat-types";
import { MARKS } from "./rich-text";

/** Shown in this order: what is established first, what is not last. */
const ORDER: { status: FormulaStatus; name: string }[] = [
  { status: "from_tool", name: "No mark" },
  { status: "quoted", name: "quoted" },
  { status: "checked", name: "checked" },
  { status: "conditional", name: "conditional" },
  { status: "contradicted", name: "contradicted" },
  { status: "unchecked", name: "? / unchecked" },
];

/**
 * What the marks next to formulas mean. The same sentences as the tooltips and the marks' own
 * explanations (they are read from the same table), so the legend cannot drift from them.
 */
export function MarksLegend({ onClose }: { onClose: () => void }) {
  return (
    <section
      aria-label="What the marks mean"
      className="mb-2 rounded-lg border border-rule bg-sunk p-3 text-sm"
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-muted">
          Minerva computes results with tools and checks them. A formula the model wrote on its own
          is marked, so you can tell it from a result.
        </p>
        <button type="button" onClick={onClose} className="shrink-0 text-xs text-accent">
          Close
        </button>
      </div>
      <dl className="mt-2 space-y-1.5">
        {ORDER.map(({ status, name }) => (
          <div key={status} className="grid gap-x-3 sm:grid-cols-[8rem_1fr]">
            <dt className={`font-medium ${MARKS[status].tone}`}>{name}</dt>
            <dd className="text-muted">{MARKS[status].title}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-muted">
        Tap or click a mark to read what it means for that formula. A number such as (3) is only a
        name for finding an equation again; it says nothing about whether it was verified.
      </p>
    </section>
  );
}
