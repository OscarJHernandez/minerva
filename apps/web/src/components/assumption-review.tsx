"use client";

import { useState } from "react";
import { Math } from "@minerva/ui/math";
import type { Assumptions, RuleCondition } from "@/lib/types";

export function AssumptionReview({
  conditions,
  assumptions,
  onApprove,
  disabled,
  action,
}: {
  conditions: RuleCondition[];
  assumptions: Assumptions;
  onApprove: (next: Assumptions) => Promise<string | null>;
  disabled?: boolean;
  action: string;
}) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (!conditions.length || conditions.some((condition) => !condition.symbol)) return null;
  async function approve() {
    const proposed = new Map(Object.entries(assumptions));
    for (const { symbol, assumption } of conditions) {
      if (!symbol) return;
      proposed.set(symbol, [...new Set([...(proposed.get(symbol) ?? []), assumption])]);
    }
    setBusy(true);
    try {
      setError(await onApprove(Object.fromEntries(proposed)));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="mt-2 space-y-2 text-sm">
      <button
        type="button"
        aria-expanded={open}
        className="rounded border border-rule px-2 py-1"
        onClick={() => setOpen(!open)}
      >
        Review assumptions
      </button>
      {open && (
        <div className="space-y-1">
          <p>
            Current assumptions:{" "}
            {Object.entries(assumptions)
              .map(([s, values]) => s + " " + values.join(", "))
              .join("; ") || "none"}
            .
          </p>
          <ul className="list-disc pl-5">
            {conditions.map(({ variable, symbol, assumption, bound_latex }) => (
              <li key={variable + assumption}>
                {bound_latex ? <Math latex={bound_latex} /> : symbol} must be {assumption}
              </li>
            ))}
          </ul>
          {error && <p role="alert">{error}</p>}
          <button
            type="button"
            disabled={busy || disabled}
            className="rounded border border-rule px-2 py-1"
            onClick={() => void approve()}
          >
            {action}
          </button>
        </div>
      )}
    </div>
  );
}
