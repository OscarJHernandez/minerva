"use client";

import { useEffect, useRef } from "react";
import { continuation } from "@/lib/derivation";
import type { DistributionNotation, HistoryItem } from "@/lib/types";
import { StepCard } from "./step-card";

interface Props {
  steps: HistoryItem[];
  /** The step the editor is continuing, if any. */
  currentId: string | null;
  onContinue: (item: HistoryItem, edit: boolean) => void;
  onDelete: (item: HistoryItem) => void;
  onAnswerNotation?: (item: HistoryItem, notation: DistributionNotation) => Promise<void>;
}

/** The derivation: every step with what produced it, in the order it was made. */
export function DerivationView({
  steps,
  currentId,
  onContinue,
  onDelete,
  onAnswerNotation,
}: Props) {
  const last = useRef<number>(0);

  useEffect(() => {
    // Bring a newly added step into view; leave the page alone when steps were removed or reloaded.
    if (steps.length > last.current && last.current > 0) {
      const id = steps[steps.length - 1]?.id;
      document.getElementById(`step-${id}`)?.scrollIntoView?.({ block: "nearest" });
    }
    last.current = steps.length;
  }, [steps]);

  if (steps.length === 0) {
    return (
      <section
        aria-labelledby="derivation-heading"
        className="rounded-lg border border-dashed border-rule p-6"
      >
        <h2 id="derivation-heading" className="font-serif text-xl">
          Derivation
        </h2>
        <p className="mt-2 max-w-prose text-sm text-muted">
          Enter an expression and apply an operation or a handbook identity. Each result becomes a
          step you can continue from, edit or copy. Results are verified before they are shown; if
          Minerva cannot verify or simplify something, it says so. No language model is involved.
        </p>
      </section>
    );
  }

  const numbers = new Map(steps.map((s, i) => [s.id, i + 1]));
  return (
    <section aria-labelledby="derivation-heading">
      <h2 id="derivation-heading" className="mb-3 font-serif text-xl">
        Derivation <span className="text-sm font-normal text-muted">({steps.length})</span>
      </h2>
      <ol className="space-y-3">
        {steps.map((item) => (
          <StepCard
            key={item.id}
            item={item}
            number={numbers.get(item.id) ?? 0}
            parentNumber={item.parent_id ? (numbers.get(item.parent_id) ?? null) : null}
            current={item.id === currentId && continuation(item) !== null}
            onContinue={onContinue}
            onDelete={onDelete}
            onAnswerNotation={onAnswerNotation}
          />
        ))}
      </ol>
    </section>
  );
}
