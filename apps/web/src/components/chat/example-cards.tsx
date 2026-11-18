"use client";

import { Math } from "@minerva/ui/math";
import { type HomepageExample, visibleExamples } from "@/lib/homepage-examples";

/**
 * The example cards of the empty chat: a two-column grid (one column on a phone). The whole card is one button, so
 * it is reachable with the keyboard and a click or Enter sends its exact prompt. The expected answer is never shown:
 * a reader finds it through Minerva.
 */
export function ExampleCards({
  onPick,
  disabled,
  examples,
}: {
  onPick: (prompt: string) => void;
  disabled: boolean;
  /** Defaults to the registry's approved examples. */
  examples?: readonly HomepageExample[];
}) {
  const shown = examples ?? visibleExamples();
  if (shown.length === 0) return null;
  return (
    <ul className="mt-8 grid gap-3 text-left sm:grid-cols-2" data-testid="example-cards">
      {shown.map((ex) => (
        <li key={ex.id} className="min-w-0">
          <button
            type="button"
            disabled={disabled}
            onClick={() => onPick(ex.prompt)}
            aria-label={ex.title}
            data-example-id={ex.id}
            className="flex h-full w-full flex-col rounded-xl border border-rule bg-raised p-4 text-left hover:border-accent focus-visible:border-accent disabled:opacity-50"
          >
            <span className="text-sm font-medium">{ex.title}</span>
            <span className="my-2 block min-w-0 overflow-x-auto text-ink" aria-hidden="true">
              <Math latex={ex.latex} display className="text-base" />
            </span>
            <span className="text-sm text-muted">{ex.description}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
