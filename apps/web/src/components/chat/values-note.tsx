import type { ValuesChosenBy } from "@/lib/chat-types";

const TEXT: Record<Exclude<ValuesChosenBy, "user">, string> = {
  assistant_delegated: "Values chosen by the assistant at your request",
  assistant: "Some values were supplied by the assistant, not by you",
};

/**
 * Who chose the numbers of a numerical run. Nothing is shown for numbers the user wrote (or typed on
 * the card's form); a run with values the assistant picked says so, and lists them.
 */
export function ValuesNote({ chosenBy, values }: { chosenBy?: ValuesChosenBy; values?: string[] }) {
  if (!chosenBy || chosenBy === "user") return null;
  return (
    <div
      role="note"
      data-testid="values-note"
      className="mb-2 rounded-md border border-warn/50 bg-warn-soft/40 px-3 py-2 text-xs"
    >
      <p className="font-medium text-warn">{TEXT[chosenBy]}</p>
      {values && values.length > 0 && <p className="mt-1 text-muted">{values.join(", ")}</p>}
    </div>
  );
}
