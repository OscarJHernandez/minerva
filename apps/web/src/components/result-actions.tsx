"use client";

import { useState } from "react";

export function ResultActions({
  text,
  latex,
  onOpen,
  onChoose,
  limited = false,
}: {
  text: string;
  latex?: string | null;
  onOpen?: () => void;
  onChoose?: (
    operation:
      "simplify" | "expand" | "factor" | "differentiate" | "integrate" | "series" | "identities",
  ) => void;
  limited?: boolean;
}) {
  const [message, setMessage] = useState("");
  async function copy(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setMessage("Copied");
    } catch {
      setMessage("Copy failed");
    }
  }
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
      {onOpen && (
        <button type="button" className="text-accent underline" onClick={onOpen}>
          Continue in workbench
        </button>
      )}
      {onChoose && (
        <label>
          <span className="sr-only">Operation on this result</span>
          <select
            className="rounded border border-rule bg-raised p-1"
            value=""
            onChange={(event) => {
              const operation = event.target.value as Parameters<typeof onChoose>[0];
              if (operation) onChoose(operation);
            }}
          >
            <option value="">Next operation…</option>
            <option value="simplify">Simplify</option>
            <option value="expand">Expand</option>
            <option value="factor">Factor</option>
            <option value="differentiate">Differentiate</option>
            {!limited && <option value="integrate">Integrate</option>}
            {!limited && <option value="series">Series</option>}
            <option value="identities">Find identities</option>
          </select>
        </label>
      )}
      <button type="button" className="text-accent underline" onClick={() => void copy(text)}>
        Copy text
      </button>
      {latex && (
        <button type="button" className="text-accent underline" onClick={() => void copy(latex)}>
          Copy LaTeX
        </button>
      )}
      {message && <span role="status">{message}</span>}
    </div>
  );
}
