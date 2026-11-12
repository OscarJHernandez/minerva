"use client";

import type { IdentitiesResult, IdentityMatch } from "@/lib/types";
import { IdentitiesPanel } from "../identities-panel";

interface Props {
  /** The lookup for the expression and assumptions as they are now; null when none was made or it is out of date. */
  result: IdentitiesResult | null;
  stale: boolean;
  searching: boolean;
  /** Rule id being applied right now. */
  applying: string | null;
  disabled: boolean;
  onFind: () => void;
  onApply: (match: IdentityMatch) => void;
}

/**
 * Handbook identities for the current expression. Retrieval proposes, the engine decides: only a
 * match the engine reports as applicable (conditions proven from the assumptions, rewrite verified)
 * gets an Apply button, and applying it recomputes and re-verifies the rewrite on the server.
 */
export function IdentityPicker({
  result,
  stale,
  searching,
  applying,
  disabled,
  onFind,
  onApply,
}: Props) {
  return (
    <div>
      <button
        type="button"
        onClick={onFind}
        disabled={disabled || searching}
        className="rounded border border-accent px-3 py-1.5 text-sm font-medium text-accent disabled:opacity-50"
      >
        {searching ? "Searching…" : "Find identities for the expression"}
      </button>
      {stale && (
        <p className="mt-2 text-xs text-muted">
          The expression or assumptions changed since the last search. Search again.
        </p>
      )}
      {result && !stale && (
        <div className="mt-3">
          <IdentitiesPanel result={result} apply={{ busy: applying, onApply }} />
        </div>
      )}
      {!result && !stale && (
        <p className="mt-2 text-xs text-muted">
          Lists the handbook identities that match, which are applicable under your assumptions, and
          why the others are not.
        </p>
      )}
    </div>
  );
}
