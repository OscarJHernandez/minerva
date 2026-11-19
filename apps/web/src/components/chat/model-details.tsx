"use client";

import type { ModelInfo } from "@/lib/chat-types";
import { modelFacts, modelLabel, modelNotes } from "@/lib/model-info";

/** Which model, in one line; a tap shows the weights, size, quantisation, digest and thinking state. */
export function ModelDetails({
  info,
  alias = "",
  suffix = "",
  open = false,
}: {
  info: ModelInfo | null | undefined;
  alias?: string;
  suffix?: string;
  open?: boolean;
}) {
  const label = modelLabel(info, alias) + suffix;
  if (!info) return <span className="text-xs text-muted">{label}</span>;
  return (
    <details className="text-xs text-muted" open={open} data-testid="model-details">
      <summary className="cursor-pointer select-none hover:text-ink">{label}</summary>
      <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        {modelFacts(info).map(([k, v]) => (
          <div key={k} className="contents">
            <dt>{k}</dt>
            <dd className="break-words text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      {modelNotes(info).map((n) => (
        <p key={n} className="mt-1">
          {n}
        </p>
      ))}
    </details>
  );
}
