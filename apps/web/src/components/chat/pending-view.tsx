"use client";

import { useEffect, useState } from "react";
import type { Card, ToolRun } from "@/lib/chat-types";
import { splitMath } from "@/lib/inline-math";
import { TOOL_LABEL } from "@/lib/tool-labels";
import { CardView } from "./card-view";
import { RichText } from "./rich-text";

export type PendingItem = { kind: "text"; text: string } | { kind: "card"; card: Card };

/** A reply that is still arriving. */
export interface Pending {
  items: PendingItem[];
  tools: ToolRun[];
  cards: number;
  /** When the turn began (ms since the epoch): for the seconds counter. */
  startedAt: number;
  /** The server says it is now checking the formulas the model wrote. */
  phase?: "checking";
}

/** After this long with nothing visible, say why: a model host may be loading its model. */
const SLOW_START_SECONDS = 20;

export function useElapsedSeconds(startedAt: number): number {
  const [now, setNow] = useState(startedAt);
  useEffect(() => {
    const tick = () => setNow(Date.now());
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [startedAt]);
  return Math.max(0, Math.floor((now - startedAt) / 1000));
}

/** What the turn is doing right now, in words. */
export function statusText(pending: Pending, seconds: number): string {
  const running = pending.tools.length > pending.cards ? pending.tools[pending.cards] : null;
  if (running) return `${TOOL_LABEL[running.name] ?? "Working"}…`;
  if (pending.phase === "checking") return "Checking the formulas in the reply…";
  const hasText = pending.items.some((i) => i.kind === "text");
  if (!hasText && seconds >= SLOW_START_SECONDS) {
    return "Still waiting for the model. If it was idle, the host may be loading it: this can take a minute.";
  }
  return pending.items.length === 0 ? "Thinking…" : "Working…";
}

export function PendingView({ pending }: { pending: Pending }) {
  const seconds = useElapsedSeconds(pending.startedAt);
  return (
    <div className="space-y-3" aria-busy="true">
      {pending.items.map((item, i) =>
        item.kind === "text" ? (
          // Closed formulas render as they arrive; an unfinished one stays text until it closes.
          <div key={i} className="leading-relaxed">
            <RichText segments={splitMath(item.text)} />
          </div>
        ) : (
          <CardView key={item.card.id} card={item.card} streaming />
        ),
      )}
      <p className="text-sm text-muted" role="status" data-testid="pending-status">
        {statusText(pending, seconds)}
        {seconds >= 3 && <span className="ml-2 tabular-nums">{seconds} s</span>}
      </p>
    </div>
  );
}
