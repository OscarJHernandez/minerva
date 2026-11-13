"use client";

import { useState } from "react";
import type { GeometryQuantity, GeometryRequest, GeometryResult } from "@/lib/types";
import { GeometryPanel } from "../geometry-panel";
import { useMotionRunner } from "./motion-runner";

/**
 * The geometry result in a conversation. Its buttons post the card's own request back with another
 * quantity, so the same coordinates, metric and assumptions are reused without the model retyping them.
 */
export function GeometryCard({
  result,
  request,
}: {
  result: GeometryResult;
  request: GeometryRequest;
}) {
  const runner = useMotionRunner();
  const [asked, setAsked] = useState<GeometryQuantity[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = runner?.geometry;

  async function more(quantity: GeometryQuantity) {
    if (!run) return;
    setBusy(true);
    setError(null);
    try {
      await run({ ...request, quantities: [quantity], tensor: null });
      setAsked((a) => [...a, quantity]);
    } catch {
      setError("That could not be computed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <GeometryPanel
        result={result}
        asked={asked}
        onMore={run ? (q) => void more(q) : undefined}
        busy={busy}
      />
      {error && (
        <p role="alert" className="mt-2 rounded bg-bad-soft px-3 py-2 text-sm text-bad">
          {error}
        </p>
      )}
    </div>
  );
}
