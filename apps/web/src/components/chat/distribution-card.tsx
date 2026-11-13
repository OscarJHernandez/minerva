"use client";

import { useState } from "react";
import type { DistributionAnswer, DistributionResult, EntrySummary } from "@/lib/types";
import { DistributionPanel } from "../distribution-panel";
import { useMotionRunner } from "./motion-runner";

/**
 * A delta card in the conversation. When the engine asks "is that a Dirac delta?", the answer is
 * posted straight to the API with the call that was made: the model never answers for the user.
 */
export function DistributionCard({
  result,
  methods,
  request,
}: {
  result: DistributionResult;
  methods: EntrySummary[];
  request: DistributionAnswer;
}) {
  const runner = useMotionRunner();
  const [error, setError] = useState<string | null>(null);
  const answer = runner?.distribution;
  return (
    <>
      <DistributionPanel
        result={result}
        methods={methods}
        onAnswer={
          answer
            ? async (notation) => {
                setError(null);
                try {
                  await answer({ ...request, notation });
                } catch (e) {
                  setError(e instanceof Error ? e.message : "Could not run that.");
                }
              }
            : undefined
        }
      />
      {error && (
        <p role="alert" className="mt-2 text-sm text-bad">
          {error}
        </p>
      )}
    </>
  );
}
