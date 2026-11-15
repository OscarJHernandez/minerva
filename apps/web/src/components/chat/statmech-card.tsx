"use client";

import { useState } from "react";
import type { StatAnswers, StatMechResult, StatToolArgs } from "@/lib/statmech-types";
import { StatMechPanel } from "../statmech-panel";
import { useMotionRunner } from "./motion-runner";

/**
 * A statistical-mechanics card in the conversation. The conventions only the user can choose (the
 * phase-space cell, whether copies are indistinguishable) are asked here and posted straight to the
 * API with the call that was made: the model never answers for the user.
 */
export function StatMechCard({
  result,
  args,
  answers,
}: {
  result: StatMechResult;
  args: StatToolArgs;
  answers: StatAnswers;
}) {
  const runner = useMotionRunner();
  const [error, setError] = useState<string | null>(null);
  const post = runner?.statmech;
  const needs = {
    cell: args.system.kind === "classical" && answers.phase_space_cell === undefined,
    identical: Boolean(args.copies_count) && answers.identical === undefined,
  };
  return (
    <>
      <StatMechPanel
        result={result}
        needs={needs}
        onAnswer={
          post
            ? async (next) => {
                setError(null);
                try {
                  await post({ ...args, ...answers, ...next });
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
