"use client";

import { createContext, useContext } from "react";
import type { MotionRequest, PerturbationRequest } from "@/lib/chat-types";
import type { OdeAnalyzeInput, OdeInput, OdeNumericInput, OdeSolveInput } from "@/lib/ode-types";
import type { StatAnswers, StatToolArgs } from "@/lib/statmech-types";
import type { DistributionAnswer, GeometryRequest } from "@/lib/types";
import type { VariationRequest } from "@/lib/variation-types";

/**
 * Lets a card start a numerical solution in the open conversation. Provided by the chat app; absent
 * where a card is shown outside a conversation, in which case the form is not offered.
 */
export interface MotionRunner {
  run: (request: MotionRequest) => Promise<void>;
  /** Solves a perturbation problem again with the user's comparison numbers. */
  perturb?: (request: PerturbationRequest) => Promise<void>;
  /** Asks for more quantities of the geometry on a card. */
  geometry?: (request: GeometryRequest) => Promise<void>;
  /** Runs a variation card with the numbers the user typed in its form. */
  variation?: (request: VariationRequest) => Promise<void>;
  /** Runs a delta card again with the user's answer to "is that a Dirac delta?". */
  distribution?: (request: DistributionAnswer) => Promise<void>;
  /** Starts a closed-form or numerical solution from an analysis card, with the user's own numbers. */
  ode?: {
    symbolic: (request: OdeInput | OdeSolveInput) => Promise<void>;
    numeric: (request: OdeNumericInput) => Promise<void>;
    qualitative: (request: OdeAnalyzeInput) => Promise<void>;
  };
  /** Runs a statistical-mechanics card again with the user's choices (a cell, numbers, ...). */
  statmech?: (request: StatToolArgs & StatAnswers) => Promise<void>;
}

const MotionRunnerContext = createContext<MotionRunner | null>(null);

export const MotionRunnerProvider = MotionRunnerContext.Provider;

export function useMotionRunner(): MotionRunner | null {
  return useContext(MotionRunnerContext);
}
