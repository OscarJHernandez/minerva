"use client";

import { createContext, useContext } from "react";
import type { MotionRequest, PerturbationRequest } from "@/lib/chat-types";

/**
 * Lets a card start a numerical solution in the open conversation. Provided by the chat app; absent
 * where a card is shown outside a conversation, in which case the form is not offered.
 */
export interface MotionRunner {
  run: (request: MotionRequest) => Promise<void>;
  /** Solves a perturbation problem again with the user's comparison numbers. */
  perturb?: (request: PerturbationRequest) => Promise<void>;
}

const MotionRunnerContext = createContext<MotionRunner | null>(null);

export const MotionRunnerProvider = MotionRunnerContext.Provider;

export function useMotionRunner(): MotionRunner | null {
  return useContext(MotionRunnerContext);
}
