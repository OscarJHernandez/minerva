import registry from "../../../../packages/ui/fixtures/homepage-examples.json";

/**
 * One example card of the chat home page. The registry is a shared fixture
 * (`packages/ui/fixtures/homepage-examples.json`) so that the Python tests can check every card against the
 * agent evaluation suite: the exact prompt, an engine-checked reference call (`eval_case`) and, for an approved
 * card, the record of five live runs. It holds no expected answer: the oracle is the evaluation case.
 */
export interface HomepageExample {
  /** Stable id, never reused. */
  id: string;
  title: string;
  /** What the example explores, one sentence, with no result in it. */
  description: string;
  /** The displayed expression (LaTeX); it is a picture of the problem, never submitted. */
  latex: string;
  /** Exactly what a click sends to the chat. */
  prompt: string;
  /** The capability id in docs/capabilities.yaml. */
  capability: string;
  /** The agent evaluation case that runs this exact prompt. */
  eval_case: string;
  /** True only after the case passes and five live runs were all correct and engine-backed. */
  approved: boolean;
}

export const ALL_EXAMPLES: readonly HomepageExample[] = registry.examples;

/** The cards the home page shows: only the approved ones, in registry order. */
export function visibleExamples(
  examples: readonly HomepageExample[] = ALL_EXAMPLES,
): HomepageExample[] {
  return examples.filter((e) => e.approved);
}
