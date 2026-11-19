import registry from "../fixtures/explorations.json";

/**
 * One "Explore in Minerva" entry point: a verified landing-page example that can be continued in the chat. The
 * registry is a shared fixture (`packages/ui/fixtures/explorations.json`) read by the site (the button) and the
 * web app (the route that puts `prompt` in the message box). It holds no result: the user runs the prompt and every
 * number comes from the engine. `services/api/tests/test_exploration_registry.py` ties each entry to an evaluation
 * case that runs this exact prompt, to the metric of the landing example, and (for `approved`) to a record of five
 * correct live runs.
 */
export interface Exploration {
  /** Stable id, never reused: it is the address `/explore/<id>`. */
  id: string;
  title: string;
  /** The landing-page example (docs/capabilities.yaml) this continues. */
  example: string;
  /** The capability id in docs/capabilities.yaml. */
  capability: string;
  /** The button label. */
  button: string;
  /** Exactly what is put in the message box. It is never sent without the user's click. */
  prompt: string;
  /** The agent evaluation case (suite `explore`) that runs this exact prompt. */
  eval_case: string;
  /** True only after five correct, engine-backed live runs of this exact prompt at the deployed time limit. */
  approved: boolean;
}

export const EXPLORATIONS: readonly Exploration[] = registry.explorations;

/** The address of an exploration in the web app (the app's origin is added by the caller). */
export const explorePath = (id: string) => `/explore/${id}`;

export function findExploration(
  id: string,
  list: readonly Exploration[] = EXPLORATIONS,
): Exploration | undefined {
  return list.find((e) => e.id === id);
}

/** The buttons a page may show: only approved explorations, optionally of one landing example. */
export function approvedExplorations(
  example?: string,
  list: readonly Exploration[] = EXPLORATIONS,
): Exploration[] {
  return list.filter((e) => e.approved && (example === undefined || e.example === example));
}
