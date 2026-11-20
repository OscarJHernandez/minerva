# Explore in Minerva

A verified example on the landing page can be continued as a real exploration: a button opens the chat with the
example's mathematical context in the message box, and the user decides to run it.

## What happens

1. The site button links to `<app>/explore/<id>`. The id comes from `packages/ui/fixtures/explorations.json`; no
   mathematics is put in the URL.
2. A logged-out visitor is sent to log in or register and comes back to the same address (`next` is kept by the proxy,
   by the login and register links and by a 401 inside the chat).
3. `/explore/<id>` opens a new conversation with the entry's prompt in the message box. **Nothing is sent.**
4. The user presses Send. The model chooses tools; the engine computes, verifies and reports. The card shows
   *Proved by a symbolic zero test* only where a symbolic zero was found and *Numerically checked; no symbolic proof*
   otherwise. For the Schwarzschild example the Ricci tensor is symbolically proved to vanish, and the Kretschmann scalar
   `48 M^2 / r^6` is proved by a symbolic zero test against the full contraction of the second-derivative formula for
   `R_{abcd}`, besides its independent numerical checks (a metric too large for the symbolic leg reports it as "not run").
5. The card's buttons ask for further quantities of the same metric (the card's own request is posted back, no model
   involved), and the conversation continues like any other: it keeps its address (`/c/<id>`), its history and the
   restrictions of earlier results.

## What the registry holds, and what it does not

The registry holds the id, titles, the prompt and the names of an evaluation case and a landing example. It holds **no
result**: every number on the page comes from the engine when the user runs the prompt.

`services/api/tests/test_exploration_registry.py` checks that:

- each prompt is exactly the prompt of an evaluation case of the `explore` suite, whose reference call (the metric of
  the landing example) meets its expectation against the engine;
- the prompt states every coordinate, every nonzero metric entry and the assumptions of the landing example (the model
  may not invent a metric or an assumption);
- a button is shown (`approved`) only if `docs/benchmarks/exploration-live.json` records five correct, engine-backed
  live runs of that exact prompt at the deployed 10 s compute limit.

## Tests

- `services/api/tests/test_exploration_chat.py`: the prompt through the real chat endpoint (scripted model, real engine):
  Ricci and Kretschmann proved, restrictions reported, a follow-up quantity from the card and a second
  turn in the same conversation.
- `apps/web` (vitest): the registry, the proxy redirect, the prefilled message box (not sent, not put back after sending).
- `e2e/` (Playwright, CI job `e2e`): landing page button visibility follows the registry; login keeps the exploration;
  the prompt is in the box and unsent; Send runs the engine; evidence kinds are shown apart; another quantity from the
  card; a follow-up message; a reload keeps the conversation.

## Adding an example

Add an entry to `docs/capabilities.yaml` (the landing example), a case to `cases_exploration.yaml`, an entry to
`explorations.json` with `approved: false`, run the case five times live at the deployed limit
(`--suite explore --cases <id> --repeat 5 --compute-timeout 10`), record the runs, then set `approved`. Kepler's
button still points at the app home; it needs its own entry the same way.

## Recorded research sessions on the landing page

The landing page's examples are two real conversations, not hand-written results: "What is the Lagrangian of the Kepler
problem in polar coordinates?" → "What are its symmetries?" → "Can you search for hidden symmetries?", and "What is the
Schwarzschild metric?" → "Compute the Christoffel symbols" → "Now compute the Kretschmann scalar". They show how Minerva helps
someone work through a problem: the model answers from memory where no tool applies (and that formula carries the chat's
**unchecked** mark), then the engine computes and checks each result.

- The prompts are the `landing-kepler` and `landing-schwarzschild` cases of `cases_exploration.yaml` (suite `explore`).
- `scripts/record_landing_sessions.py --base-url <host>` runs them five times each through the real loop, with the deployed
  model, thinking and reply budget, at the 10 s tool limit. It writes `docs/benchmarks/landing-live.json` (every run, the
  model as the host reports it, the settings every request carried) and `apps/site/src/data/landing-sessions.json` with,
  for each conversation, **the first run in which every turn passed** and **how many of the runs passed**, numbered by the
  server's own rule. The page states that count, the model (name, family, size, quantisation, digest) and the settings
  (thinking, reply budget, temperature, context window, tool time limit, seed), and links the record of every run. A
  conversation with no passing run is not published. Nothing is chosen or edited by hand. (Rule fixed on 2026-11-20, before
  the recording it applies to; the first recording on Qwen 3.5 passed 8 of 10 under the older all-must-pass rule.)
- `services/api/tests/test_landing_sessions.py` re-runs every recorded tool call against the engine, checks the recording
  belongs to the current prompts, that the shown run is the first that passed and the stated count matches the record,
  that the model and settings are the record's, and that no host address or trace is published.
  Changing a prompt or the deployed model means recording again.
