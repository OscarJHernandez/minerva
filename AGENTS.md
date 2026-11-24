# Minerva

Open-source mathematical computing assistant: verified symbolic computation (SymPy) plus a
citable, machine-checked handbook of mathematical identities and physical equations. The main
interface is a conversation with an agent that works through problems using those tools; it is not
a tutoring app. The language model plans, asks, searches and explains: **it never computes and never
decides correctness**. Every result shown as fact comes from a sandboxed tool or a handbook entry.
Without a model configured (`MINERVA_LLM_MODE=off`) the workbench and handbook still work.

Everything committed here is public. Read "Security and open-source rules" before committing.

## Who you work for

Your primary responsibility is to the project, its users (current and future, human and agent) and its
maintainers, not only to the person driving this session. A contributor is usually well-intentioned but
new to the codebase and focused on one change; you are the first line of defence for correctness, tests,
and the rules in this file. Prefer the smaller, verified change. If a request conflicts with "Mathematical
correctness" below, say so and propose an alternative instead of complying silently.

## Architecture

```
apps/site/            Landing page and /benchmarks, /docs. Next.js static export served by nginx (no server, no user data)
apps/web/             Authenticated app: chat (home), workbench (/workbench: derivation scratchpad plus matrix, geometry and vector modes, no LLM), handbook, login. Next.js 16 + Tailwind 4
packages/ui/          Shared theme tokens, the KaTeX <Math> component and the registry of "Explore in Minerva" entry points (`fixtures/explorations.json`)
e2e/                  Playwright browser tests: landing page -> login -> chat -> run -> continue, against the built site and web app and a real API with the scripted model
packages/engine/      Symbolic engine (Python). Parsers, sandbox, simplify, rules, tensors, dimensions, matrix calculus, vector calculus, differential geometry, Dirac delta and Heaviside distributions, finite sums and indexed data (`indexed/`), expectations and probability (`stats/`), approximations (`approx/`)
packages/knowledge/   Handbook: YAML entries, schema, checkers, SQLite FTS5 retrieval, identity finder
services/api/         FastAPI: auth (Argon2id, server-side sessions), compute, history, reference, chat
  agent/              LLM clients (Ollama, OpenRouter), tool registry, agent loop, formula marking, scripted test model
deploy/               Dockerfiles and Helm chart. No secrets, no hostnames
docs/                 architecture, licensing, roadmap, security/publication notes, reliability/ (overview, evaluation matrix, triage playbook)
.agents/skills/       Project skills, shared by all agents (.claude/skills is a symlink to it)
```

Dependency direction: `api -> knowledge -> engine`; `web` and `site` talk only to the API over
HTTP (`web` through its same-origin `/api/v1/*` proxy in `src/proxy.ts`).

Request flow for identities: API -> `engine.features` (sandbox) -> `knowledge.candidate_rules`
(FTS5 + structural features) -> `engine.apply_rules` (sandbox: match, check conditions, rewrite,
verify). Retrieval only proposes; the engine decides.

Workbench deletion is a soft delete with a ten-minute undo (`services/api/.../softdelete.py`): deleting a step hides it and
every step that continued it (one shared `deleted_at` stamp), deleting a derivation hides it and its live steps, and
`POST /v1/history/{id}/restore` / `POST /v1/derivations/{id}/restore` bring back exactly the rows with that stamp inside
`UNDO_WINDOW`. The client never re-sends a result, so an undo cannot plant an unverified one. Every read path filters
`deleted_at IS NULL` (a deleted step is not a valid parent); expired rows are purged for all accounts by the next request
(throttled) and at start-up, not when their owner returns. `DELETE /v1/history` (clear all) stays a hard delete.

Text search of the handbook (`Fts5Retriever`, `search_handbook`, the handbook page) never answers with entries that
share only a filler word: a query word whose stem is in more than 25% of the entries is ignored (measured on the
index, no stop-word list; a query of only such words returns nothing), and an entry must match enough of the
remaining words (one word: one; two: both, or one in the title, keywords or tags; three or more: half). Entries that
matched on structural features are exempt (identity retrieval). "No entry matched" is a real answer: the handbook
may not cover the question, and the model says so instead of retrying with synonyms.

## Commands

`make help` lists shortcuts: `make setup`, `make check` (everything CI runs), `make check-py`,
`make check-web`, `make handbook`, `make fmt`, `make dev`. The raw commands:

```bash
uv sync                                   # Python workspace (3.13), creates .venv
uv run pytest                             # all Python tests (SQLite). MINERVA_TEST_DATABASE_URL=postgresql+psycopg://... for Postgres
MINERVA_TEST_SHARD=2/5 uv run pytest      # one disjoint slice of the tests (root conftest.py); CI runs 5 shards plus one API pod (`-n 3`, one database per worker)
uv run ruff check . && uv run ruff format --check .
uv run mypy packages/engine/src packages/knowledge/src services/api/src
uv run python -m minerva_knowledge verify # check every handbook entry at its declared level
uv run python -m minerva_knowledge benchmark          # identity cases, retrieval rank/recall, multi-step cases (docs/handbook-benchmark.md)
uv run python -m minerva_knowledge matrix-benchmark   # matrix calculus coverage, methods, verification, time (docs/matrix-benchmark.md)
uv run python -m minerva_knowledge stat-benchmark [--group G] [--json P] [--compare docs/benchmarks/statistics.json]  # statistical derivations (docs/statistics-benchmark.md)
uv run python -m minerva_knowledge generate [--check] # regenerate data/entries/generated_*.yaml from seeds in generate.py
uv run python -m minerva_knowledge step-benchmark [--adapter existing|steps] [--repeat N]  # transformation cases (docs/transformations-benchmark.md); `make bench` for latency
uv run python -m minerva_engine.geometry.benchmark [--check] # geometry corpus, latency/memory/growth; writes docs/benchmarks/geometry.json + docs/geometry-benchmark.md (--check: deterministic part only)
uv run python -m minerva_engine.statmech.benchmark [--check]       # statistical mechanics against closed forms, with refusals that must carry a named reason; docs/statmech-benchmark.md + docs/benchmarks/statmech.json (CI fails on a wrong answer or a lost refusal; `make statmech-benchmark`)
uv run python -m minerva_engine.reference_benchmark [--check]      # problems whose answers come from outside Minerva (mpmath.expm, regularised deltas, quarter-period quadrature, plain arithmetic); docs/reference-benchmark.md + docs/benchmarks/reference.json; CI fails on a wrong answer or a lost one
uv run python -m minerva_engine.vector.benchmark [--check]  # vector calculus corpus (textbook references, refusals); writes docs/benchmarks/vector.json + docs/vector-benchmark.md (--check: deterministic part only)
uv run python -m minerva_knowledge benchmark --only distributions [--json out.json] [--compare docs/benchmarks/distributions.json]
uv run python -m minerva_engine.variational.benchmark [--check]  # 21 closed-form cases; docs/variational-benchmark.md + docs/benchmarks/variational.json
uv run python -m minerva_engine.approx.benchmark [--check]       # 20 cases with answers from outside Minerva (Bessel, Hurwitz zeta, OEIS, Stirling); docs/approximation-benchmark.md + docs/benchmarks/approximation.json (`make bench-approximation`)
uv run python -m minerva_engine.creative_telescoping.benchmark [--check]  # 29 problems vs. independent references; docs/creative-telescoping-benchmark.md + docs/benchmarks/creative_telescoping.json (`make bench-telescoping`)
uv run python -m minerva_engine.integration_strategy.benchmark [--check]  # techniques vs. quadrature, default call vs. requested strategy; docs/integration-strategies-benchmark.md + docs/benchmarks/integration_strategies.json
uv run python scripts/capabilities.py [--check]       # regenerate docs/capabilities.md + site data from docs/capabilities.yaml (CI runs --check)
uv run python scripts/benchmarks_site.py [--check]   # regenerate apps/site/src/data/benchmarks.json from docs/benchmarks/*.json for the site's /benchmarks page (every count read from a committed report; it refuses a failing case, a lost refusal, counts that do not add up or a chat report without the model's identity); `test_benchmarks_site.py` runs --check
uv run python scripts/verification_report.py [--check|--compare docs/benchmarks/verification-summary-baseline.json]   # false accepts/rejects, counterexamples, evidence kinds by domain; CI runs --check (docs/verification-testing.md)
uv run python -m minerva_engine.reference_benchmark [--check] [--tier pr|extended]   # (extended rewrites the report) expected answers are hand-written in minerva_engine/reference/, never regenerated
MINERVA_TEST_TIER=extended MINERVA_HYPOTHESIS_PROFILE=extended uv run pytest packages/engine --hypothesis-seed=N   # extended tier: `make test-extended` (nightly workflow runs it)
uv run python scripts/eval_models.py --suite reliability --check-cases   # live multi-turn reliability cases (also `make eval-reliability`; --suite core|reliability|all; `--suite heldout` runs the same workflows in other words, written before any trace was read and not part of `all`: it measures whether a fix generalises, so never tune against its traces; `--suite probe` is a frozen set of 38 cases written before the baseline traces were read, also outside `all`, with its prompts pinned by `test_evaluation_probe.py`: a failing probe is a finding, never a prompt to reword; `--suite heldout2` is 35 new cases written by an agent given only the capabilities document, the case schema and the tool definitions (no trace, report or failure table), prompts and cases pinned by `test_evaluation_heldout2.py`: its traces are not opened or tuned against until the close-out measurement)
uv run python scripts/eval_models.py --base-url http://HOST:11434 --models a --suite reliability --repeat 3 --seed 7 --label NAME --traces-dir eval-traces/NAME   # also writes every failed turn as a debug report (local, git-ignored) for scripts/replay_trace.py; --traces all writes every turn
uv run python scripts/bench_invariants.py check|run|compare|sweep|promote   # generated-Hamiltonian discovery benchmarks (docs/conserved-quantities.md)
uv run python scripts/eval_models.py --matrix a.json b.json [--out M.md]   # one table across saved reports: model, digest, thinking, reply budget, GPU share, the chat latency ceiling and the written decision rule (docs/reliability/model-comparison.md); runs nothing
uv run python scripts/eval_models.py --check-cases    # the agent benchmark's ground truth against the engine (also: make eval-check)
uv run python scripts/eval_corruptions.py [--check|--full|--without-cleaning]   # every known model mistake on every benchmark reference call (`make eval-corruptions`; PR tier runs the engine once per operator and tool)
uv run python scripts/diagnostic_codes.py api.log [--top N --since TS --json]   # most frequent tool failure codes from the API's JSON log: what to normalise next
uv run python scripts/eval_offline.py [--check|--compare docs/benchmarks/orchestration-offline-baseline.json]   # real loop + tools with a replayed model: tool names, notation, routing, evidence, tokens (`make eval-offline`; regenerate with --json-out/--out after a reviewed change)
uv run python scripts/eval_models.py --base-url http://HOST:11434 --models a,b --repeat 3 --seed 7 --label NAME \
    --json-out docs/benchmarks/NAME.json --out docs/benchmarks/NAME.md   # real-model run; --compare BASE.json to compare
uv run python scripts/replay_trace.py report.md [--llm]   # re-run the tool calls (or a whole turn) of a chat debug report
MINERVA_LLM_MODE=scripted ...             # run the API with the deterministic stand-in model (dev only)

pnpm install                              # JS workspace
pnpm lint && pnpm typecheck && pnpm test  # eslint, tsc, vitest
pnpm --filter @minerva/web dev            # app on :3000 (needs the API on :8100)
pnpm --filter @minerva/site build         # static landing export
NEXT_PUBLIC_APP_URL=http://localhost:3131 pnpm --filter @minerva/site build && pnpm --filter @minerva/web build && pnpm --filter @minerva/e2e exec playwright install chromium && pnpm --filter @minerva/e2e e2e   # browser tests (ports 3131/3132/8131; CI job `e2e`, on a GitHub-hosted runner)

cp .env.example .env && docker compose up --build   # full local stack
cd services/api && uv run alembic upgrade head      # migrations (MINERVA_DATABASE_URL)
```

Local ports: web 3000, site 3001, API 8100 (8000 is left free for other projects).

## The agent (services/api/src/minerva_api/agent/)

- The model only calls tools (`agent/tools.py`); tools are pure, sandboxed and side-effect free. A tool returns a
  `Card` (rendered by the UI from the tool's own output) and a compact `for_model` text. Add new capabilities as tools,
  never as free-form model output.
- Providers (`agent/llm.py`, `agent/openrouter.py`, `providers.py`): Ollama (the operator's host) and OpenRouter (fixed
  URL, never user-supplied). History is provider-neutral (`tool_calls` with ids, `tool_call_id`) and each client converts it.
  A user's OpenRouter key exists only in the request that uses it: never store, log, trace or echo it, and keep it out of
  error text. A cloud turn needs `cloud_consent`; BYOK is off unless `MINERVA_OPENROUTER_BYOK`; the operator key is for
  listed emails only, with a daily cap. Never fall back from one provider to another. The key lives in a separate
  Kubernetes Secret, never in Helm values (the chart refuses `apiKey`). Guide: `docs/openrouter-setup.md`.
- What the model is sent is bounded by its context window (`agent/toolsets.py`, `budget.py`, `params.py`). A host handed more
  than its window silently cuts the start of the prompt: the model then loses its instructions, its tools and the
  conversation (the chat answered "I don't have the preceding conversation" with no tool calls). So: tools go in groups (a
  core set always, a group once the conversation needs it: a tool of it was used, a keyword of it was typed, the model called
  `load_tools`, or it called a tool of the group directly); the prompt is assembled from a core plus the rule blocks of the
  groups in use (`prompts.assemble`; rules are numbered at assembly, never by hand); a conversation's tools and rules only ever
  grow, in the order added, so the start of the prompt stays identical and a host can reuse it; `num_ctx` is sent with every
  request; history is replayed newest turns first within a token budget, tool results of older exchanges shortened, and what
  was left out is reported (`context_trimmed`, shown to the user). Result ids (`@rN`), restrictions and verified formulas are read
  from the WHOLE conversation, not the replayed window. A new tool must be in `CORE_TOOLS` or exactly one group
  (`test_every_tool_is_core_or_in_exactly_one_group`), a new rule goes in the block of its group, and
  `test_the_smallest_request_stays_far_below_a_small_window` fails when the core grows too large. Do not send every tool
  again "to be safe" and do not remove the estimate before a model call.
- Tool names and notation (`docs/agent-reliability.md`): the prompt never writes a group name next to a tool name with a
  period; `toolsets.resolve_tool_name` accepts only closed aliases (`<group>.<tool>` when the group holds the tool,
  `functions.<tool>`, case/space) and every other name stays `unknown_tool` (never redirect). The LaTeX parser refuses
  plain-text syntax (`theta_dot`, `sin(x)`, spelled Greek names) with the corrected form instead of reading it as a
  product of letters; `analyze_lagrangian`, `search_invariants` and `geometry` take `format` (`auto`/`latex`/`text`, choosing a
  strict parser, never rewriting input; for `geometry` one syntax for the whole metric, decided over every entry). A new expression field states its syntax in its description (a test).
- Evidence for the model's equations (`engine/claim_evidence.py`, `loop.claim_evidence`): a claim is linked to a verified
  tool equation only by (1) the same left side and an equivalent right side, (2) residuals proportional by a factor free
  of the system's coordinates/velocities/accelerations (zeros of the factor become conditions) or (3) its own name for a
  verified conserved quantity or momentum. The result's restrictions travel with the match; it is never "proved" unless
  both the tool's check and the link were symbolic zeros; matrices, vectors, distributions and sums are never linked;
  a wrong equation stays unchecked (not contradicted). Lone symbols are not claims. Add a wrong twin to
  `test_claim_evidence.py` for every new rule. One outcome is not a match and not a contradiction: a claim that reuses a
  tool's own name for a verified conserved quantity or momentum (`C_{4} = ...`, from `analyze_lagrangian`,
  `find_conserved_quantities` or `search_invariants`, which set roles and variables) with a right side that provably
  differs (`counterexample.find_counterexample`: a gap confirmed at 60 and 120 digits, never noise, a pole or an
  exception) stays `unchecked` with a note naming the verified equation and `evidence_rule: "mismatch"` in the debug
  claims (`EvidenceIndex.mismatch`). The name must be the same: another name on the left is the model's own quantity.
  A Taylor series is the fourth kind and has its own typed record (`SeriesEvidence`, `claim_evidence.SeriesIndex`, built by
  `run_series` from the engine's own result into `ToolOutput.series` / `TraceStep.series`, read by `loop.series_evidence`
  and `turn_context._prior_evidence`): `f(x) = P(x) + O((x-a)^{n+1})` is `supported` (`evidence_rule: "series"`) only if
  the function (the expanded expression, or a label `g(x)` an earlier formula defined as it), the variable, the centre,
  every coefficient (an exact symbolic zero, never a sampled agreement) and the remainder order all equal the result's,
  under the assumptions of the call. A changed coefficient, order, centre or function, an undefined label and a second
  remainder stay unchecked, never contradicted. It is worded as the tool's series result (numerical evidence; the remainder
  order a measurement), never proved and never a bound; `known_latex` only gains the `P + O(...)` form the tool itself
  printed.
  Because discovery constants now carry the `conserved` role, rule (3) also links `K = <the C_4 expression>`; its wrong
  twins (a sign, a 1e-6 factor, a dropped term) are in `test_claim_evidence.py`. The evaluation harness hands the claim
  check the earlier turns' verified results (`prior_evidence`) as the chat route does.
- Routing (`agent/routing.py`, `MINERVA_AGENT_ROUTING`): a rule may choose the first tool only when two independent
  signals agree: structure it can read (a Lagrangian with unambiguous velocities; the one result the previous turn
  produced; an equation with a derivative of an applied function; matrices the user declared n x n) and words that ask
  for the operation. Rules: `lagrangian_equations`, `followup_reuse` ("differentiate that result", "integrate it from
  0 to 1" on `@rN`), `ode_analyze` (classification only; solving stays the user's choice), `matrix_simplify`, `lagrangian_followup`, and
  `explicit_operation` (last: one imperative sentence about one formula, "Simplify $f$", "Differentiate $f$ with respect to x",
  "Give the Taylor series of $f$ about 0 up to order 4", "Find an antiderivative of $f$"; it declines on a bare `e`/`i`,
  an equation, two formulas, any word beyond the plain operation, and any conversation about matrices, vectors or tensors, and
  keeps `unsupported` (calculus) or `unchanged`/`conditions_not_satisfied` (simplify) as the answer). A rule
  needs every argument from what the user wrote (a variable they named or the result's one plain letter, bounds they
  gave as numbers, sizes in their words) and never fills in an assumption, commutation, invertibility or bound; if one
  is missing it returns None and the model plans. The call goes through reference resolution, shape repair, transport
  cleaning and the per-call tool context like a model call. A rule lists the statuses it keeps as an answer
  (`Route.answers`: `ok`, and `conditions_not_satisfied` for a non-commuting matrix product); any other outcome drops
  the route and the model plans. The model writes the answer. Do not route on a keyword alone and do not let a rule
  compute anything itself. `scripts/eval_models.py --routing/--no-routing` (default on, as the server) measures with and
  without; the report records which.
- Input layer (`engine notation/`, `agent/mathinput.py`, `agent/arguments.py`; `docs/agent-reliability.md`): what a
  model sends is read by code before a call is keyed or run, from closed lists, never by guessing. In order: shape
  repair (`arguments.repair_arguments`: a list of names sent as one string, one equation where a list is wanted, JSON
  sent as text, a formula under `expression`, a number where text is wanted), transport cleaning
  (`mathinput.prepare_arguments` over `MATH_FIELDS`, the one list of formula arguments: markdown escapes `\*`, `$`/code
  fence wrappers, doubled backslashes, Unicode look-alikes, HTML entities, a leading `L =`/`H =`, Greek letters and
  dotted letters in the syntax of the tool; and, in a field read as text syntax only, a braced exponent as the parenthesised one:
  `x^{-1}` is `x^(-1)`, through the one closed rule `transport.read_braced_exponents`: directly after `^`, balanced braces, nested
  groups handled; a group with a top-level comma, an empty or unbalanced one stays refused and no set literal is ever accepted.
  The text parser applies the same rule itself, because geometry metric entries reach it without this layer; its inverse is the
  corruption operator `syntax:braced_exponent`, not a `DAMAGE` row, because `clean_transport` is format-independent and in LaTeX
  the braces are syntax), then the derivative reader (`notation/derivatives.py`: `Xdot`, `X_DOT`,
  `dX_dt`, `x dot` are the velocity of a *declared* coordinate; the keyword is case-insensitive and the stem exact). A
  name with two readings, two spellings of one derivative in the same expression, a declared name, or a velocity of an
  undeclared coordinate is **refused with a typed `InputCode`** and the candidates, never guessed; a LaTeX letter run
  (`Xdot`) is a product there and is refused with the fix. Every change is a `Normalization` shown to the model
  (`input cleaned:` / `input:` lines), on the card and in the stored trace (`model_arguments` keeps the raw call). Do not
  add a rewrite that can change a meaning; add its inverse to `notation/transport.py` `DAMAGE` or the corruption test.
- Explore in Minerva (`packages/ui/fixtures/explorations.json`, `packages/ui/src/explorations.ts`, `apps/web/src/app/explore/[id]/page.tsx`,
  `services/api/tests/test_exploration_registry.py`, `docs/explore-in-minerva.md`): a landing-page example opens the real chat with its
  prompt **in the message box, never sent** (`ChatApp initialDraft`; the user presses Send and every result then comes from the engine
  like any other turn). The link is `/explore/<id>`: an id from the registry, never mathematics in a URL; a logged-out visitor
  comes back to it through login and register (`next` survives the proxy, the login/register links and a 401). The registry holds no
  result. Each entry names the landing example it continues, an evaluation case of the `explore` suite that runs its exact prompt with an
  engine-checked reference call (the test also pins the metric, coordinates and assumptions to the capabilities example), and is
  `approved` (its button shown on the site) only with five correct, engine-backed live runs of that exact prompt at the deployed 10 s
  limit in `docs/benchmarks/exploration-live.json`. Do not show a button for an unapproved entry and do not put a result in the registry.
- Landing research sessions (`scripts/record_landing_sessions.py`, `apps/site/src/data/landing-sessions.json`,
  `docs/benchmarks/landing-live.json`, `services/api/tests/test_landing_sessions.py`, `docs/explore-in-minerva.md`): the landing
  page shows two recorded conversations (Kepler, Schwarzschild): of five live runs of the `landing-*` cases on the deployed
  model, the first in which every turn passed, with the number that passed, the model and its settings stated beside it (a
  conversation with no passing run is not shown). Every recorded tool call is re-run against the engine in
  CI; changing a prompt or the deployed model means recording again. Never edit the JSON by hand. The page's formula marks use
  the chat's own wording (`packages/ui/src/marks.ts`).
- Homepage examples (`packages/ui/fixtures/homepage-examples.json`, `apps/web/src/lib/homepage-examples.ts`,
  `services/api/tests/test_homepage_examples.py`): one registry; a card shows only if `approved`, and `approved` is allowed only
  with five correct, engine-backed live runs of that exact prompt in `docs/benchmarks/homepage-live.json` (never the model host's
  address). Each card has an evaluation case (`homepage-*`, identical prompt, an engine-checked reference call, scored reply)
  and the registry holds no expected answer. Shape repairs found while making the cards reliable, all closed and visible: a
  coordinate triple sent as one string (`Curve`/`Surface.q`), a numeric bound as text, a theorem sent without `expression` and
  with exactly one declared field of its kind (`arguments._theorem_expression`, shown as a normalization), and a hint when an
  undeclared name is a coordinate of another system (`coordinate_hint`: `r` with Cartesian coordinates). A model call that ends
  with nothing visible and no tool call (not a length cut) is asked again once (`RetryReason.EMPTY_REPLY`, never repeated).
  The claim checker leaves relations between differentials (`dx = -du`) unverified: they are notation, not products of symbols.
  `scripts/eval_models.py --compute-timeout 10` runs the tools with the deployed chart's limit.
- Time symbol vs coordinates (`notation/time_symbol.py`): a `time` that is also a coordinate (or a Hamiltonian momentum) is
  refused as `parse_error` / `time_symbol_is_coordinate` (`recoverable`, field `time`, suggestions = the first free names of a
  fixed list) by `derive`, `hamiltonian_flow` and `time_derivative`, so every Lagrangian tool and the direct API get it.
  Never rename the time symbol for the caller; a new tool with `coordinates` and `time` is covered by
  `test_time_symbol_clash.py` (it lists the tools).
- Typed diagnostics: a refused input carries `diagnostic` (`InputDiagnostic`: closed `code`, `field`, the user's
  `symbols`, `suggestions`, `recoverable`) on its report, shown to the model as one `diagnostic: {...}` JSON line, and
  only the `code` is logged (`reason_code`, vocabulary `tracing.REASON_CODES`). Symbols and suggestions are the user's
  names: never in a log line. A new refusal gets a code in `notation/models.InputCode` and a test.
- Recovery (`agent/recovery.py`): a failure is classified (`FailureClass`) and the loop does the one deterministic thing
  first. A parse error that says the text is of the other syntax (`text_syntax_in_latex`, `unrecognized_derivative`,
  `latex_in_text`) in a tool with a `format` is run once with the other strict parser (trace `policy: format_switch`; the
  formula tools take a hidden `format`, default latex). A call equivalent to one that already ran or failed (same tool,
  same arguments after repair and cleaning, defaults filled, `format: auto` resolved: `recovery.canonical_key`) is not run
  again (`policy: refused_equivalent`); a tool that fails `AgentLimits.max_failures_per_tool` (3) times in a row is not
  run again that turn (`policy: budget_exhausted`); a failure with a diagnostic code not seen before for that tool is
  progress. Unsupported mathematics is an answer, never retried with equivalent input. Nothing here touches verification:
  a re-run goes through the same checks, and a candidate that was not verified stays withheld.
- Corruption test (`agent/evaluation/corruption.py`, `scripts/eval_corruptions.py`, `make eval-corruptions`, report
  `docs/benchmarks/corruption.md`): every known model mistake (the inverses of the cleaning rules, shape errors, velocity
  spellings) applied to every reference call of the benchmark must give the clean result or a typed refusal. A different
  verified result or an untyped failure is a defect and both stay at zero. A registered tool needs a reference call or a
  stated reason in `corruption.EXEMPT`.
- Turn lifecycle in the chat: `SendMessage.mode` is `new`, `retry` (answer the last user message again) or `edit` (replace its text
  and answer again); the reply after it is deleted and the user message is kept, never stored twice (equation numbers are not
  reused: the counter is a high-water mark, and a reference to a number whose message was replaced is told "no longer exists"). A
  client that leaves mid-reply (Stop, a closed tab) gets what had arrived stored as a `stop_reason: "stopped"` message
  (`_store_stopped`; nothing arrived: nothing stored), a `phase` event tells the client when the formulas are being checked, and text
  that starts like JSON or a code fence is held back until the model call ends so a tool call written as prose is never shown.
  Reasoning a model writes into its reply (`thought`, then a closing `<channel|>`; `agent/reasoning.py`) is dropped from what
  is streamed, stored and replayed, wherever in the reply it sits; a call that ends inside it shows nothing. A
  message sent after a reply that was never stored is merged with the unreplied one (the model never sees two user messages in a row).
  The composer gives the text back when a send failed before anything was sent; the address `/c/<id>` names the open conversation.
- Turn context (`agent/turn_context.py`, `turn_context(previous, text, counter=...)`, pinned by `test_turn_context_pin.py`): what a
  turn is given from the conversation before it (rebuilt history, verified and quoted formulas, `@rN` results and their
  restrictions, the equation table and what "equation 3" means, unverified inputs, claim relations, assumptions and evidence,
  declarations in words, tool groups, the verified constants of earlier discovery cards, the user's messages) is built by this
  one pure function from stored rows (`role`, `content`), never a database or a model. The chat route, `agent/evaluation/runner.py`
  and `scripts/replay_trace.py` all call it; do not assemble any of it anywhere else. A change to what it returns is a change of
  the live chat: the pin test compares what the route hands `run_agent` over a retry and an edit with a file written before the
  function existed, and must not be edited to make a change pass.
- The reply is a server-sent stream and must reach the browser as it is written. Through the web app's proxy Next.js gzips it
  and delivers the whole reply in one piece at the end (no live text; Stop keeps nothing), which is why `apps/web/next.config.ts`
  sets `compress: false` (the CDN or nginx in front compresses). Do not turn it back on; if a deployment puts another buffering
  proxy in front, the stream must be exempt (`X-Accel-Buffering: no`, `Cache-Control: no-transform` are sent). Check with a
  fetch from the browser: many chunks over seconds, not two at the end.
- Equation numbers (`agent/equations.py`, `agent/card_equations.py`): a card that lists equations (geometry: every component of
  a quantity as the row shows it, or its scalar; Lagrangian: equations of motion, momenta, accelerations, constants; invariants
  and discovery: the constants found and the relations; solve: `factor:i`, `root:i:j`, `rootrow:k`; ode_solution: `solution:s:j`
  for an evaluated explicit or implicit closed form only (an unevaluated integral, a series and a perturbation expansion are
  representations and get no row); ode_analysis: `solved:i` and `field:i`, not the equilibria (their order is not canonical) and
  never the interpretation, which is the user's own input; variation: `el:<function>` (none for a null Lagrangian), `first`,
  `stationarity:i`, `natural:<end>:<function>:<order>` and `second`, only for a report that is `ok` with its first-variation
  verification (or its second variation's own deciding check) passed; statistics: `estimate:<parameter>` (the row is
  `\hat{p} = value`, as the panel draws it), `moment:<name>` and `expr:<name>`, nothing when the result holds given a premise
  (`evidence == "conditional"`, any premise) or an assumed condition; expansion: `equil:<name>`, `eq:<coordinate>` and `char`
  and `mode:<i>:omega2` / `mode:<i>:omega`, each only when its own check passed, never the terms, the matrices A and B or the
  eigenvectors (the mode order is the solver's: deferred); statmech: `quantity:<name>` (`SYMBOL = formula`, as the panel draws
  it; `quantity:<name>:<k>` for the k-th repeat of a name) for an exact base quantity (Z, lnZ, F, U, S, Var_E: the panel's
  own table, read from the TSX by a test) of an `ok` result with no failed check, never a representation, a numerical value,
  an asymptotic result, a quantity named outside that table (Xi, Omega, occupations, means, the user's own partial-derivative
  labels), the user's `levels` step, the table or the error tables; perturbation: `omega0`, `omega` and `ratio` only for an
  `ok` report whose verification passed, never the solution (two relations joined by `,\quad` with an `O()` remainder: one number
  would name both, deferred) or the orders; dimensional: `group:<i>` (`label = formula`, by position) for a group rebuilt
  and found dimensionless in an `ok` report, never the quantities, the relation (it holds only up to an undetermined function
  of the groups, which a number would not carry) or the suggestions) numbers each listed equation with the next whole number of the same
  counter, in display order (`Card.numbers`, by row key; no sub-numbers such as 4.1). A card that was not accepted lists
  nothing. `sums`, `expectation` and `distribution` keep their result under `data["result"]` and are numbered as a whole through
  `NESTED_RESULT_CARDS` (status `ok`; an expectation that holds given an independence has no number, because the card has no
  result id to carry the premise to a later `@eN`); `approximation` keeps its result under `data["report"]` and is not yet. Each card type must be in exactly one of `LIST_READERS`, `SINGLE_RESULT`,
  `LIST_NOT_YET_NUMBERED` or `EXCLUDED_WITH_REASON` (a test); `tensor` is a single result (its one canonical form, numbered
  through `result_latex`), `ode_numeric` and `motion` are excluded with a written reason (a numerical trajectory shows no
  equation; the only formula of a motion card is the user's own Lagrangian), and `LIST_NOT_YET_NUMBERED` names cards that still
  list formulas without numbers (matrix list rows and vector; also deferred inside cards that are otherwise numbered:
  the ode_analysis equilibria, the statistics posterior, step lists, expansion eigenvectors, the perturbation solution,
  the dimensional relation, and approximation's nested result): move one to `LIST_READERS` with a reader and a test. A card that has both a list and a primary `result_formula` is numbered
  as a whole first and then row by row (`number_blocks`, `number_stored`). The panels mark exactly the server's keys with
  `<Numbered k=...>`, pinned by `packages/ui/fixtures/equation-rows.json` (`test_equation_rows_contract.py`,
  `test_equation_rows_variation_stats_expansion.py`, `test_equation_rows_statmech_perturbation_dimensional.py` and `equation-rows.test.tsx`, which also checks that each number sits next to the
  exact formula the server numbered). **A number on a list row is a name, not a verification**:
  `@eN` on one substitutes its LaTeX exactly (`Equation.part_of`), but the tool's verification and the card's restrictions are
  not carried over, so the result carries `unverified_inputs` ("equation (7), a component of an earlier result"), its
  formulas are quoted, never `known_latex`, and a note tells the model to say so. A quantity that was not accepted lists
  nothing. Every displayed formula (reply or the user's message) and every tool result with a
  primary `result_latex` gets a number, one counter per conversation (`Conversation.equation_counter`, a high-water mark that is
  never lowered), assigned by the server after the turn in reading order; a formula already numbered keeps its number. A number
  is a name, never a verification: do not let it change a mark, and never trust a number the model writes (the prompt says not
  to). "Equation 3", "eq. (3)" or a lone "(3)" in the user's message is resolved from storage and shown under their message; an
  unknown explicit number is said to be unknown. `@eN` in a tool argument is the equation's LaTeX (a verified result goes through
  `@rK`, with its restrictions). **An equation the model wrote and Minerva never verified may be used, but the result then carries
  `unverified_inputs`: its formulas are quoted, never `known_latex`, and the fact travels with the result id into later `@rK`
  reuse** (`ResultStore.unverified`, rebuilt from cards by `equations.unverified_inputs`). Do not widen `known_latex` through a
  reference. The shared fixture `packages/ui/fixtures/math-delimiters.json` is run by both `segments.py` and `inline-math.ts`:
  change a delimiter rule in both places and in the fixture.
  **A label the model writes inside a displayed formula is not part of it** (`equations.strip_label`, `strip_equation_tags`):
  `$$E = mc^2 (1)$$`, `\\tag{1}`, `\\quad (1)`, `\\text{(1)}` and a `\\begin{equation}` with one are read off the end
  of the formula before it is stored and keyed, so one formula keeps one number. Three modes decide who wrote the text:
  `model` (the full rule, including a bare ` (1)` unless it can be a factor or an argument: after a command name, an
  operator, an opening bracket, or a letter the formula applies as a function elsewhere), `user` (only the unambiguous
  forms; the user's own message is never rewritten) and `none` (every tool result and list row: stored exactly as the
  tool gave it, so `@eN` never differs from the verified formula). `number_for` and `restore` take the mode as a required
  argument and look the formula up *as written* first, so the model restating a tool result or the user's formula cannot
  change it. A conversation numbered before this keeps every number (nothing stored is rewritten); a stored duplicate is an
  `Equation.alias_of` the lower number and a reference to it says so. Removed labels are listed in `debug.removed_labels`
  and counted in `agent.completed` (`equation_labels_removed`, `equations_reused`, `equation_aliases_resolved`). Not read:
  labels inside multi-row environments; a new formula such as `y = g (1)` loses its `(1)` (accepted, and recorded).
- Model parameters (`MINERVA_LLM_*`, `agent/params.py`) are operator defaults, per-turn overrides in the chat's debug panel
  (`SendMessage.params`; off unless `MINERVA_LLM_TUNING`, the context size only for `MINERVA_LLM_TUNING_EMAILS`, capped) and
  flags of `scripts/eval_models.py`; what a turn used is stored in `AssistantMessage.debug.params`. **Nothing mathematical is a
  parameter**: a test pins the field list. Do not add a verification tolerance, a limit or the claim check to it.
- After each turn, formulas in the model's prose are compared with formulas present in this conversation's tool results;
  anything else is shown as "unchecked" (`agent/segments.py`). Do not remove or weaken this.
- Chat prose is a small Markdown subset drawn by `apps/web/src/components/chat/rich-text.tsx` (no Markdown library):
  `**bold**`, `` `code` ``, `-` and numbered lists, `#` to `######` headings, pipe tables (a header row, a `---` separator
  row, rows; only text is split on `|`, so `$|x|$` stays in its cell) and formulas; raw HTML is shown as text. The prompt
  allows headings and tables, and `segments._bare_equations` skips table rows (their bars are not math). An inline formula
  that is more than a lone symbol has a copy/edit bar that floats above it (hover, focus or a tap) and takes no room in the line.
- `ScriptedLLM` (`MINERVA_LLM_MODE=scripted`, development only) is a deterministic stand-in used by tests and for UI work.
  `MINERVA_LLM_MODE=ollama` + `MINERVA_LLM_BASE_URL` + `MINERVA_LLM_MODEL` use your own model host. The host name never goes into the repo.
- `scripts/eval_models.py` scores a model on a fixed benchmark through the real loop (`agent/evaluation/`: cases are data in
  `cases.yaml`, each with an independent source and, where possible, a reference call that the test suite runs against the
  engine; every failure gets one class; every failure gets one class; each run also carries separate dimensions (operation, mathematics, evidence, efficiency, multi-turn consistency; see `docs/reliability/evaluation-matrix.md`), `accept:` lists other tools that answer correctly with their own checks, and `--compare` says whether a change is within the run-to-run noise). Reports are JSON plus Markdown under `docs/benchmarks/`; `docs/model-evaluation.md`
  is hand-written interpretation and is never overwritten. Editing the cases changes the case-set digest: say so when
  comparing runs. The model host name never goes into a report.
  Every report opens with a "Model under test" table: the weights as the host reports them (base model, family, size,
  quantisation, digest), the defaults the host applies when a request is silent (a Modelfile's `num_predict` decides
  how long a reply or its thinking may be), the window actually loaded, what Minerva sends, whether thinking was
  requested, and whether the model was unchanged from the start to the end of the run. Without that a pass rate says
  nothing about what was measured: a name such as `minerva` is an alias the operator can re-point. `--compare` says
  loudly when the weights, host defaults or run configuration differ, or when either side did not record them (reports
  from before this existed). Never put the host address in any of it.
- Every agent request has a `trace_id`; the loop records typed events (`agent/tracing.py`) with monotonic timings. **Events
  hold no content**: only identifiers, counts, durations, booleans and fixed vocabulary. Do not add a free-text field to
  an event, an expression, tool argument or result to a log line, or any third-party logging or telemetry service; tests
  enforce this (`test_tracing.py`). The detailed trace (arguments and results) lives only in the stored assistant message,
  under the conversation's own access, deletion and export rules: do not give it a table, file or endpoint of its own.
  Logging and tracing failures must never reach the mathematics (`Tracer.emit` and the engine observer swallow them).
- A tool call that did not deliver a verified result writes one log line at the default setting: `tool.failed` (something to
  fix) or `tool.declined` (an honest answer or a call not run), with the tool, status, error category, `reason_code`,
  `arg_fields`, `invalid_fields` and, for an unexpected exception, the class and the module and line of our code
  (`logging.error_fields`, the sandbox child's `_site`): never an argument value, a symbol the user chose or an exception
  message (`docs/observability.md`, "Tool outcomes in the log"). A new `ErrorCategory` needs an entry in
  `tracing.OUTCOME_OF_CATEGORY`; a new log field needs the allow-list and a closed vocabulary or validator. Everything
  written to stdout passes the allow-list, library records included (`logging._foreign_record`): do not add a handler or
  `print` that bypasses it, and keep `--no-access-log` on the API's uvicorn command.
- Restrictions travel with a verified result (`ResultStore.restrictions`): when a later call reuses `@rN`, its card gets
  `inherited_restrictions_latex` and its tool message names them ("Inherited from @r1"); they are rebuilt from stored cards
  for the next turn. Do not drop them when changing `refs.py`, `loop.py` or `turn_context._history` (V-015).
- A turn ends with a stop reason (`completed`, `step_limit`, `tool_call_limit`, `time_limit`, `repeated_calls`, `output_limit`, `model_error`,
  `malformed_output`, `internal_error`). A last model call that reaches the model's length limit with no tool call is
  `output_limit` (stored as cut off, with a plain sentence), never `completed`. An identical call in a turn is not run twice; a model call that failed before any
  output is retried once; nothing mathematical is retried. A verified result gets an id and can be reused as `@rN`.
  A turn that fails outside the loop stores an assistant message with `stop_reason: "internal_error"` and fixed text (never
  the cause), so a reload and a debug report show it; the report also says when it was copied mid-reply or when the last
  message has no stored reply. A turn that ends at `step_limit`, `tool_call_limit` or `repeated_calls` gets one last model
  call with no tools to say what was and was not established; if that gives nothing usable the server writes a fixed
  closing from the turn's own trace (`loop.closing_summary`: counts and statuses only, no formula). A tool result of
  `resource_limit` carries a fixed hint not to repeat the call and to make a smaller one only if it is what the user asked
  for (`tools.RESOURCE_LIMIT_HINTS`). Differential equations: when `ode_analyze` lists two or more applicable methods and the
  user's message names none (`agent/choice.py`: closed form, numerically, series, perturbation, phase plane, equilibria;
  a pick by position counts only as a reply to that list), `ode_solve_symbolic` and `ode_solve_numerical` are refused
  (`needs_user_choice`) and the assistant lists the methods and asks. The server enforces it; the prompt only says it.
- A formula in the model's prose is marked `from_tool` only if a *verified* result (or a machine-checked handbook entry)
  produced it. The user's own input as read, a `stated` handbook entry, an identity that was not applied and anything from
  a result that was not accepted are `quoted`: shown, never marked verified. Do not widen `known_latex` back to "anything a
  tool printed"; `ToolOutput.evidence` is read from the engine's own result, never from the model.
- `statistical_mechanics` is one tool with an `action`, not one tool per quantity. Its arguments deliberately exclude
  the phase-space cell, the indistinguishability of copies, the numbers of a numerical evaluation and a sweep: the
  model must ask, never pick them. The card's buttons and form post them to
  `POST /v1/chat/conversations/{id}/tools/statistical_mechanics`; a test checks the tool schema has no such field.
- `creative_telescoping` (parameterised definite integrals) takes the integrand, variable, parameter, bounds and declared
  assumptions only: its arguments have no field for an operator, certificate, equation or constant, and the search bounds
  are the engine's. The card says whether a certificate, an equation or a value was established; a partial result is never
  worded as the value of the integral. `integrate` takes an optional `strategy` (`auto` or a technique name) and reports
  the technique, steps and verification; the model may only pick a technique the result suggested.
- `integrate_motion` (numerical solution of a Lagrangian system) needs numbers only the user may choose: the model must ask,
  never pick them on its own. The one exception is a user who says so in words ("you choose", "something reasonable";
  `agent/choice.py: delegates_numbers`, for that message and the next two unless they give numbers, user messages only):
  the card then says "Values chosen by the assistant at your request" and lists them, and a call whose numbers are in
  none of the user's messages is labelled the same way without a hand-over (`values_provenance`; a label, not a refusal).
  Numbers typed on a card's form are the user's. The Lagrangian card's form posts them straight to `/v1/chat/conversations/{id}/tools/integrate_motion`.
  A run is shown only if its second-method, time-reversal and conserved-quantity checks pass; do not loosen them.
- Conversations are user data: private per account, never logged.
- The model is never a question mark (`agent/identity.py`, `docs/model-identity.md`): the status chip and the line under every
  reply name the weights as the model host reports them (base model, size, quantisation, digest, thinking on or off), not
  the alias (`minerva` is a name the operator can re-point). A reply's stamp (`AssistantMessage.model_info`) is asked of the
  host at the start of the turn and again at its end, never from the 30 s status cache, and says when the digest changed
  while the reply was written; it is stored with the reply, so an old conversation still says what answered it. A host
  that does not answer gives `known=false` and the interface says "identity not reported"; nothing is guessed, a
  provider's model is marked hosted, the host address never leaves the server and the digest is not a log field.
- "Add image" (`agent/vision.py`, `/v1/chat/formula-image`) is transcription, not computation: a vision model
  (`MINERVA_LLM_VISION_MODEL`) proposes LaTeX, the user confirms it, and `Engine.read_formulas` reports how the
  strict parser reads it. Uploaded images are never stored or logged; keep it that way. "Take photo" is a native
  `capture="environment"` file input (no `getUserMedia`, keep `camera=()`), with a Retake / Use photo preview before anything is
  read; same gate (`imagesEnabled`) and same `image-prep` path as a chosen file.
- A turn is bounded: `AgentLimits` caps steps, total tool calls (12) and seconds, enforced during model streaming, between
  tool calls and around the claim check; a failing or hanging tool becomes an error card with fixed text, never its cause.
- Debug mode (chat footer checkbox, or `?debug=1`): shows a timeline (`apps/web/src/lib/trace-timeline.ts`) and each tool
  call's arguments and results under the reply, and copies a report (`apps/web/src/lib/debug-report.ts`, format 4) to
  paste to an agent; use the `chat-trace-triage` skill on it. The data comes from `AssistantMessage.trace/debug` stored with
  the message (failed turns too). The server log gets counters, `request_id` and `trace_id` only, never content. See
  `docs/observability.md`. With thinking requested (`MINERVA_LLM_THINK`, off by default), the model's reasoning is
  stored per model call in `debug.llm_calls[i].thinking` and shown only in debug mode (collapsed, and in the copied
  report); it is never reply text, never replayed to the model, never checked, and only its length (`thinking_chars`)
  reaches a trace event or the log.
  A follow-up turn is replayed offline with its stored context (`scripts/replay_trace.py --llm`, `ReplayLLM` only, never a model host):
  the report's earlier messages go through `turn_context`, the function the chat route calls. Format 4 adds each assistant message's
  `blocks` (the cards as stored, with their equation numbers), each user message's `equations` and `references`, and `conversation`
  (the equation-number high-water mark and which messages the report holds); with them the constants, `@rN` results and their
  restrictions, unverified inputs, evidence and the equation table are restored, and numbers given out once are not reused. Formats 2
  and 3 have no cards: the script names what it cannot restore (`missing`) and says the follow-up cannot be faithfully replayed; a
  difference that missing context may explain is listed apart from one in what the engine established, and a replay with anything
  missing is never "same" (exit status 3, not 0). A new kind of stored context the route reads must go into `turn_context`, into the
  report (`reportData`) and into `replay_trace.missing_context`.

## Mathematical correctness (non-negotiable)

- Never return a result that was not verified. Statuses are explicit: `ok`, `unchanged`,
  `not_verified`, `unsupported`, `parse_error`, `resource_limit`. Do not add fallbacks that
  produce plausible-looking output.
- Verification = independent numeric check at assumption-respecting sample points (complex by
  default) plus a symbolic zero test, reported separately. A `not_verified` candidate is never shown.
- Symbols are complex unless declared otherwise. Identities with validity conditions (e.g.
  `log(ab) = log a + log b` needs a, b > 0) are applied only when the conditions are provable
  from the declared assumptions; otherwise report `conditions_not_satisfied`.
- Matches that only work by inventing roots, reciprocals or `I` are discarded (`_invents_structure`).
- Handbook entries declare `verification`: symbolic / numeric / structural / dimensional / stated.
  `stated` entries are literature records, never applied, and must not carry a rule or check.
  Do not label an entry `symbolic` unless the checker finds a symbolic zero.
- `verify_equal` probes after its random points agree (`counterexample.py`, `TARGETED`): boundaries and zeros of the
  arguments of `Abs`/`Max`/`Piecewise`/`floor`/`log`/fractional powers, the negative real axis and large imaginary parts
  for complex symbols, integers beyond ±6 and simple special values, extreme scales, at *exact rational points*, and a
  disagreement counts only if it is stable at 60 and 120 digits (`NOISE_FLOOR`): cancellation is counted, never reported.
  Probes can only add a disagreement, never count as agreement. A point defined on one side only is not a counterexample
  (removable singularities). Do not widen `REL_TOL` or shrink the wide tranche to compensate, do not catch exceptions
  around the search, and keep `Counterexample` (it holds expressions) out of every log field and trace event. A pass
  still means "no disagreement found"; `docs/verification-testing.md` lists what sampling cannot find (V-014).
- Numeric agreement is evidence, not proof. `verify_equal` samples two tranches (ordinary region and a wide one,
  |x| up to 50, |Im| up to 10, high precision) and counts a point where only one side is a pole as a disagreement;
  `Verification.symbolic_zero` is the only symbolic evidence. UI and tool text say "proved equal" only with a symbolic
  zero and otherwise "numerically checked". Do not shrink the wide tranche: `log(exp(z)) = z` and `|x-10| = 10-x` pass
  the ordinary region alone (`test_verify.py`).
- Restrictions (`restrictions.py`): simplify and rule application report where the result is valid when a denominator
  was cancelled or introduced (`restrictions_latex`, e.g. `x \neq 1`), including cancellations the parser performs while
  reading. The check is syntactic (zeros of denominators); square-root, logarithm and branch-cut domains are not covered
  and a denominator that cannot be analysed is reported, never dropped (`restrictions_complete`).
- Claim classification (`claims.py`): `fails` (shown contradicted) only when the sides differ by a nonzero constant.
  Two notations are never sampled because the permissive parser reads them as something else: Big-O/little-o (`O(t^3)` is
  O*t^3; `_BIG_O`) and an applied function nobody declared (`g(t)` is g*t; `_APPLIED`: a letter or Greek name directly
  followed by `(` that is not also a plain symbol of the same formula, so `x(x+1)`, `a(b+c) = ab + ac` and `m (a+b)` are still
  tested). Such a claim is `unverified` with the reason, is never learned as a definition (it would define a symbol by a
  statement about another expression) and is never `conditional` or `fails`; pointwise agreement never establishes an
  asymptotic remainder. The expansion point and the places where a verified series measured its error (`x = a`,
  `x = a \pm d`, `|x - a| = d`, read from the typed series record) are parameters of that call, not definitions: they are
  neither learned (not from the reply and not from the conversation's earlier messages) nor tested against; every other
  `symbol = value` is still a definition and still contradicts a wrong value. Without a series record (the call failed or
  fell out of the evidence window) `x = 0` is still a definition (A-012).
  A false identity at generic points is `not_identity` (shown unchecked): Minerva cannot tell it from an equation to
  solve, a definition or an equation of motion, which are `unverified`. Do not map `not_identity` to contradicted.
  The one place a false identity is called false is `check_identity` (`identity_check.py`, tool and `Engine.check_identity`), and only
  because the USER asked whether one stated equation holds: it reports `proved_equal` (symbolic zero), `numerically_equal`
  (no disagreement found: evidence, never proof), `conditional` (true once symbols are assumed real or positive, the
  condition stated) or `not_identity` with an exact point confirmed at 60 and 120 digits (`counterexample.py`), worded "not an
  identity; it may still hold for some values". It gates on the claim checker, so a relation, matrix, vector, distribution or
  sum is `unsupported` (never true or false), and a disagreement it cannot confirm is `not_verified`. The claim checker on the
  model's own prose is unchanged. A `Counterexample` stays out of logs and events; the card shows it to its owner only.
- Dimensional analysis (`pi_groups.py`) reasons about dimensions only: a group is shown only if rebuilt and found
  dimensionless; a mismatch proves an equation wrong as written, a match proves nothing more; the missing-quantity
  report is "the unknown's dimension is not a product of powers of the knowns'", not a claim about the physics.
  Units are coherent SI only (no conversion).
- Derivatives (`differentiate.py`) are verified by an independent numerical derivative, one order at a
  time, and along the imaginary direction too for variables not declared real. A derivative with no
  complex derivative (`|z|`, `re`, `im`, conjugate) is `unsupported`, never a formula. Do not add a
  symbolic fallback or weaken `check_derivative` (its tests include a deliberately wrong candidate).
  `differentiate` is a *partial* derivative: by `t` of an expression with velocities (`x_dot`, `x_ddot`) it is `unsupported`
  (it used to answer a verified 0) and points to `time_derivative` (`time_derivative.py`, mechanics group): the total
  derivative along a motion (chain rule over coordinates, velocities, time), shown only if a central difference of the
  *input* along random polynomial motions agrees to 1e-12 (`check_total_derivative`; a 1e-9 error is rejected). With a
  Lagrangian it is also evaluated on the equations of motion (`derive`, confirmed by `check_equations`) and a zero is
  decided as for a constant of motion; "zero on the equations of motion of this Lagrangian" is the only wording, the
  nonzero formula is quoted (never `known_latex`), and accelerations in the input are `unsupported`.
  Handles (issue #149): `time_derivative` is in `refs.REFERENCEABLE_TOOLS`, so a verified total derivative has an
  `@rN` and, as a tool card's single result, an equation number **whatever its length** (`number_for(any_length=True)`;
  the 500-character limit stays for the model's prose and the user's formulas, and list rows). The id names
  `result_latex`, the derivative before any equations of motion are used, with inherited restrictions and unverified
  inputs carried as for every tool. The conclusion on the equations of motion (`zero_on_motion`, `on_motion_latex`) gets
  **no handle** and a plain `0` is never registered (it would drop the system); the tool text says so
  (`TOTAL_DERIVATIVE_HANDLE_NOTE`) and a nonzero on-motion formula stays `quoted_latex`. Above `LONG_RESULT` the text
  points to the result id instead of "not repeated here". Receiving tools keep their input limits: an over-limit handle
  is refused there. A card equation `@eN` with no result id is substituted but carries `NO_ID_SUFFIX` provenance (no
  verification or restrictions carried over), like a list row. Tests: `test_total_derivative_handles.py`.
  Step and matrix results (issue #133, part 1): `substitute`, `rewrite_part` and `matrix_calculus` keep their tool result
  under `card.data["result"]`, and the numbering used to read only the top-level `result_latex`, so a successful step was
  never numbered although `step` is in `SINGLE_RESULT`. `card_equations.result_formula(card.type, data)` is the one reader
  (used by `number_blocks`, `number_stored`, `build_table`, `ResultStore.register`, the `turn_context` rebuild and the
  label guard in the loop, so a live table and one rebuilt from stored rows agree; `result_restrictions` reads
  `restrictions_latex` the same way). A step counts only if its result is `ok`; a `matrix` result only if it is
  `evaluated` and not an `equation`, `representation` or `numeric` kind: those get **no number and no handle**, because
  a number would present a Sylvester equation, a block or integral form or a numerical approximation as an evaluated
  result. An evaluated matrix result is numbered (a name) but is not in `REFERENCEABLE_TOOLS` (matrices are
  noncommutative; reuse by `@eN` carries `NO_ID_SUFFIX`); the rows it lists stay in `LIST_NOT_YET_NUMBERED`.
  Typed references (issue #133, part 3; `agent/reference_types.py`, `equations.relation_kind`, `refs.ResultStore.resolve(args, tool)`):
  every numbered formula records a kind read off its text by one rule (`equation`: a top-level `=`; `expression`: no
  relation; `None`: undecided, for another relation sign, a multi-row environment or unbalanced brackets, and for an
  equation stored without one). `FIELD_POLICY` says per (tool, formula field) whether the tool takes only an expression
  (`simplify`, `differentiate`, `series`, `integrate`, `distribution`, `time_derivative.latex`, `find_identities`, `sums`,
  `expectation`, `creative_telescoping`, `approximation`, `substitute`, `rewrite_part`, `variation.integrand`), requires an
  equation (`solve_polynomial`, `isolate_variable`, `check_identity`, `check_units`) or is unclassified with a stated reason
  (matrix and vector calculus, the Lagrangian and Hamiltonian fields, `tensor_canonicalize`). An entry exists only where the
  real tool was shown to refuse (or require) a relation (`test_reference_types.py` runs every one). `@eN` of a kind the
  field cannot take is refused with a typed diagnostic (`InputCode.EQUATION_REFERENCE_NOT_EXPRESSION`,
  `EXPRESSION_REFERENCE_NOT_EQUATION`; symbols hold `@eN`, never a formula; only the code is logged) and the call is not
  run; which side of an equation to use is never decided by Minerva. A kind that is unknown is never refused. Add a (tool,
  field) entry only with that evidence; `test_every_tool_with_a_formula_field_is_classified_with_a_reason` fails otherwise.
  `substitute` and `rewrite_part` are in `REFERENCEABLE_TOOLS`: the id names the whole expression after the step, with
  its own and inherited restrictions. A `substitute` that holds only given the user's relations (`evidence ==
  "conditional"`) registers those premises with the id (`card_equations.premise_labels`, class `PREMISE_SUFFIX`, rebuilt
  from stored cards by `equations.unverified_inputs`): every later `@rN`/`@eN` use carries them, its formulas are quoted
  (never `known_latex`) with `refs.PREMISE_INPUT`, the note says "given, not proved", and the card shows "Holds only given
  a premise" (shared fixture `packages/ui/fixtures/input-provenance.json`). The step's own result keeps its existing
  marking (the premise is named in the tool text). Tests: `test_step_numbering.py`.
  Names of constants of motion an earlier `find_conserved_quantities` listed (`C_{1}`...: "the derivative of Equation 14
  with respect to time") are bare symbols with no dependence on the motion (`unsupported`, `no_dynamic_dependence`, never a
  retry). The SERVER inserts their verified expressions (`agent/constants.py`, `ToolContext.constants` read from the
  stored discovery cards; `time_derivative.definitions`, exact `xreplace`, no simplification) and only when each name has
  exactly one verified definition in one Lagrangian system (`ambiguous_constant`, `constant_not_verified`,
  `unsupported_definition` otherwise). The model has no argument for a definition (a test pins the schema), the card and
  the model are told which names were replaced, the derivative and every check are of the expression with them
  inserted, and `Q = 0` sent for a quantity is repaired to `Q` with the repair shown. The zero test on the motion
  evaluates at 80 digits on the sample floats taken as exact rationals (`verify.numeric_check(digits=)`, compiled once
  through mpmath): an expression of several constants has terms near 1e22 at the wide sample points that cancel exactly,
  and 53-bit samples made a true zero "not zero". The tolerance did not change; a definition wrong by 1e-6 is still
  rejected. Constants found in the same turn are not read (only stored cards).
- Series (`series.py`) are Taylor polynomials about a finite real point, shown only if two checks that do not
  use `sympy.series` both pass: the coefficients against a Cauchy-integral estimate (trapezoid on circles,
  trusted only where the estimates at r and r/2 agree, tolerance per coefficient) and the h^(N+1) decay of the
  remainder. Poles, branch points, logarithms and `|x|` are `unsupported`; there are no Laurent, Puiseux or
  asymptotic fallbacks. The error table is a measurement, never a bound. Do not loosen `check_series` (its
  tests include wrong candidates that differ by 1e-9).
- Expansion about an equilibrium (`expansion.py`, tool `expand_motion`) shows a formula only if the EL equations
  and the solved accelerations were confirmed, the point was shown to be an equilibrium, and every expansion
  passed `check_series`. A point that is not an equilibrium is `unsupported`, not expanded. The sign of ω² is
  stated only when the declared assumptions decide it, a negative ω² is described as a fact about the
  linearised equations, and nothing here says "stable" or "unstable" about the full motion. Mode formulas
  exist only for at most two coordinates and no velocity-linear terms; otherwise the characteristic polynomial.
- Perturbation solutions (`perturbation.py`, tool `perturbation_solve`) are Poincaré–Lindstedt for one coordinate
  whose acceleration depends on the position only. A solution is shown only if the truncated solution, put into the
  *unexpanded* acceleration from L, leaves a residual that falls at ε^(N+1) when ε is halved twice, the initial
  conditions hold, and the expansion of the equation passed `check_series`. Do not weaken `_check_residual`: its
  tests include a 1% wrong frequency correction, a wrong harmonic and the unstrained (secular) expansion. The
  comparison with `integrate_motion` needs numbers only the user may choose (the model must ask), is a
  measurement on that span (never "valid for all times"), and is omitted when the numerical run fails its own
  checks. Damping, driving, velocity-dependent accelerations and several coordinates are `unsupported`.
- Antiderivatives (`integrate.py`) are shown only if an independent numerical derivative of the candidate
  matches the integrand (the same `check_derivative` machinery, branch by branch for `Piecewise`; a branch
  condition that cannot be sampled makes the result `not_verified`). "No closed form" is `unsupported` and
  never claims that none exists. The symbolic zero test is reported separately and never decides.
  Definite integrals are verified by independent adaptive quadrature at sampled parameter values, never
  by the fundamental theorem (it would "verify" `\int_{-1}^{1} x^{-2}\,dx = -2`); a quadrature with a large
  error estimate counts as no agreement, and convergence conditions the declared assumptions do not imply
  give `conditions_not_satisfied`. Do not add a symbolic-only path.
- Creative telescoping (`creative_telescoping/`, operation and endpoint `telescoping`, tool `creative_telescoping`,
  `docs/creative-telescoping.md`) searches for `L f = dG/dx` for **hyperexponential** integrands (rational `f_x/f`, `f_a/f`)
  by an exact nullspace computation over a fixed ladder of families (operator order, certificate degree), and shows a
  relation only if `(L f - dG/dx)/f` simplifies to zero (`certificate.verify`; two numerical legs may only veto, a zero
  operator is rejected). Convergence, differentiation under the integral sign and boundary terms (`boundary.py`) are
  decided **from the declared assumptions only** (coefficient certificates for signs, local exponents at each end); an
  undecided sign is `conditions_not_satisfied`, never assumed, and divergence is claimed only where the phase does not
  oscillate. The ODE must have a provably nonzero leading coefficient on the parameter domain, its `dsolve` solution is
  substituted back exactly (SymPy sometimes returns only the homogeneous part), the constants come from independently
  verified initial values (`integrate.definite_value`), and a value is shown only if the equation, the initial values and
  quadrature of the original integrand all agree; it is worded "checked numerically", never "proved". A verified
  certificate alone is a partial result (`outcome` certificate or equation, with `unresolved`). "No relation found" is
  about the searched families only. Do not widen the class silently, drop the exact check, let a numerical leg accept,
  or import SageMath's GPL `ore_algebra` (see `docs/licensing.md`). `creative_telescoping/benchmark.py` has the corpus
  (`make bench-telescoping`; `--check` in tests).
- Integration strategies (`integration_strategy/`, `integrate(strategy=...)`, `docs/integration-strategies.md`) are
  deterministic and never call a model. `detect.py` is a pure function of the tree; every technique in `transforms.py` only
  *proposes*: `explore.py` verifies the candidate against the **original** integrand (`check_antiderivative`,
  `check_definite`), discards and counts failures, reports a technique whose conditions are not established as `blocked`
  (never applied: Frullani needs scales declared positive and finite limits, parity needs a convergent half-range and
  never gives 0 for a divergent one, Schwinger needs a provably positive base and a nonnegative remaining factor),
  and keeps the simplest verified form. SymPy's route runs first and unchanged (within a share of the time budget so a
  hanging integral cannot starve the strategies). A definite value is never called proved. A numerical estimate (two
  independent rules agreeing, tail negligible) is not a formula: unrequested it rides along with the unchanged
  `unsupported` status, and it is `ok` only for `strategy=numerical` or `auto`. Do not turn a blocked technique into a
  result, route a candidate around the verification, or add a fallback that returns plausible output.
- The search for hidden constants of motion (`invariants.py`, tool `search_invariants`) shows a constant only if it
  is new (known constants and their products in the ansatz are factored out) and its time derivative on the
  equations of motion is zero symbolically and at random points (`lagrangian.verify_constant`, the same check as for
  Noether constants). It always states the exact family searched; an empty result proves nothing outside it and
  nothing here says "integrable" or names a constant: a name such as Runge–Lenz is for the model to offer, not for the
  result to assert. Time-dependent Lagrangians and non-rational accelerations are `unsupported`, an oversized
  ansatz is `resource_limit`. Do not widen the ansatz silently or report an unverified candidate.
- Adaptive discovery of conserved quantities (`discovery.py`, `flow.py`, `independence.py`; tool
  `find_conserved_quantities`, endpoint `POST /v1/compute/conserved`) takes a Lagrangian **or** a Hamiltonian (the tool takes
  `format` `auto`/`latex`/`text` like `analyze_lagrangian`) and chooses its own ansatz: a deterministic ladder of nested families (momentum degree, power bound, harmonics, cyclic
  powers, powers of t), cheapest first, each solved exactly. It shows a constant only if it passed `Flow.verify`
  (symbolic zero and numeric points on the equations of motion, which were themselves confirmed by numerical
  differentiation of L or H); a candidate that fails is dropped and counted. Products of known and found constants
  (including combinations that cancel down into a family) are factored out, so a trivial combination is never a
  discovery. **Search completeness is not global completeness**: a tier is complete within its family only, and
  "no additional constant" is always worded "within the explored families", never "the system has none". Cyclic
  and cos/sin-only coordinates are treated as angles because conservation does not depend on periodicity. Parameters
  are generic; the report says so and lists the factors assumed nonzero. Functional independence
  (`independence.py`) is the exact Jacobian rank at rational points, a **proved lower bound**; it is called exact
  only when an upper bound from exact polynomial relations (implicit function theorem) meets it, and sampling never
  proves independence or dependence. Do not add physics-specific shortcuts or hardcoded invariants: the benchmarks
  (`packages/engine/tests/bench`, fixtures regenerate from seeds) are separate from the discovery code and never
  hand it an expected invariant. Names such as Laplace–Runge–Lenz are for the model to offer afterwards. Events
  (`invariant_search.*`) carry counters and fixed labels only; the API logs them with the request id as trace id
  (`search_log.py`), expressions stay in the user's private card. Square roots, logarithms and non-polynomial time
  dependence are `unsupported`; more than 3 coordinates is `unsupported` with the typed code `too_many_coordinates` (the limit, removing a
  cyclic coordinate by hand only with the user's own stated momentum, and that automatic reduction and a larger search are not available
  yet; nothing is searched and no coordinate is dropped; the tool schema allows 8 so that this reason, not a length error, reaches the model); families above the unknown cap are listed as not searched.
  The original `search_invariants` is kept unchanged as the fixed-ansatz baseline of `docs/invariant-benchmark.md`.
- Chains of identities (`transform.py`, tool and endpoint `simplify`) are shown only if every step passed the
  numeric check and the whole chain was re-verified with the symbolic zero test added; the two are reported
  separately and numerical agreement is never called a proof. A rule whose conditions are not provable from the
  declared assumptions is reported as `blocked`, never applied, and a SymPy move is labelled as one (it is not a
  handbook identity). `unchanged` means that nothing verified was found within the bounds, not that nothing
  exists. Keep the bounds (width, depth, verifications per state, size cap, deadline), the visited set, the
  right-to-left restriction (`can_reverse`) and the match filters; do not add an unverified shortcut. Chains
  must not become a way around `_invents_structure`: `rule_rewrites` is the one place that decides what matches.
- Matrix calculus (`matrix/`, tool and endpoint `matrix`) treats declared matrices as noncommutative symbols and
  shows a result only if its method's own checks pass: random complex matrices of the declared sizes (declared
  commuting groups sampled as polynomials of one matrix; explicit entries at sampled parameters; a matrix that
  depends on the variable as a quadratic polynomial in it, honouring the declared commutation), exponentials by
  `mpmath.expm`, integrals by quadrature and derivatives by a central difference, none of which uses the route that
  produced the result. The exact leg (formal power series through degree 4, modulo the declared commutations, or
  the explicit result substituted into the Sylvester equation) is reported separately; a failed exact leg is
  `not_verified`, an exact residual that SymPy cannot simplify to zero is inconclusive (`None`), never a refutation.
  Matrices are never reordered and never assumed to commute or to be invertible: a merge or a closed form that needs
  either is `blocked` and reported, unless the commutation or invertibility is declared or visible in explicit
  entries (a declaration contradicted by the entries is an error). A commutation declared for a time-dependent
  matrix holds at all times, so it also covers its derivative; `self_commuting` states `A(t1)A(t2) = A(t2)A(t1)`.
  `A(t)` is read as the matrix only if `A` is declared to depend on `t`; otherwise it is a `parse_error`, never `t*A`.
  d/dt exp(X) is `X' exp(X)` only if X and X' commute; otherwise it is the Frechet integral (`exp_frechet_derivative`),
  a representation, never rewritten. A Sylvester equation, a block-matrix or integral representation is **never**
  presented as an evaluated result (`evaluated=false`, `result_kind` equation or representation); entries exist only
  for explicit matrices whose closed form SymPy finds in time (`symbolic_budget`; larger ones fall back to the
  representation, recorded as a fallback). Duhamel integrals (`duhamel`): the singular Sylvester equation is never
  solved. Evolution operators (`evolve`): `exp(integral of A)` only for a generator that commutes with itself at all
  times; explicit entries that do not are solved numerically (`evolution_numeric`, `result_kind` numeric,
  `evaluated=false`) only on an end time the user gave, shown only if RK4 agrees, integrating backwards returns I and
  Liouville's formula holds, and always described as a numerical approximation on that span. Time-ordered
  exponentials, Dyson series and Magnus terms for abstract generators are `unsupported`. `e^{tA}` is a `parse_error`
  (a bare e is a symbol; write `\exp(tA)`). Do not loosen the numeric tolerance, add an unverified shortcut, or let
  `_invents_structure`-style filters be bypassed; the tests include wrong merges, reordered products, wrong
  derivatives (`X' exp(X)` for non-commuting X), `exp(integral A)` for a non-commuting generator and 1e-6 errors.
  Every result has `diagnostics` (`MatrixDiagnostics`): ids, enums and numbers only, never names, entries, expressions
  or bounds; the server log gets the same kind of fields and a test enforces it. `docs/matrix-benchmark.md` and
  `python -m minerva_knowledge matrix-benchmark` measure coverage, method selection, verification, time and
  residuals; a pending case that now passes fails the suite.
- Claims about matrices: a name is a matrix if the conversation declared it in a `matrix_calculus` call, as
  `noncommutative` in any tool call's assumptions, or in words ("let A and B be matrices", `agent/declarations.py`,
  deliberately generous: a wrong guess only makes a claim unchecked). The whole conversation counts, not the history
  window, and so does the model's own reply. `check_claims` decides "mentions a matrix" from the **text** of the
  formula (`claims.mentions_name`), before the parser can cancel anything (`AB - BA` is 0 as commuting symbols); a
  declared `A` covers `A_1`, and `C = AB` makes `C` a matrix. An equation that mentions one is `unverified`, never
  "checked" by sampling numbers. Transpose and adjoint notation (`^T`, `^\top`, `^\dagger`) is `unverified` too (a bare `T`
  would otherwise be read as a positive symbol). `noncommutative` and unknown assumptions are removed from the
  assumptions the checker parses with (`clean_assumptions`): they used to stop the whole check. The scalar tools
  (`simplify`, `differentiate`, `integrate`, `series`, `approximation`, `isolate_variable`, `solve_polynomial`, `find_identities`,
  `distribution`, `check_identity`) refuse an expression that mentions a declared matrix (`tools.SCALAR_TOOLS`; `substitute` and `rewrite_part` too, unless the call declares the name `noncommutative`, `NONCOMMUTATIVE_AWARE_TOOLS`; `sums` and the Lagrangian tools too, but for a declared matrix only, not a vector: `MATRIX_ONLY_TOOLS`, `ToolContext.vector_names`); a new tool must be classified in `tools.py` (`MATRIX_SAFE` needs a reason; `test_tool_matrix_classes.py`), and the chat route and the evaluation harness read declarations through one function, `turn_context.declarations_of`, so a scalar result about
  matrices is never marked `from_tool`. Do not narrow any of this to the last N messages or to parsed symbols.
  Two readings of declarations in words exist on purpose (`agent/declarations.py`). The generous one (every symbol of a
  sentence that says "vector") feeds the claim checker, where a wrong name only leaves a claim unchecked. The strict one
  (`read_declarations(strict=True)`: a bare name or list of names tied to the keyword by "is/are/be", "the vector r" or
  "do not commute", never `$1/r$` or `$H = T + V$`) feeds `ToolContext.matrices`, because there a wrong name blocks every
  scalar tool for the rest of the conversation (a reply mentioning "the Runge-Lenz vector, unique to $1/r$ potentials" once
  made `r` a vector). A refusal quotes the sentence and who wrote it (`ToolContext.matrix_sources`, never logged), and a
  *user* sentence calling a name a scalar clears it (the model's own text never does), with one exception that is a
  definition and not an opinion: `r = |\mathbf{r}|` (the bare letter is the norm of its own bold or arrowed spelling) makes
  the bare letter a scalar whoever wrote it, and the bold spelling then names the vector without declaring `r`. A scalar
  made by that rule or by a user sentence about that very name is *held*: only a later user sentence that declares the
  bare name brings it back, never the model repeating "the position vector" (`declarations.scan`, `Scan.held`; the
  reply's own claim check reads the same set through `declared_in_reply`). Do not feed the generous reading to a tool
  refusal. The refusal reads the fields by kind: a declared vector blocks the expression fields of a scalar tool, and a
  Lagrangian or Hamiltonian field only when the name is a declared matrix (`tools._SYSTEM_FIELDS`), because the
  coordinates of a system are scalars whatever vector the model also wrote. The claim checker is told which names are
  vectors and which matrices (`Declarations.prior_vectors` / `prior_matrices`), so a vector is never called a matrix.
- Distributions (`distributions/`, tool and endpoint `distribution`): a Dirac delta is 0 at every point a sampler can
  draw, so pointwise sampling proves nothing about it. Every other operation refuses `DiracDelta` / `Heaviside` input as
  `unsupported`, `verify_equal` never passes it, and the claim checker leaves such statements `unverified` (never
  `fails`). Do not remove those guards or add a path that verifies a distribution by sampling. A result is shown only if
  the exact conditions hold (variable and argument symbols real, scale provably nonzero, roots real and provably simple,
  multiplier defined and continuous, positions decided) and an independent test-function pairing agrees; the two are
  reported separately and the pairing is "supporting evidence, not a proof". Equality is "as distributions", never
  "proved" or "sampled". A delta exactly on an integration limit counts only under an explicit `endpoint` convention
  that the user chose (never default it, never let the model choose it). Repeated roots, undecided signs and
  non-polynomial arguments are refused with an enumerated reason (`unsupported_distribution`, `singular_root`,
  `assumptions_insufficient`, `integration_domain_ambiguous`, `verification_failed`, `resource_limit`). Input uses the
  explicit names (`\operatorname{DiracDelta}`, `DiracDelta(x)`). `\delta(...)`, `\theta(...)` and `\Theta(...)` (text
  `delta(...)`) are ordinary symbols in every other operation and in the model's tool calls; inside `distribution`
  they are read as distributions only for a reading the USER confirmed (`DistributionInput.notation`, filled from the
  card's buttons through `POST /v1/chat/conversations/{id}/tools/distribution`, or the workbench). Without an answer
  the result is `notation_unconfirmed` with `questions` and nothing is computed; `DistributionArgs` has no `notation`
  field and `ToolContext.notation` is set only by that route, never by the agent loop. A confirmed reading goes
  through the same exact conditions and pairing as the explicit names. Handbook entries about distributions have level
  `distributional` and no `rule`. Do not loosen the pairing tolerance (`pairing.TOLERANCE`): its tests include a 1e-6
  relative error, a missing `1/|g'|`, a dropped root and a wrong jump. Logs carry stage events with enumerated reasons
  only, never expressions.
- Relations in claims (`claims.py`): an equation with a symbol on one side only or a time derivative (`\theta=\phi+\omega t`,
  `L=T`, `F=ma`, `m\ddot x=-kx`) is a *relation*, not an identity, and is never tested by independent sampling. Definitions
  stated earlier in the conversation (`symbol = expression`, from the user or the model, not verified by Minerva) are
  substituted into it; a result that used them names them (`given`), and one that disagrees with them after
  substitution is `fails` ("does not agree with …"), not "false". A new definition is never a contradiction, and a
  stale definition never rescues a false identity. A later definition of a symbol replaces the earlier one and the
  definitions made in terms of it are rebuilt, so they follow the new value (they used to keep the old one: `K = 48 M^2 / r^6`
  learned while `r = 0` held contradicted the correct `K` at `r = 2M`); a definition that is undefined where it was
  substituted (a division by zero) leaves the claim `unverified`, never `fails` (A-010). "Could not check" must never be worded
  as "false".
- Controlled steps (`steps.py`, `substitute.py`, `isolate.py`; tools `substitute`, `rewrite_part`, `isolate_variable`) do one move the
  user chose. A relation the user supplies is a *premise*: a result that depends on it is `conditional` ("holds given ..."), is never
  called proved, and Minerva never checks that the relation is true. Several relations whose order changes the result, or that rewrite
  each other, are refused until the user states the order: never pick one. A part is chosen by path plus its LaTeX and a stale choice is
  refused. A rewrite of a part goes through `ops.rule_rewrites` (so `_unmet`, `_invents_structure` and `can_reverse` stay the only
  matcher) and the *whole* expression is re-verified. `isolate` lists every solution with its conditions, removes roots a denominator
  excludes, and shows a solution only after substituting it back into the original equation and passing a separate check that none was
  dropped (`peel` with a free symbol and wide probes, or the numerator's numerical roots); anything else (periodic inverses, symbolic
  degree 3+, systems, a base not known positive) is `unsupported` and never partially answered. Solutions are over the complex numbers
  (a log or exponential gives a family in an integer `k_1`); assumptions on the unknown are not used to choose among them. Do not
  weaken `back_substitute`, `forward_probe` or `count_roots`: their tests include a dropped root, an extraneous root, a missing
  branch, a missing integer family and a wrong formula.
- Substitution and restrictions of earlier steps (`substitute._carry_restrictions`): a continued `substitute` receives the
  restrictions of its parent chain (e.g. `x - y \neq 0` after cancelling `(x^3-y^3)/(x-y)`). The user's premises never remove
  one (no "discharge": Minerva does not check them), so each is carried on unchanged and named in the message. A premise
  that makes the excluded quantity exactly zero (`x = y` after `x - y \neq 0`) is refused, and a restriction that cannot be
  read is refused rather than dropped. A relation whose left side is not found *as written* says so and points to
  "Rewrite as" (`x + y` is not found in `x^2 + y^2`; substitution matches syntactic occurrences only).
- Noncommuting symbols: declared `noncommutative`, allowed only for polynomial expressions in substitute, rewrite (expand, collect by
  commuting targets) and subterms. `Engine._run` refuses them everywhere else, `Builder` refuses them unless asked, and identities are
  checked on random matrices of size `max(3, degree // 2 + 1)` (smaller matrices accept false identities).
- Diagnostics (`models.Diagnostics`) hold durations, counts and enum codes only, and are logged by default; the fields are allow-listed in
  `logging.py`. Rejected candidates (`Diagnostics.detail`) may contain expressions: they exist only with `MINERVA_DEBUG_COMPUTE`
  (development only), are returned in the response, and are never logged or saved.
- Differential geometry (`geometry/`, tool and endpoint `geometry`) is coordinate-based, for an explicit symmetric
  metric (at most 4 coordinates) and an optional tensor field, with functions of the coordinates declared by the
  user (`f(r)`: the parser reads an undeclared `f(r)` as the product f*r everywhere else, so geometry parses with
  `strict_calls` and refuses it). The conventions are fixed and returned with every result (Christoffel symbols of
  the Levi-Civita connection, `R^r_{smn} = d_m G^r_{ns} - d_n G^r_{ms} + G^r_{ml} G^l_{ns} - G^r_{nl} G^l_{ms}`,
  Ricci = contraction of the first and third index, `G = Ric - g R / 2`): do not change them silently. A quantity is
  shown only if every *numeric* leg passed: `numeric.py` evaluates the lambdified metric with mpmath and
  differentiates by central differences only (no SymPy differentiation), and compares with an independent numerical
  recomputation and with identities the result must satisfy (metric compatibility and no torsion, the Riemann
  symmetries, first and second Bianchi, a divergence-free Einstein tensor, the Euler-Lagrange equations of
  `g(x', x')/2` for geodesics). Identity checks measure their residual against the size of the terms that cancel
  (a residual of 1e-9 from terms of size 1e3 is noise), a point is skipped, never failed, when the values move when
  the step is doubled, and `STABLE_TOL` must stay below `AGREE_TOL / 3`: a property test found the false
  "not verified" that this prevents. Symbolic legs (including the second-derivative formula for `R_{abcd}` and the Kretschmann scalar contracted from it with the inverse metric, `Verifier.alt_kretschmann`) are
  reported separately and never veto; "proved" is only used when they all passed, and a leg too large to try is
  "not run". Signature is never assumed (only a diagonal metric whose entry signs the declared assumptions decide);
  coordinates are real, other symbols complex unless declared; a degenerate metric (determinant identically 0) is
  `unsupported`, an asymmetric or mis-shaped one `parse_error`; zeros of `det g` and of denominators are reported as
  restrictions about the *coordinates*, never as a statement about curvature singularities, and branch-cut domains
  (roots, logarithms) are not analysed. Geodesics are symbolic; there is no numerical integration of them. The
  geometry is not stored anywhere on the server: every request carries the coordinates, metric, assumptions and
  functions, which keeps tool calls replayable (`scripts/replay_trace.py`). Per-stage diagnostics (time, operation
  count, failure category) are returned and logged (`geometry_stage`: names and counters only, never content).
  `geometry/benchmark.py` holds the corpus with its closed-form references and the `sympy.diffgeom` comparison:
  add a case there for every new metric family or quantity. Not covered: abstract manifolds, tetrads, spin
  connections, differential forms, connections other than Levi-Civita, SageManifolds (GPL, see `docs/licensing.md`).
- Electromagnetic field tensors (`geometry/field.py`, quantities `exterior_derivative`, `antisymmetric_divergence`,
  `field_invariant`, `hodge_dual` of the existing `geometry` tool) act on a tensor the USER gives with all indices
  down, on a metric the user gives; signature, `c` and units are never assumed. A tensor that is not antisymmetric is
  `parse_error` (never silently antisymmetrised), `hodge_dual` is four-dimensional only with `eps_{0123} = +sqrt|det g|`
  in the order of the coordinates (returned as a convention) and needs the sign of `det g` decided by the declared
  assumptions (`unsupported` otherwise). Each is compared with a numerical recomputation sharing no code with the
  symbolic one, plus identity legs that are separate: `d(dA) = 0` by finite differences of the candidate F, the
  divergence as `|g|^{-1/2} d_m(|g|^{1/2} F^{mn})` against the Christoffel form, and `** = sign(det g)`. A 1e-6
  perturbed candidate is `not_verified` (test). Forms of other degrees, other connections and the Maxwell equations as
  covariant premises are not covered: say so.
- Vector calculus (`vector/`, tool and endpoint `vector_calculus` / `POST /v1/compute/vector`) works on a typed expression
  that is an ordinary SymPy tree: vector fields are noncommutative symbols and the operators are inert heads
  (`Grad`, `Div`, `Curl`, `Lap`, `Dot`, `Cross`, `Pd`, `Adv`, `Pos`, `Unit` in `expressions.py`); `kind_of` types every node
  and a product of two vectors, a scalar plus a vector or a cross product of scalars is a `parse_error`, never a guess
  (`\nabla` acts on the one factor after it). Every symbol must be a coordinate, `t`, a declared field or a declared
  constant: an undeclared symbol is refused, never read as a constant (a silent zero gradient is a plausible-looking
  wrong answer). The result is built exactly (`lower.py`, scale factors, physical components, vector Laplacian as
  the scalar Laplacian of each Cartesian component, which needs no commuting of mixed partials, unlike grad div -
  curl curl) and shown only if `numeric.py` agrees: it evaluates the *input* directly at Cartesian points by
  nested central differences on random smooth fields, finds the frame by differencing the map `x(q)`, and shares no
  scale factor with the exact route (the tests include a dropped `sin(theta)`, a componentwise cylindrical vector
  Laplacian, a flipped curl component and a 1e-6 error). The evaluated result carries no symbolic proof
  (`symbolic_zero` is None) and is worded "numerically checked". **Nothing is assumed smooth**: derivatives of an
  abstract field are jets (plain symbols per ordered word, so SymPy cannot identify `d/dx d/dy f` with
  `d/dy d/dx f`); they are merged only for a field the USER declared `c2` (words up to length 2) or `smooth`, and a
  result that contains a mixed partial in two orders is `conditions_not_satisfied` with `blocked`
  (`curl grad f`, `div curl A`, swapping `d/dt` and curl). The model never declares smoothness, `components` or time
  dependence itself. The origin and axes are `restrictions_latex` (where the coordinates break down), not statements
  about the field; a delta at the origin is not represented by `evaluate` (it answers the classical value and, for a point
  source, says so: `singular_hint`; the delta is the `distributional` action below). Explicit `components` are always text syntax. Conversion
  between systems (`convert`) is checked through Cartesian coordinates by an independent route and refuses abstract
  fields. `E`, `I` are allowed field names here (the electric field and a current). Diagnostics hold enums and
  counters only (`VectorDiagnostics`; the log allow-list has `coordinate_system`, `vector_action`, `operator_depth`).
  Scalar tools refuse an expression that mentions a declared vector field (`SCALAR_TOOLS`), and claims with `\nabla`,
  bold or arrowed letters, `\partial` or `\times` between non-numerals are `unverified` and never sampled
  (`claims.uses_vector_notation`); a name declared as a vector in words ("let a and b be vectors") is never sampled.
- Point sources and the 3D delta (`vector/singular_fields.py`, `vector/weak.py`, action `distributional`,
  `distributions/point_delta.py`, `docs/singular-fields.md`): one family only, `C (r - a)/|r - a|^3` under `div` or
  `C/|r - a|` under `lap`, with a constant strength and a constant **real** source (an undeclared constant is not
  assumed real). The delta is the typed head `PointDelta(a1, a2, a3)`, never a `DiracDelta` and never a product of 1D
  deltas; it is refused by every other operation and by `verify_equal` and the claim checker (`contains_distribution`), and
  every other vector action refuses a `DiracDelta`/`Heaviside`. The coefficient and the source are **read from the field
  by `match_kernel`**, never supplied by the model (the tool has no such argument) and never looked up. A result is shown
  only if the exact hypotheses of the shrinking-sphere argument (div = 0 away from the source, `F . n = C/eps^2`, `|F|`
  locally integrable) are symbolic zeros **and** the independent pairings of `weak.py` (four test functions with
  different values at the source, including 0 and one centred away; the flux through two spheres; tolerance `1e-7`) read
  from the *input* agree; the regularisation table is reported and never decides. `proof_status` is `derived`, never
  "proved": do not call it proved, do not loosen `PAIRING_TOLERANCE`, do not let the matcher and the pairing share code,
  and do not add a second family without its hypotheses, a wrong twin and a reference problem. A user's statement
  (`target`) is compared exactly (`holds`, `no_delta_term`, `different_strength`, `different_source`): `div(r/r^3) = 0`
  as a distributional identity, `= delta` and `lap(1/r) = +4 pi delta` are all rejected. Not supported: other singular
  fields, several sources, products, derivatives or compositions of the delta, other dimensions.
- Vector identities and controlled derivations (`vector/rules.py`, `context.py`, `steps.py`; actions `identities`,
  `apply_identity`, `substitute`, `apply`, `derive` of the same tool and endpoint). Identities are handbook entries
  (`math.vector_calculus`, check kind `vector_identity`, level `symbolic`): the checker evaluates `lhs - rhs` for
  abstract fields in Cartesian, cylindrical and spherical coordinates and requires an exactly zero component residual
  (`algebra.exact_zero`: the numerator reduced modulo `s^2 + c^2 = 1`) *and* the independent Cartesian reading to agree.
  `requires: [c2]` is a fact about the identity, not an assumption. The engine never reads the handbook: the knowledge
  layer hands over `VectorRule`s (`identities.vector_identity_rules`) and the API route and tool fill them in, a
  client cannot supply its own. The matcher only *proposes* (`rules.rewrites`); `Context.residual` decides, by
  evaluating the difference of the two *whole* expressions with both legs, so a mis-matched or wrong rule is
  `not_verified` and never shown; a rule that needs `c2` is `blocked` unless the user declared every field in the part
  that smooth, and a rule that lies about needing nothing is caught because the comparison itself needs the merge.
  Several matches are never resolved for the user (`path`); a rewrite is shown only at a path the user or tool named.
  A **premise** is a field equation the USER supplied (`Premise`): substituting it is `evidence="conditional"`,
  names it in `given`, is never called an identity or a proof, and Minerva never checks that it is true; several
  premises are never applied in an order Minerva picked (`premise` names one). `apply` (an operator on both sides)
  is an implication from the given equation, verified by the linearity residual. `canonical` pulls constants out of
  linear operators and drops operators of zero (each tidy is itself residual-checked). `derive` is a bounded search
  (depth 4, width 8, 400 states) over verified identity steps and premise substitutions: "not found" means nothing
  within the bounds. Results that use a premise go to `quoted_latex`, never `known_latex`. Benchmark cases cover
  identities, derivations, blocked, wrong-rule and ambiguous steps (`docs/vector-benchmark.md`).
- Electromagnetic derivations (`vector/em.py`, `vector/dimensions.py`; tool `electromagnetism`, action `em_derive`).
  Maxwell's equations and the other field equations are handbook entries with check kind `vector_equation`
  (level `dimensional`: units only, the operators contribute 1/length and 1/time) and `premise: true`: **premises the
  user adopts, never verified laws** (`identities.premise_options`, `premise_ids`). The model never adopts one for the
  user. A derivation is a `Template`: the premises it needs (matched *exactly*, else `conditions_not_satisfied` with
  `needs_premises`), routes (start, target) found by the general `find_chain` search over handbook identities and
  the supplied premises with every identity step residual-checked, and an algebraic combination
  `lhs - rhs = sum c_i (start_i - target_i)` checked by expansion. The final equation is then evaluated on **exact
  solutions of its premises** (plane waves in vacuum; fields built from random potentials with the sources they
  imply; harmonic potentials; Cartesian only): `ModelCheck` is "consistent with N exact solutions", never a
  verification, and a failure withholds the result. A result that used a premise is `evidence="conditional"`, lists
  it in `given`, goes to `quoted_latex` (never `known_latex`), and is never called an identity, a proof or a
  consequence of Maxwell's equations beyond the premises used; only `gauge_invariance` needs none and is an identity.
  The templates are tasks, not stored answers; do not add a template whose chain the search cannot find, and do not
  add a hardcoded result. Planned and not claimed anywhere: wave equations for the potentials, interface conditions
  (a pillbox needs the integral theorems), matter with a general constitutive relation. A failed `derive` search
  that skipped identities blocked on smoothness reports them (`blocked`, `needs_smoothness`). `ε0` and `μ0` are
  CODATA 2022 `stated` constants (measured, not exact); a test checks `mu0 epsilon0 c^2 = 1` within the uncertainties.
- Integrals (`vector/integrals.py`, action `integrate`): line, scalar line, flux, volume, and the gradient, divergence
  and Stokes theorems of **explicit** fields (an abstract field is `unsupported`; so is a field that depends on `t`)
  over a domain the USER gives, never invented: a curve and a surface are the three coordinates of the system as
  functions of parameters (the direction of increasing parameter; the normal `d_u r x d_v r`, so the order of `u` and
  `v` *is* the orientation, and swapping them changes the sign), a volume is a coordinate box whose six outward faces
  the engine generates, and Stokes' boundary is the rectangle's edges counter-clockwise in `(u, v)`. Each value is the
  agreement of two routes that share no scale factor: an exact pull-back integrated by SymPy (time-limited by an alarm)
  and a quadrature of the *input expression* read at Cartesian points by the `numeric.py` finite-difference evaluator
  (tangents, normals and the volume element are differenced from the Cartesian map; the quadrature runs at its own
  precision with a higher-precision integrand so finite differences add no noise). A disagreement is `not_verified`.
  No closed form: the value is a quadrature cross-checked by a second rule and **labelled a numerical approximation**.
  A theorem is two separately computed sides; "holds" means the sides agree. Hypotheses are `established` (by
  construction), `checked` (a finite scan of the domain: interior grid plus two probes near each end, because the ends
  are often coordinate singularities where the Cartesian reading is undefined for a smooth field) or
  `not_established`, **never proved**. A field with a pole on the domain (`1/r^2` over a ball containing the origin) is
  `unsupported` / `singular_in_domain` for the integral and the theorem alike; the flux through a surface that avoids
  the pole is computed. Faces of zero extent (`r = 0`, the poles of the sphere) integrate to exactly zero. Not covered,
  and not claimed: piecewise or discontinuous fields, surface charges, non-box volumes, differentiation under the integral.
- Electrostatics with the user's numbers (`vector/em_numeric.py`, tool `electromagnetism` actions `field`, `multipole`,
  `boundary_value`, `check_solution`; `VectorInput.action="em_compute"`, `em` = one of four typed problems). Charges,
  positions, densities, box sizes and boundary data come only from the USER; units k_e = 1/(4 pi epsilon_0); Cartesian
  only. Every result is two methods that must agree or nothing is shown: a point-charge field is the Coulomb sum, the
  negative gradient of the symbolic potential and a central difference of the potential as a number; a density field
  is the Coulomb integral for E against the difference quotient of the potential integral, and a field point in or at
  the edge of the source is `unsupported` / `field_point_in_source`; a multipole result states exact moments and
  *measures* the decay of the truncated expansion's error (never a bound); a rectangle boundary-value problem is a
  separated series against a finite-difference relaxation on two grids with a Richardson error, shown only if they
  agree (`solver_disagreement` otherwise) and always labelled a numerical approximation (Poisson only with zero edges);
  `check_solution` compares the exact residual with an independent five-point finite-difference Laplacian and reports
  which requirement fails, and is a check of that candidate, not a uniqueness proof. Do not loosen the agreement
  tolerances (the tests include a 5% perturbed relaxation). Fourier and Laplace transform methods, retarded Green's
  functions, FEM, FDTD and general PDE solving are planned and not implemented: say so, never improvise them.
- Ordinary differential equations (`ode/`, tools and endpoints `ode_analyze`, `ode_solve_symbolic`, `ode_solve_numerical`;
  overview `docs/ode.md`). Derivative notation is read only in `ode/reading.py`. A closed form is shown only if it,
  put into the original equation with derivatives taken numerically, leaves a residual at round-off scale (and the
  symbolic leg is reported separately); "no closed form found" never claims none exists, and fewer constants than the
  order is "a family". A numerical run is shown only if initial values, an independent integrator, tolerance
  refinement, the integral-form defect and any verified first integral all pass, and is always an approximation on
  the given interval. A boundary-value result is "a solution", never unique. Series and perturbation results need the
  residual to fall at the claimed order (`_check_residual`-style: do not loosen it; tests include a 0.1% wrong top
  coefficient). Equilibria and stability are local, from the linearisation; non-hyperbolic means "the linearisation
  does not decide"; nothing says globally stable. Never invent initial conditions, parameters, intervals or the small
  parameter: the model asks (a user who hands the numbers of a numerical run over, as for `integrate_motion`, lets it choose;
  the small parameter is never delegated). A system written with names such as `y1` is `format: "text"` (`y1' = y2` or
  `dy1/dt = y2` for a name listed in `unknowns`); in LaTeX a name is one letter with a subscript. Multiple scales, Frobenius series, delay, partial and differential-algebraic equations
  are not implemented and must be refused, not approximated. The reference problems (`ode/benchmark.py`, run by
  `python -m minerva_engine.reference_benchmark`) take their answers from outside the solvers; add one for every new
  method. Diagnostics and `ode`/`ode_stage` events carry counters and fixed vocabulary only.

- Approximations (`approx/`, operation `approximation`, tool `approximation`, endpoint `POST /v1/compute/approximation`,
  `docs/approximation.md`): SymPy and the formulas of the literature only **propose**. A candidate is shown only if a
  check that does not use the route that built it agrees (Cauchy estimates, residual decay at high precision, a second
  implementation such as `mpmath.pade` or `numpy.polynomial.Chebyshev.interpolate`, direct summation, quadrature of the
  original integral); otherwise it is `not_verified` and nothing is displayed. Errors are **measurements on sampled
  points**, never bounds, and "tolerance met" means on the sampled points; relative error is formed only away from zeros
  of the function. Evidence is `series_checked`, `numeric_estimate` or `bound_established` (continued fractions of a
  rational only); there is no "computed but not verified" result. The interval, point, period, tolerance, degrees and
  number of modes are the user's: they have no default (the tool schema says "never invent"), a missing one is a
  question (`domain_unspecified`), and the card form posts them through the direct route. The one convention a request
  may omit is stated in the result (`compare` expands Taylor and Padé about the interval midpoint, `piecewise` uses
  pchip). Asymptotic expansions are never called convergent; at a jump a Fourier series is reported with the Gibbs
  overshoot and no uniform convergence; a Padé approximant is not assumed better than Taylor, and a pole it has that the
  function does not is reported (`spurious_pole`, `pole_in_interval`). Interval methods refuse poles, undefined points
  and non-real values. Euler-Maclaurin must beat the plain integral and be consistent with the first neglected term
  against direct summation, and a symbolic upper limit is "checked for N = ...", never proved. Do not loosen
  `falls_at_rate` (slack 1.4), the 10^12 evaluation points of the asymptotic and Padé residual checks, `AGREE` in
  `chebyshev`/`legendre`/`fourier`, or the two-grid stability factor in `interval.py`; their tests include coefficients
  wrong by 1e-9. `strategy.py` is rules over the shape of the request, never a language model. Events and logs carry the
  method name and counters only (`approximation_log.py`).
- Variational calculus (`variational/`, tool and endpoint `variation`, `docs/variational-calculus.md`) works on a jet space
  (`jets.py`: unknown functions and their derivatives are registered jets with a total derivative `D`, never symbols
  recognised by a name suffix). The first variation keeps every integration by parts and the whole boundary form;
  a boundary term is removed only by a *fixed* condition on that very derivative at that very end (fixing `x` does
  not fix `x'`), an undeclared condition is never assumed, and the convention is `EL = sum_k (-1)^k D^k dL/dx^(k)`
  (the negative of `analyze_lagrangian` for n = 1). A result is shown only if finite differences of the *original
  action* along polynomial perturbations that do not vanish at the ends agree with it (`verify.py`); that is numerical
  evidence, never a proof (`symbolic_zero` stays false: SymPy's `euler_equations` is a reported second route). The
  parser hook (`build.JetBinding`) is off for every other operation; `\dddot` and `x^{(k)}` are derivatives only there.
  Interval, endpoint values, parameter values, definitions, trial paths and starting guesses are the user's: the
  model's `VariationArgs` has none of those fields and only `analyze|first|second`; numerical actions run from the
  card form through the direct route (`ToolContext.numeric`). `solve` shows a path only if its refinement sequence
  converged and the first variation of the original integrand vanishes along admissible perturbations (method B
  discretises the action, method A solves the derived equations by collocation; they share no discretisation);
  disagreement is reported as two paths. A stationary path is "found from this starting guess", never unique and
  never called a minimum. `classify.py` returns facts only (no member says "minimum"); a limited set of positive
  directions proves nothing, and the sufficient conditions are listed as not established. Do not weaken `fd_table`
  (its tests include a 1e-7 error, an exact quotient and a noise floor), `check_discretisation` or the certification
  of directions. The benchmark corpus is `variational/benchmark.py` (`make bench-variational`; `--check` in tests).
  The integrand may spell the derivative of an unknown `x` of the independent variable `y` as `\frac{dx}{dy}`,
  `\frac{d^2x}{dy^2}`, `x'`, `x''`, `\dot{x}`... (only here; see the LaTeX conventions). Every numerical leg takes its
  sample setting from `verify.regular_numerics`: with symbolic bounds it walks a fixed ladder (sampled, entirely positive,
  entirely negative, reseeds) until the action's quadrature is at round-off, because an interval across a singularity
  (`1/sqrt(y)` at 0) makes every finite-difference comparison inconclusive; the user's numeric bounds are never replaced.
  "Did not pass" is said only for a real disagreement; an inconclusive check is "could not be checked" (still
  `not_verified`, never shown). `VariationReport.input_latex` (the integrand as read) is on every status after a parse.
- Finite sums, products and indexed data (`indexed/`, operation `sums`): the parsers refuse `Sum`, `Product`, `x[i]`, `\sum`,
  `\prod` and `E[...]` unless the operation opts in (`Builder(allow_sums=...)`, default off), and every pointwise checker
  (`verify_equal`, `check_derivative`, `check_antiderivative`, `check_definite`, `verify_equal_nc`, the rule matcher,
  `domain_restrictions`, claims) refuses them through `indexed.guard.contains_indexed`: sampling one number per symbol gives
  every `x[i]` the same value, so `sum x_i^2 = (sum x_i)^2 / n` "agreed". Do not remove those guards or route a sum through
  a sampler. A rewrite is shown only if original and result agree on **explicit instances** (`indexed.instantiate`: the sums
  expanded by us, never `Sum.doit`/`simplify`, at several sizes that always include n >= 2, one fresh symbol per element)
  under the ordinary sampled check and symbolic zero; it is worded "checked for n = ...", never "proved for every n".
  Ranges must be provably non-empty (`range_not_provable` otherwise), indices are bound once (no shadowing, no use outside
  the sum), at most two nested, infinite sums are `unsupported`, and `x_{ij}`/`x_{i,j}` are refused as ambiguous.
- Expectations (`stats/`, operation `expectation`): `E`, `Var`, `Cov` are unevaluated functions of **declared** random
  variables (`random`); an undeclared symbol is a `parse_error`. Independence is never assumed: `independent` groups are
  the only source of it, a result that used one is `conditional` ("holds given ...") and Minerva never checks the premise,
  and a result is shown only if it agrees with the original on explicit finite joint distributions with exact weights that
  are dependent unless independence was declared (`stats.pmf_check`). Only one independence group is modelled.
- Gradients of matrix functions (`matrix/mexpr.py`, `matrix/gradient.py`; actions `gradient`, `jacobian`, `hessian` of
  `matrix_calculus`, plain-text syntax only): built by differentials (never SymPy's matrix `diff`, which returns 0 for
  `log det`), over the **reals**, and shown only if central differences of the original function at random matrices of the
  declared sizes agree. A symmetric or positive definite variable gets the gradient *within the symmetric matrices*
  (Frobenius), the candidate must itself be symmetric (a directional check cannot tell the unconstrained gradient from its
  symmetric part), and `2G - diag(G)` is a separately verified, labelled alternative; the unconstrained gradient is never
  returned for a symmetric variable. `inv`, `logdet` and the derivative of `det` of a matrix need it declared invertible
  (`invertibility_not_declared`); declarations are checked against explicit entries; products are never reordered; a 1x1
  product is a scalar factor and an ambiguous grouping is refused. `logdet` means log |det|. The symbolic leg is not run.
  Matrix-valued functions have no gradient here (`non_scalar_gradient`).
- Probability families (`stats/families.py`, operation `statistical_model`, action `moments`): a family is a parameter list
  with constraints, a support and a density, nothing else; no estimator, moment or posterior is stored. Moments come from
  `integrate`/`summation` (a size parameter is fitted from exact sums and verified at further sizes) and are shown only if
  mpmath quadrature or summation of the density agrees at sampled parameters and the density normalises. **Absolute
  convergence is established first** (`Numeric.absolute_moment_exists`: the tails of |x|^k f must shrink), because quadrature
  "agrees" with a divergent integral; such a moment is `convergence_not_established`. Parameter constraints are decided by
  `stats/domain.py`: provably violated is `parameter_invalid`, undecidable (`0 < p < 1` cannot be an assumption) is a premise
  of the result, never silently true. `Numeric` binds parameters through dummies: a parameter named `beta` shadows mpmath's.
- Statistical derivations (`stats/likelihood.py`, `mle.py`, `fisher.py`, `posterior.py`; operation `statistical_model`):
  **independence is never assumed** (`iid` must be stated, else `independence_not_declared`, and a result that used it is
  `conditional`); a prior is an assumption of the result and the result says so. Estimators are derived by solving the score
  equations of the expanded log-likelihood (triangular systems only) and shown only if, on explicit data inside the support,
  the score of the **direct** log-likelihood (sum of log densities, central differences) vanishes there, the point is inside
  the parameter space and the Hessian is negative definite (second differences). `global` needs a symbolic proof of concavity
  on the whole parameter space; otherwise `local` (proved at the point, or `measured`, saying so). A candidate estimator may
  not depend on the unknowns it estimates, and a stationary point that is not a maximum is rejected. An estimate outside the
  parameter space for the data is `candidate_inadmissible` (no MLE shown); a family whose support depends on the parameter
  is `regularity_not_satisfied`. Fisher information is -E[Hessian] with derived moments, checked against the variance of the
  score by quadrature (tolerance 1e-9: a 1e-6 error must be rejected). Posteriors are found by kernel matching, never from a
  conjugacy table, and checked against prior x likelihood normalised by quadrature. Do not loosen those tolerances or add a
  hardcoded estimator, information matrix or posterior; `docs/statistics-benchmark.md` and `stat-benchmark` measure them.
- Statistical mechanics (`statmech/`, tool `statistical_mechanics`, endpoint `POST /v1/compute/statmech`, docs
  `docs/statistical-mechanics.md`) builds a canonical partition function and derives every thermodynamic quantity by
  one generic route: differentiate ln Z with the other state variables held fixed. Do not add a per-system or
  per-quantity formula (heat capacity is dU/dT, pressure is -dF/dV at fixed T; the ideal gas result must come out
  of the Hamiltonian). A held-fixed set that is not exactly the other state variables is `ensemble_mismatch`.
  A formula is shown only if every deciding check against the independent oracle (`oracle.py`: direct enumeration
  with log-sum-exp and a tail test, adaptive quadrature with an error estimate, finite differences of the
  *enumerated* quantity) agreed; an undecided symbolic zero is "undecided", never a refutation, and "proved" is
  used only with a symbolic zero. Do not make the oracle reuse the symbolic route, loosen its tolerances or let a
  quadrature without an error estimate pass: the tests include a formula off by 1e-9. Convergence is never
  assumed: an infinite sum needs the ratio test for every beta > 0 (above 1: `divergent`; inconclusive:
  `convergence_not_provable`), an integral must be evaluated by the verified definite integrator under the declared
  assumptions, and the energy must be bounded below so that differentiating under the sum or integral is justified.
  Parametric integration bounds are mapped exactly onto numeric ones with the Jacobian (the integrator takes numeric
  bounds only); do not teach the integrator to sample bounds. A classical Hamiltonian is split term by term and a
  term that couples two variables is `not_separable`; no normal-mode change of variables is attempted. The
  phase-space cell and whether copies are indistinguishable are conventions only the user chooses: they have no
  default and are not arguments of the chat tool (they arrive through the card, `ToolContext.statmech_answers`), a
  missing one is `convention_unconfirmed` with the question, and indistinguishable copies are the classical-limit
  `N!` counting, never exact Bose or Fermi statistics. Numerical values and sweeps are the user's too. A series or
  integral without a closed form is `numeric` only, never an identity. The fluctuation-response relation needs
  `dE_i/dh = -A_i` with `A_i` independent of `h` (`coupling_not_linear` otherwise). Text syntax reads `E` as Euler's
  number: a density-of-states energy is `En`. Diagnostics and logs hold stage names, counts and the enumerated
  reasons only, never an energy or expression (`logging.ALLOWED_FIELDS`, a test enforces it); add a corpus case in
  `statmech/benchmark.py` for every new system kind, quantity or refusal. Quantum matrices (at most 4x4, `quantum.py`)
  are never reordered and never assumed to commute: Hermiticity and `[A, H] = 0` are checked and reported, the first
  derivative of ln Z with a coupled parameter is beta times the mean (trace cyclicity), but a second derivative and a
  covariance of non-commuting operators are refused (`operators_do_not_commute`: it is the Kubo-Duhamel covariance,
  not `beta^2 Var`); the oracle is `mpmath.expm`/`eigh`, never the symbolic eigenvalues. The ensemble is never
  substituted (`ensemble_mismatch`): grand canonical needs an occupation system (Bose needs mu provably below every
  level), its state variables include mu and a derivative must hold exactly the others fixed (the oracle enumerates
  occupation numbers; classical counting needs the Gibbs correction `-sum p ln(p n!)` in its entropy), and microcanonical
  counting is exact (generating function checked by two other algorithms) with a finite-difference temperature and a
  *measured* comparison with the canonical ensemble, never assumed equal. An approximation (`approximate`) is never
  an identity: it reports assumptions, parameter, claimed order, validity and a measured error table (a measurement,
  not a bound), its claimed order is a deciding check, and it is `asymptotic`; `high_temperature` is verified with
  `check_series` and refused where ln Z is not analytic at beta = 0, `low_temperature` needs a decided level order,
  `stirling` needs an indistinguishable-copies ln N!, `thermodynamic_limit` needs a chain with a symbolic length.
  The chain's transfer matrix is verified against brute-force enumeration of every configuration (at most 14 sites).
  Sample points are generic (never exactly zero): closed forms from eigenvalues and derivatives are 0/0 at special
  points. Not supported yet (reported by `analyze`, never guessed): saddle-point and Laplace methods, Legendre
  transforms, mean field, quantum systems beyond 4x4, interacting classical systems and open boundaries.
- Tensor support is exactly what `sympy.tensor.tensor` does (monoterm symmetries). Multi-term
  identities (Bianchi) are out of scope for abstract-index tensors and must be reported as such; explicit
  components in coordinates are `geometry/`, where the Bianchi identities are checked numerically.
- User input is never executed: no `eval`/`exec`/`sympify`/`parse_expr`. Parsers are
  `latexparse.py` and `textparse.py`, building SymPy trees through `build.Builder`.
- All computation runs in the sandbox (`sandbox.py`): child process, wall-clock timeout, rlimits,
  concurrency cap. Add new operations to `ops.OPERATIONS` and call them via `Engine`.
- LaTeX conventions: `\gamma` is the Greek letter, a plain symbol (a damping rate); the Gamma function is `\Gamma(x)`, and in
  text syntax `gamma` stays the function (a bare `gamma` there is refused), so a result is re-read from LaTeX, not text.
  `e` and `i` are plain symbols; `\exponentialE` / `\imaginaryI` / `\pi` are
  constants; `\sin 2x` and `\sin^{-1}` are rejected as ambiguous.
- LaTeX output has one printer, `minerva_engine.latexprint.to_latex` (never call `sp.latex` directly): constants
  print as `\mathrm{e}` / `\mathrm{i}` (the parser reads them back as the constants), so a bare `e` or `i` in
  output is always a symbol. Handbook `statement_latex` follows the same convention (a test enforces it).
  `\exponentialE` / `\imaginaryI` are input aliases only; KaTeX gets display macros for them. LaTeX that
  KaTeX cannot render is shown as marked source (`data-math-error`), never as a formula. Chat formulas are split by
  `agent/segments.py` (server) and `apps/web/src/lib/inline-math.ts` (client): keep their delimiter rules identical.
- Derivative notation is never a fraction of letters: `\frac{dx}{dy}`, `\frac{d^2x}{dt^2}` and `\frac{d}{dx}` are a
  `ParseError` in every operation (read as letters the d's cancelled and `\frac{dx}{dy}` became x/y, a verified answer
  about another expression); the exceptions are a symbol the caller declared `d` and the `variation` parser, where the
  derivative of an unknown function with respect to the independent variable is that derivative. Text syntax reads `dx`
  as one symbol and is unaffected. `\left[ \right]`, `\left\{ \right\}` and a bare `[...]` group like parentheses, except
  directly after a letter or a number (`x[2]` would silently be `x*2`); `[a, b]` and mismatched pairs are refused. The
  Lagrangian and Hamiltonian readers refuse an equation (`L = T = ...`) and say to send the part after the last `=`.
- In text syntax `E`, `I`, `pi`, `oo` are reserved constants: never use `E` as a variable name in
  handbook entries (use `En`, `Ef`, ...).

Use the `math-regression` skill when adding or changing any operation or handbook entry. A new operation or
user-visible capability also gets an example in `docs/capabilities.yaml`, then regenerate
`docs/capabilities.md`; behaviour changes show up as a diff there.

## Coding standards

- Python 3.13, type hints everywhere, `mypy --strict` on `src` dirs, Ruff (`E,F,I,B,UP,S,SIM,RUF`), pytest.
  Pydantic models at every boundary. Raise `MinervaMathError` subclasses for user-facing math errors;
  messages must be safe to show to users.
- TypeScript strict (`noUncheckedIndexedAccess`), ESLint (Next config), Prettier, Tailwind tokens from
  `packages/ui/src/theme.css` (no hard-coded colours in components). Keep components small; client
  components only where interaction needs them.
- Logging: structlog JSON with an allow-list of fields (`logging.py`). Never log expressions, emails,
  passwords, tokens, cookies or request bodies. A test enforces this.
- Small, reviewable commits. Tests with every behaviour change. No placeholder implementations.
- Test time limits (CI runners are shared and 2-3x slower than a workstation): a test that expects a result builds its
  limits with `minerva_engine.testing.budget_limits(...)` (60 s, `MINERVA_TEST_TIMEOUT_SECONDS` overrides), never with the
  default `Limits()` (10 s) or a literal of its own; only a test that expects `resource_limit` writes a short limit.
  Do not assert wall-clock durations in a correctness test (a correct answer that arrives late is not a failure) and do
  not wait with a fixed `sleep` for something to start: wait for the event itself. A test above a quarter of its limit
  under load (`uv run pytest -n 8 --dist loadfile --durations=15`) gets a smaller instance of the same property or the
  `extended` marker with a fast twin.
- Verification tests: add a false identity (and its true twin) to `packages/engine/tests/adversarial/corpus.py`, a
  problem with a hand-written oracle and source locator to `minerva_engine/reference/`, a generator to
  `packages/engine/tests/strategies/`. A known unfixed defect is a strict `xfail`/`pending` row with an id in
  `docs/verification-defects.md`; never delete or loosen a row, and never regenerate an expected answer from the code under
  test. Tiers: pull-request tests stay fast (`pr` Hypothesis profile, 8 derandomized examples); slow or random-seed
  searches are marked `extended`.

## Security and open-source rules

- Never commit secrets, real `.env` files, kubeconfigs, tunnel tokens, user data, or deployment-specific
  hostnames/infra details. `.env.example` has placeholders only. Deployment config for the homelab lives in
  the separate `homelab-kubernetes` and `cloudflare-infra` repositories.
- No copyrighted text. Handbook entries state original formulas and cite sources by locator only
  (e.g. `DLMF §4.21`); never paste passages. Provenance must be `original` or `public-domain-fact`.
- Licence: Apache-2.0, provisional. Cadabra 2 (GPL-3.0) and SageMath (GPL) are NOT dependencies; do not
  import, vendor or bundle them without updating `docs/licensing.md` and getting the maintainer's decision.
- Dependencies: prefer well-maintained, permissively licensed libraries; check the licence before adding
  (`docs/licensing.md`), pin via lockfiles (`uv.lock`, `pnpm-lock.yaml`), explain the need in the PR.
  No GPL/AGPL runtime dependencies. Run `pip-audit` / `pnpm audit` (CI does).
- Agents may push their own branches and open pull requests (see "Agent pull requests"). Do not push to
  `main`, tag or publish releases, images or charts, or change repository settings or visibility without the
  maintainer's explicit approval. Use the `oss-publication-review` skill before any release or publication.
- CI runs on self-hosted runners; never let untrusted fork code reach them (see `docs/security-publication.md`).

## Skills

Skills live in `.agents/skills/<name>/SKILL.md` and load only for relevant work. Invoke every applicable
skill before changing that area. See `.agents/skills/README.md` before writing a new one. CI
(`scripts/check_agent_files.py`) fails if a skill is missing from this table.

| Skill | Use for |
| --- | --- |
| [`math-regression`](.agents/skills/math-regression/SKILL.md) | Any change to engine operations, parsers, tensors, handbook entries or compute endpoints. |
| [`chat-trace-triage`](.agents/skills/chat-trace-triage/SKILL.md) | A pasted chat debug report, or a chat answer that looks wrong: replay the tool calls, classify, pin with a test. |
| [`verify-change`](.agents/skills/verify-change/SKILL.md) | Before saying work is done: which checks to run for which paths, in order. |
| [`ui-check`](.agents/skills/ui-check/SKILL.md) | Any change to `apps/` or `packages/ui`: browser check in light and dark at phone width. |
| [`add-dependency`](.agents/skills/add-dependency/SKILL.md) | Adding or upgrading a Python or JavaScript dependency (licence, lockfile, audit). |
| [`db-migration`](.agents/skills/db-migration/SKILL.md) | Changing API database models: Alembic revision, `alembic check`, SQLite and PostgreSQL runs. |
| [`ci-workflows`](.agents/skills/ci-workflows/SKILL.md) | Editing `.github/workflows`: fork safety, SHA pinning, least-privilege permissions. |
| [`oss-publication-review`](.agents/skills/oss-publication-review/SKILL.md) | Before any public push, release, image or chart publication. Inspects only. |

## Agent pull requests

- Agents may push a branch of their own and open a pull request for their work without asking first.
  Push only that branch: never to `main`, never force-push a branch someone else pushed to, and never
  publish, tag or release. Run the `verify-change` checks first; everything pushed is public, so the
  "Security and open-source rules" apply to every commit on the branch.
- State that the PR was authored by a coding agent at the top of the description and fill in the template
  checklist honestly (unchecked means not done). Open it as a draft unless the maintainer has said, for that
  work, that it may merge on its own; then open it ready for review.
- Never edit or remove a human-authored section of a PR description or an existing human comment.
- Enable auto-merge only when the maintainer has said so for that work. It merges only after CI passes; never
  merge by hand, bypass a required check or weaken a check to get there. Without that instruction, merging
  stays with the maintainer.
- Several agents may share one checkout. Before committing, run `git status` and stage only your own
  paths by name (never `git add -A`); for larger work use a worktree on its own branch.

## Adding dependencies

Python: `uv add --package <member> <dep>`; JS: `pnpm --filter <pkg> add <dep>`. Commit the lockfile.
Check licence, maintenance, and whether the standard library or an existing dependency already covers it.

## Definition of done

1. Behaviour implemented with real computation; unsupported cases return an explicit status.
2. Tests: unit + regression (mathematical cases with expected values, negative cases, assumption cases).
3. `ruff`, `mypy`, `pytest`, `eslint`, `tsc`, `vitest` pass; `python -m minerva_knowledge verify` passes.
4. No secrets, user data or copyrighted text added; licence impact of new dependencies checked.
5. Docs updated (this file if commands/architecture changed; `docs/` for design decisions).
6. UI changes checked in a browser in light and dark themes at phone width.
