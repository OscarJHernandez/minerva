# Observability: traces, debug reports, benchmarks

Minerva's mathematical behaviour should be measurable, explainable and reproducible. When the agent
fails, the evidence should say whether the cause was tool selection, parsing, missing assumptions,
verification, the model, or a resource limit. This page describes what is recorded, where it lives, what
is never recorded, and how to use it. Nothing here needs infrastructure beyond the API process and its
database: there is no log shipper, metrics server or tracing backend.

## One trace id per request

Every agent request (a chat message, or a card form that runs a tool directly) gets a `trace_id`: 32 hex
characters, new for each request. It is a different thing from `request_id`, which identifies the HTTP
request. The trace id is:

- returned in the `X-Trace-ID` response header and in the data of every server-sent event;
- stored with the assistant message (`debug.trace_id`);
- on every trace event and on the `chat_turn` log line;
- bound to the engine for the duration of the turn, so each sandbox call the engine makes is attributed to
  the request that caused it.

To follow one request: take the trace id from the debug panel or report, then search the server log for it.

## What a turn records

The agent loop emits typed events in order (`agent/tracing.py`). Each has `trace_id`, `seq`, `t_ms` (milliseconds since the
turn started, from a monotonic clock) and `step` (the model call it belongs to).

| Event | Important fields |
| --- | --- |
| `agent.started` | model, prompt fingerprint, limits (steps, seconds, tool calls), number of history messages |
| `llm.request` | message count, number of tools offered |
| `llm.response` | latency, time to first output, finish reason, tokens in and out (when the host reports them), number of tool calls, the first proposed tool (a registry name or `unknown`) |
| `tool.started` | tool, whether the arguments validated, whether it is a repeat, how many `@rN` references were replaced |
| `tool.completed` | status, duration, time in the sandbox and number of sandbox calls, error category, result id |
| `verification.completed` | for a tool: the kind of check (`symbolic_zero`, `numeric`, `quadrature`, `series_checks`, `structural`, `dimensional`, `lookup`), sample points agreed, restrictions, whether the result was accepted and why. For the model's own formulas: how many hold, are conditional, contradicted or unchecked |
| `agent.retry` | reason, attempt number |
| `agent.stopped` | stop reason, steps, tool calls |
| `agent.completed` | total latency, model, tool, sandbox and claim-check time, tokens, tool calls and errors, repeated calls, formulas with engine evidence |

Stop reasons: `completed`, `step_limit`, `tool_call_limit`, `time_limit`, `repeated_calls`, `model_error`,
`malformed_output`, `internal_error`. Error categories: `invalid_arguments`, `unknown_tool`,
`parse_error`, `conditions_not_satisfied`, `unsupported`, `not_verified`, `timeout`, `resource_limit`,
`sandbox_killed`, `busy`, `tool_failure`, `model_host`, `malformed_model_output`, `repeated_call`,
`limit_reached`.

Each tool call in the stored trace also carries an `evidence` record: what the engine established
(numbers and flags, read from the engine's own result), and `accepted` — whether the result may be shown as
an engine result. `unchanged`, `not_verified`, `unsupported`, `parse_error` and a blocked identity are
never accepted, whatever the model says about them.

The share of formulas in the model's prose that have engine evidence is recorded per turn
(`evidence_pct`): formulas marked as coming from a verified result, or tested by the claim check.
Formulas that are only *quoted* (the user's own input, a handbook entry that is only stated, an identity
that was not applied) are shown, but not counted as verified.

## Privacy: what is and is not recorded

The rule is simple: **the server log and the events hold no content.**

- An event has no field for an expression, a tool argument, a tool result, model text or an exception
  message. Its fields are identifiers, counts, durations, booleans and members of fixed vocabularies. A
  test fails if a free-text field is added to an event, if the log allow-list lacks an event field, or if
  it contains anything that could hold content. A tool name the model invents is logged as `unknown`.
- Nothing is sent to a third-party logging or telemetry service: events go to standard output through
  structlog only, and the API has no telemetry dependency (a test checks this). If you ship stdout to a
  log service, what it receives has no mathematics in it.
- Detailed content (the arguments the model sent and what the tool returned) lives **only in the stored
  assistant message of the conversation it belongs to**. It follows the conversation: private to the
  account (another account gets 404), deleted when the conversation is deleted, and covered by any
  retention or export feature added to conversations later, because it has no table, file or endpoint of
  its own. Do not add one; keep it that way.
- Raw model responses and the model's reasoning are not recorded. `thinking` output is dropped by the
  client.
- The debug panel and the copied report say that arguments and results are conversation content. The
  report is copied by the user in their own browser; share it only where you would share the
  conversation.
- Logging and tracing failures never reach the mathematics: `Tracer.emit` swallows its own errors and
  counts them (`events_dropped`), and the engine ignores an observer that raises.

Settings (see `.env.example`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `MINERVA_AGENT_EVENTS` | `summary` | `off`: only the request lines. `summary`: one `agent.completed` event per request. `all`: every event |
| `MINERVA_AGENT_MAX_STEPS` | 6 | model calls per answer |
| `MINERVA_AGENT_MAX_SECONDS` | 180 | wall clock per answer |
| `MINERVA_AGENT_MAX_TOOL_CALLS` | 12 | tool calls per answer |

The stored trace and its events are kept at every setting; the setting only controls the server log.

## Reading a trace

Switch on "Debug mode" in the chat footer (or open the chat with `?debug=1`). Under each answer, the
panel shows a timeline: what the model asked for, whether the arguments were valid, what the tool
returned, what was verified and why the result was accepted or rejected, and why the turn went on,
retried or stopped. Below it are the raw arguments and results (conversation content). "Copy debug
report" copies a Markdown report with the trace id, stage timings, status codes, the same timeline, and
a JSON block for scripts. A turn that failed is stored too, so its trace survives a reload.

### What the loop does

- **Repeated calls.** A call identical to one that already ran in the turn is not run again; the model is
  told, and the step is recorded as `repeated`. After two repeats, or three calls in a row that establish
  nothing, the turn is wrapped up: one last model call with no tools asks for a summary of what was and
  was not established, and the stop reason is `repeated_calls`.
- **Retries.** A model call that fails before any output is retried once (`agent.retry`); nothing
  mathematical is ever retried into a different answer. A tool call written as text, or with arguments that
  are not a JSON object, gets one correction; a second one ends the turn (`malformed_output`).
- **Result references.** A verified result has an id (`r1`). A later call may write `@r1` in an expression
  argument, and the server substitutes the engine's own LaTeX, so the model does not retype it. The trace
  shows both what the model wrote and what was substituted.
- **Assumptions.** The model declares them in each call. A call that uses a symbol declared earlier in the
  conversation without declaring it again is flagged (`assumptions_dropped`); Minerva never adds an
  assumption itself.

## Replaying a report

```bash
uv run python scripts/replay_trace.py report.md          # re-run every recorded tool call
uv run python scripts/replay_trace.py report.md --llm    # replay a whole turn, no model
```

The first compares what the model read back, the tool's status and the evidence record. The second
rebuilds the model's replies of one turn from the recorded events and runs the real loop on them: tools,
references, formula marks, the claim check and stop reasons are real; only the model is replayed. It
reproduces what the agent did with those replies, not the model. Reports of format 1 (before tracing) can
be replayed call by call.

## The benchmark

`scripts/eval_models.py` runs a fixed set of cases through the real loop against an Ollama host. The cases
are data (`services/api/src/minerva_api/agent/evaluation/cases.yaml`) in ten categories (simple symbolic,
multi-step, calculus, retrieval, tensor and matrix, physics, ambiguous, unsupported, false or hallucinated
claims, follow-ups). Expected outcomes come from hand-written expectations in the engine tests and handbook
benchmark, not from any agent's output; each case names its source, and the test suite runs every case's
reference call against the real engine and requires it to meet the expectation.

```bash
uv run python scripts/eval_models.py --check-cases                       # ground truth against the engine
uv run python scripts/eval_models.py --list --categories followup
uv run python scripts/eval_models.py --base-url http://HOST:11434 --models a,b \
    --repeat 3 --seed 7 --label baseline --json-out baseline.json --out baseline.md
uv run python scripts/eval_models.py --compare-only baseline.json new.json
```

The host comes from the command line or `MINERVA_EVAL_BASE_URL`, and is never written to a report. Each repeat
uses `seed + repeat` (when the host honours seeds). A report records the code revision, prompt fingerprint,
case-set digest, sampling settings, limits and versions; reports with different case-set digests are not
directly comparable. `make eval-check` runs the fast deterministic checks; `make eval` runs a real
evaluation (needs `MINERVA_EVAL_BASE_URL` and `MINERVA_EVAL_MODELS`). The fast checks run in CI as part of
the test suite; real-model runs are expensive and are run separately.

Metrics, per model: tool-selection accuracy (the first tool called is the expected one), valid tool
arguments, completion (a reply arrived and no limit ended the turn), unsupported formulas shown as
verified (must be 0), unnecessary and repeated tool calls, latency distributions (turn, model call, tool
call), and every failure classified into exactly one class. The targets are goals, not results; a
report shows both.

| Failure class | Meaning |
| --- | --- |
| `tool_selection` | wrong tool, no tool where one was needed, a forbidden tool, or a sequence not followed |
| `arguments` | invalid arguments, or arguments that do not say what the user asked |
| `parsing` | the engine could not read what the model passed |
| `missing_assumptions` | assumptions dropped, or invented |
| `verification` | a false or unsupported statement shown as verified |
| `model_honesty` | the reply misstates what the tools said, or is not what was asked |
| `claim_checker_false_positive` | a correct statement was marked contradicted: a checker problem, not a model one |
| `resource_limit` | step, tool-call or time limit, or a tool hit a resource limit |
| `unproductive_loop` | the agent kept calling tools that could not help and was stopped |
| `model_host` | the model host failed |
| `no_reply` | the turn ended without an answer for another reason |

## Troubleshooting

| Symptom | Where to look | Likely cause |
| --- | --- | --- |
| Wrong tool, or no tool | timeline: "Model call N: answered in words" with no tool, or a different first tool | the model; try another model, or check the tool descriptions. Benchmark: `tool_selection` |
| Tool refused the call | timeline "arguments INVALID", category `invalid_arguments` | the model's JSON; the message the model read back names the first problem |
| The formula was read wrongly | `parse_error`, and the "input as read" in the card | the model's LaTeX (`cosx` without a backslash); replay shows it |
| Result is `unchanged` or `conditions_not_satisfied` | the evidence record and "reason" | not a failure: the engine found nothing verified, or an assumption is needed |
| A formula is marked `quoted` | the note on the formula | it is the user's input, a stated handbook entry, or an identity that was not applied |
| A formula is marked `unchecked` | claim check row in the timeline | the model wrote it; no tool produced it and it could not be tested |
| Turn ends early | "Ended early" line and the stop reason | see the stop reasons above; `repeated_calls` usually means the model could not make progress |
| Slow | stage timings: model, tools, sandbox, claim check | model latency dominates on a small host; tool time is the engine; sandbox time includes waiting for a free slot (`MINERVA_COMPUTE_MAX_CONCURRENCY`) |
| `resource_limit` | tool.completed category: `timeout`, `sandbox_killed` or `busy` | the computation is too heavy (raise `MINERVA_COMPUTE_TIMEOUT_SECONDS` or simplify), or the server is busy |
| Host failed | `agent.retry` then `model_error` | the model host; see [self-hosting.md](self-hosting.md) |

## Performance analysis

- Per turn: the stage timings in the report (model, tools, sandbox, claim check) add up to the total
  except for time spent waiting between them; `llm.response` events show time to first output and tokens.
- Across turns: with `MINERVA_AGENT_EVENTS=summary`, the `agent.completed` log lines carry the same
  numbers; `jq` over the log gives distributions without any content.
- Across models: the benchmark's JSON has per-run `seconds`, `llm_ms` and `tool_ms`; the report gives
  median, p95 and max. Token counts are shown only when the host reports them; nothing is estimated.
- Sandbox time includes the wait for a free slot, and parsing and verification are inside the sandbox
  operation: they are not separated.

## Limits

- Timings come from the API process; the model host's own metrics are not visible beyond what it reports.
- `--llm` replay reproduces the loop, not the model, and restores earlier turns only as plain text.
- A benchmark of a sampling model is a measurement on those runs; use repeats before reading a difference
  of one case into a change.
