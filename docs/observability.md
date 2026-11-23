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
| `tool.failed` / `tool.declined` | one per tool call that did not deliver a verified result (see "Tool outcomes in the log"): tool, status, error category, `outcome_kind`, a fixed `reason_code`, `arg_fields`, `invalid_fields`, and for an unexpected exception its class and where our code raised it |
| `verification.completed` | for a tool: the kind of check (`symbolic_zero`, `numeric`, `quadrature`, `series_checks`, `structural`, `dimensional`, `lookup`), sample points agreed, restrictions, whether the result was accepted and why. For the model's own formulas: how many hold, are conditional, contradicted or unchecked |
| `agent.routed` | a rule chose the first tool without asking the model (`routed`) or its result was not verified so the model plans (`declined`): decision, tool, and a fixed reason (see `docs/agent-reliability.md`) |
| `agent.retry` | reason, attempt number |
| `agent.stopped` | stop reason, steps, tool calls |
| `agent.completed` | total latency, model, tool, sandbox and claim-check time, tokens, tool calls and errors, repeated calls, formulas with engine evidence, equation numbering counts (`equation_labels_removed`, `equations_reused`, `equation_aliases_resolved`; counts only: the removed labels themselves are `debug.removed_labels`, stored with the message, shown in debug mode, never logged) |

Stop reasons: `completed`, `step_limit`, `tool_call_limit`, `time_limit`, `repeated_calls`, `output_limit`
(the last model call hit its length limit with no tool call; the reply is cut off and says so), `model_error`,
`malformed_output`, `internal_error`. Error categories: `invalid_arguments`, `unknown_tool`,
`parse_error`, `conditions_not_satisfied`, `unsupported`, `not_verified`, `timeout`, `resource_limit`,
`sandbox_killed`, `busy`, `tool_failure`, `model_host`, `malformed_model_output`, `repeated_call`,
`limit_reached`.

Each tool call in the stored trace also carries an `evidence` record: what the engine established
(numbers and flags, read from the engine's own result), and `accepted` — whether the result may be shown as
an engine result. `unchanged`, `not_verified`, `unsupported`, `parse_error` and a blocked identity are
never accepted, whatever the model says about them.

The share of formulas in the model's prose that have engine evidence is recorded per turn
(`evidence_pct`): formulas marked as coming from a verified result, or tested by the claim check. A lone
symbol (`\theta`) asserts nothing and is not counted. A model equation that is the same statement as a verified
result (equivalent right side, proportional residual, or its own name for a verified conserved quantity) is
marked from the tool and its claim record names the result, the rule and whether the link was symbolic or
sampled. Each model call also records the estimated tokens of the system prompt, tool schemas, earlier
conversation and the turn's own messages (`est_system`, `est_tools`, `est_history`, `est_turn`); a trace step
keeps the tool name the model wrote when it was an accepted alias, and whether a rule chose the call.
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
- A cloud model provider (OpenRouter, only when the user or operator chooses it, with consent; see
  [openrouter-setup.md](openrouter-setup.md)) receives the conversation, because it *is* the model host. That is
  not logging: trace events, timings, error categories and the debug record are never sent to any provider, and
  the benchmark's report never contains a key or host.
- Detailed content (the arguments the model sent and what the tool returned) lives **only in the stored
  assistant message of the conversation it belongs to**. It follows the conversation: private to the
  account (another account gets 404), deleted when the conversation is deleted, and covered by any
  retention or export feature added to conversations later, because it has no table, file or endpoint of
  its own. Do not add one; keep it that way.
- Raw model responses are not recorded. The model's reasoning (Ollama's `message.thinking`, returned only when
  thinking is requested: `MINERVA_LLM_THINK`, the debug panel's per-turn override, `eval_models.py --think`) is
  conversation content like the tool arguments: it is kept, cut at 20,000 characters per model call, in the stored
  assistant message only (`debug.llm_calls[i].thinking`, with `thinking_chars` and `thinking_truncated`), under that
  conversation's access, deletion and export rules and in the local evaluation trace files. It is never reply text,
  never replayed to the model, never checked or numbered. The log and the trace events carry its length only
  (`thinking_chars`). The UI shows it only in debug mode, collapsed per call, and the copied report includes it.
- The debug panel and the copied report say that arguments and results are conversation content. The
  report is copied by the user in their own browser; share it only where you would share the
  conversation.
- Logging and tracing failures never reach the mathematics: `Tracer.emit` swallows its own errors and
  counts them (`events_dropped`), and the engine ignores an observer that raises.

Settings (see `.env.example`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `MINERVA_AGENT_EVENTS` | `summary` | `off`: only the request lines. `summary`: one `agent.completed` event per request, plus a `tool.failed` or `tool.declined` line for every tool call that did not succeed. `all`: every event |
| `MINERVA_AGENT_MAX_STEPS` | 6 | model calls per answer |
| `MINERVA_AGENT_MAX_SECONDS` | 180 | wall clock per answer |
| `MINERVA_AGENT_MAX_TOOL_CALLS` | 12 | tool calls per answer |

The stored trace and its events are kept at every setting; the setting only controls the server log.

## Tool outcomes in the log

At the default setting the log answers three questions about a turn without opening the conversation:
which tool failed, what kind of call it was, and why. A tool call that did not deliver a verified result
writes one line, with the turn's `trace_id`:

| Event | `outcome_kind` | Which calls |
| --- | --- | --- |
| `tool.failed` | `failure` | Something to fix in Minerva or in how the model called it: `invalid_arguments`, `unknown_tool`, `parse_error`, `not_verified`, `timeout`, `resource_limit`, `sandbox_killed`, `busy`, `tool_failure`, `malformed_model_output` |
| `tool.declined` | `answer` | What the engine or handbook established, or a call that was not run: no handbook entry, `unchanged`, `unsupported`, `conditions_not_satisfied`, `needs_user_choice`, `limit_reached`, a repeated call |

The mapping is `tracing.OUTCOME_OF_CATEGORY`; a test fails when an error category is added without a
decision. A verified result writes no line. Fields:

- `tool`, `call_id`, `status`, `error_category`, `outcome_kind`, `arguments_valid`, `repeated`, `references`,
  `duration_ms`, `engine_ms`.
- `arg_fields`: the argument names the model filled that belong to the tool's own schema, sorted. A name the
  schema does not have is dropped, so the model cannot choose what is logged.
- `reason_code`: a fixed code: the validation error type (`missing`, `string_type`, `literal_error`, ...) of an
  `invalid_arguments` failure, or, for a mathematical input that could not be read, one of the engine's
  `InputCode` values (`unrecognized_derivative`, `ambiguous_derivative`, `mixed_derivative_spellings`,
  `undeclared_dependent`, `declared_name_absent`, `acceleration_in_input`, `text_syntax_in_latex`, `latex_in_text`,
  `equation_not_expression`, ...; `minerva_engine.notation.models`). It is absent when there is no code, never a
  copy of the category. The symbols the code is about and the spellings that would work are the user's names and
  are in the stored message only (the model sees them in a `diagnostic: {...}` line), never in a log line.
- `invalid_fields`: the schema paths that failed (`quantities.*.name`), rebuilt level by level from the
  argument model. Only the model's own field names are written; a list index or a dictionary key, which can be
  a symbol the user chose, becomes `*`, and a path that cannot be placed is cut with `*`.
- `error_type`, `error_module`, `error_line`: for an exception nobody expected (a tool that crashed, the sandbox
  child, the agent loop, the chat stream, an endpoint), the class and the module and line of *our* code that
  raised it. The class is reported only if it belongs to Minerva or a library we depend on, else `other`. The
  message is never written. The sandbox child sends the same triple (`SandboxCall`), so an engine error points
  into `minerva_engine`, not at the sandbox's plumbing.
- `agent.completed` adds `tools_called`, `failed_tools`, `declined_tools` (registry names in the order first
  called; a name the model invented is `unknown`) and `first_failure_code`.

Every value is a member of a closed set (the tool registry, the argument models' field names, enums, pydantic's
error types), and the event models drop anything else (`ToolOutcome` and `ErrorSite` validators), so an e-mail
address or an expression cannot get into these fields even by mistake.

There is no separate cap on these lines: a turn is limited to `MINERVA_AGENT_MAX_TOOL_CALLS` calls (12) and a
repeated call counts towards it, so one turn writes at most that many.

**Which codes are most common.** `scripts/diagnostic_codes.py api.log` (or `kubectl logs ... | ... -`) counts the
`tool.failed` and `tool.declined` lines by tool, `reason_code` and `error_category`, most frequent first. It reads
only those closed fields. The most frequent code in real use says which normalisation (`notation/transport.py`,
`notation/derivatives.py`), shape repair (`agent/arguments.py`) or schema change to add next.

**What the loop did itself.** A stored trace step carries `policy` when the recovery policy (`agent/recovery.py`)
acted: `format_switch` (the call failed in one syntax and was re-run once in the other; the result says so),
`refused_equivalent` (an equivalent call had already run or failed) or `budget_exhausted` (the tool failed
`max_failures_per_tool` times in a row). `model_arguments` keeps what the model sent when shape repair, cleaning or
`@rN` references changed it. The debug report prints both.

**From a log line to a fix.** Take the `trace_id`; open the conversation's debug panel (or the stored message's
`debug.events`) for the arguments and results; copy the debug report and replay it with
`scripts/replay_trace.py` (the `chat-trace-triage` skill). What the log does not hold, on purpose: the
arguments, the results and *why the model chose that tool*. The last is answered by the report and the
benchmark cases, not by a log.

### One logging path

Everything written to standard output is one JSON line that has passed the allow-list, including what the
libraries log. Records from uvicorn, SQLAlchemy, httpx, asyncio and the other libraries we run under are
reduced to `event = foreign:<logger>` (or `foreign_log` for a logger not on the list), the level and, for an
exception, the class and location as above: their messages, arguments and tracebacks are dropped, because
a message can hold an expression, an e-mail address or a SQL parameter. The access log is switched off
(`--no-access-log`; our `request` line replaces it without the client address or query string). An exception
that escapes an endpoint gets a fixed 500 body (`detail`, `request_id`), an `unhandled_error` line with the
class and location, and its `request` line with status 500.

One limit: the chat reply is a stream whose 200 status is sent before the turn runs. A failure inside it still
has a `request` line with 200; `chat_turn_failed` (class and location) and the stored message are the signal,
and the debug report prints `raised: <class> at <module>:<line>` for it.

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
  was not established, and the stop reason is `repeated_calls`. The same last call is made when the turn
  ends at `step_limit` or `tool_call_limit` (a turn that spent every step on tool calls used to end with
  the stop sentence alone, issue 137). Its formulas go through the usual claim check; any tool call it
  returns is dropped. When that call cannot be made (no time left) or gives nothing usable (empty, a
  tool call written as text, a host error), the server writes the closing itself (`loop.closing_summary`):
  fixed words from counts and statuses of the turn's own trace (how many calculations gave a verified
  result, how many did not and why), with no formula, argument or model text.
- **Retries.** A model call that fails before any output is retried once (`agent.retry`); nothing
  mathematical is ever retried into a different answer. A tool call written as text, or with arguments that
  are not a JSON object, gets one correction; a second one ends the turn (`malformed_output`).
- **Result references.** A verified result has an id (`r1`). A later call may write `@r1` in an expression
  argument, and the server substitutes the engine's own LaTeX, so the model does not retype it. The trace
  shows both what the model wrote and what was substituted.
- **Assumptions.** The model declares them in each call. A call that uses a symbol declared earlier in the
  conversation without declaring it again is flagged (`assumptions_dropped`); Minerva never adds an
  assumption itself.

## Report format 3 and the diagnosis

Format 3 only adds fields (format 2 reports are still read). Each turn's `debug.diagnosis`
(`agent/diagnosis.py`) names the subsystem that owns the first failure (`model`, `parser`, `orchestrator`,
`engine`, `verifier`, `infrastructure`), the step and its category, and whether the turn can be replayed.
Two subsystems among the failures, or a record with dropped events, gives `undetermined` with the candidates
listed: the stage is never guessed. Each model call stores `request_sha` (a fingerprint of the system
prompt and tool schemas it was sent, also on `llm.request`) and the tool groups it had loaded; `prompt_sha`
is the catalogue fingerprint. The timeline shows `tool.failed` and `tool.declined` events with their reason
code and category.

## Report format 4 and follow-up turns

Format 4 adds what a later turn is rebuilt from: each assistant message's `blocks` (the cards as stored: their
data, `number` and `numbers`), each user message's `equations` and `references`, and `conversation`
(`equation_counter`: the server's high-water mark, never below the numbers shown; `messages_total`,
`first_message_index` and `messages_included`, because "Copy this turn" holds only one exchange). Formats 2 and 3
are still read. The report holds conversation content, as before.

## Replaying a report

```bash
uv run python scripts/replay_trace.py report.md          # re-run every recorded tool call
uv run python scripts/replay_trace.py report.md --llm    # replay a whole turn, no model
```

The first compares what the model read back, the tool's status and the evidence record. The second
rebuilds the model's replies of one turn from the recorded events and runs the real loop on them: tools,
references, formula marks, the claim check and stop reasons are real; only the model is replayed. It
reproduces what the agent did with those replies, not the model. Reports of format 1 (before tracing) can
be replayed call by call. `--llm` also compares the arguments that were executed and the evidence verdict
(check, accepted, reason) of each step, not only tool names and statuses.

A follow-up is given the context the live chat gave it, by the same function (`agent/turn_context.py`): the history,
the verified constants of earlier discovery cards (`time_derivative` inserts them), `@rN` results and the
restrictions they carry, the equation table and what "equation 3" meant, the equations a result was computed from
that Minerva never verified, the claim checker's relations, assumptions and verified evidence, the declarations in
words and the tool groups. Numbering is replayed from the report's high-water mark: the smallest mark under which
the numbers and references recorded on the user's message come out again, so a number that vanished (a retry or an
edit) is not handed out again and a reference to it is not recreated; when the report cannot say which, it says so.
Only a `ReplayLLM` is used: no model host is contacted, ever.

What a report cannot give back is listed as `missing` (`earlier messages`, `constants`, `result handles`,
`restrictions`, `unverified inputs`, `equation table`): formats 2 and 3 hold the trace but not the cards, and a
per-turn copy holds only one exchange. The verdict is then `NOT FAITHFUL`, or `NOT COMPARABLE` when a call that
depends on the missing part (a reference to an earlier result or equation, a total derivative, formula marks and
the reply text) came out differently: those are listed as "differences that missing context may explain", apart
from a difference in what the engine did, which stays `DIFFERENT`. Exit status: 0 the replay was faithful and
unchanged, 1 a difference the missing context cannot explain, 2 the report could not be read, 3 not faithful (not
a pass). A replay with a first assistant turn, or earlier turns that had no tool results and no numbered
formulas, misses nothing and says nothing about earlier turns.

```bash
uv run python scripts/replay_trace.py report.md --to-fixture /tmp/failure.json   # outside the repository
```

writes the executed calls of the last turn and what the engine said (no prose) as a regression fixture. It
still holds the user's mathematics: replace the arguments with a minimal example of your own, set an `id`
(the file name) and `contains_user_mathematics: false`, and commit it under
`services/api/tests/fixtures/failures/`. `test_failure_fixtures.py` replays every committed fixture.

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

Models can name their provider: a bare name or `ollama:name` is an Ollama model; `openrouter:vendor/model`
runs through OpenRouter, reading `OPENROUTER_API_KEY` from the environment only (never an argument; it costs
real money and needs no `--base-url`). Reports also record first-output time, tokens and cost when the provider
reports them, tool failures, results that were not verified, and rate-limit and timeout counts. The first
measurements, taken before and after the reliability changes, are in
[model-evaluation.md](model-evaluation.md).

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

## Context use and model memory in the chat

The chat always shows a compact meter above the composer (no debug mode needed): the window of the model
that answered last, what its last request used, a warning near the window (80%) or when earlier messages
were left out (`context_trimmed`), and, folded, prompt and completion totals for the last reply and the
conversation. It reads only counts already stored with each reply (`debug.llm_calls[].prompt_tokens` /
`completion_tokens`, `debug.context_window`, `debug.prompt_estimate_tokens`, `context_trimmed`); there is no
new field and nothing is logged. A count the provider did not report is left out of the sums; when no call
reported a prompt count the server's pessimistic estimate is shown and labelled "(estimate)". A hosted
(OpenRouter) reply has no window figure.

`GET /v1/chat/status` adds optional numbers to each local model in `providers.ollama.models`:
`size_bytes` (the file size from Ollama's `/api/tags`, replaced by the size in memory from `/api/ps` when
the model is loaded), `size_vram_bytes` (the part in GPU memory, loaded models only) and `loaded`. Each is
present only if the host reported a positive integer, so a host that reports nothing shows nothing. Only
these numbers leave the server (not the address, digests or other fields); they are cached for 30 s with
the installed-model probe (`providers.ollama_memory`) and never logged. The model settings panel shows
them, and the server's effective context window and temperature (`status.params.defaults`, read-only here;
changing them stays in the debug panel under the existing tuning flags).
