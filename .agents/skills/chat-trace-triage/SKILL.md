---
name: chat-trace-triage
description: Diagnose a wrong or odd chat answer from a pasted Minerva debug report (the text copied with "Copy debug report" in chat debug mode). Use when a user pastes a report, "debug report", "trace" or minerva-debug-json block, or says a chat derivation, simplification or card looks wrong.
---

# chat-trace-triage

A debug report is what the browser built from one conversation: the user's messages, the model's raw
prose, every tool call with the arguments the model sent, what the model read back, the card's
status, input as read and result, formula marks, claim checks, a timeline of the turn, stage timings, evidence
records, error categories, a stop reason and a `trace_id`. It ends
with a JSON block (`<!-- minerva-debug-json -->`) that the replay script reads.

The report is the user's own mathematics. Save it under the scratchpad or `/tmp`, never in the
repository, and never paste it into an issue or a commit.

## 1. Replay

```bash
uv run python scripts/replay_trace.py path/to/report.md          # every recorded tool call
uv run python scripts/replay_trace.py path/to/report.md --llm    # one whole turn, replayed without a model
```

It re-runs each recorded tool call through `tools.execute` (sandboxed, no model, no network) and prints
`same` or `DIFFERENT` against what the model read back, the tool's status and the evidence record. Exit 1
means a result changed. `--llm` rebuilds the model's replies of one turn from the recorded events and runs the
real loop on them (tools, `@rN` references, formula marks, claim check, stop reason): use it when the question
is what the agent did with the model's output, not what the model said. Calls that were never run (repeats, calls
over a limit) are skipped. The script
works from a file: do not pipe Python through stdin to the engine, because its sandbox starts child
processes that need a real `__main__` file.

## 2. Read the timeline, then classify

The report's `Timeline:` lists what the agent chose, whether the arguments were valid, what was verified and why
a result was accepted or rejected, and why the turn went on, retried or stopped. Start there, with the `stop
reason` and each tool call's `evidence` and `category`. The categories and the benchmark's failure classes use the
same words (see `docs/observability.md`).

Read in this order and stop at the first that applies.

| Symptom in the report | Class | Where to fix |
| --- | --- | --- |
| `arguments` is not what the user asked (bare `cos`/`sin`, missing braces, a guessed integrand, wrong variable) | Model sent bad input | `agent/prompts.py` and tool descriptions in `agent/tools.py`; consider making the tool reject it (next row) |
| `input as read` differs from `arguments` (`c o s x`, `i n s x`, dropped factors, wrong grouping) | Parser misread | `latexparse.py` / `textparse.py`, under the `math-regression` skill. Ambiguity must be rejected, never guessed |
| `input as read` is right, `result` is wrong or `unchanged` when it should not be | Engine result | `ops.py`, `simplify`, rules; `math-regression` |
| status `ok` and verification passes for an input that was misread | Verification gap | Verification checks the result against the *parsed* input, so a misparse verifies perfectly; fix the parse, and say so in the report to the user |
| Formula marked `unchecked`/`from_tool` unexpectedly, claim check `unparsed` or `unverified` | Formula marking | `agent/segments.py`, `engine/claims.py` |
| A valid definition, coordinate change or equation of motion marked `contradicted` | Claim checker | `claim check ... [identity]` means it was tested as an identity; `[relation]` should never be `fails` without `given` definitions. `engine/claims.py`, with a regression test in `test_claims.py` |
| Model prose contradicts the card (a different result, a "verified" claim with no tool) | Model honesty | prompt wording; `scripts/eval_models.py` case |
| Card data is right, the screen is wrong (clipped, unrendered, overflowing) | UI | `apps/web`, `ui-check` skill |
| Stop reason `repeated_calls`, `repeated` steps, many calls with status `unchanged`/`unsupported` | Model loop | the model could not make progress or ignored a result: tool descriptions and prompt rules 15-16; consider whether the tool should say more |
| Category `malformed_model_output`, a tool call written as prose, `arguments_not_json` | Model output format | the model or its template does not make real tool calls; try another model |
| `assumptions_dropped` on a call, or an assumption the user never stated in `arguments` | Missing or invented assumptions | prompt wording; the model must carry the user's assumptions and never add its own |
| Formula marked `quoted` | Expected | the user's input, a stated handbook entry or an identity that was not applied; not a verification failure |
| Formulas with engine evidence low, claim check `timeout` or `error` | Verification budget | `agent/loop.py` claim-check budget; the reason is in `claim_check` |
| `stop reason: model_error` or `last turn failed`, `error` set in debug | Model host or limits | `agent/llm.py`, `settings.py` limits; the `trace_id` and `request_id` join the API log lines (events never hold content) |

The server log never contains expressions; `trace_id`, `request_id`, durations and counters only. Use them to find
the right log lines, not to recover content. To measure rather than guess how often a class happens, run the
benchmark (`docs/observability.md`) and read its failure-class table.

## 3. Reduce and pin

1. Reduce the failing input to the smallest expression that still shows the class (replay it by hand
   with `execute(...)`, as in `scripts/tests/test_replay_trace.py`).
2. Write the regression test first (engine, API or web, matching the class) and watch it fail.
3. Fix, then re-run the replay on the original report: the affected call should now be `DIFFERENT`
   and correct, or the tool should refuse the input with an explicit status.
4. Follow `verify-change` before saying it is done.

## What not to do

- Do not paper over a parser misread in the prompt alone; the model will send the same input again.
- Do not add a fallback that guesses what the model meant. Report `parse_error` and let the model retry.
- Do not weaken the "unchecked" marking to make a report look cleaner.
