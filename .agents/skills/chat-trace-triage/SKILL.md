---
name: chat-trace-triage
description: Diagnose a wrong or odd chat answer from a pasted Minerva debug report (the text copied with "Copy debug report" in chat debug mode). Use when a user pastes a report, "debug report", "trace" or minerva-debug-json block, or says a chat derivation, simplification or card looks wrong.
---

# chat-trace-triage

A debug report is what the browser built from one conversation: the user's messages, the model's raw
prose, every tool call with the arguments the model sent, what the model read back, the card's
status, input as read and result, formula marks, claim checks, timings and a `request_id`. It ends
with a JSON block (`<!-- minerva-debug-json -->`) that the replay script reads.

The report is the user's own mathematics. Save it under the scratchpad or `/tmp`, never in the
repository, and never paste it into an issue or a commit.

## 1. Replay

```bash
uv run python scripts/replay_trace.py path/to/report.md
```

It re-runs each recorded tool call through `tools.execute` (sandboxed, no model, no network) and prints
`same` or `DIFFERENT` against what the model read back. Exit 1 means a result changed. The script
works from a file: do not pipe Python through stdin to the engine, because its sandbox starts child
processes that need a real `__main__` file.

## 2. Classify: find the first step that went wrong

Read in this order and stop at the first that applies.

| Symptom in the report | Class | Where to fix |
| --- | --- | --- |
| `arguments` is not what the user asked (bare `cos`/`sin`, missing braces, a guessed integrand, wrong variable) | Model sent bad input | `agent/prompts.py` and tool descriptions in `agent/tools.py`; consider making the tool reject it (next row) |
| `input as read` differs from `arguments` (`c o s x`, `i n s x`, dropped factors, wrong grouping) | Parser misread | `latexparse.py` / `textparse.py`, under the `math-regression` skill. Ambiguity must be rejected, never guessed |
| `input as read` is right, `result` is wrong or `unchanged` when it should not be | Engine result | `ops.py`, `simplify`, rules; `math-regression` |
| status `ok` and verification passes for an input that was misread | Verification gap | Verification checks the result against the *parsed* input, so a misparse verifies perfectly; fix the parse, and say so in the report to the user |
| Formula marked `unchecked`/`from_tool` unexpectedly, claim check `unparsed` or `unverified` | Formula marking | `agent/segments.py`, `engine/claims.py` |
| Model prose contradicts the card (a different result, a "verified" claim with no tool) | Model honesty | prompt wording; `scripts/eval_models.py` case |
| Card data is right, the screen is wrong (clipped, unrendered, overflowing) | UI | `apps/web`, `ui-check` skill |
| `last turn failed` section, `error` set in debug | Model host or limits | `agent/llm.py`, `settings.py` limits; `request_id` joins the API log line |

The server log never contains expressions; `request_id`, durations and counters only. Use them to find
the right log line, not to recover content.

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
