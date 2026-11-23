# Reliability overview

Minerva has two reliability problems that need different evidence. **Engine reliability**: is a result that is
shown mathematically right? **Agent reliability**: does the loop around a small model call the right tool, read its
input, stop in bounded time and say only what a tool established? The model never computes and never decides
correctness, so a failure of the second kind can waste a turn or mislead about a result's origin, but it cannot
make an unverified formula look verified. Several guards exist to keep it that way.

Latest live measurements, what changed and what is still open: `progress-2026-11-18.md` (triage: `triage-2026-11-18.md`). The current reference point (core suite, three repeats, noise, weak categories): `baseline-core-x3-2026-11-18.md`. Every live run, with the model, its configuration and the decision taken: `experiment-log.md`.

## Guarantees (and the test that holds each)

| Guarantee | Held by |
| --- | --- |
| A shown result passed a check that shares no code with the route that built it | engine tests, `verification_report.py --check`, domain benchmarks |
| A formula in the prose is marked verified only if a verified tool result or machine-checked handbook entry produced it | `agent/segments.py`, `test_agent_reliability.py`, `test_claim_evidence.py` |
| A turn is bounded: steps, tool calls, seconds; it ends with one stop reason | `AgentLimits`, `test_agent_reliability.py`, `test_fault_injection.py` |
| A tool that outlives the turn does not keep a sandbox child or slot | `sandbox.cancel_event`, `test_sandbox_cancel.py`, A-003 |
| A model's known input mistakes give the clean result or a typed refusal, never a different verified result | `eval_corruptions.py`, `test_argument_corruption.py` |
| Logs and trace events hold no user content | `test_tracing.py`, the log allow-list |
| A failed turn explains itself: first failure, owning subsystem, whether it can be replayed | `agent/diagnosis.py`, report format 3 |

## Limits

- Sampling agreement is evidence, not proof (`docs/verification-testing.md`); the symbolic zero is reported separately.
- "Not found" and "no closed form" are statements about what was searched.
- The diagnosis is a rule over recorded events. It names a subsystem only when one rule decides; otherwise it says
  `undetermined`. It is a pointer for a human, not a verdict.
- Replayed and scripted runs show the loop and tools, not a real model. Only a live run (`scripts/eval_models.py`)
  says something about a model, and it is noisy: compare pass counts to the run-to-run spread and do not compare
  latency across days or hosts.

## How the pieces fit

```
user message -> prepare arguments (cleaning, derivative reader) -> tool (sandbox child) -> verification -> card
                      |                                              |
                 typed diagnostic                              evidence kinds
model <- tool text (for_model) ...                     claim check on the prose afterwards
        every step is a trace event (ids, counts, durations, hex digests only)
turn end -> debug.diagnosis -> debug report (format 4) -> replay_trace.py -> fixture -> regression test
```

See `evaluation-matrix.md` for what covers each tool and `triage-playbook.md` for what to do with a failure.

## Risk register

Known and not fixed (ids in `docs/verification-defects.md`):

| Risk | State |
| --- | --- |
| A turn stopped by a client that left stores no trace or result ids (A-006) | open; a later turn cannot reuse its results |
| The failed wrap-up call, `steps` counts, claim-check error site, queue time and the tool-level `verify_ms` label are imprecise (A-007) | open; `test_tracing.py` pins the current `verify_ms` |
| No engine adversarial corpora beyond `verify_equal`; no property tests for `stats/` and `approx/`; geometry, statistics and approximation lack `reference/` problems | not started |
| No per-tool breakdown (selection, argument validity, recovery) in live reports | not started |
| Few multi-turn cases (follow-ups, changed assumptions, correcting a mistake) in offline and live suites | not started |
| Benchmark reports carry little provenance (commit, dataset digest, replay or live) | not started |
| Provider behaviour on the wire (HTTP 5xx, streaming cut) is tested through `LLMError` only | open |
