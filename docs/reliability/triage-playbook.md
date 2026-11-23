# Triage playbook

From "this answer looks wrong" to a regression test, without another model and without user content in the repo.

1. **Export.** Open the chat in debug mode (footer checkbox or `?debug=1`) and use "Copy debug report" (format 4;
   formats 1 to 3 are still read, but a follow-up in them cannot be replayed faithfully). The report starts with a `diagnosis:` line: the stage, the step, the first
   failure code, and whether the turn can be replayed.
2. **Read the diagnosis.** Stages: `model`, `parser`, `orchestrator`, `engine`, `verifier`, `infrastructure`,
   `undetermined`. `undetermined` is an honest answer (several subsystems, or events were dropped): read the
   timeline and the step table instead of forcing a label.
3. **Replay the tools, no model.** `uv run python scripts/replay_trace.py report.md` re-runs the recorded tool
   calls and compares status and evidence. A difference means the code changed behaviour since the report.
4. **Replay the whole turn.** `--llm` plays the recorded model replies through the real loop and compares the
   executed arguments and the evidence verdicts per step, so a change in routing, cleaning or recovery shows.
   A follow-up turn gets its stored context back (constants, `@rN` results and restrictions, evidence, the equation
   table) when the report is format 4; `missing` says what an older report cannot restore, and exit status 3 means
   "no difference found, but not an equivalent replay" (see `observability.md`).
5. **Classify** with the closed vocabulary (`FailureClass`, `InputCode`, stop reasons) and `scripts/diagnostic_codes.py`
   over the API log for how often a code occurs. Decide: model mistake the input layer should read (add the inverse to
   `notation/transport.py` `DAMAGE` or a corruption operator), orchestration defect (fix and pin), engine
   defect (`math-regression` skill), or honest "unsupported".
6. **Make a fixture.** `--to-fixture PATH` writes the executed calls and expected status and evidence, no prose,
   `contains_user_mathematics: true`. Minimize it to a synthetic example that still fails, remove anything the
   user wrote, review it, and commit it under `services/api/tests/fixtures/failures/`. Nothing is written into the
   repository for you. `test_failure_fixtures.py` replays every fixture.
7. **Fix with the test first.** The test fails before the fix and passes after. Record a defect you cannot fix as a
   strict `xfail` with an id in `docs/verification-defects.md`.
8. **Run the gates.** `verify-change` skill; at least `scripts/eval_offline.py --check`,
   `scripts/eval_corruptions.py --check` and `verification_report.py --check`.

## Privacy

A report holds the conversation. Never paste one into an issue or a commit. A fixture in the repo has no prose
and no user mathematics. Server logs hold counters, identifiers and fixed vocabulary only.

## Reading the first failure

| Diagnosis stage | Usual first move |
| --- | --- |
| `parser` | read the typed diagnostic; is the model's spelling one a cleaning rule can handle without changing a meaning? |
| `model` | a stalled or failing host, or a reply cut off; check provider, `context_trimmed`, `request_sha` and the groups loaded |
| `orchestrator` | step, budget or recovery decision; compare `policy` on the step |
| `engine` / `verifier` | replay the tool; a status other than `ok` is usually the honest answer |
| `infrastructure` | sandbox timeout or resource limit; check the queue and concurrency |
