"use client";

import { Math } from "@minerva/ui/math";
import Link from "next/link";
import { useState } from "react";
import type {
  DistributionNotation,
  DistributionNotationName,
  DistributionQuestion,
  DistributionReason,
  DistributionResult,
  EntrySummary,
} from "@/lib/types";
import { EntryCard } from "./entry-card";
import { StatusPill } from "./result-panel";

const REASON: Record<DistributionReason, string> = {
  unsupported_distribution: "Outside what this version supports",
  singular_root: "A root of the argument is repeated",
  assumptions_insufficient: "A condition is not established by the declared assumptions",
  verification_failed: "The test-function check did not confirm the candidate",
  integration_domain_ambiguous: "A delta sits on an integration limit: a convention is needed",
  notation_unconfirmed: "Minerva needs to know whether a name is a Dirac delta or a step function",
  resource_limit: "Resource limit reached",
  parse_error: "The expression could not be read",
};

const WRITTEN: Record<DistributionNotationName, string> = {
  delta: "\\delta(\\ldots)",
  theta: "\\theta(\\ldots)",
  Theta: "\\Theta(\\ldots)",
};

/**
 * Outside this operation `\delta(x)` is the symbol delta times x. The user, never Minerva or the model,
 * says whether it is a distribution; nothing is computed until they do.
 */
function NotationQuestions({
  questions,
  onAnswer,
}: {
  questions: DistributionQuestion[];
  onAnswer: (notation: DistributionNotation) => void | Promise<void>;
}) {
  const [answers, setAnswers] = useState<DistributionNotation>({});
  const [sent, setSent] = useState(false);

  function choose(q: DistributionQuestion, reading: "dirac_delta" | "heaviside" | "symbol") {
    const next = { ...answers, [q.name]: reading };
    setAnswers(next);
    if (!sent && questions.every((item) => next[item.name] !== undefined)) {
      setSent(true);
      void Promise.resolve(onAnswer(next)).catch(() => setSent(false));
    }
  }

  return (
    <div className="mt-3 space-y-3" data-testid="notation-questions">
      {questions.map((q) => {
        const yes = q.candidate === "dirac_delta" ? "a Dirac delta" : "a step function";
        const label = q.candidate === "dirac_delta" ? "Dirac delta" : "step function";
        return (
          <div
            key={q.name}
            role="group"
            aria-label={`Is ${q.name}(…) ${yes}?`}
            className="rounded border border-rule bg-sunk p-3 text-sm"
          >
            <div className="flex flex-wrap items-center gap-x-2">
              <span>Is</span>
              <Math latex={WRITTEN[q.name]} />
              <span>a {label}?</span>
            </div>
            <p className="mt-1 text-xs text-muted">
              Outside this tool it is an ordinary symbol, so nothing is computed until you say.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                disabled={sent}
                aria-label={`Yes, ${q.name}(…) is ${yes}`}
                className="rounded bg-accent px-3 py-1.5 text-accent-ink hover:opacity-90 disabled:opacity-50"
                onClick={() => choose(q, q.candidate)}
              >
                Yes, {yes}
              </button>
              <button
                type="button"
                disabled={sent}
                aria-label={`No, ${q.name}(…) is an ordinary symbol`}
                className="rounded border border-rule px-3 py-1.5 hover:bg-raised disabled:opacity-50"
                onClick={() => choose(q, "symbol")}
              >
                No, an ordinary symbol
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Delta and step functions: which handbook identity was applied at each step, the conditions that
 * were established exactly, and the test-function evidence. The equality is as distributions, never
 * pointwise, and the wording never calls the pairing check a proof.
 */
export function DistributionPanel({
  result,
  methods = [],
  onAnswer,
}: {
  result: DistributionResult;
  methods?: EntrySummary[];
  /** Runs the call again with the user's answers; without it a question is shown but cannot be answered. */
  onAnswer?: (notation: DistributionNotation) => void | Promise<void>;
}) {
  const [copied, setCopied] = useState(false);
  const v = result.verification;
  const shown = result.status === "ok" && result.result_latex !== null;
  const asValue = result.equality === "value";

  return (
    <section
      aria-labelledby="distribution-heading"
      className="rounded-lg border border-rule bg-raised p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="distribution-heading" className="font-serif text-xl">
          Delta and step functions
        </h2>
        <StatusPill
          status={result.status}
          okLabel={asValue ? "Evaluated, test-function checked" : "Equal as distributions"}
        />
      </div>

      {result.input_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">Input as read</div>
          <div className="overflow-x-auto">
            <Math latex={result.input_latex} display />
          </div>
        </div>
      )}

      {shown && result.result_latex && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">
            {asValue ? "Value" : "Result (as distributions)"}
          </div>
          <div className="overflow-x-auto">
            <Math latex={result.result_latex} display className="text-xl" label="Result" />
          </div>
          <button
            type="button"
            className="mt-1 text-xs text-accent hover:underline"
            onClick={async () => {
              await navigator.clipboard?.writeText(result.result_latex ?? "");
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
          >
            {copied ? "Copied" : "Copy LaTeX"}
          </button>
        </div>
      )}

      {shown && !asValue && (
        <p className="mt-2 text-sm text-muted" data-testid="distributional-note">
          Equal as distributions: both sides act the same way on every test function. This is not a
          pointwise statement; a delta has no values to compare.
        </p>
      )}

      {shown && result.endpoint && (
        <p className="mt-2 text-sm" data-testid="endpoint-note">
          Endpoint convention used: <strong>{result.endpoint}</strong>, as you chose.
        </p>
      )}

      {!shown && result.reason && (
        <p
          role="note"
          className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn"
          data-testid="refusal-reason"
        >
          {REASON[result.reason]}.
        </p>
      )}

      {result.reason === "notation_unconfirmed" &&
        (result.questions?.length ?? 0) > 0 &&
        onAnswer && <NotationQuestions questions={result.questions ?? []} onAnswer={onAnswer} />}

      {shown && result.steps.length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm font-medium">
            Steps <span className="font-normal text-muted">({result.steps.length})</span>
          </h3>
          <ol className="mt-2 space-y-2">
            {result.steps.map((step, i) => (
              <li
                key={`${i}-${step.entry_id}`}
                className="min-w-0 rounded border border-rule bg-sunk p-3 text-sm"
              >
                <div>
                  {i + 1}. {step.title}{" "}
                  <Link
                    href={`/reference/${step.entry_id}`}
                    className="break-all text-xs text-accent hover:underline"
                  >
                    <code>{step.entry_id}</code>
                  </Link>
                </div>
                <div className="overflow-x-auto">
                  <Math latex={step.before_latex} display />
                </div>
                <div className="overflow-x-auto">
                  <Math latex={`\\Rightarrow\\; ${step.after_latex}`} display />
                </div>
                {step.conditions.length > 0 && (
                  <ul className="mt-1 space-y-1 text-xs">
                    {step.conditions.map((c) => (
                      <li key={`${c.latex}-${c.how}`} className="flex min-w-0 flex-wrap gap-x-2">
                        <span className={c.established ? "text-ok" : "text-bad"}>
                          {c.established ? "established" : "not established"}
                        </span>
                        <Math latex={c.latex} />
                        <span className="text-muted">{c.how}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}

      {result.blocked.length > 0 && (
        <div className="mt-4 rounded bg-warn-soft p-3 text-sm text-warn">
          <div>Not applied: a condition is not established.</div>
          <ul className="mt-2 space-y-1">
            {result.blocked.map((b) => (
              <li key={`${b.entry_id}-${b.reason}`}>
                <code className="text-xs">{b.entry_id}</code>: {b.detail}
              </li>
            ))}
          </ul>
        </div>
      )}

      {!result.blocked.some((b) => b.detail === result.message) && (
        <p className="mt-3 text-sm text-muted">{result.message}</p>
      )}
      {result.notes.map((note) => (
        <p key={note} className="mt-1 text-sm text-warn">
          {note}
        </p>
      ))}

      {v && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Conditions</dt>
          <dd>{v.conditions_established ? "Established exactly" : "Not all established"}</dd>
          <dt className="text-muted">Test-function check</dt>
          <dd>
            {v.pairings_agreed} of {v.pairings_checked} pairings agree (supporting evidence, not a
            proof)
          </dd>
          <dt className="text-muted">Samples</dt>
          <dd>
            {v.parameter_samples} parameter sample{v.parameter_samples === 1 ? "" : "s"}
          </dd>
          <dt className="text-muted">Computed in</dt>
          <dd>{result.duration_ms} ms</dd>
        </dl>
      )}

      {result.diagnostics.events.length > 0 && (
        <details className="mt-3 text-xs text-muted">
          <summary className="cursor-pointer">Stages</summary>
          <ol className="mt-1 list-decimal pl-5">
            {result.diagnostics.events.map((e, i) => (
              <li key={`${i}-${e.stage}`}>
                {e.stage.replace(/_/g, " ")}: {e.status}
                {e.reason ? ` (${e.reason})` : ""}, {e.duration_ms} ms
              </li>
            ))}
          </ol>
        </details>
      )}

      {methods.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium">
            Handbook identities{" "}
            <span className="font-normal text-muted">({methods.length}): show</span>
          </summary>
          <div className="mt-2 space-y-3">
            {methods.map((m) => (
              <EntryCard key={m.id} entry={m} />
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
