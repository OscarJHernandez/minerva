"use client";

import { Math } from "@minerva/ui/math";
import type { IdentitiesResult, IdentityMatch } from "@/lib/types";
import { EntryCard } from "./entry-card";
import { Restrictions, StatusPill } from "./result-panel";

interface Apply {
  /** Rule id being applied right now, if any. */
  busy: string | null;
  onApply: (match: IdentityMatch) => void;
}

function Group({
  title,
  hint,
  items,
  collapsed = false,
  apply,
}: {
  title: string;
  hint: string;
  items: IdentityMatch[];
  collapsed?: boolean;
  apply?: Apply;
}) {
  if (items.length === 0) return null;
  const body = (
    <div className="space-y-3">
      {items.map((m) => (
        <EntryCard key={m.entry.id} entry={m.entry}>
          <Detail match={m} apply={apply} />
        </EntryCard>
      ))}
    </div>
  );
  if (collapsed) {
    return (
      <details className="mt-5">
        <summary className="cursor-pointer text-sm font-medium">
          {title} <span className="font-normal text-muted">({items.length}): show</span>
        </summary>
        <p className="mb-2 mt-1 text-xs text-muted">{hint}</p>
        {body}
      </details>
    );
  }
  return (
    <div className="mt-5">
      <h3 className="text-sm font-medium">
        {title} <span className="font-normal text-muted">({items.length})</span>
      </h3>
      <p className="mb-2 text-xs text-muted">{hint}</p>
      {body}
    </div>
  );
}

function Detail({ match, apply }: { match: IdentityMatch; apply?: Apply }) {
  const a = match.application;
  if (a.applicability === "applicable") {
    return (
      <div className="mt-3 rounded bg-ok-soft p-3 text-sm">
        <div className="text-ok">Applied to this part of your expression:</div>
        {a.matched_subexpression_latex && <Math latex={a.matched_subexpression_latex} display />}
        <div className="text-ok">Rewritten expression (numerically checked):</div>
        {a.rewritten_latex && <Math latex={a.rewritten_latex} display />}
        <Restrictions latex={a.restrictions_latex} complete={a.restrictions_complete ?? true} />
        {a.verification && (
          <div className="text-xs text-muted">
            {a.verification.numeric_points_agreed}/{a.verification.numeric_points_checked} sample
            points agree
            {a.verification.symbolic_zero ? "; difference simplifies to 0" : ""}
          </div>
        )}
        {apply && (
          <button
            type="button"
            onClick={() => apply.onApply(match)}
            disabled={apply.busy !== null}
            className="mt-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
          >
            {apply.busy === match.entry.id ? "Applying…" : "Apply as next step"}
          </button>
        )}
      </div>
    );
  }
  if (a.applicability === "conditions_not_satisfied") {
    return (
      <div className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn">
        <div>
          Matches
          {a.matched_subexpression_latex && (
            <>
              {" "}
              <Math latex={a.matched_subexpression_latex} />
            </>
          )}
          , but it is not applied: these conditions cannot be shown from your assumptions.
        </div>
        <ul className="mt-1 list-disc pl-5">
          {a.unmet_conditions.map((c) => (
            <li key={`${c.variable}-${c.assumption}`}>
              <code>{c.variable}</code> must be {c.assumption} (the pattern variable bound to the
              matched part)
            </li>
          ))}
        </ul>
      </div>
    );
  }
  if (a.applicability === "not_verified") {
    return <p className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn">{a.message}</p>;
  }
  return <p className="mt-3 text-sm text-muted">{a.message}</p>;
}

export function IdentitiesPanel({
  result,
  apply,
}: {
  result: IdentitiesResult;
  /** When given, applicable matches get an Apply button (the Workbench). Others never do. */
  apply?: Apply;
}) {
  const by = (kind: string) => result.matches.filter((m) => m.application.applicability === kind);
  const related = by("no_match");
  const applicable = by("applicable");
  const conditional = by("conditions_not_satisfied");
  const unverified = by("not_verified");
  const none = result.status === "ok" && applicable.length + conditional.length === 0;

  return (
    <section
      aria-labelledby="identities-heading"
      className="rounded-lg border border-rule bg-raised p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="identities-heading" className="font-serif text-xl">
          Identities
        </h2>
        {result.status !== "ok" && <StatusPill status={result.status} />}
      </div>
      {result.input_latex && <Math latex={result.input_latex} display />}
      {result.message && <p className="mt-2 text-sm text-muted">{result.message}</p>}
      {none && (
        <p className="mt-3 text-sm">
          No identity from the knowledge base applies to this expression. Minerva does not guess:{" "}
          {result.candidates_considered} candidate
          {result.candidates_considered === 1 ? " was" : "s were"} checked.
        </p>
      )}
      <Group
        title="Applicable"
        hint="Matched, conditions proven from your assumptions, and the rewrite checked numerically."
        items={applicable}
        apply={apply}
      />
      <Group
        title="Needs assumptions"
        hint="The pattern matches but validity conditions are not met. Add assumptions to unlock."
        items={conditional}
      />
      <Group title="Matched but not verified" hint="Not shown as results." items={unverified} />
      <Group
        title="Related, not applicable"
        hint="Structurally similar entries that do not occur in this expression."
        items={related}
        collapsed
      />
    </section>
  );
}
