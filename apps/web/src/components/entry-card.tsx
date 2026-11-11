import { Math } from "@minerva/ui/math";
import Link from "next/link";
import type { EntrySummary } from "@/lib/types";
import { VerificationBadge } from "./verification-badge";

export function EntryCard({
  entry,
  children,
}: {
  entry: EntrySummary;
  children?: React.ReactNode;
}) {
  return (
    <article className="rounded-lg border border-rule bg-raised p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-serif text-lg">
          <Link href={`/reference/${entry.id}`} className="hover:text-accent hover:underline">
            {entry.title}
          </Link>
        </h3>
        <VerificationBadge level={entry.verification} />
      </div>
      <Math latex={entry.statement_latex} display />
      {entry.validity && <p className="text-sm text-muted">Valid: {entry.validity}</p>}
      {entry.reference && (
        <p className="mt-1 text-xs text-muted">
          Source: {entry.reference.source}
          {entry.reference.locator ? `, ${entry.reference.locator}` : ""}
        </p>
      )}
      {children}
    </article>
  );
}
