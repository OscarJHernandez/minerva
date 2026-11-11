import { Math } from "@minerva/ui/math";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { VerificationBadge } from "@/components/verification-badge";
import { reference } from "@/lib/server-api";

type Params = Promise<{ id: string }>;

export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const { id } = await params;
  const entry = await reference.entry(id);
  return { title: entry?.title ?? "Not found" };
}

export default async function EntryPage({ params }: { params: Params }) {
  const { id } = await params;
  const entry = await reference.entry(id);
  if (!entry) notFound();

  return (
    <article className="mx-auto max-w-3xl pt-8">
      <Link
        href={`/reference?chapter=${entry.chapter}`}
        className="text-sm text-accent hover:underline"
      >
        ← {entry.chapter}
      </Link>
      <div className="mt-3 flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="font-serif text-3xl">{entry.title}</h1>
        <VerificationBadge level={entry.verification} />
      </div>
      <div className="mt-1 font-mono text-xs text-muted">{entry.id}</div>

      <div className="my-6 rounded-lg border border-rule bg-raised p-5 text-xl">
        <Math latex={entry.statement_latex} display />
      </div>

      <dl className="space-y-4 text-sm">
        {entry.validity && (
          <div>
            <dt className="font-medium">Validity</dt>
            <dd className="text-muted">{entry.validity}</dd>
          </div>
        )}
        {entry.conditions.length > 0 && (
          <div>
            <dt className="font-medium">Conditions checked before applying</dt>
            <dd className="text-muted">
              {entry.conditions.map((c) => `${c.variable} ${c.assumption}`).join("; ")}
            </dd>
          </div>
        )}
        {entry.constant && (
          <div>
            <dt className="font-medium">Value</dt>
            <dd className="text-muted">
              {entry.constant.value} {entry.constant.unit}
              {entry.constant.uncertainty
                ? ` (standard uncertainty ${entry.constant.uncertainty})`
                : " (exact)"}
            </dd>
          </div>
        )}
        {entry.symbols.length > 0 && (
          <div>
            <dt className="font-medium">Symbols</dt>
            <dd>
              <table className="mt-1 w-full text-left">
                <tbody>
                  {entry.symbols.map((s) => (
                    <tr key={s.name} className="border-t border-rule">
                      <td className="py-1 pr-4 font-mono">{s.name}</td>
                      <td className="py-1 pr-4 text-muted">{s.meaning}</td>
                      <td className="py-1 font-mono text-xs text-muted">{s.unit ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </dd>
          </div>
        )}
        {entry.notes && (
          <div>
            <dt className="font-medium">Notes</dt>
            <dd className="text-muted">{entry.notes}</dd>
          </div>
        )}
        {entry.reference && (
          <div>
            <dt className="font-medium">Source</dt>
            <dd className="text-muted">
              {entry.reference.url ? (
                <a
                  href={entry.reference.url}
                  className="text-accent hover:underline"
                  rel="noreferrer"
                >
                  {entry.reference.source}
                  {entry.reference.locator ? `, ${entry.reference.locator}` : ""}
                </a>
              ) : (
                <>
                  {entry.reference.source}
                  {entry.reference.locator ? `, ${entry.reference.locator}` : ""}
                </>
              )}
            </dd>
          </div>
        )}
      </dl>
      {entry.applicable_as_rule && (
        <p className="mt-8 text-sm text-muted">
          This entry is applied by the engine in{" "}
          <Link href="/" className="text-accent hover:underline">
            the workspace
          </Link>{" "}
          when its pattern matches and its conditions hold.
        </p>
      )}
    </article>
  );
}
