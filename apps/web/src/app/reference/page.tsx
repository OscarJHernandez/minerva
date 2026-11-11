import type { Metadata } from "next";
import Link from "next/link";
import { EntryCard } from "@/components/entry-card";
import { reference } from "@/lib/server-api";

export const metadata: Metadata = { title: "Reference" };

export default async function ReferencePage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string; chapter?: string }>;
}) {
  const { q, chapter } = await searchParams;
  const [toc, entries] = await Promise.all([
    reference.toc(),
    q || chapter ? reference.entries({ q, chapter }) : Promise.resolve(null),
  ]);
  const books = [
    { id: "mathematics", title: "Mathematics" },
    { id: "physics", title: "Physics" },
  ] as const;

  return (
    <div className="pb-16 pt-8">
      <h1 className="font-serif text-3xl">Reference</h1>
      <p className="mt-1 max-w-2xl text-sm text-muted">
        A growing handbook of identities, tables and physical equations. Each entry shows how it was
        checked; entries marked “stated” are recorded from the literature and never applied by the
        engine.
      </p>

      <form action="/reference" className="mt-5 flex max-w-xl gap-2" role="search">
        <input
          name="q"
          defaultValue={q ?? ""}
          placeholder="Search, e.g. gaussian, euler, kinetic energy"
          aria-label="Search the reference"
          className="flex-1 rounded border border-rule bg-raised px-3 py-2 text-sm"
        />
        <button className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink">
          Search
        </button>
      </form>

      {entries ? (
        <section className="mt-8" aria-label="Results">
          <div className="mb-3 flex items-center justify-between text-sm text-muted">
            <span>
              {entries.length} {entries.length === 1 ? "entry" : "entries"}
              {q ? ` for “${q}”` : ""}
              {chapter ? ` in ${chapter}` : ""}
            </span>
            <Link href="/reference" className="text-accent hover:underline">
              Back to contents
            </Link>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {entries.map((e) => (
              <EntryCard key={e.id} entry={e} />
            ))}
          </div>
          {entries.length === 0 && <p className="text-sm">No entries match.</p>}
        </section>
      ) : (
        <div className="mt-8 grid gap-10 md:grid-cols-2">
          {books.map((b) => (
            <section key={b.id} aria-labelledby={`book-${b.id}`}>
              <h2 id={`book-${b.id}`} className="border-b border-rule pb-1 font-serif text-xl">
                {b.title}
              </h2>
              <ul className="mt-3 space-y-3">
                {toc
                  .filter((c) => c.book === b.id)
                  .map((c) => (
                    <li key={c.id}>
                      <Link
                        href={`/reference?chapter=${c.id}`}
                        className="font-medium hover:text-accent hover:underline"
                      >
                        {c.title}
                      </Link>{" "}
                      <span className="text-xs text-muted">{c.entry_count}</span>
                      <div className="text-xs text-muted">
                        {c.sections.map((s) => s.title).join(" · ")}
                      </div>
                    </li>
                  ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
