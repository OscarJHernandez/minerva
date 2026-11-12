import Link from "next/link";
import { APP_URL, REPO_URL, SELF_HOSTING_URL } from "@/lib/links";

export function SiteHeader() {
  return (
    <header className="mx-auto flex h-16 max-w-5xl items-center justify-between px-5 sm:px-8">
      <Link href="/" className="font-serif text-2xl italic tracking-tight">
        Minerva
      </Link>
      <nav aria-label="Main" className="flex items-center gap-5 text-sm">
        <Link href="/#examples" className="hidden text-muted hover:text-ink md:inline">
          Examples
        </Link>
        <Link href="/docs" className="text-muted hover:text-ink">
          Docs
        </Link>
        <a href={SELF_HOSTING_URL} className="hidden text-muted hover:text-ink md:inline">
          Self-host
        </a>
        <a href={REPO_URL} className="text-muted hover:text-ink">
          GitHub
        </a>
        <a
          href={APP_URL}
          className="rounded bg-accent px-3.5 py-1.5 font-medium text-accent-ink hover:opacity-90"
        >
          Try Minerva
        </a>
      </nav>
    </header>
  );
}
