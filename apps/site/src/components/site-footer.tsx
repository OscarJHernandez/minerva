import Link from "next/link";
import { APP_URL, CONTRIBUTING_URL, REPO_URL, SELF_HOSTING_URL } from "@/lib/links";

export function SiteFooter({ wide = false }: { wide?: boolean }) {
  return (
    <footer className="border-t border-rule py-10 text-sm text-muted">
      <div
        className={`mx-auto flex ${wide ? "max-w-6xl" : "max-w-5xl"} flex-wrap items-center justify-between gap-x-6 gap-y-3 px-5 sm:px-8`}
      >
        <span>
          <span className="font-serif italic">Minerva</span> · Apache-2.0
        </span>
        <nav aria-label="Footer" className="flex flex-wrap gap-x-5 gap-y-2">
          <Link href="/docs" className="hover:text-ink">
            Docs
          </Link>
          <a href={SELF_HOSTING_URL} className="hover:text-ink">
            Self-hosting
          </a>
          <a href={CONTRIBUTING_URL} className="hover:text-ink">
            Contributing
          </a>
          <a href={REPO_URL} className="hover:text-ink">
            GitHub
          </a>
          <a href={`${APP_URL}/login`} className="hover:text-ink">
            Log in
          </a>
        </nav>
        <span>This site stores nothing about you: no cookies, no analytics.</span>
      </div>
    </footer>
  );
}
