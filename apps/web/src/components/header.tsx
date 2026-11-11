import Link from "next/link";
import { UserMenu } from "./user-menu";

export function Header() {
  return (
    <header className="border-b border-rule">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-6">
          <Link href="/" className="font-serif text-xl italic tracking-tight">
            Minerva
          </Link>
          <nav aria-label="Main" className="flex gap-4 text-sm text-muted">
            <Link href="/" className="hover:text-ink">
              Workspace
            </Link>
            <Link href="/reference" className="hover:text-ink">
              Reference
            </Link>
          </nav>
        </div>
        <UserMenu />
      </div>
    </header>
  );
}
