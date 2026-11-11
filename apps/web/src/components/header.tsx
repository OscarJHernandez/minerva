import Link from "next/link";
import { UserMenu } from "./user-menu";

export function Header() {
  return (
    <header className="border-b border-rule">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3 sm:gap-6">
          <Link href="/" className="font-serif text-lg italic tracking-tight sm:text-xl">
            Minerva
          </Link>
          <nav
            aria-label="Main"
            className="flex gap-2.5 text-[13px] text-muted sm:gap-4 sm:text-sm"
          >
            <Link href="/" className="hover:text-ink">
              Chat
            </Link>
            <Link href="/workbench" className="hover:text-ink">
              Workbench
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
