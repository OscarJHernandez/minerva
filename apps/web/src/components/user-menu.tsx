"use client";

import { modelSettings } from "@/lib/model-settings";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { User } from "@/lib/types";

export function UserMenu() {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    api
      .me()
      .then((u) => !cancelled && setUser(u))
      .catch(() => !cancelled && setUser(null));
    return () => {
      cancelled = true;
    };
    // Re-check the session after navigation (login, logout, expiry).
  }, [pathname]);

  if (user === undefined) return <div className="h-5 w-24" aria-hidden />;
  if (user === null) {
    return (
      <div className="flex gap-3 text-sm">
        <Link href="/login" className="text-muted hover:text-ink">
          Log in
        </Link>
        <Link href="/register" className="font-medium text-accent hover:underline">
          Create account
        </Link>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="hidden text-muted sm:inline">{user.email}</span>
      <button
        type="button"
        className="shrink-0 whitespace-nowrap rounded border border-rule px-2 py-1 text-[13px] hover:bg-sunk sm:px-2.5 sm:text-sm"
        onClick={async () => {
          await api.logout().catch(() => undefined);
          modelSettings.clear(); // forget any OpenRouter key held in memory
          setUser(null);
          router.push("/login");
          router.refresh();
        }}
      >
        Log out
      </button>
    </div>
  );
}
