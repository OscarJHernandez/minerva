"use client";

import { useEffect, useState } from "react";

export type NavGroup = { id: string; title: string; items: { id: string; title: string }[] };
export type NavLink = { id: string; title: string };

/** Ids of every section the nav points at, in page order. */
function sectionIds(groups: NavGroup[], extra: NavLink[]): string[] {
  return [...groups.flatMap((g) => g.items.map((i) => i.id)), ...extra.map((e) => e.id)];
}

/**
 * Left navigation of the docs. The links are plain anchors, so they work without JavaScript; the script
 * only marks the section being read (`aria-current="location"`). `closeOnNavigate` collapses the phone
 * disclosure that wraps it after a link is followed.
 */
export function DocsNav({
  groups,
  extra,
  closeOnNavigate = false,
}: {
  groups: NavGroup[];
  extra: NavLink[];
  closeOnNavigate?: boolean;
}) {
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const ids = sectionIds(groups, extra);
    const visible = new Set<string>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) visible.add(e.target.id);
          else visible.delete(e.target.id);
        }
        const first = ids.find((id) => visible.has(id));
        if (first) setActive(first);
      },
      { rootMargin: "-10% 0px -70% 0px" },
    );
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [groups, extra]);

  const link = (item: NavLink) => (
    <a
      href={`#${item.id}`}
      aria-current={active === item.id ? "location" : undefined}
      className={`block border-l py-1 pl-3 text-sm ${
        active === item.id
          ? "border-accent font-medium text-ink"
          : "border-rule text-muted hover:text-ink"
      }`}
    >
      {item.title}
    </a>
  );

  return (
    <nav
      aria-label="Docs"
      onClick={(e) => {
        if (closeOnNavigate && (e.target as HTMLElement).closest("a")) {
          (e.currentTarget.closest("details") as HTMLDetailsElement | null)?.removeAttribute(
            "open",
          );
        }
      }}
    >
      {groups.map((g) => (
        <div key={g.id} className="mb-5">
          <p className="mb-1 text-xs uppercase tracking-[0.14em] text-muted">{g.title}</p>
          <ul>
            {g.items.map((i) => (
              <li key={i.id}>{link(i)}</li>
            ))}
          </ul>
        </div>
      ))}
      <ul className="border-t border-rule pt-4">
        {extra.map((e) => (
          <li key={e.id}>{link(e)}</li>
        ))}
      </ul>
    </nav>
  );
}
