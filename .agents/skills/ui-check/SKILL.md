---
name: ui-check
description: Verify a Minerva UI change in a real browser in light and dark themes at phone width. Use after any change under apps/web, apps/site or packages/ui (components, pages, theme tokens, KaTeX rendering).
---

# ui-check

Type checks and unit tests do not show layout, theming or rendering faults. Look at the page.

## 1. Start the app

`.claude/launch.json` defines `minerva-api` (port 8100) and `minerva-web` (port 3000). Start the API
first when the page calls it (workspace, handbook, chat); the landing page (`apps/site`) is static and
needs only `pnpm --filter @minerva/site build` or its dev server on port 3001.

## 2. Check, in this order

1. Open the changed page at desktop width. No console errors or warnings (`read_console_messages`),
   no failed requests (`read_network_requests`).
2. Phone width (375x812): no horizontal scroll, controls reachable, text not clipped, math not overflowing
   its container.
3. Dark theme at both widths: no hard-coded colours (use tokens from `packages/ui/src/theme.css`),
   adequate contrast, KaTeX output readable.
4. Light theme at both widths, same checks.
5. Exercise the interaction you changed (type, submit, error state, empty state) and confirm the DOM text.

## 3. Rules to keep

- Tailwind tokens only; no hard-coded colours in components.
- Client components only where interaction needs them.
- Rendered maths comes from verified API results; do not generate or "fix" LaTeX in the UI.

## 4. Report

State which widths and themes you checked and what you saw. If a browser was unavailable, say so.
