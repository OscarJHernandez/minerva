# Licensing assessment

**Status: provisional.** The repository is Apache-2.0 (the `LICENSE` file from the initial commit).
This document records the compatibility analysis. It is an engineering assessment, not legal advice;
have a lawyer review it before relying on it for anything that matters (for example distributing
container images that contain GPL software).

## Decision

Keep **Apache-2.0** for now. Cadabra 2 and SageMath are **not** dependencies, are not vendored and are
not imported anywhere. The decision must be revisited **before** any of these happens:

1. Minerva code imports `cadabra2` / `sage.*` in the same process, or shares in-memory objects with them.
2. A Minerva artifact that is *distributed* (a public container image, a release tarball, a package)
   contains Cadabra or Sage.
3. Minerva copies Cadabra/Sage source into this repository.

## Why this matters: Apache-2.0 and GPL

- Apache-2.0 is **one-way compatible** with GPL-3.0: Apache-2.0 code can be included in a GPL-3.0 work,
  but GPL-3.0 code cannot be relicensed as Apache-2.0. A combined work that is distributed must satisfy
  the GPL for the whole combination.
- Apache-2.0 is **not** compatible with GPL-2.0-only. SageMath is GPL-2.0-or-later overall (verify the
  current statement and per-component licences at integration time), which permits use under GPL-3.0.

## Cadabra 2 (GPL-3.0)

What we know from upstream: Cadabra 2 is licensed GPL-3.0 and exposes a Python module (`cadabra2`) and a
notebook interface.

A separate process or service does **not automatically** solve the problem:

| Scenario | Assessment |
| --- | --- |
| Importing `cadabra2` inside the API or engine process | The combination is a single program. If distributed, the whole must be offered under GPL-3.0. **Avoid.** |
| Separate process, narrow text/JSON protocol (expression strings in, strings out) | Generally treated as separate programs communicating at arm's length (FSF FAQ on pipes, sockets and command-line arguments), but this is a matter of degree: exchanging complex internal data structures makes it look like one program. Keep the protocol minimal and documented. |
| Running Cadabra on **our own server** for users, without distributing it | GPL-3.0 (unlike AGPL) has no network-use clause; hosting does not by itself trigger source obligations. |
| **Distributing** an image or bundle that contains Cadabra (including pushing a public image to a registry) | That is conveying Cadabra: its licence terms (source offer, notices, GPL-3.0 text) apply to the Cadabra part, and a combined distribution must not impose extra restrictions. |

Recommended integration approach when the time comes:

1. Cadabra stays an **optional, user-installed backend** behind the `SymbolicBackend` interface, discovered at
   runtime; Minerva's default images do not contain it.
2. Communicate over a narrow, documented protocol (text in, text out), never by sharing Python objects.
3. Publish no public images that contain Cadabra unless the GPL obligations for that image are met
   (source for Cadabra, notices, licence text) and reviewed.
4. If tighter coupling is ever required (in-process use, shared data structures), either relicense the
   combined work under GPL-3.0-or-later or do not couple.

## SageMath / SageManifolds

SageMath is GPL (v2 or later overall, per its documentation; individual components vary). The same
reasoning applies: optional backend, separate process, narrow protocol, no distribution of Sage inside
Minerva artifacts without a compliance review. Confirm the exact licence of the SageManifolds components
you intend to use.

## Current dependencies

Collected from installed package metadata (Python) and `pnpm licenses` (production JavaScript).

**Python (runtime + dev tooling in the lockfile):** MIT, BSD-2/3-Clause, Apache-2.0, ISC, MIT-0,
Unlicense, PSF-2.0, plus:

| Package | Licence | Note |
| --- | --- | --- |
| `psycopg`, `psycopg-binary` | LGPL-3.0-only | Used unmodified as a library. Compatible with an Apache-2.0 project; anyone distributing a build must permit replacing the library (satisfied by shipping it as an ordinary installed package). `psycopg-binary` bundles `libpq` (PostgreSQL licence). Alternative if this becomes a concern: `asyncpg` (Apache-2.0) with an async API |
| `certifi`, `pathspec` | MPL-2.0 | File-level copyleft; used unmodified |
| `sympy`, `mpmath` | BSD-3-Clause | Core computation |
| `fastapi`, `pydantic`, `sqlalchemy`, `alembic`, `argon2-cffi`, `pyyaml`, `lark`, `uvicorn`, `structlog` | MIT / BSD / Apache-2.0 | |

**JavaScript (production):** MIT (majority), Apache-2.0, ISC, BSD-3-Clause, 0BSD, plus:

| Package | Licence | Note |
| --- | --- | --- |
| `@fontsource-variable/inter`, `@fontsource-variable/newsreader` | OFL-1.1 | Fonts; OFL allows embedding and redistribution with the licence |
| `@img/sharp-libvips-*` | LGPL-3.0-or-later | Optional native dependency of Next.js image optimisation; the app does not use `next/image`. Prebuilt binary, unmodified. Can be excluded from images if a strictly permissive set is wanted |
| `caniuse-lite` | CC-BY-4.0 | Browser data used at build time |
| `mathlive`, `katex`, `next`, `react`, `tailwindcss` | MIT | |

**Infrastructure images:** PostgreSQL (PostgreSQL licence), nginx (BSD-2-Clause), Python (PSF), Node (MIT).

Re-run the collection when dependencies change:

```bash
uv run --no-sync python -c "import importlib.metadata as m; [print(d.metadata['Name'], d.metadata.get('License-Expression') or d.metadata.get('License')) for d in m.distributions()]"
pnpm --filter @minerva/web licenses list --prod
```

## Handbook content

Entries are original statements of mathematical facts with locator-only citations (for example
`DLMF §4.21`). No text, tables or figures are copied from books or papers. Mathematical facts are not
copyrightable but a publisher's selection, arrangement and wording are, so contributors must not
transcribe tables. Physical constants cite the SI Brochure and CODATA by name only.
