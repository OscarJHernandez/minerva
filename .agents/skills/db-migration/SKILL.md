---
name: db-migration
description: Change the Minerva API database schema safely. Use when editing SQLAlchemy models in services/api/src/minerva_api/db.py or adding an Alembic revision under services/api/alembic/versions.
---

# db-migration

Production runs PostgreSQL; the default tests run SQLite. A migration must work on both.

## 1. Change the model and add a revision

Edit `services/api/src/minerva_api/db.py`, then from `services/api`:

```bash
export MINERVA_DATABASE_URL=sqlite+pysqlite:///./dev.sqlite     # local scratch DB, ignored by git
uv run alembic revision --autogenerate -m "short description"
```

Review the generated file in `alembic/versions/`: autogenerate misses renames, server defaults and data
migrations. Name files `NNNN_topic.py` following the existing sequence.

## 2. Rules

- Migrations are additive and reversible where possible; give `downgrade()` a real body.
- No user data, emails or real identifiers in migrations, fixtures or tests.
- New columns on populated tables need a default or a backfill step.
- Never log or print row contents.

## 3. Prove it

```bash
uv run alembic upgrade head && uv run alembic check      # models and migrations agree
uv run pytest services/api                                # SQLite
MINERVA_TEST_DATABASE_URL=postgresql+psycopg://... uv run pytest services/api   # PostgreSQL, if available
```

CI repeats `upgrade head` and `check` against PostgreSQL. Delete the scratch `dev.sqlite` when done.
