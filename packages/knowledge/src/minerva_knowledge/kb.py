"""The knowledge base: loaded entries plus an SQLite FTS5 retrieval index."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from .loader import load_chapters, load_entries
from .models import Chapter, Entry
from .render import feature_token, rule_features, search_text, statement_latex

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class Hit(BaseModel):
    entry_id: str
    score: float  # lower is better (bm25); 0.0 for non-ranked listings


@dataclass(frozen=True)
class SearchQuery:
    text: str = ""
    features: Collection[str] = ()
    kind: str | None = None
    chapter: str | None = None
    rules_only: bool = False
    limit: int = 20


class Retriever(Protocol):
    """Interface for retrieval backends.

    `Fts5Retriever` is lexical (BM25 over titles, keywords and text) plus structural feature
    tokens. A semantic or structural-embedding retriever can implement the same protocol and
    be combined with it later; the engine only ever relies on the returned candidates being
    *proposals* that it checks itself.
    """

    def search(self, query: SearchQuery) -> list[Hit]: ...


class Fts5Retriever:
    def __init__(self, entries: Iterable[Entry]) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE entries (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, chapter TEXT NOT NULL,
              verification TEXT NOT NULL, has_rule INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE entries_fts USING fts5(
              id UNINDEXED, title, keywords, tags, body, features,
              tokenize = 'porter unicode61'
            );
            """
        )
        for e in entries:
            feats = " ".join(feature_token(f) for f in sorted(rule_features(e)))
            self._db.execute(
                "INSERT INTO entries VALUES (?,?,?,?,?)",
                (e.id, e.kind, e.chapter, e.verification, 1 if e.rule else 0),
            )
            self._db.execute(
                "INSERT INTO entries_fts(id, title, keywords, tags, body, features) "
                "VALUES (?,?,?,?,?,?)",
                (e.id, e.title, " ".join(e.keywords), " ".join(e.tags), search_text(e), feats),
            )
        self._db.commit()

    @staticmethod
    def _match_expression(query: SearchQuery) -> str | None:
        clauses: list[str] = []
        words = _TOKEN_RE.findall(query.text)[:12]
        if words:
            clauses.append("(" + " OR ".join(f'"{w}"' for w in words) + ")")
        feats = [feature_token(f) for f in sorted(set(query.features))][:24]
        if feats:
            clauses.append("features : (" + " OR ".join(f'"{t}"' for t in feats) + ")")
        return " OR ".join(clauses) if clauses else None

    def search(self, query: SearchQuery) -> list[Hit]:
        match = self._match_expression(query)
        limit = max(1, min(query.limit, 200))
        where, args = [], []
        if query.kind:
            where.append("e.kind = ?")
            args.append(query.kind)
        if query.chapter:
            where.append("e.chapter = ?")
            args.append(query.chapter)
        if query.rules_only:
            where.append("e.has_rule = 1")
        # `where` holds only fixed fragments with `?` placeholders; user values go in `args`.
        extra = (" AND " + " AND ".join(where)) if where else ""
        with self._lock:
            if match is None:
                rows = self._db.execute(
                    "SELECT e.id, 0.0 FROM entries e WHERE 1=1"  # noqa: S608
                    + extra
                    + " ORDER BY e.id LIMIT ?",
                    [*args, limit],
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT f.id, bm25(entries_fts, 0.0, 6.0, 4.0, 2.0, 1.0, 3.0) AS score "
                    "FROM entries_fts f JOIN entries e ON e.id = f.id "
                    "WHERE entries_fts MATCH ?" + extra + " ORDER BY score, f.id LIMIT ?",
                    [match, *args, limit],
                ).fetchall()
        return [Hit(entry_id=r[0], score=float(r[1])) for r in rows]


class KnowledgeBase:
    def __init__(self, entries: list[Entry], chapters: list[Chapter]) -> None:
        self.entries = entries
        self.chapters = chapters
        self._by_id = {e.id: e for e in entries}
        self.retriever: Retriever = Fts5Retriever(entries)

    @classmethod
    def load(cls) -> KnowledgeBase:
        return cls(load_entries(), load_chapters())

    def get(self, entry_id: str) -> Entry | None:
        return self._by_id.get(entry_id)

    def statement_latex(self, entry: Entry) -> str:
        return statement_latex(entry)

    def toc(self) -> list[dict[str, object]]:
        """Chapters that have entries, with per-section entry counts."""
        counts: dict[str, dict[str | None, int]] = defaultdict(lambda: defaultdict(int))
        for e in self.entries:
            counts[e.chapter][e.section] += 1
        result: list[dict[str, object]] = []
        for c in self.chapters:
            if c.id not in counts:
                continue
            result.append(
                {
                    "id": c.id,
                    "book": c.book,
                    "title": c.title,
                    "summary": c.summary,
                    "entry_count": sum(counts[c.id].values()),
                    "sections": [
                        {"id": s.id, "title": s.title, "entry_count": counts[c.id].get(s.id, 0)}
                        for s in c.sections
                        if counts[c.id].get(s.id, 0)
                    ],
                }
            )
        return result

    def search(self, query: SearchQuery) -> list[Entry]:
        return [e for h in self.retriever.search(query) if (e := self.get(h.entry_id))]

    def candidate_rules(
        self, features: Collection[str], text: str = "", limit: int = 25
    ) -> list[Entry]:
        """Rule entries whose left-hand side shares structure with the expression."""
        if not features and not text.strip():
            return []
        return self.search(SearchQuery(text=text, features=features, rules_only=True, limit=limit))
