"""The knowledge base: loaded entries plus an SQLite FTS5 retrieval index."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from .loader import load_chapters, load_entries
from .models import Chapter, Entry
from .render import feature_token, rhs_features, rule_features, search_text, statement_latex

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# A query word whose stem is in more than this share of the entries says nothing about which entry is
# meant ("of" is in more than half): it is ignored, unless that would leave no word at all. The share is
# measured on the index itself (no list of stop words), and only on a handbook large enough to measure.
COMMON_SHARE = 0.25
COMMON_MIN_ENTRIES = 40
NO_QUERY_SHARE = 0.4
STRONG_COLUMNS = "{title keywords tags}"


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
    # Which side of a rule the features must come from: "lhs" (the rule fits the expression as
    # written) or "both" (a chain may also read the rule right to left).
    sides: Literal["lhs", "both"] = "lhs"


class Retriever(Protocol):
    """Interface for retrieval backends.

    `Fts5Retriever` is lexical (BM25 over titles, keywords and text) plus structural feature
    tokens. A semantic or structural-embedding retriever can implement the same protocol and
    be combined with it later; the engine only ever relies on the returned candidates being
    *proposals* that it checks itself.
    """

    def search(self, query: SearchQuery) -> list[Hit]: ...


def enough_words(count: int, matched: int, in_strong_column: bool) -> bool:
    """Whether an entry that matched `matched` of the `count` content words of a text query is a hit.

    One word: a match. Two words: both, or one that is in the title, keywords or tags (a match only in
    the body of the entry is too weak). Three or more: at least half of them."""
    if count <= 1:
        return matched >= 1
    if count == 2:
        return matched >= 2 or (matched == 1 and in_strong_column)
    return matched >= (count + 1) // 2


class Fts5Retriever:
    def __init__(self, entries: Iterable[Entry]) -> None:
        self._lock = threading.RLock()
        self._words: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE entries (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, chapter TEXT NOT NULL,
              verification TEXT NOT NULL, has_rule INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE entries_fts USING fts5(
              id UNINDEXED, title, keywords, tags, body, features, rhs_features,
              tokenize = 'porter unicode61'
            );
            """
        )
        for e in entries:
            feats = " ".join(feature_token(f) for f in sorted(rule_features(e)))
            rhs_feats = " ".join(feature_token(f) for f in sorted(rhs_features(e)))
            self._db.execute(
                "INSERT INTO entries VALUES (?,?,?,?,?)",
                (e.id, e.kind, e.chapter, e.verification, 1 if e.rule else 0),
            )
            self._db.execute(
                "INSERT INTO entries_fts(id, title, keywords, tags, body, features, rhs_features) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    e.id,
                    e.title,
                    " ".join(e.keywords),
                    " ".join(e.tags),
                    search_text(e),
                    feats,
                    rhs_feats,
                ),
            )
        self._db.commit()
        self._count = len(self._db.execute("SELECT id FROM entries").fetchall())

    # -- the words of a text query ---------------------------------------------------------------
    def _word_ids(self, word: str) -> tuple[frozenset[str], frozenset[str]]:
        """The entries a word (stemmed like the index) occurs in, and those where it is in the title,
        keywords or tags. `word` holds letters and digits only."""
        key = word.lower()
        cached = self._words.get(key)
        if cached is None:
            with self._lock:
                every = self._db.execute(
                    "SELECT id FROM entries_fts WHERE entries_fts MATCH ?", [f'"{key}"']
                ).fetchall()
                strong = self._db.execute(
                    "SELECT id FROM entries_fts WHERE entries_fts MATCH ?",
                    [f'{STRONG_COLUMNS} : "{key}"'],
                ).fetchall()
            cached = (frozenset(r[0] for r in every), frozenset(r[0] for r in strong))
            self._words[key] = cached
        return cached

    def _content_words(self, text: str) -> list[str]:
        """The words of a query that can tell entries apart, without repeats, at most 12."""
        words = list(dict.fromkeys(w.lower() for w in _TOKEN_RE.findall(text)))[:12]
        if self._count < COMMON_MIN_ENTRIES or len(words) < 2:
            return words
        rare = [w for w in words if len(self._word_ids(w)[0]) <= COMMON_SHARE * self._count]
        if rare:
            return rare
        # Every word is common ("derivative of the"): keep the rarest, unless even that is in nearly
        # half of the entries ("of the"), which is no query at all.
        rarest = min(words, key=lambda w: len(self._word_ids(w)[0]))
        return [rarest] if len(self._word_ids(rarest)[0]) <= NO_QUERY_SHARE * self._count else []

    @staticmethod
    def _match_expression(words: list[str], query: SearchQuery) -> str | None:
        clauses: list[str] = []
        if words:
            clauses.append("(" + " OR ".join(f'"{w}"' for w in words) + ")")
        feats = [feature_token(f) for f in sorted(set(query.features))][:24]
        if feats:
            columns = "{features rhs_features}" if query.sides == "both" else "features"
            clauses.append(columns + " : (" + " OR ".join(f'"{t}"' for t in feats) + ")")
        return " OR ".join(clauses) if clauses else None

    def _feature_ids(self, query: SearchQuery) -> frozenset[str]:
        match = self._match_expression([], query)
        if match is None:
            return frozenset()
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM entries_fts WHERE entries_fts MATCH ?", [match]
            ).fetchall()
        return frozenset(r[0] for r in rows)

    def search(self, query: SearchQuery) -> list[Hit]:
        words = self._content_words(query.text)
        if query.text.strip() and not words and not query.features:
            return []  # something was asked, but nothing in it can be searched for
        match = self._match_expression(words, query)
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
                    "SELECT f.id, bm25(entries_fts, 0.0, 6.0, 4.0, 2.0, 1.0, 3.0, 1.0) AS score "
                    "FROM entries_fts f JOIN entries e ON e.id = f.id "
                    "WHERE entries_fts MATCH ?" + extra + " ORDER BY score, f.id LIMIT ?",
                    [match, *args, 200 if words else limit],
                ).fetchall()
        hits = [Hit(entry_id=r[0], score=float(r[1])) for r in rows]
        if not words or match is None:
            return hits
        # Entries that matched on structure are hits as they are; the others matched on words alone and
        # must match enough of them.
        structural = self._feature_ids(query)
        found = [h for h in hits if h.entry_id in structural or self._covers(h.entry_id, words)]
        return found[:limit]

    def _covers(self, entry_id: str, words: list[str]) -> bool:
        matched, strong = 0, False
        for word in words:
            every, in_strong = self._word_ids(word)
            if entry_id in every:
                matched += 1
                strong = strong or entry_id in in_strong
        return enough_words(len(words), matched, strong)


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
        self,
        features: Collection[str],
        text: str = "",
        limit: int = 100,
        sides: Literal["lhs", "both"] = "lhs",
    ) -> list[Entry]:
        """Rule entries whose left-hand side (or, with `sides="both"`, either side) shares structure
        with the expression."""
        if not features and not text.strip():
            return []
        return self.search(
            SearchQuery(text=text, features=features, rules_only=True, limit=limit, sides=sides)
        )
