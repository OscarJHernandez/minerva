"""Load and validate the YAML knowledge base."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from .models import Chapter, Entry

DATA_DIR = Path(str(resources.files("minerva_knowledge") / "data"))


class KnowledgeBaseError(Exception):
    """Raised for an invalid knowledge base (duplicate ids, unknown chapters, bad schema)."""


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)  # safe_load: no arbitrary object construction


def load_chapters(data_dir: Path = DATA_DIR) -> list[Chapter]:
    raw = _read_yaml(data_dir / "chapters.yaml")
    if not isinstance(raw, list):
        raise KnowledgeBaseError("chapters.yaml must contain a list")
    chapters = [Chapter(**item) for item in raw]
    ids = [c.id for c in chapters]
    if len(ids) != len(set(ids)):
        raise KnowledgeBaseError("duplicate chapter ids")
    return chapters


def load_entries(data_dir: Path = DATA_DIR) -> list[Entry]:
    chapters = {c.id: c for c in load_chapters(data_dir)}
    entries: list[Entry] = []
    for path in sorted((data_dir / "entries").glob("*.yaml")):
        raw = _read_yaml(path)
        if not isinstance(raw, list):
            raise KnowledgeBaseError(f"{path.name} must contain a list of entries")
        for item in raw:
            try:
                entry = Entry(**item)
            except Exception as exc:
                label = item.get("id", "?") if isinstance(item, dict) else "?"
                raise KnowledgeBaseError(f"{path.name}: entry {label}: {exc}") from exc
            chapter = chapters.get(entry.chapter)
            if chapter is None:
                raise KnowledgeBaseError(f"{entry.id}: unknown chapter {entry.chapter!r}")
            if entry.section and entry.section not in {s.id for s in chapter.sections}:
                raise KnowledgeBaseError(f"{entry.id}: unknown section {entry.section!r}")
            entries.append(entry)
    ids = [e.id for e in entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise KnowledgeBaseError(f"duplicate entry ids: {duplicates}")
    return entries
