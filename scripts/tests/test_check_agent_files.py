import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "check_agent_files", Path(__file__).resolve().parents[1] / "check_agent_files.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_repo(tmp_path: Path, *, listed: bool = True, name: str = "demo") -> Path:
    (tmp_path / ".agents" / "skills" / "demo").mkdir(parents=True)
    (tmp_path / ".agents" / "skills" / "README.md").write_text("# skills\n")
    (tmp_path / ".agents" / "skills" / "demo" / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Does a thing. Use when asked.\n---\n\n# demo\n"
    )
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "skills").symlink_to("../.agents/skills")
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n")
    row = "| [`demo`](.agents/skills/demo/SKILL.md) | x |\n" if listed else ""
    (tmp_path / "AGENTS.md").write_text(f"# Rules\n\n{row}")
    return tmp_path


def test_repository_is_consistent() -> None:
    assert MODULE.check(REPO_ROOT) == []


def test_valid_layout_passes(tmp_path: Path) -> None:
    assert MODULE.check(make_repo(tmp_path)) == []


def test_skill_missing_from_agents_table_fails(tmp_path: Path) -> None:
    errors = MODULE.check(make_repo(tmp_path, listed=False))
    assert errors == ["demo: not listed in the AGENTS.md Skills table"]


def test_name_must_match_directory(tmp_path: Path) -> None:
    errors = MODULE.check(make_repo(tmp_path, name="other"))
    assert any("!= directory name" in e for e in errors)


def test_claude_md_must_only_import_agents(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "CLAUDE.md").write_text("# duplicated rules\n")
    assert any("CLAUDE.md" in e for e in MODULE.check(repo))


def test_dangling_table_entry_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "AGENTS.md").write_text(
        "[`gone`](.agents/skills/gone/SKILL.md)\n[`demo`](.agents/skills/demo/SKILL.md)\n"
    )
    assert any("gone" in e and "does not exist" in e for e in MODULE.check(repo))


def test_broken_symlink_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / ".claude" / "skills").unlink()
    assert any(".claude/skills" in e for e in MODULE.check(repo))
