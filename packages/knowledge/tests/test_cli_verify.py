"""`verify` must not accept numeric agreement for an entry declared `symbolic`."""

import pytest
from minerva_knowledge import __main__ as cli
from minerva_knowledge.checks import CheckReport


def test_symbolic_entry_without_symbolic_zero_fails_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [e for e in cli.load_entries() if e.verification == "symbolic"][:1]
    monkeypatch.setattr(cli, "load_entries", lambda: entries)
    monkeypatch.setattr(
        cli,
        "run_check",
        lambda e: CheckReport(e.id, "rule", True, "numeric 12/12", symbolic_proof=False),
    )
    assert cli.main(["verify"]) == 1
    assert "declared symbolic but no symbolic zero" in capsys.readouterr().out


def test_symbolic_entry_with_symbolic_zero_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [e for e in cli.load_entries() if e.verification == "symbolic"][:1]
    monkeypatch.setattr(cli, "load_entries", lambda: entries)
    monkeypatch.setattr(
        cli, "run_check", lambda e: CheckReport(e.id, "rule", True, "ok", symbolic_proof=True)
    )
    assert cli.main(["verify"]) == 0
