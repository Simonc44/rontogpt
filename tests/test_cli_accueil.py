"""Accueil du CLI : `rontogpt` sans argument doit ouvrir la conversation.

Le TUI est le defaut, mais il depend de Bun et de `tui/node_modules`. Sur une
installation de developpeur fraiche (`uv sync` seul), il est absent : le CLI
retombait alors sur la DEMO mathematique X3, ce qui n'est pas une reponse a
« je veux parler au modele ».
"""
import pytest


def test_sans_tui_le_cli_ouvre_la_conversation(monkeypatch):
    from aura import __main__ as cli, tui

    monkeypatch.setattr(tui, "lancer", lambda: tui.INDISPONIBLE)
    monkeypatch.setattr(cli, "RontoGPT", lambda: object())
    vus = []
    monkeypatch.setattr(cli, "_chat", lambda ia: vus.append(ia) or 0)
    monkeypatch.setattr(cli, "_demo",
                        lambda ia: pytest.fail("la demo n'est pas un accueil"))

    assert cli.main([]) == 0
    assert len(vus) == 1


def test_demo_reste_explicite(monkeypatch):
    from aura import __main__ as cli, tui

    monkeypatch.setattr(tui, "lancer", lambda: tui.INDISPONIBLE)
    monkeypatch.setattr(cli, "RontoGPT", lambda: object())
    monkeypatch.setattr(cli, "_chat",
                        lambda ia: pytest.fail("--demo doit lancer la demo"))
    monkeypatch.setattr(cli, "_demo", lambda ia: 0)

    assert cli.main(["--demo"]) == 0
