"""Le lanceur du TUI n'ecrit plus dans le terminal qu'il partage avec l'ecran.

Defaut reel : `python -m aura.tui` demarre l'API dans SON processus, donc dans
SON terminal. Chaque `logging.warning` (les caches du routeur non scelles, par
exemple) tombait PAR-DESSUS l'interface Bun : en-tete dedouble, lignes
`[integrite] ...` collees dans la barre de saisie. `_isoler_sorties` detourne
les descripteurs 1 et 2 vers `logs/tui-api.log`, et rend les anciens a Bun.
"""
import os
import sys

import pytest

from aura import tui


@pytest.fixture(autouse=True)
def journal_propre():
    """`_isoler_sorties` remplace la journalisation (`force=True`).

    Sans cette restitution, pytest perd son gestionnaire de capture pour la
    suite du run : les tests passeraient ou non selon l'ordre tire.
    """
    import logging

    handlers, niveau = logging.root.handlers[:], logging.root.level
    yield
    logging.root.handlers[:] = handlers
    logging.root.setLevel(niveau)


@pytest.fixture()
def racine(tmp_path, monkeypatch):
    """Un faux projet : le journal atterrit dans `tmp_path/logs/`."""
    monkeypatch.setattr(tui, "_RACINE", tmp_path)
    return tmp_path


def test_les_sorties_partent_dans_le_journal(racine):
    anciennes = tui._isoler_sorties()
    if anciennes is None:                 # systeme sans dup2 utilisable
        pytest.skip("detournement des sorties indisponible ici")
    try:
        # `os.write` et pas `print` : on teste les DESCRIPTEURS, pas le
        # tampon de `sys.stdout` (que pytest a deja remplace).
        os.write(1, b"[tui] stdout detourne\n")
        os.write(2, b"[tui] stderr detourne\n")
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        tui._rendre_sorties(anciennes)

    journal = racine / "logs" / "tui-api.log"
    assert journal.is_file(), "aucun journal ecrit"
    texte = journal.read_text(encoding="utf-8", errors="replace")
    assert "stdout detourne" in texte
    assert "stderr detourne" in texte


def test_les_sorties_sont_rendues_apres_le_tui(racine):
    """Ce qui est ecrit APRES la restauration ne doit pas finir au journal."""
    anciennes = tui._isoler_sorties()
    if anciennes is None:
        pytest.skip("detournement des sorties indisponible ici")
    tui._rendre_sorties(anciennes)
    os.write(1, b"[tui] apres restauration\n")
    sys.stdout.flush()

    journal = racine / "logs" / "tui-api.log"
    texte = journal.read_text(encoding="utf-8", errors="replace") if journal.is_file() else ""
    assert "apres restauration" not in texte


def test_restaurer_sans_avoir_isole_ne_casse_rien():
    """`_rendre_sorties(None)` est le chemin « detournement impossible »."""
    tui._rendre_sorties(None)              # ne doit rien lever
