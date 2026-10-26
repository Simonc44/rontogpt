"""Le lanceur du TUI n'ecrit plus dans le terminal qu'il partage avec l'ecran.

Defaut reel : `python -m aura.tui` demarre l'API dans SON processus, donc dans
SON terminal. Chaque `logging.warning` (les caches du routeur non scelles, par
exemple) tombait PAR-DESSUS l'interface Bun : en-tete dedouble, lignes
`[integrite] ...` collees dans la barre de saisie. `_isoler_sorties` detourne
les descripteurs 1 et 2 vers `logs/tui-api.log`, et rend les anciens a Bun.

Les tests tournent dans un PROCESSUS ENFANT : `_isoler_sorties` fait `dup2`
sur les descripteurs du processus, et le faire dans pytest detournerait la
sortie de la suite elle-meme (le run peut alors se bloquer dans la capture).
"""
import subprocess
import sys

SCRIPT = r"""
import os
import sys
from pathlib import Path

from aura import tui

tui._RACINE = Path(sys.argv[1])
anciennes = tui._isoler_sorties()
if anciennes is None:
    print("DISPONIBLE", flush=True)
    raise SystemExit(0)
os.write(1, b"AVANT stdout\n")
os.write(2, b"AVANT stderr\n")
tui._rendre_sorties(anciennes)
print("APRES restauration")          # doit revenir sur la sortie normale
sys.stdout.flush()
"""


def _jouer(tmp_path):
    return subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path)],
                          capture_output=True, text=True, timeout=120)


def test_les_sorties_partent_dans_le_journal(tmp_path):
    r = _jouer(tmp_path)
    assert r.returncode == 0, r.stderr
    if "DISPONIBLE" in r.stdout:          # systeme sans dup2 utilisable
        return
    journal = tmp_path / "logs" / "tui-api.log"
    assert journal.is_file(), "aucun journal ecrit"
    texte = journal.read_text(encoding="utf-8", errors="replace")
    assert "AVANT stdout" in texte and "AVANT stderr" in texte


def test_les_sorties_sont_rendues_apres_le_tui(tmp_path):
    """Ce qui est ecrit APRES la restauration ne doit pas finir au journal."""
    r = _jouer(tmp_path)
    assert r.returncode == 0, r.stderr
    if "DISPONIBLE" in r.stdout:
        return
    assert "APRES restauration" in r.stdout
    journal = tmp_path / "logs" / "tui-api.log"
    assert "APRES restauration" not in journal.read_text(encoding="utf-8")


def test_restaurer_sans_avoir_isole_ne_casse_rien():
    """`_rendre_sorties(None)` est le chemin « detournement impossible »."""
    from aura import tui
    tui._rendre_sorties(None)              # ne doit rien lever
