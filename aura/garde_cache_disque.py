"""Portails devant le cache disque de llama.cpp (avis PYSEC-2026-2447).

POURQUOI CE MODULE
------------------
`diskcache` 5.6.3 — tire par `llama-cpp-python`, qui l'importe au chargement
du module et ne peut donc pas en etre separe — serialise en **pickle** : qui
peut ecrire dans le dossier du cache peut y deposer un objet dont la LECTURE
execute du code. Aucune version corrigee n'existe (PyPI s'arrete a 5.6.3,
l'amont est dormant depuis mars 2024). `aura` ne se sert jamais de ce cache :
le seul utilise est `LlamaRAMCache`, en memoire, qui ne deserialise rien.

Ce module ne supprime pas l'avis (le paquet reste installe, pip-audit le voit
toujours) : il rend le CHEMIN VULNERABLE inatteignable, et il faut ouvrir les
portails dans l'ordre pour l'atteindre.

1. ACTIVATION   `AURA_CACHE_DISQUE=1`. Absent, vide ou different : refus.
2. COFFRE       `AURA_COFFRE_CACHE` doit designer un chemin ABSOLU et un VRAI
                dossier — pas un lien symbolique, donc pas un dossier monte
                depuis ailleurs.
3. MARQUEUR     le coffre doit avoir ete PREPARE :
                    python -m aura.garde_cache_disque --preparer <dossier>
                Un fichier `.coffre` y est ecrit avec le chemin canonique. Un
                coffre recopie ou deplace ne passe plus (le chemin enregistre
                ne correspond plus a celui ou il se trouve).
4. PERMISSIONS  (POSIX) dossier en 0700, marqueur en 0600, proprietaires du
                compte courant : aucun autre compte ne peut y ecrire.

Windows : les modes POSIX n'y existent pas, le portail 4 y est donc sans objet
(les ACL ne sont pas inspectees) ; les portails 1 a 3 s'appliquent.

LE PORTAIL 0 EST UN TEST, PAS DU CODE
-------------------------------------
`tests/test_garde_cache_disque.py` echoue si `set_cache` ou `LlamaDiskCache`
apparait dans `aura/` : aucune autre ligne du produit ne peut ouvrir ce
chemin, et l'invariant se verifie a chaque execution de la suite.
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger("aura.garde_cache")

PORTE_ACTIVATION = "AURA_CACHE_DISQUE"
PORTE_COFFRE = "AURA_COFFRE_CACHE"
MARQUEUR = ".coffre"
_ATTR_INSTALLE = "_aura_garde_cache"


class Refus(PermissionError):
    """Un portail est reste ferme ; `portail` dit lequel et pourquoi."""

    def __init__(self, portail: str, raison: str):
        super().__init__(f"[{portail}] {raison}")
        self.portail = portail
        self.raison = raison


def _dossier_coffre(dossier: str | os.PathLike[str] | None) -> Path:
    """Portail 2 : le coffre est absolu, reel, et ce n'est pas un lien."""
    brut = dossier if dossier is not None else os.environ.get(PORTE_COFFRE)
    if not brut:
        raise Refus("coffre", f"{PORTE_COFFRE} n'est pas pose : aucun coffre "
                              "n'est designe")
    chemin = Path(brut)
    if not chemin.is_absolute():
        raise Refus("coffre", f"{chemin} n'est pas un chemin absolu — un "
                               "coffre relatif suivrait le dossier courant")
    if chemin.is_symlink():
        raise Refus("coffre", f"{chemin} est un lien symbolique : il peut "
                               "designer un dossier ecrit par un autre compte")
    canonique = chemin.resolve()
    if not canonique.is_dir():
        raise Refus("coffre", f"{canonique} n'est pas un dossier existant")
    return canonique


def _lire_marqueur(canonique: Path) -> str:
    """Portail 3a : le coffre a ete prepare, et il le dit."""
    marqueur = canonique / MARQUEUR
    if not marqueur.is_file():
        raise Refus("marqueur", f"{canonique} n'est pas un coffre prepare "
                                f"(aucun {MARQUEUR}) — lancer : python -m "
                                "aura.garde_cache_disque --preparer "
                                f"\"{canonique}\"")
    try:
        return marqueur.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise Refus("marqueur", f"{marqueur} illisible : {e}") from e


def _verifier_permissions(canonique: Path) -> None:
    """Portail 4 : personne d'autre ne peut ecrire dans le coffre."""
    if not hasattr(os, "getuid"):
        return                      # Windows : pas de modes POSIX
    uid = os.getuid()
    for cible, attendu in ((canonique, 0o700), (canonique / MARQUEUR, 0o600)):
        vu = cible.stat()
        if vu.st_uid != uid:
            raise Refus("permissions", f"{cible} appartient a un autre compte "
                                       f"(uid {vu.st_uid})")
        mode = stat.S_IMODE(vu.st_mode)
        if mode & 0o077:
            raise Refus("permissions", f"{cible} est en {oct(mode)} : les "
                                       "autres comptes peuvent y ecrire "
                                       f"(attendu {oct(attendu)})")


def verifier(dossier: str | os.PathLike[str] | None = None) -> Path:
    """Ouvre les portails dans l'ordre et rend le coffre canonique.

    Leve `Refus` au premier portail ferme : l'appelant ne peut donc pas
    atteindre le cache disque par accident.
    """
    if os.environ.get(PORTE_ACTIVATION, "0") != "1":
        raise Refus("activation", f"{PORTE_ACTIVATION}=1 absent : le cache "
                                  "disque est ferme")
    canonique = _dossier_coffre(dossier)
    ecrit = _lire_marqueur(canonique)
    if os.path.normcase(ecrit) != os.path.normcase(str(canonique)):
        raise Refus("marqueur", f"{canonique / MARQUEUR} designe {ecrit!r} : "
                                "coffre deplace ou recopie")
    _verifier_permissions(canonique)
    return canonique


def autorise(dossier: str | os.PathLike[str] | None = None) -> bool:
    """Variante silencieuse : `True` seulement si TOUS les portails sont ouverts."""
    try:
        verifier(dossier)
    except (Refus, OSError) as e:
        LOG.debug("[garde] cache disque refuse : %s", e)
        return False
    return True


def preparer(dossier: str | os.PathLike[str]) -> Path:
    """Prepare un coffre : dossier 0700 + marqueur 0600 portant son chemin."""
    chemin = Path(dossier)
    if not chemin.is_absolute():
        raise Refus("coffre", f"{chemin} n'est pas un chemin absolu")
    chemin.mkdir(parents=True, exist_ok=True, mode=0o700)
    canonique = chemin.resolve()
    marqueur = canonique / MARQUEUR
    if os.name != "nt":
        os.chmod(canonique, 0o700)
    marqueur.write_text(str(canonique) + "\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(marqueur, 0o600)
    return canonique


# ── Le garde lui-meme ──────────────────────────────────────────────────

def _est_cache_disque(cache: Any) -> bool:
    """Reconnait le cache disque par sa classe, sans en instancier un."""
    if type(cache).__name__ == "LlamaDiskCache":
        return True
    try:
        from llama_cpp import LlamaDiskCache
    except ImportError:
        return False
    return isinstance(cache, LlamaDiskCache)


def installer(classe_llama: Any = None) -> bool:
    """Enveloppe `Llama.set_cache` : un cache disque doit passer les portails.

    Idempotent, et sans effet si `llama_cpp` est absent (venv de CI, machine
    sans le moteur) : l'import du cerveau ne doit jamais echouer a cause d'un
    garde. Le cache MEMOIRE (`LlamaRAMCache`) n'est pas concerne : il ne
    deserialise rien, c'est le seul que `aura` utilise.
    """
    if classe_llama is None:
        try:
            from llama_cpp import Llama as classe_llama
        except ImportError:
            LOG.debug("[garde] llama_cpp absent : rien a garder")
            return False
    if getattr(classe_llama, _ATTR_INSTALLE, False):
        return True
    # Une classe SANS `set_cache` n'a aucun point d'entree vers le cache
    # disque : il n'y a rien a garder, et refuser de s'installer serait une
    # panne gratuite (les doubles de test de llama_cpp n'exposent pas tout).
    original = getattr(classe_llama, "set_cache", None)
    if original is None:
        LOG.debug("[garde] %r n'expose pas set_cache : rien a garder",
                  classe_llama)
        return False

    @functools.wraps(original)
    def set_cache(self: Any, cache: Any) -> Any:
        if _est_cache_disque(cache):
            coffre = verifier()
            LOG.warning("[garde] cache disque de llama.cpp ouvert sur %s — "
                        "PYSEC-2026-2447 redevient atteignable", coffre)
        return original(self, cache)

    classe_llama.set_cache = set_cache
    setattr(classe_llama, _ATTR_INSTALLE, True)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aura.garde_cache_disque",
        description="Portails devant le cache disque de llama.cpp "
                    "(avis PYSEC-2026-2447).")
    groupe = ap.add_mutually_exclusive_group(required=True)
    groupe.add_argument("--preparer", type=Path, metavar="DOSSIER",
                        help="cree/repare un coffre (0700 + marqueur 0600)")
    groupe.add_argument("--verifier", action="store_true",
                        help="dit si les portails sont ouverts (code 1 sinon)")
    a = ap.parse_args(argv)

    if a.preparer is not None:
        canonique = preparer(a.preparer)
        print(f"coffre prepare : {canonique}")
        print(f"  {PORTE_ACTIVATION}=1")
        print(f'  {PORTE_COFFRE}="{canonique}"')
        return 0

    try:
        canonique = verifier()
    except (Refus, OSError) as e:
        print(f"refus : {e}", file=sys.stderr)
        return 1
    print(f"ouvert : {canonique}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
