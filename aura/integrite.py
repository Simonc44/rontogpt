"""Empreintes SHA-256 des fichiers charges comme des POIDS ou du CODE.

POURQUOI
--------
`diskCache`, `joblib.load` et `pickle` executent du code a la
deserialisation, et un GGUF remplace change le cerveau sans que rien ne le
dise. Un attaquant qui peut ecrire dans `modeles/` n'a besoin d'aucune CVE :
il remplace le fichier, et le processus le charge avec ses propres droits.
On scelle donc chaque fichier sensible avec son empreinte, dans un voisin
`<nom>.sha256`, et un fichier qui ne correspond plus est REFUSE (le cerveau
ne se charge pas, le classifieur n'est pas deserialise).

LIMITE, ECRITE NOIR SUR BLANC
-----------------------------
C'est une protection contre un fichier MODIFIE ou ECRASE — telechargement
interrompu, mauvaise version, malware qui substitue un modele — pas contre un
attaquant qui tourne deja sous ton compte : celui-la peut reecrire le sceau
en meme temps que le fichier. Seule une cle qui ne vit pas sur la machine
fermerait ce cas, et `aura/signature.py` le fait pour le `.aef`, pas pour les
modeles.

REGLAGES
--------
    AURA_INTEGRITE=0        aucune verification (chargement plus rapide)
    AURA_INTEGRITE=strict   un fichier NON SCELLE est refuse
    (defaut)                non scelle : avertissement ; sceau present : verifie

USAGE
-----
    python -m aura.integrite --sceller modeles/*.gguf modeles/routeur/*.joblib
    python -m aura.integrite --verifier modeles/cerveau.gguf
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

LOG = logging.getLogger("aura.integrite")

#: Suffixe du sceau : `cerveau.gguf` -> `cerveau.gguf.sha256`.
SCEAU = ".sha256"
#: Taille de lecture : 1 Mio. Assez gros pour ne pas payer 1 000 appels sur
#: un modele de 1 Go, assez petit pour ne pas doubler la memoire du fichier.
BLOC = 1 << 20


class IntegriteRompue(RuntimeError):
    """Le fichier ne correspond plus a son sceau."""


def empreinte(chemin: str | Path) -> str:
    """SHA-256 hexadecimal du fichier, lu par blocs (jamais en entier)."""
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(BLOC), b""):
            h.update(bloc)
    return h.hexdigest()


def chemin_du_sceau(chemin: str | Path) -> Path:
    p = Path(chemin)
    return p.with_name(p.name + SCEAU)


def lire_sceau(chemin: str | Path) -> str | None:
    """Empreinte enregistree, ou None (pas de sceau / illisible)."""
    sceau = chemin_du_sceau(chemin)
    try:
        texte = sceau.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not texte:
        return None
    return texte.split()[0].lower()


def sceller(chemin: str | Path) -> Path:
    """Ecrit `<fichier>.sha256` (format `sha256sum`) et rend son chemin."""
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"fichier introuvable : {chemin}")
    sceau = chemin_du_sceau(chemin)
    sceau.write_text(f"{empreinte(chemin)}  {chemin.name}\n", encoding="utf-8")
    LOG.info("[integrite] sceau ecrit : %s", sceau)
    return sceau


def _desactive() -> bool:
    return os.environ.get("AURA_INTEGRITE", "1").strip() == "0"


def _strict() -> bool:
    return os.environ.get("AURA_INTEGRITE", "").strip().lower() == "strict"


def verifier(chemin: str | Path, strict: bool | None = None) -> str:
    """Controle un fichier avant de le charger.

    Rend ``"desactive"``, ``"absent"`` (averti) ou ``"ok"``. Leve
    `IntegriteRompue` si l'empreinte differe, et aussi si le fichier n'est
    pas scelle en mode strict.
    """
    chemin = Path(chemin)
    if _desactive():
        return "desactive"
    attendu = lire_sceau(chemin)
    if attendu is None:
        if strict if strict is not None else _strict():
            raise IntegriteRompue(
                f"{chemin} n'est pas scelle : AURA_INTEGRITE=strict refuse "
                "un fichier dont l'empreinte n'a jamais ete enregistree "
                f"(sceller : python -m aura.integrite --sceller \"{chemin}\")")
        LOG.warning("[integrite] %s n'est pas scelle : aucune verification "
                    "possible (python -m aura.integrite --sceller)", chemin)
        return "absent"
    vu = empreinte(chemin)
    if vu != attendu:
        raise IntegriteRompue(
            f"{chemin} ne correspond plus a son sceau : attendu {attendu}, "
            f"lu {vu}. Le fichier a ete modifie ou remplace ; il n'est PAS "
            "charge.")
    return "ok"


def charger_joblib(chemin: str | Path):
    """`joblib.load` precedee de la verification du sceau.

    Un `.joblib` est du pickle : le deserialiser execute du code. C'est le
    meme risque que le cache disque de llama.cpp, mais dans NOS fichiers.
    """
    import joblib

    chemin = Path(chemin)
    verifier(chemin)
    return joblib.load(chemin)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aura.integrite",
        description="Scelle et verifie les fichiers sensibles (SHA-256).")
    groupe = ap.add_mutually_exclusive_group(required=True)
    groupe.add_argument("--sceller", nargs="+", type=Path, metavar="FICHIER",
                        help="ecrit <fichier>.sha256")
    groupe.add_argument("--verifier", nargs="+", type=Path, metavar="FICHIER",
                        help="code 1 si un fichier ne correspond plus")
    a = ap.parse_args(argv)

    if a.sceller:
        for chemin in a.sceller:
            try:
                sceau = sceller(chemin)
            except OSError as e:
                print(f"[integrite] {e}", file=sys.stderr)
                return 1
            print(f"scelle : {sceau}")
        return 0

    code = 0
    for chemin in a.verifier:
        try:
            etat = verifier(chemin, strict=False)
        except (IntegriteRompue, OSError) as e:
            print(f"[integrite] ROMPU : {e}", file=sys.stderr)
            code = 1
            continue
        print(f"{'ok' if etat == 'ok' else etat:>8} : {chemin}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
