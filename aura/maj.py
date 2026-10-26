"""Verification des mises a jour (GitHub Releases Simonc44/aura-1b).

Philosophie : l'auto-update ne JAMAIS remplacer les fichiers tout seul
(un update qui casse une install locale est pire qu'une maj oubliee) —
on detecte, on affiche, on ouvre la page officielle. L'installation reste
le fait de l'utilisateur (git pull, ou relance de l'installeur).

Usage :
    python -m aura.maj            # verifie et affiche
    python -m aura.maj --json     # sortie machine
    python -m aura.maj --ouvrir   # ouvre la page de la release dans le navigateur

Aucune interface ne l'appelle au demarrage depuis le retrait de la GUI web :
se lance a la demande, ou depuis un front via ``--json``.
AURA_MAJ=0 desactive toute verification reseau.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.error
import urllib.request
import webbrowser

from . import config

LOG = logging.getLogger(__name__)

# Nom ACTUEL du depot. L'ancien (`aura-1b`) repond encore par une redirection
# 301, mais l'auto-mise a jour ne doit pas dependre d'une redirection : elle
# lit la derniere release, et c'est elle qui decide de dire « a jour ».
_REPO = "Simonc44/rontogpt"
_URL = f"https://api.github.com/repos/{_REPO}/releases/latest"
_AGENT = "aura-maj"


def _norme(valeur: str) -> tuple[int, ...]:
    """'v0.2.1' / '0.2' -> (0, 2, 1) ; non numerique -> (0,) (comparaison sure)."""
    chiffres = re.findall(r"\d+", valeur or "")
    return tuple(int(c) for c in chiffres) if chiffres else (0,)


def derniere_release(timeout: float = 4.0) -> dict | None:
    """Release publiee sur GitHub, ou None (aucune release / hors-ligne)."""
    req = urllib.request.Request(
        _URL, headers={"Accept": "application/vnd.github+json",
                       "User-Agent": _AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode() or "{}")
    tag = data.get("tag_name")
    if not tag:
        return None
    return {"tag": tag, "url": data.get("html_url") or f"https://github.com/{_REPO}/releases",
            "notes": (data.get("body") or "").strip()[:400]}


def verifier(timeout: float = 4.0) -> dict:
    """{dispo, version, nouvelle?, url?, notes?, erreur?} — jamais d'exception."""
    locale = config.version()
    if os.environ.get("AURA_MAJ", "1") == "0":
        return {"dispo": False, "version": locale, "desactive": True}
    try:
        rel = derniere_release(timeout)
    except (urllib.error.URLError, OSError, ValueError) as e:
        # hors-ligne : silencieux, une IA locale doit marcher sans internet
        return {"dispo": False, "version": locale, "erreur": str(e)}
    if not rel:
        return {"dispo": False, "version": locale,
                "erreur": "aucune release publiee"}
    dispo = _norme(rel["tag"]) > _norme(locale)
    return {"dispo": dispo, "version": locale, "nouvelle": rel["tag"],
            "url": rel["url"], "notes": rel["notes"]}


def main(argv=None) -> int:
    # console Windows en cp1252 : les notes de release contiennent des
    # emojis -> on ne veut jamais d'UnicodeError sur l'affichage
    try:
        import sys
        sortie = getattr(sys, "stdout", None)
        if sortie is not None and hasattr(sortie, "reconfigure"):
            sortie.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        # Sans reconfigure, un emoji dans les notes de release ferait une
        # UnicodeError a l'affichage. On continue : la verification des mises
        # a jour vaut mieux qu'un plantage sur un caractere.
        LOG.debug("stdout non reconfigurable en utf-8 : %s", e)
    ap = argparse.ArgumentParser(prog="aura.maj")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    ap.add_argument("--ouvrir", action="store_true",
                    help="ouvre la page de la release si une maj est dispo")
    args = ap.parse_args(argv)

    rap = verifier()
    if args.json:
        print(json.dumps(rap, ensure_ascii=False, indent=2))
    elif rap.get("erreur") and not rap.get("dispo"):
        print(f"[maj] {rap['version']} — verification impossible "
              f"({rap['erreur']})")
    elif rap["dispo"]:
        print(f"[maj] mise a jour disponible : {rap['version']} -> "
              f"{rap['nouvelle']}\n       {rap['url']}")
        if rap.get("notes"):
            print(f"       {rap['notes'][:200]}")
        if args.ouvrir and rap.get("url"):
            webbrowser.open(rap["url"])
    else:
        print(f"[maj] {rap['version']} a jour"
              + (" (verification desactivee)" if rap.get("desactive") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
