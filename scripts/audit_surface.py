"""Audit de SURFACE : quels modules du paquet servent reellement ?

Pourquoi
--------
Le projet compte ~76 modules dans `aura/`. L'etude de marche (docs/ETUDE_
MARCHE_ET_ARCHITECTURE_2026-10.md) en annonce une dizaine non atteignables et
plusieurs caches/memoires qui se recouvrent, sans les nommer. Ce script les
NOMME, a partir du code (AST), jamais d'un souvenir : il suit les imports
(y compris ceux faits dans une fonction) depuis les points d'entree du produit.

Lecture seule par defaut. Rien n'est deplace sans `--archiver --oui`.

Usage
-----
    python scripts/audit_surface.py                 # rapport dans le terminal
    python scripts/audit_surface.py --ecrire        # + docs/SURFACE.md
    python scripts/audit_surface.py --json          # sortie machine
    python scripts/audit_surface.py --archiver      # simulation du deplacement
    python scripts/audit_surface.py --archiver --oui  # deplace vraiment

Classement d'un module
----------------------
ATTEIGNABLE   importe (directement ou non) depuis un point d'entree du produit
SCRIPTS       non atteignable depuis le produit, mais importe par scripts/
TESTS         non atteignable, importe seulement par tests/ (le deplacer
              casserait ces tests : jamais deplace automatiquement)
ORPHELIN      importe par personne : candidat a l'archivage
REF-TEXTE     orphelin MAIS nomme dans une chaine du code (importlib,
              `python -m aura.x`) : a verifier a la main

Limite assumee : un import construit dynamiquement (nom calcule) n'est pas vu.
Les chaines litterales `aura.xxx` sont signalees (REF-TEXTE), pas ignorees.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

#: Points d'entree du PRODUIT (noms de modules, relatifs au paquet).
ENTREES = ("__init__", "__main__", "api", "tui", "gui",
           "superviseur", "winmain", "kernel", "coucou")

#: Un fichier de cache : `.cache_xxx.jsonl`, `.graphe_faits.jsonl`, etc.
_RE_FICHIER = re.compile(
    r"""["'](\.?[A-Za-z0-9_]*(?:cache|graphe|conversation|corrections|"""
    r"""feedback|index|vecteurs)[A-Za-z0-9_.\-]*\.(?:jsonl|json|npy|bin|"""
    r"""joblib|db|sqlite|toml))["']""")
_RE_REF_TEXTE = re.compile(r"""["'](?:python -m )?(?:aura|rontogpt)\.([a-z_][a-z0-9_]*)""")


def _modules_du_paquet(dossier: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(dossier.glob("*.py"))}


def _imports(chemin: Path, paquet: str, connus: set[str]) -> tuple[set[str], set[str]]:
    """(imports reels, references textuelles) vers des modules du paquet."""
    try:
        source = chemin.read_text(encoding="utf-8-sig")
        arbre = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set(), set()
    reels: set[str] = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            for alias in noeud.names:
                parties = alias.name.split(".")
                if parties[0] == paquet and len(parties) > 1 and parties[1] in connus:
                    reels.add(parties[1])
        elif isinstance(noeud, ast.ImportFrom):
            base = (noeud.module or "").split(".") if noeud.module else []
            if noeud.level >= 1:                   # import relatif : dans le paquet
                if not base:                       # from . import a, b
                    reels.update(a.name for a in noeud.names if a.name in connus)
                elif base[0] in connus:            # from .x import y
                    reels.add(base[0])
            elif base and base[0] == paquet:       # from aura import x / aura.x import y
                if len(base) > 1 and base[1] in connus:
                    reels.add(base[1])
                elif len(base) == 1:
                    reels.update(a.name for a in noeud.names if a.name in connus)
    textes = {m.group(1) for m in _RE_REF_TEXTE.finditer(source)} & connus
    return reels, textes - reels


def _fermeture(depart: set[str], graphe: dict[str, set[str]]) -> set[str]:
    vus: set[str] = set()
    pile = list(depart)
    while pile:
        m = pile.pop()
        if m in vus:
            continue
        vus.add(m)
        pile.extend(graphe.get(m, ()))
    return vus


def _importes_par(dossier: Path, paquet: str, connus: set[str]) -> dict[str, set[str]]:
    """module -> noms des fichiers de `dossier` qui l'importent."""
    res: dict[str, set[str]] = {}
    if not dossier.is_dir():
        return res
    for f in sorted(dossier.rglob("*.py")):
        reels, textes = _imports(f, paquet, connus)
        for m in reels | textes:
            res.setdefault(m, set()).add(f.name)
    return res


def analyser(racine: Path = RACINE, paquet: str = "aura") -> dict:
    dossier = racine / paquet
    modules = _modules_du_paquet(dossier)
    connus = set(modules)
    graphe: dict[str, set[str]] = {}
    textes: dict[str, set[str]] = {}
    for nom, chemin in modules.items():
        reels, ref = _imports(chemin, paquet, connus)
        graphe[nom] = reels - {nom}
        textes[nom] = ref - {nom}

    # le paquet `rontogpt/` (alias du produit) fait partie des entrees
    entrees = {e for e in ENTREES if e in connus}
    for f in sorted((racine / "rontogpt").glob("*.py")) if (racine / "rontogpt").is_dir() else []:
        reels, ref = _imports(f, paquet, connus)
        entrees |= reels | ref
    atteints = _fermeture(entrees, graphe)

    par_scripts = _importes_par(racine / "scripts", paquet, connus)
    par_tests = _importes_par(racine / "tests", paquet, connus)
    scripts_atteints = _fermeture(set(par_scripts), graphe)

    # qui importe qui, dans le produit seulement (pour afficher les importeurs)
    importeurs: dict[str, set[str]] = {m: set() for m in modules}
    for src, cibles in graphe.items():
        for c in cibles:
            importeurs[c].add(src)
    ref_texte: dict[str, set[str]] = {m: set() for m in modules}
    for src, cibles in textes.items():
        for c in cibles:
            ref_texte[c].add(src)

    lignes = {m: sum(1 for _ in p.open(encoding="utf-8-sig", errors="replace"))
              for m, p in modules.items()}
    classement: dict[str, str] = {}
    for m in modules:
        if m in atteints:
            classement[m] = "ATTEIGNABLE"
        elif m in scripts_atteints:
            classement[m] = "SCRIPTS"
        elif m in par_tests or any(m in _fermeture({t}, graphe) for t in par_tests):
            classement[m] = "TESTS"
        elif ref_texte[m] or textes_hors(m, racine, paquet):
            classement[m] = "REF-TEXTE"
        else:
            classement[m] = "ORPHELIN"

    # caches / memoires : fichier -> modules qui le nomment
    fichiers: dict[str, set[str]] = {}
    for nom, chemin in modules.items():
        try:
            src = chemin.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        for m in _RE_FICHIER.finditer(src):
            fichiers.setdefault(m.group(1), set()).add(nom)

    return {
        "paquet": paquet, "modules": sorted(modules), "lignes": lignes,
        "entrees": sorted(entrees), "classement": classement,
        "importeurs": {m: sorted(v) for m, v in importeurs.items()},
        "par_scripts": {m: sorted(v) for m, v in par_scripts.items()},
        "par_tests": {m: sorted(v) for m, v in par_tests.items()},
        "ref_texte": {m: sorted(v) for m, v in ref_texte.items() if v},
        "fichiers_cache": {k: sorted(v) for k, v in sorted(fichiers.items())},
    }


def textes_hors(module: str, racine: Path, paquet: str) -> bool:
    """Le module est-il nomme dans une chaine de scripts/ ou tests/ ?"""
    motif = re.compile(r"""["'](?:python -m )?%s\.%s\b""" % (paquet, re.escape(module)))
    for sous in ("scripts", "tests"):
        d = racine / sous
        if not d.is_dir():
            continue
        for f in d.rglob("*.py"):
            try:
                if motif.search(f.read_text(encoding="utf-8-sig", errors="replace")):
                    return True
            except OSError:
                continue
    return False


# -- rapport ------------------------------------------------------------------

def rapport(r: dict) -> str:
    cl = r["classement"]
    compte = {k: sum(1 for v in cl.values() if v == k)
              for k in ("ATTEIGNABLE", "SCRIPTS", "TESTS", "REF-TEXTE", "ORPHELIN")}
    out = ["# Surface du paquet `%s`" % r["paquet"], "",
           "%d modules : %s." % (len(r["modules"]),
                                 ", ".join("%d %s" % (n, k) for k, n in compte.items() if n)),
           "", "Points d'entree suivis : %s." % ", ".join(r["entrees"]), ""]
    for titre, cle, conseil in (
            ("Orphelins", "ORPHELIN", "personne ne les importe : archivables"),
            ("Texte seulement", "REF-TEXTE",
             "nommes dans une chaine (importlib, -m) : verifier a la main"),
            ("Tests seulement", "TESTS",
             "non atteignables du produit ; les deplacer casse ces tests"),
            ("Scripts seulement", "SCRIPTS", "utilises par scripts/, pas par le produit")):
        mods = [m for m in r["modules"] if cl[m] == cle]
        if not mods:
            continue
        out += ["## %s (%d) : %s" % (titre, len(mods), conseil), "",
                "| module | lignes | tests | scripts |", "|---|---|---|---|"]
        for m in mods:
            out.append("| `%s` | %d | %s | %s |" % (
                m, r["lignes"][m],
                ", ".join(r["par_tests"].get(m, [])) or "-",
                ", ".join(r["par_scripts"].get(m, [])) or "-"))
        out.append("")
    recouv = {f: ms for f, ms in r["fichiers_cache"].items() if len(ms) > 1}
    out += ["## Fichiers de cache / memoire nommes par plusieurs modules", ""]
    if recouv:
        out += ["| fichier | modules |", "|---|---|"]
        out += ["| `%s` | %s |" % (f, ", ".join(ms)) for f, ms in recouv.items()]
    else:
        out.append("Aucun fichier n'est nomme par deux modules.")
    out += ["", "## Tous les fichiers de cache / memoire", "",
            "| fichier | module(s) |", "|---|---|"]
    out += ["| `%s` | %s |" % (f, ", ".join(ms)) for f, ms in r["fichiers_cache"].items()]
    out.append("")
    return "\n".join(out)


def archiver(r: dict, racine: Path, vraiment: bool) -> list[str]:
    """Deplace les ORPHELINS vers aura/experimental/. Jamais les autres."""
    cibles = [m for m in r["modules"] if r["classement"][m] == "ORPHELIN"]
    if not cibles:
        return ["Aucun orphelin : rien a archiver."]
    dest = racine / r["paquet"] / "experimental"
    msgs = []
    if vraiment:
        dest.mkdir(exist_ok=True)
        init = dest / "__init__.py"
        if not init.exists():
            init.write_text('"""Modules sans importeur, mis de cote par '
                            'scripts/audit_surface.py."""\n', encoding="utf-8")
    for m in cibles:
        src = racine / r["paquet"] / (m + ".py")
        msgs.append("%s %s -> experimental/" % ("deplace" if vraiment else "(simulation)", src.name))
        if vraiment:
            shutil.move(str(src), str(dest / src.name))
    return msgs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--racine", default=str(RACINE))
    p.add_argument("--paquet", default="aura")
    p.add_argument("--json", action="store_true")
    p.add_argument("--ecrire", action="store_true", help="ecrit docs/SURFACE.md")
    p.add_argument("--archiver", action="store_true",
                   help="archive les ORPHELINS (simulation sans --oui)")
    p.add_argument("--oui", action="store_true", help="confirme le deplacement")
    a = p.parse_args(argv)
    racine = Path(a.racine)
    r = analyser(racine, a.paquet)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    texte = rapport(r)
    print(texte)
    if a.ecrire:
        (racine / "docs").mkdir(exist_ok=True)
        (racine / "docs" / "SURFACE.md").write_text(texte, encoding="utf-8")
        print("-> docs/SURFACE.md ecrit")
    if a.archiver:
        print("\n".join(archiver(r, racine, a.oui)))
        if not a.oui:
            print("Simulation seulement. Relance avec --oui, puis `pytest -q`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
