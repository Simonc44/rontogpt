"""Audit des avis de securite sur les versions FIGEES du verrou (`uv.lock`).

Usage :
    uv run python scripts/audit_dependances.py
    uv run python scripts/audit_dependances.py --venv .venv --verrou uv.lock

POURQUOI CE SCRIPT EXISTE : `pip-audit --no-deps -r <requirements.txt>` ne
passe PAS sur Windows pour ce projet, et ce n'est pas un probleme de
configuration. Deux erreurs successives, mesurees le 16 oct. 2026 :

1. AVEC les empreintes (`uv export --frozen --no-emit-project -o req.txt`) :
   `--require-hashes` s'active des qu'UNE ligne en porte une, et
   `llama-cpp-python==0.3.36` n'en a pas — il vient de l'index
   `abetlen/llama-cpp-python`, pas de PyPI :
   « Hashes are required in --require-hashes mode, but they are missing ».

2. SANS les empreintes (`uv export ... --no-hashes`) : pip installe le paquet
   pour lire ses metadonnees, extrait le sdist de `llama-cpp-python` et
   depasse la limite de chemin de Windows :
   « [Errno 2] No such file or directory: '...\\ChatAttachmentsListItemMcpResource.svelte' ».

`--no-deps` ne dispense pas de cette installation : il empeche seulement
l'exploration des DEPENDANCES, pas la preparation des paquets demandes.

Ce script fait donc exactement ce que fait le job `securite` de la CI : il
audite le dossier site-packages du venv — les versions figees, telles
qu'INSTALLEES — avec le meme pip-audit epingle et le meme avis ignore. Rien a
installer, rien a compiler, donc rien qui puisse depasser un chemin Windows.

Avant l'audit, il compare le venv au verrou. Un venv en retard a annonce deux
avis `urllib3` (2.7.0 installe contre 2.8.0 fige) qui n'existent pas dans le
verrou, et c'est le genre de faux positif qu'on lit comme un vrai probleme :
l'ecart est donc annonce AVANT les avis, avec les deux versions, puis l'audit
a lieu — il porte sur ce qui est INSTALLE, ce qui est justement l'interet.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent

#: Meme epingle que la CI : deux versions de pip-audit ne rendent pas
#: forcement le meme verdict, et comparer des resultats differents ne veut
#: rien dire.
_PIP_AUDIT = "pip-audit==2.10.1"

#: 1. `diskcache` (tire par llama-cpp-python) : aucune version corrective
#:    n'est publiee, et le retirer voudrait dire retirer le moteur du cerveau.
#: 2. `opentelemetry-api` (tire par google-adk) : le correctif existe (1.45.1)
#:    mais google-adk 2.13.0 pinne la 1.42.1, donc `uv lock
#:    --upgrade-package` ne peut pas la faire monter.
#: Le jour ou l'un des deux avis tombe, retirer sa ligne d'ici ET du job de CI
#: (`.github/workflows/tests.yml`) : les deux listes doivent rester identiques,
#: sinon le local et la CI ne rendent plus le meme verdict.
_AVIS_IGNORES = ("PYSEC-2026-2447", "CVE-2026-114603")


def interpreteur(venv: Path) -> Path:
    """Python du venv, ou FileNotFoundError (avec le chemin regarde)."""
    for relatif in ("Scripts/python.exe", "bin/python"):
        candidat = venv / relatif
        if candidat.is_file():
            return candidat
    raise FileNotFoundError(f"aucun interpreteur dans {venv} (essaye "
                            f"Scripts/python.exe et bin/python)")


def site_packages(python: Path) -> Path:
    """Dossier des paquets installes, CALCULE par l'interpreteur du venv.

    Jamais recopie (`.venv/Lib/site-packages`) : un chemin fige redevient faux
    des que la version de Python change, et pip-audit sur un dossier
    INEXISTANT ne trouve aucune vulnerabilite — il rend un audit vert qui n'a
    rien regarde.
    """
    out = subprocess.run(
        [str(python), "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _canonique(nom: str) -> str:
    """PEP 503 : `Foo.Bar_baz` et `foo-bar-baz` sont le meme paquet."""
    return re.sub(r"[-_.]+", "-", nom).lower()


def versions_verrou(chemin: Path) -> dict[str, set[str]]:
    """{nom: versions figees} lu dans `uv.lock`.

    Un meme nom peut apparaitre PLUSIEURS fois (numpy, scipy : une entree par
    jeu de marqueurs), donc un ENSEMBLE de versions, pas une seule. Et toutes
    les entrees du verrou ne sont pas installees ici : celles qui ne visent
    que Python 3.15 n'existent pas sur un 3.12. C'est pourquoi l'appelant ne
    compare que les paquets PRESENTS DES DEUX COTES.
    """
    versions: dict[str, set[str]] = {}
    nom: str | None = None
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        if ligne.startswith("[[package]]"):
            nom = None
        elif ligne.startswith("name = "):
            nom = _canonique(ligne.split('"')[1])
        elif nom and ligne.startswith("version = "):
            versions.setdefault(nom, set()).add(ligne.split('"')[1])
    return versions


def versions_installees(sp: Path) -> dict[str, str]:
    """{nom: version} des distributions REELLEMENT presentes dans `sp`."""
    vues: dict[str, str] = {}
    for dist in metadata.distributions(path=[str(sp)]):
        # `dist.name` et non `dist.metadata["Name"]` : la lecture par cle d'un
        # champ ABSENT est devenue un DeprecationWarning sur 3.12 (et une
        # exception ensuite), et une .dist-info sans nom arrive.
        nom = dist.name
        if nom:
            vues[_canonique(nom)] = dist.version or ""
    return vues


def ecarts(figees: dict[str, set[str]], installees: dict[str, str]
           ) -> list[tuple[str, list[str], str]]:
    """(nom, versions du verrou, version installee) pour chaque desaccord."""
    return [(nom, sorted(figees[nom]), vu)
            for nom, vu in sorted(installees.items())
            if nom in figees and vu not in figees[nom]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="audit_dependances",
        description="Audit des avis de securite sur les versions du verrou.")
    ap.add_argument("--venv", type=Path, default=_RACINE / ".venv",
                    help="venv a auditer (defaut : .venv)")
    ap.add_argument("--verrou", type=Path, default=_RACINE / "uv.lock",
                    help="verrou de reference (defaut : uv.lock)")
    a = ap.parse_args(argv)

    try:
        python = interpreteur(a.venv)
    except FileNotFoundError as e:
        print(f"[audit] {e}", file=sys.stderr)
        print("[audit] creer l'environnement : uv sync", file=sys.stderr)
        return 2

    sp = site_packages(python)
    if not sp.is_dir():
        print(f"[audit] site-packages introuvable : {sp}", file=sys.stderr)
        return 2

    figees = versions_verrou(a.verrou)
    installees = versions_installees(sp)
    desaccords = ecarts(figees, installees)
    if desaccords:
        print(f"[audit] ATTENTION : le venv n'est pas le verrou"
              f" ({len(desaccords)} ecart(s)) :", file=sys.stderr)
        for nom, attendues, vue in desaccords:
            print(f"[audit]   {nom} : verrou {' / '.join(attendues)}"
                  f" — installe {vue}", file=sys.stderr)
        print("[audit] les avis ci-dessous portent donc sur ces versions"
              " INSTALLEES, pas sur le verrou. Pour auditer le verrou :"
              " uv sync --frozen", file=sys.stderr)
    else:
        print(f"[audit] {len(figees)} paquets figes, {len(installees)} installes"
              f" — venv conforme au verrou")
    print(f"[audit] site-packages : {sp}")
    commande = ["uvx", "--from", _PIP_AUDIT, "pip-audit", "--path", str(sp)]
    for avis in _AVIS_IGNORES:
        commande += ["--ignore-vuln", avis]
    return subprocess.run(commande).returncode


if __name__ == "__main__":
    raise SystemExit(main())
