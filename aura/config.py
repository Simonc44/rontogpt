"""Configuration centralisee : aura.toml + profils (rapide / equilibre / qualite).

Pourquoi un module dedie : chaque module d'RontoGPT lit ses variables ``AURA_*``
au moment de l'IMPORT (llama_cerveau._CHEMIN_GGUF, orchestrateur._SEUIL_*
...). ``appliquer()`` doit donc tourner AVANT d'importer l'orchestrateur —
c'est le cas dans ``aura.api``, ``aura.gui_web`` et ``aura.__main__``.

Priorite (de la plus forte a la plus faible) :
    1. variable d'environnement deja posee par l'utilisateur
    2. valeur du fichier aura.toml (racine du projet)
    3. valeurs du profil selectionne
    4. defauts du code (inchanges)

Usage :
    python -m aura.config            # affiche la configuration effective
    appliquer()                      # au demarrage d'un point d'entree
"""
from __future__ import annotations

import json
import logging
import os
import tomllib
from pathlib import Path

# Declare ICI (et non plus bas) : `_lire_toml` peut etre appele pendant
# l'import du module, AVANT l'ancien emplacement de cette ligne — le logger
# aurait alors ete un nom indefini au moment precis ou on en a besoin.
logger = logging.getLogger(__name__)

_RACINE = Path(__file__).resolve().parent.parent
_CHEMIN = _RACINE / "aura.toml"

# Profils : valeurs posees SEULEMENT si l'utilisateur ne les a pas posees.
# Rapide = on coupe tout ce qui n'est pas la generation (agents = plan +
# double passe + compresseurs, verification web) et on resserre le budget.
PROFILS: dict[str, dict[str, str]] = {
    "rapide": {"AURA_AGENTS": "0", "AURA_VERIF_WEB": "0",
               "AURA_BUDGET_S": "15"},
    "equilibre": {},
    "qualite": {"AURA_AGENTS": "1", "AURA_VERIF_WEB": "1",
                "AURA_BUDGET_S": "180"},
}

# Cle TOML -> variable d'environnement. Les booleens deviennent "1"/"0",
# ce qui colle aux conventions existantes (AURA_X != 0 = actif,
# AURA_X == 0 = coupe) pour toutes les cibles de cette table.
_CLES: dict[tuple[str, str], str] = {
    ("profil", "nom"): "AURA_PROFIL",
    ("llama", "gguf"): "AURA_GGUF",
    ("llama", "chemin"): "AURA_GGUF_CHEMIN",
    ("llama", "gpu"): "AURA_GPU",
    # Verrouillage des poids en RAM : use_mlock cote in-process,
    # --load-mode mmap+mlock cote llama-server. Une seule cle pour les deux
    # chemins, sinon la MEME config donnait deux empreintes memoire selon
    # celui qui repondait.
    ("llama", "mlock"): "AURA_MLOCK",
    # Prompt systeme : "defaut" (mesure sur le 1B) ou "fr" (variante Gemini 3,
    # a reserver a un cerveau plus grand / aux adaptateurs formes dessus).
    ("llama", "systeme"): "AURA_SYSTEME",
    ("serveur", "actif"): "AURA_SERVEUR",
    ("serveur", "adresse"): "AURA_SERVEUR_ADRESSE",
    # AURA_SERVEUR_SPEC = speculative decoding COTE SERVEUR (rentable :
    # x2 mesure sur texte repetitif). AURA_SPEC (in-process) reste OFF par
    # defaut : la il ralentit (logits_all impose par llama_cpp).
    ("serveur", "spec"): "AURA_SERVEUR_SPEC",
    ("qualite", "budget_s"): "AURA_BUDGET_S",
    ("qualite", "verif_web"): "AURA_VERIF_WEB",
    ("qualite", "agents"): "AURA_AGENTS",
    # Redactions longues (dissertation, essai, analyse) : Gemini d'emblee
    # au lieu des trois blocs locaux. Opt-in explicite : le sujet quitte
    # l'ordinateur. AURA_CLOUD=non reste prioritaire.
    ("qualite", "redaction_gemini"): "AURA_GEMINI_REDACTION",
    ("qualite", "seuil_confiance"): "AURA_SEUIL_CONFIANCE",
    # Compaction automatique a 90 % (aura/contexte.py) :
    # "importance" oublie les echanges vieux ou peu importants ;
    # "recent" garde simplement les derniers tours.
    ("memoire", "compaction"): "AURA_COMPACTION",
    # Majorite d'experts (self-consistency) sur puzzles/enigmes :
    # N echantillons valides par les solveurs, vote sur la reponse.
    ("qualite", "majorite"): "AURA_MAJORITE_N",
    ("api", "port"): "AURA_API_PORT",
    ("maj", "verifier"): "AURA_MAJ",
    # Briques vectorielles (embeddings MiniLM) : encodeur partage,
    # detecteur de concepts, 2e avis flou du routeur.
    ("minilm", "actif"): "AURA_MINILM",
    ("concepts", "actif"): "AURA_CONCEPTS",
    ("concepts", "seuil"): "AURA_CONCEPTS_SEUIL",
    # Bibliotheque etendue : fichier TOML externe (vide = concepts.toml
    # a la racine du projet).
    ("concepts", "fichier"): "AURA_CONCEPTS_TOML",
    ("flou", "actif"): "AURA_FLOU",
}


def _lire_toml(chemin: Path | None = None) -> dict:
    """Fichier de configuration, ou {} (absent/corrompu = defauts du code)."""
    c = chemin or _CHEMIN
    try:
        if c.is_file():
            with open(c, "rb") as f:
                return tomllib.load(f)
    except Exception as e:
        # Absent ou corrompu : les defauts du code prennent la main. Ce n'est
        # pas une erreur, mais c'est LA premiere chose a regarder quand un
        # reglage ne s'applique pas — donc on le dit au lieu de le taire.
        logger.debug("config %s ignoree : %s", c, e)
    return {}


def _vers_env(valeur) -> str:
    if isinstance(valeur, bool):
        return "1" if valeur else "0"
    return str(valeur)


def _conflits(fichier: dict, nom: str) -> dict[str, tuple[str, str]]:
    """Variables qu'une valeur EXPLICITE du fichier ecrase alors que le profil
    selectionne les regle AUSSI. env -> (valeur du fichier, valeur du profil)."""
    regles = PROFILS.get(nom, {})
    out: dict[str, tuple[str, str]] = {}
    for (section, cle), env in _CLES.items():
        bloc = fichier.get(section)
        if not isinstance(bloc, dict) or cle not in bloc or env not in regles:
            continue
        valeur = _vers_env(bloc[cle])
        if valeur != regles[env]:
            out[env] = (valeur, regles[env])
    return out


def conflits_profil(chemin: Path | None = None,
                    profil: str | None = None) -> dict[str, tuple[str, str]]:
    """Le fichier defait-il SILENCIEUSEMENT le profil selectionne ?

    La priorite « fichier > profil » est assumee : c'est un choix explicite de
    l'utilisateur. Mais l'appliquer sans rien dire est un piege — mesure du
    8 oct. 2026 : `aura.toml` epingle `budget_s = 45` ; le profil `qualite`
    (budget 180 s, calibre pour les longues redactions) devient donc
    inoperant, et rien ne le signalait : le budget declare d'un cote et
    celui reellement applique de l'autre, sans une ligne pour le dire.

    Renvoie {} quand il n'y a rien a signaler.
    """
    fichier = _lire_toml(chemin)
    nom = (profil
           or str(((fichier.get("profil") or {}).get("nom")) or "")
           or os.environ.get("AURA_PROFIL", "")
           or "equilibre").strip().lower()
    if nom not in PROFILS:
        nom = "equilibre"
    return _conflits(fichier, nom)


def appliquer(chemin: Path | None = None,
              profil: str | None = None) -> dict[str, str]:
    """Pose les variables AURA_* du fichier + du profil. Renvoie ce qui a
    ete pose (pour affichage/logs). Les variables deja presentes dans
    l'environnement ne sont JAMAIS ecrasees (priorite utilisateur)."""
    deja = set(os.environ)                      # snapshot = choix utilisateur
    fichier = _lire_toml(chemin)
    pose: dict[str, str] = {}

    # 1) profil : argument > fichier > env > defaut
    nom = (profil
           or str(((fichier.get("profil") or {}).get("nom")) or "")
           or os.environ.get("AURA_PROFIL", "")
           or "equilibre").strip().lower()
    if nom not in PROFILS:
        nom = "equilibre"
    pose["AURA_PROFIL"] = nom
    for k, v in PROFILS[nom].items():
        pose[k] = v

    # 2) valeurs explicites du fichier (elles battent le profil)
    for (section, cle), env in _CLES.items():
        # SAUF le choix du profil lui-meme : l'etape 1 l'a deja resolu avec la
        # bonne priorite (« argument > fichier > env > defaut »). Le reecrire
        # ici faisait GAGNER le fichier sur l'argument : `--profil qualite`
        # etait silencieusement annule par `[profil] nom = "equilibre"`, en
        # contradiction avec la priorite documentee en tete de ce module.
        if env == "AURA_PROFIL":
            continue
        bloc = fichier.get(section)
        if isinstance(bloc, dict) and cle in bloc:
            pose[env] = _vers_env(bloc[cle])

    # Le fichier peut DEFAIRE le profil sans le dire : ici, on le dit. Le
    # message nomme la variable, les deux valeurs ET la sortie (retirer la
    # ligne), parce qu'un avertissement sans remede n'est qu'un bruit.
    for env, (du_fichier, du_profil) in _conflits(fichier, nom).items():
        logger.warning(
            "aura.toml ecrase le profil %s : %s = %s (profil : %s) — retire "
            "la ligne du fichier pour laisser le profil decider",
            nom, env, du_fichier, du_profil)

    for k, v in pose.items():
        if k not in deja:
            os.environ[k] = v
    return {k: os.environ.get(k, "") for k in pose}


def efficace() -> dict[str, str]:
    """Les variables AURA_* actuellement visibles (affichage / diagnostic)."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("AURA_")}


def version() -> str:
    """Version du projet, lue dans pyproject.toml (source de verite)."""
    try:
        with open(_RACINE / "pyproject.toml", "rb") as f:
            donnees = tomllib.load(f)
        # version sous [project] (PEP 621) ; fallback sur un toplevel plat
        v = (donnees.get("project") or {}).get("version") or donnees.get("version")
        if v:
            return str(v)
    except Exception as e:
        logger.debug("version illisible dans pyproject.toml (%s) : 0.0.0", e)
    return "0.0.0"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="aura.config", description="affiche la configuration effective")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    ap.add_argument("--profil", default=None,
                    choices=sorted(PROFILS), help="forcer un profil (test)")
    args = ap.parse_args(argv)
    pose = appliquer(profil=args.profil)
    conflits = conflits_profil(profil=args.profil)
    if args.json:
        print(json.dumps({"version": version(), "pose": pose,
                          "conflits_profil": conflits,
                          "environnement": efficace()},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"RontoGPT {version()} — profil : {os.environ.get('AURA_PROFIL', '?')}")
    print(f"Fichier : {_CHEMIN if _lire_toml() else _CHEMIN.name + ' (absent -> defauts)'}")
    for k, v in efficace().items():
        print(f"  {k:26s} = {v}")
    # Le profil dit une chose et le fichier une autre : a dire AVANT de
    # chercher pourquoi une reponse est trop courte.
    for env, (du_fichier, du_profil) in conflits.items():
        print(f"  [!] {env} : le fichier impose {du_fichier}, le profil "
              f"voulait {du_profil} — retire la ligne pour laisser le profil")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
