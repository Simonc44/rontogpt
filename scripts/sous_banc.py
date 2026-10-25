"""Sous-banc de PROVENANCE — qui a repondu, mesure en secondes.

NB (7 oct. 2026) : la liste des tiers « exacts » n'est plus ecrite a la
main, elle est DERIVEE de `aura/registre_experts.py`. Une liste recopiee
avait deja rate un expert ajoute — la mesure comptait une amelioration
comme un simple changement de couche.

Pourquoi ce fichier existe
--------------------------
`banc_100.py` prend 13,7 h et ne repond qu'a « combien de bonnes reponses ».
Il ne dit pas QUI a repondu : un changement d'ORDRE dans la cascade peut donc
degrader le systeme sans que le score bouge d'un point (et inversement). C'est
precisement ce qui rendait l'architecture intestable.

Ici on ne mesure pas la qualite, on mesure la PROVENANCE. La generation est
COUPEE (elle renvoie une sentinelle) et le reseau est COUPE (recherche vide) :

  * le banc tourne en quelques secondes et n'est pas bruite par le LLM ;
  * « resolu » = une couche a rendu la BONNE valeur sans generer ;
  * toute question qui tombe sur `llama-3.2-1b` est, par definition, non
    couverte — c'est la liste des chantiers.

Deux usages
-----------
    .venv/Scripts/python.exe scripts/sous_banc.py --baseline
        ecrit la reference (scripts/sous_banc_attendu.json). A LANCER AVANT
        tout changement d'ordre ou d'expert.

    .venv/Scripts/python.exe scripts/sous_banc.py
        compare a la reference et liste les ecarts :
          PERDU  = resolu avant, confie au LLM maintenant  (regression)
          GAGNE  = l'inverse                                (amelioration)
          CHANGE = meme couche, valeur differente           (a examiner)

Le cache semantique est ISOLE, VIDE par defaut, et le banc n'y ECRIT plus
(7 oct. 2026) : on mesure la couche deterministe, pas la memoire. Sans
cette coupure d'ecriture, le banc se repondait a lui-meme — la 2e
dissertation ressemblait a la 1re et recevait sa sentinelle en 1 ms.
`--cache <fichier>` permet de mesurer l'effet d'un cache donne (c'est ainsi
qu'on prouve que l'exact prime sur le flou) : les hits servent, les
ecritures restent coupees.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RACINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_ATTENDU = Path(__file__).resolve().parent / "sous_banc_attendu.json"
_SENTINELLE = "[GENERE]"

# Tiers qui resolvent SANS generer : tout le reste est « non couvert ».
#
# DERIVE DU REGISTRE, jamais recopie. Cette liste etait une seconde source
# de verite, et elle a rate le nouvel expert `problemes-arithmetiques` des
# sa sortie : le banc a compte 3 victoires comme « changement de couche »
# au lieu d'« amelioration ». En derivant les noms des experts EXACT et
# VERIFIE declares, ajouter un solveur ne peut plus fausser la mesure.
_TIERS_HORS_REGISTRE = {
    "constante", "constante-capacite", "niveau0-instantane",
    "extraction-api", "extraction-web", "cache-semantique",
    # sous-solveur EXACT d'`experts-speciaux` (les autres sous-solveurs
    # appellent le modele : ils sont listes plus bas)
    "tri-transitif",
}

# Etiquettes qui GENERENT (elles appellent llama_cerveau). Le solveur
# derriere peut etre exact (le SAT verifie la formalisation), la reponse
# n'est pas « sans generer » pour autant — les compter comme couvertes
# gonflerait le taux.
_TIERS_GENERATIFS = {"logique-1b+sat", "potcode-1b+sandbox", "pot-1b+ast"}


def _tiers_exacts() -> set[str]:
    """Les tiers qui resolvent sans generer — DERIVES du registre.

    A APPELER DANS main(), JAMAIS AU NIVEAU MODULE : importer `aura` ici
    charge `filtre_instantane`, qui fige `_FICHIER` depuis
    `AURA_FILTRE_CACHE` — une variable que main() ne pose qu'apres. Ce
    piege a ete paye une fois : la reference est sortie a 97/100, mesuree
    sur le VRAI cache de production au lieu du cache vide du banc (11
    questions « repondues » par la memoire du projet, pas par le systeme).
    """
    from aura import registre_experts as R
    noms = {e.nom for e in R.EXPERTS if e.genre in (R.EXACT, R.VERIFIE)}
    noms.discard("experts-speciaux")      # publie les noms de ses sous-solveurs
    return (noms | _TIERS_HORS_REGISTRE) - _TIERS_GENERATIFS


def _couper_generation() -> dict[str, int]:
    """Neutralise TOUTE generation et TOUT reseau. Renvoie les compteurs."""
    compteurs = {"llm": 0, "web": 0}

    from aura import llama_cerveau as lc

    def _generer(*a, **k):
        compteurs["llm"] += 1
        return _SENTINELLE

    lc.generer = _generer                     # type: ignore[assignment]

    # Le cache s'appuie sur les embeddings : on garde le vrai chemin
    # (`filtre_instantane`), c'est lui qu'on veut mesurer.

    from aura import serveur_lora as sl

    def _chat(*a, **k):
        compteurs["llm"] += 1
        return _SENTINELLE

    sl.chat = _chat                           # type: ignore[assignment]
    sl._actifs = lambda: False                # type: ignore[assignment]

    # RESEAU : les experts web/API ne doivent pas faire varier le banc.
    from aura import memoire_web as mw

    def _vide(*a, **k):
        compteurs["web"] += 1
        return ""

    mw.chercher = _vide                       # type: ignore[assignment]
    mw.chercher_enrichi = _vide               # type: ignore[assignment]

    # API PUBLIQUES (meteo, devises...) : detection LOCALE mais appel HTTP
    # reel des qu'elle matche. La reference mesurait 0 question servie par
    # `extraction-api` : la couper ne change donc aucune provenance et rend
    # le banc totalement hors-ligne (donc reproductible).
    from aura import apis_publiques as ap

    ap.detecte = lambda q: False              # type: ignore[assignment]
    ap.chercher = _vide                       # type: ignore[assignment]

    # CACHE DE REPONSES : le run ne doit pas pouvoir se repondre a lui-meme.
    # `AURA_FILTRE_CACHE` isole le FICHIER, mais le banc ECRIVAIT dedans a
    # chaque question — et `enregistrer` reconstruit l'index MiniLM au
    # passage. Consequence mesuree le 7 oct. 2026 : la 2e dissertation
    # « ressemblait » a la 1re (cosinus >= 0,70), elle recevait donc la
    # sentinelle [GENERE] en 1 ms, et 11 questions etaient comptees
    # « couvertes » sans avoir rien produit. La reference de 97/100 etait
    # un artefact du banc, pas du systeme. On coupe l'ECRITURE : un cache
    # passe en `--cache` sert alors ses HITS (ce qu'on veut mesurer), jamais
    # les reponses que le banc vient de fabriquer.
    from aura import filtre_instantane as f0

    f0.enregistrer = lambda q, r: None        # type: ignore[assignment]
    f0.mettre_a_jour = lambda q, r: False     # type: ignore[assignment]
    return compteurs


def _une(ia, question: str) -> tuple[str, dict]:
    """Meme contrat que bench_qualite._executer, mais SANS l'importer :
    `bench_qualite` ecrit AURA_FILTRE_CACHE des son import, ce qui
    ecraserait le cache isole de ce banc."""
    t0 = time.time()
    r = ia.executer_detaille(question)
    meta = {"latence": round(time.time() - t0, 2),
            "cerveau": str(r.get("cerveau_choisi", "-")),
            "experts": sorted(r.get("experts") or set())}
    return str(r.get("reponse", "")), meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sous_banc")
    ap.add_argument("--baseline", action="store_true",
                    help="ecrit la reference au lieu de comparer")
    ap.add_argument("--cache", metavar="CHEMIN", default="",
                    help="cache semantique a utiliser (defaut : vide)")
    ap.add_argument("--cat", metavar="CAT", default="",
                    help="ne mesurer qu'une categorie")
    ap.add_argument("--json", metavar="CHEMIN", default="",
                    help="ecrit le detail complet en JSON")
    args = ap.parse_args(argv)

    cache_temporaire = not args.cache
    vide = (Path(args.cache) if args.cache else
            _RACINE / f".sous_banc_cache_{os.getpid()}_{time.time_ns()}.jsonl")
    vide.parent.mkdir(parents=True, exist_ok=True)
    vide.touch(exist_ok=True)
    os.environ["AURA_FILTRE_CACHE"] = str(vide)

    from aura import config
    config.appliquer()
    tiers_exacts = _tiers_exacts()      # APRES le cache isole (cf. docstring)
    compteurs = _couper_generation()
    from aura.memoire_conversation import MemoireConversation
    from aura.orchestrateur import RontoGPT

    import banc_100 as B  # noqa: E402  (fournit QUESTIONS + les controles)

    ia = RontoGPT()
    lignes: list[dict] = []
    t0 = time.time()
    for index, (qid, cat, question, controle) in enumerate(B.QUESTIONS):
        if args.cat and cat != args.cat:
            continue
        # Le sous-banc mesure la couverture experte, pas la memoire longue.
        # Isoler chaque exemple empeche les questions personnelles ou les
        # sentinelles precedentes de contaminer le resultat et l'historique.
        historique = (_RACINE / f".sous_banc_{os.getpid()}_{index}.jsonl")
        ia._historique = MemoireConversation(chemin=historique)
        try:
            reponse, meta = _une(ia, question)
            ok, detail = controle(reponse, meta)     # type: ignore[operator]
        except Exception as e:
            ok, detail = False, f"exception: {type(e).__name__}: {e}"
            reponse, meta = "", {"latence": 0.0, "cerveau": "?", "experts": []}
        finally:
            for chemin in (historique, historique.with_suffix(".etat.json")):
                try:
                    chemin.unlink()
                except FileNotFoundError:
                    pass
        cerveau = str(meta.get("cerveau", "?"))
        lignes.append({
            "id": qid, "categorie": cat, "cerveau": cerveau,
            "couvert": cerveau in tiers_exacts,
            "ok": bool(ok), "mesure": detail,
            "reponse": reponse[:160],
        })
    duree = time.time() - t0

    if cache_temporaire:
        try:
            vide.unlink()
        except FileNotFoundError:
            pass

    couverts = sum(1 for ligne in lignes if ligne["couvert"])
    justes = sum(1 for ligne in lignes if ligne["ok"])
    print(f"{len(lignes)} questions en {duree:.1f} s "
          f"({compteurs['llm']} generation(s) tente(s), "
          f"{compteurs['web']} recherche(s))")
    print(f"COUVERT   : {couverts}/{len(lignes)} "
          f"({100 * couverts / max(len(lignes), 1):.0f} %)  "
          "<- resolu sans generer")
    print(f"JUSTE     : {justes}/{len(lignes)} "
          f"({100 * justes / max(len(lignes), 1):.0f} %)  "
          "<- la bonne valeur, prouvee")

    if args.baseline:
        _ATTENDU.write_text(json.dumps(lignes, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"-> reference ecrite : {_ATTENDU}")
        return 0

    if not _ATTENDU.is_file():
        print(f"pas de reference ({_ATTENDU}) : relancer avec --baseline")
        return 2

    attendu = {ligne["id"]: ligne for ligne in json.loads(
        _ATTENDU.read_text(encoding="utf-8"))}
    ecarts = {"PERDU": [], "GAGNE": [], "CHANGE": [], "VALEUR": []}
    for ligne in lignes:
        a = attendu.get(ligne["id"])
        if a is None:
            continue
        if a["couvert"] and not ligne["couvert"]:
            ecarts["PERDU"].append(ligne)
        elif ligne["couvert"] and not a["couvert"]:
            ecarts["GAGNE"].append(ligne)
        elif a["cerveau"] != ligne["cerveau"]:
            ecarts["CHANGE"].append(ligne)
        elif a["reponse"] != ligne["reponse"]:
            ecarts["VALEUR"].append(ligne)

    print("-" * 74)
    for nom, sens in (("PERDU", "resolu -> LLM  (REGRESSION)"),
                      ("GAGNE", "LLM -> resolu  (amelioration)"),
                      ("CHANGE", "meme valeur, autre couche"),
                      ("VALEUR", "meme couche, autre reponse")):
        lot = ecarts[nom]
        print(f"{nom:7s} {len(lot):3d}   {sens}")
        for ligne in lot[:12]:
            print(f"          {ligne['id']:16s} {ligne['categorie']:11s} "
                  f"{ligne['cerveau']:20s} {ligne['reponse'][:44]!r}")
        if len(lot) > 12:
            print(f"          ... et {len(lot) - 12} autre(s)")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"duree_s": round(duree, 1), "lignes": lignes,
                        "ecarts": {k: [ligne["id"] for ligne in v]
                                   for k, v in ecarts.items()}},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> {args.json}")

    regression = bool(ecarts["PERDU"] or ecarts["VALEUR"])
    return 1 if regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
